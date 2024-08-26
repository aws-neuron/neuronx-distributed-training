# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import neuronx_distributed as nxd
import torch
import torch_xla.core.xla_model as xm
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_all_params_for_weight_decay_optimization,
    get_params_for_weight_decay_optimization,
)
from nemo.collections.nlp.modules.common.text_generation_utils import (
    generate,
    get_computeprob_response,
    get_default_length_params,
    get_default_sampling_params,
    megatron_gpt_generate,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam,
    OutputType,
    SamplingParam,
    TextGeneration,
)
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layer_norm import (
    LayerNorm as MixedFusedLayerNorm,
)
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from neuronx_distributed.modules.rms_norm import RMSNorm
from neuronx_distributed_training.models.megatron.gpt_model import GPTModel
from neuronx_distributed_training.models.megatron.transformer import (
    ParallelTransformerLayer,
)

from .megatron_base_model import MegatronBaseModel


class ModelModule:
    def __init__(self, cfg, padded_vocab_size, is_resuming_from_checkpoint=False):
        self.padded_vocab_size = padded_vocab_size
        self.config = cfg
        self.is_resuming_from_checkpoint = is_resuming_from_checkpoint

    def build_model(self, nxd_config):
        leaf_module_cls = [MixedFusedLayerNorm.__name__, RMSNorm.__name__]
        nxd_config["pipeline_config"].update(
            {
                "transformer_layer_cls": ParallelTransformerLayer,
                "output_loss_value_spec": (True, False) if self.config.model.get("save_logits", False) else (True),
                "input_names": ["input_ids", "position_ids", "labels", "loss_mask"],
                "leaf_module_cls": leaf_module_cls,
            }
        )
        return nxd.initialize_parallel_model(nxd_config, self.model_provider_func)

    def model_provider_func(self):
        moe_config = self.config.model.get("moe", {})
        num_moe_experts = moe_config.get('num_experts', 1)
        moe_top_k = moe_config.get('top_k', 1)
        return GPTModel(
            vocab_size=self.padded_vocab_size,
            hidden_size=self.config.model.hidden_size,
            max_position_embeddings=self.config.model.max_position_embeddings,
            num_layers=self.config.model.num_layers,
            num_attention_heads=self.config.model.num_attention_heads,
            apply_query_key_layer_scaling=self.config.model.get("apply_query_key_layer_scaling", True),
            kv_channels=self.config.model.get("kv_channels", None),
            ffn_hidden_size=self.config.model.ffn_hidden_size,
            num_tokentypes=0,
            parallel_output=True,
            init_method_std=self.config.model.get("init_method_std", 0.02),
            use_scaled_init_method=self.config.model.get("use_scaled_init_method", True),
            fp16_lm_cross_entropy=self.config.model.get("fp16_lm_cross_entropy", False),
            resume_from_checkpoint=self.is_resuming_from_checkpoint,
            use_cpu_initialization=self.config.model.get("use_cpu_initialization", False),
            hidden_dropout=self.config.model.get("hidden_dropout", 0.1),
            attention_dropout=self.config.model.get("attention_dropout", 0.0),
            ffn_dropout=self.config.model.get("ffn_dropout", 0.0),
            precision=self.config.model.get("precision", 32),
            fp32_residual_connection=self.config.model.get("fp32_residual_connection", False),
            activations_checkpoint_granularity=self.config.model.get("activations_checkpoint_granularity", None),
            activations_checkpoint_method=self.config.model.get(
                "activations_checkpoint_method", "uniform"
            ),  # Change the default to uniform
            activations_checkpoint_num_layers=self.config.model.get("activations_checkpoint_num_layers", 1),
            activations_checkpoint_layers_per_pipeline=self.config.model.get("activations_checkpoint_layers_per_pipeline", None),
            normalization=self.config.model.get("normalization", "layernorm"),
            layernorm_epsilon=self.config.model.get("layernorm_epsilon", 1e-5),
            bias_activation_fusion=self.config.model.get("bias_activation_fusion", False),  # gpu based fusions are not supported
            bias_dropout_add_fusion=self.config.model.get(
                "bias_dropout_add_fusion", False
            ),  # gpu based fusions are not supported
            share_embeddings_and_output_weights=self.config.model.get("share_embeddings_and_output_weights", True),
            position_embedding_type=self.config.model.get("position_embedding_type", "learned_absolute"),
            rotary_percentage=self.config.model.get("rotary_percentage", 1.0),
            activation=self.config.model.get("activation", "gelu"),
            bias=self.config.model.get("has_bias", True),
            transformer_block_type=self.config.model.get("transformer_block_type", "pre_ln"),
            masked_softmax_fusion=self.config.model.get("masked_softmax_fusion", True),
            gradient_accumulation_fusion=self.config.model.get("gradient_accumulation_fusion", False),
            persist_layer_norm=self.config.model.get("persist_layer_norm", False),
            sequence_parallel=self.config.distributed_strategy.get("sequence_parallel", False),
            use_emha=self.config.model.get("use_emha", False),
            save_logits=self.config.model.get("save_logits", False),
            position_interpolation_factor=self.config.model.get("positition_interpolation_factor", 1.0),
            position_freq_base=self.config.model.get("position_freq_base", 10000),
            position_abf_factor=self.config.model.get("position_abf_factor", 1),
            num_kv_heads=self.config.model.get("num_kv_heads", None),
            sliding_window=self.config.model.get("sliding_window_size", None),
            expert_model_parallel_size = self.config.distributed_strategy.get('expert_model_parallel_size', 1),
            token_shuffle_group_size = self.config.distributed_strategy.get('token_shuffle_group_size', 1),
            num_moe_experts = num_moe_experts,
            moe_top_k = moe_top_k,
            moe_frequency = moe_config.get('frequency', 1),
            moe_dropout = moe_config.get('dropout', 0.0),
            moe_sinkhorn_iterations = moe_config.get('sinkhorn_iterations', 30.0),
            moe_sinkhorn_tol = moe_config.get('sinkhorn_tol', None),
            moe_routing_algorithm = moe_config.get('routing_algorithm', 'top_k'),
            moe_router_activation = moe_config.get('router_activation', 'sigmoid'),
            moe_capacity_factor = moe_config.get('capacity_factor', num_moe_experts / moe_top_k),
            output_router_logits = moe_config.get('output_router_logits', False),
            router_aux_loss_coef = moe_config.get('router_aux_loss_coef', 0.0),
            normalize_top_k_affinities = moe_config.get('normalize_top_k_affinities', True if moe_top_k > 1 else False)
        )


class MegatronGPTModel(MegatronBaseModel):
    """
    Megatron GPT pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        # this prevents base constructor from initializing tokenizer
        self.tokenizer = None
        super().__init__(cfg, trainer=trainer, no_lm_init=True)

        if self.trainer.precision != 32:
            raise ValueError("precision must be in 32")

    def build_model(self):
        self.model_module = ModelModule(
            self.config, self.trainer.datamodule.padded_vocab_size, self.trainer.is_resuming_from_checkpoint()
        )
        return self.model_module.build_model(self.nxd_config)

    def setup_optimizer_param_groups(self):
        if self.config.model.get("do_layer_norm_weight_decay", False):
            self._optimizer_param_groups = get_all_params_for_weight_decay_optimization(self.model)
        else:
            self._optimizer_param_groups = get_params_for_weight_decay_optimization(self.model)

    def get_batch_iterator(self, batch):
        """Create a list of microbatches from a list of local minibatches.

        This function creates a list of `k`th microbatches from a list of local minibatches.
        `a local minibatch` consists of `global_batch_size / data_parallel_size` samples.
        """
        from torch_xla.distributed.parallel_loader import MpDeviceLoader

        micro_batch_size = self.config.data.micro_batch_size
        all_batches = []
        for k in range(self.num_microbatches):
            start = k * micro_batch_size
            end = start + micro_batch_size
            microbatch = list()
            for x in batch:
                size = len(x)
                assert size > start and size >= end, "size issue microbatch"
                microbatch.append(x[start:end])
            assert len(microbatch) > 0, "Microbatch lenght less than 0"
            all_batches.append(microbatch)
        return MpDeviceLoader(all_batches, xm.xla_device())

    def forward_backward_step(self, batch, is_training=False):
        device = xm.xla_device()
        running_loss = torch.zeros(1, device=device)
        if parallel_state.get_pipeline_model_parallel_size() == 1:
            batch_for_pipeline = self.get_batch_iterator(self.trainer.datamodule.process_global_batch(batch))
            total_batches = len(batch_for_pipeline)
            for batch in batch_for_pipeline:
                tokens, labels, loss_mask, _, position_ids = batch
                with torch.autocast(enabled=self.config.precision.get("type") == "autocast", dtype=torch.bfloat16, device_type="cuda"):
                    output = self.model(
                        tokens.to(device),
                        position_ids.to(device),
                        labels=labels.to(device),
                        loss_mask=loss_mask.to(device),
                    )
                loss = output[0] if self.save_logits else output
                # Want to run the loss in fp32 so that the division and sum are not affect by dp degree
                if os.environ.get("XLA_DOWNCAST_BF16", None) == "1":
                    loss = loss.to(torch.float64)
                loss = loss / total_batches
                if is_training:
                    loss.backward()
                running_loss += loss.detach()
        else:
            tokens, labels, loss_mask, _, position_ids = self.trainer.datamodule.process_global_batch(batch)
            fwd_bwd_fn = self.model.run_train if is_training else self.model.run_eval
            with torch.autocast(enabled=self.config.precision.get("type") == "autocast", dtype=torch.bfloat16, device_type="cuda"):
                output = fwd_bwd_fn(
                    input_ids=tokens,
                    position_ids=position_ids,
                    labels=labels,
                    loss_mask=loss_mask,
                )
            if parallel_state.get_pipeline_model_parallel_rank() == (
                parallel_state.get_pipeline_model_parallel_size() - 1
            ):
                running_loss = output[0].detach() if self.save_logits else output.detach()
        xm.mark_step()
        torch.distributed.all_reduce(running_loss, group=parallel_state.get_data_parallel_group())
        loss_mean = running_loss / torch.distributed.get_world_size(group=parallel_state.get_data_parallel_group())
        return loss_mean

    def init_weights(self, module):
        """
        Re-init weights after partition
        Referred from HF transformers https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L690
        """
        # Last else should always call super().init_weights() to allow initializing
        # pre-defined layers.
        if isinstance(module, RMSNorm):
            module.reset_parameters()
        elif isinstance(module, MixedFusedLayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1)
        else:
            super().init_weights(module)
