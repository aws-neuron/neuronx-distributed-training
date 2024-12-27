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

"""GPT-2 model."""

import warnings

import torch
from apex.transformer.enums import AttnMaskType
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    init_method_normal,
    scaled_init_method_normal,
)
from neuronx_distributed.modules.moe.loss_function import load_balancing_loss_func
from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy

from .language_model import get_language_model
from .module import MegatronModule


def post_language_model_processing(
    lm_output,
    labels,
    get_key_value,
    parallel_output,
    forward_method_parallel_output,
    fp16_lm_cross_entropy,
    return_logits=False,
    sequence_parallel=False,
    gradient_accumulation_fusion=False,
    share_embeddings_and_output_weights=True,
):
    lm_output = lm_output.transpose(0, 1).contiguous()
    # Output.
    logits = None
    if labels is None:
        return lm_output
    else:
        if return_logits:
            logits = lm_output.clone()
        if fp16_lm_cross_entropy:
            assert lm_output.dtype == torch.half
            loss = parallel_cross_entropy(lm_output, labels)
        else:
            loss = parallel_cross_entropy(lm_output.float(), labels)

        return loss, logits


class GPTModel(MegatronModule):
    """GPT-2 Language model."""

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        num_layers,
        num_attention_heads,
        ffn_hidden_size,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        num_tokentypes=0,
        parallel_output=True,
        init_method_std=0.02,
        use_scaled_init_method=True,
        fp16_lm_cross_entropy=False,
        resume_from_checkpoint=False,
        use_cpu_initialization=False,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.0,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_granularity=None,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        activations_checkpoint_layers_per_pipeline=None,
        normalization="layernorm",
        layernorm_epsilon=1e-5,
        bias=True,
        bias_activation_fusion=True,
        bias_dropout_add_fusion=True,
        masked_softmax_fusion=True,
        activation="gelu",
        headscale=False,
        transformer_block_type="pre_ln",
        normalize_attention_scores=True,
        position_embedding_type="learned_absolute",
        rotary_percentage=1.0,
        attention_type="multihead",
        share_embeddings_and_output_weights=True,
        gradient_accumulation_fusion=False,
        persist_layer_norm=False,
        openai_gelu=False,
        sequence_parallel=False,
        reduce_amax=True,
        use_emha=False,
        multi_query_attention=False,
        save_logits=False,
        position_interpolation_factor=1.0,
        position_freq_base=10000,
        position_abf_factor=1,
        num_kv_heads=None,
        sliding_window=None,
        num_moe_experts=1,
        moe_frequency=1,
        moe_dropout=0.0,
        moe_capacity_factor=1.0,
        moe_sinkhorn_iterations=30,
        moe_sinkhorn_tol=None,
        moe_routing_algorithm="top_k",
        moe_router_activation="sigmoid",
        moe_top_k=1,
        output_router_logits=False,
        expert_model_parallel_size=1,
        router_aux_loss_coef=0.02,
        normalize_top_k_affinities=True,
    ):
        super(GPTModel, self).__init__(share_token_embeddings=share_embeddings_and_output_weights)

        if sliding_window:
            if sliding_window > max_position_embeddings:
                warnings.warn(
                    f"Warning sliding window attention value:{sliding_window} is set higher than max position embeddings value:{max_position_embeddings}."
                )
            elif sliding_window < max_position_embeddings:
                warnings.warn(f"Warning sliding window attention value:{sliding_window} is enabled.")

        self.parallel_output = parallel_output
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.sequence_parallel = sequence_parallel
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.save_logits = save_logits
        self.output_router_logits = output_router_logits
        self.num_moe_experts = num_moe_experts
        self.moe_top_k = moe_top_k
        self.save_logits = save_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), "hidden_size must be divisible by num_attention_heads if kv_channels is None"
            kv_channels = hidden_size // num_attention_heads

        scaled_init_method = (
            scaled_init_method_normal(init_method_std, num_layers)
            if use_scaled_init_method
            else init_method_normal(init_method_std)
        )

        self.language_model, self._language_model_key = get_language_model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            num_tokentypes=num_tokentypes,
            max_position_embeddings=max_position_embeddings,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            ffn_hidden_size=ffn_hidden_size,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            init_method=init_method_normal(init_method_std),
            scaled_init_method=scaled_init_method,
            init_method_std=init_method_std,
            resume_from_checkpoint=resume_from_checkpoint,
            use_cpu_initialization=use_cpu_initialization,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_granularity=activations_checkpoint_granularity,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            activations_checkpoint_layers_per_pipeline=activations_checkpoint_layers_per_pipeline,
            normalization=normalization,
            layernorm_epsilon=layernorm_epsilon,
            rotary_percentage=rotary_percentage,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            bias=bias,
            bias_activation_fusion=bias_activation_fusion,
            bias_dropout_add_fusion=bias_dropout_add_fusion,
            masked_softmax_fusion=masked_softmax_fusion,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            activation=activation,
            headscale=headscale,
            transformer_block_type=transformer_block_type,
            normalize_attention_scores=normalize_attention_scores,
            position_embedding_type=position_embedding_type,
            attention_type=attention_type,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            sequence_parallel=sequence_parallel,
            reduce_amax=reduce_amax,
            use_emha=use_emha,
            multi_query_attention=multi_query_attention,
            position_interpolation_factor=position_interpolation_factor,
            position_freq_base=position_freq_base,
            position_abf_factor=position_abf_factor,
            num_kv_heads=num_kv_heads,
            sliding_window=sliding_window,        
            num_moe_experts=num_moe_experts,
            moe_frequency=moe_frequency,
            moe_dropout=moe_dropout,
            moe_capacity_factor=moe_capacity_factor,
            moe_sinkhorn_iterations=moe_sinkhorn_iterations,
            moe_sinkhorn_tol=moe_sinkhorn_tol,
            moe_routing_algorithm=moe_routing_algorithm,
            moe_router_activation=moe_router_activation,
            moe_top_k=moe_top_k,
            output_router_logits=output_router_logits,
            expert_model_parallel_size=expert_model_parallel_size,
            router_aux_loss_coef=router_aux_loss_coef,
            normalize_top_k_affinities=normalize_top_k_affinities,
        )

        if self.share_embeddings_and_output_weights:
            self.language_model.output_layer.weight = self.word_embeddings_weight()

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids,
        position_ids,
        labels,
        loss_mask,
        attention_mask=None,
        token_type_ids=None,
        layer_past=None,
        get_key_value=None,
        forward_method_parallel_output=None,
        encoder_input=None,
        set_inference_key_value_memory=None,
        inference_max_sequence_len=None,
        checkpoint_activations_all_layers=None,
    ):
        # input_ids: [b, s]
        # position_ids: [b, s]
        # attention_mask: [1, 1, s, s]

        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            layer_past=layer_past,
            get_key_value=get_key_value,
            encoder_input=encoder_input,
            set_inference_key_value_memory=set_inference_key_value_memory,
            inference_max_sequence_len=inference_max_sequence_len,
            checkpoint_activations_all_layers=checkpoint_activations_all_layers,
        )

        loss, logits = post_language_model_processing(
            lm_output[0] if self.output_router_logits else lm_output,
            labels,
            get_key_value,
            self.parallel_output,
            forward_method_parallel_output,
            self.fp16_lm_cross_entropy,
            return_logits=self.save_logits,
            sequence_parallel=self.sequence_parallel,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
        )
        loss = loss.float()
        loss_mask = loss_mask.view(-1).float()
        # TODO: add nemo version here
        loss = torch.sum(loss.view(-1) * loss_mask) / loss_mask.sum()  # sequence level nll
            
        if self.output_router_logits:
            aux_loss = load_balancing_loss_func(
                lm_output[-1],
                self.num_moe_experts,
                self.moe_top_k
            )

            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss

        return (loss, logits) if self.save_logits else loss