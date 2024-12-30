# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import neuronx_distributed as nxd
import torch
from transformers import LlamaConfig

from neuronx_distributed_training.models.hf_models.modeling_llama import (
    CoreAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaRMSNorm,
    LlamaMLP
)

from .base_model import BaseHfModel


class HFLLamaModule(BaseHfModel):
    def _get_model(self):
        config = LlamaConfig.from_pretrained(self.config.model.model_config)
        config.use_cache = False
        config.return_dict = False
        config.sequence_parallel_enabled = self.config.distributed_strategy.get("sequence_parallel", False)
        config.qkv_linear = self.config.model.get("qkv_linear", False)
        config.fuse_qkv = self.config.model.get("fuse_qkv", True)
        config.kv_shared_group_size = self.config.distributed_strategy.get("kv_replicator", 1)
        config.max_position_embeddings = self.config.model.get("max_position_embeddings", config.max_position_embeddings)
        config.use_flash_attention = self.config.model.fusions.flash_attention
        if self.config.model.get('num_layers', -1) != -1:
            config.num_hidden_layers = self.config.model.get('num_layers')
        if self.config.model.get('hidden_size', -1) != -1:
            config.hidden_size = self.config.model.get('hidden_size')
        if self.config.model.get('rope_theta', -1) != -1:
            config.rope_theta = self.config.model.get('rope_theta')

        leaf_module_cls = [LlamaRMSNorm.__name__]
        if self.config.model.get("activations_checkpoint_granularity", None) == "selective":
            if self.config.model.get("activations_checkpoint_recompute_mlp", False) and self.config.model.encoder_seq_length>=8192:
                self.nxd_config["activation_checkpoint_config"] = (CoreAttention, LlamaMLP)
            else:
                self.nxd_config["activation_checkpoint_config"] = CoreAttention
        elif self.config.model.get("activations_checkpoint_granularity", None) == "full":
            self.nxd_config["activation_checkpoint_config"] = "full"
        self.nxd_config["pipeline_config"].update(
            {
                "transformer_layer_cls": LlamaDecoderLayer,
                "output_loss_value_spec": (True, False),
                "input_names": ["input_ids", "attention_mask", "labels"],
                "leaf_module_cls": leaf_module_cls,
            }
        )
        include_buffers = True
        return nxd.initialize_parallel_model(self.nxd_config, self.model_provider_func, include_buffers, config)

    def model_provider_func(self, config):
        model = LlamaForCausalLM(config)
        # Here we make sure we use the same sine and cosine matrices for all layers.
        # Making use of same tensors would make the CSE algorithm eliminate the lookup call
        # from layers, keeping only lookup from first layer.
        with torch.no_grad():
            cos, sin = self.get_sin_cos_matrix(config)
            for layer in model.model.layers:
                layer.self_attn.rotary_emb.cos_cached = cos
                layer.self_attn.rotary_emb.sin_cached = sin

        return model

    def get_sin_cos_matrix(self, config):
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(config.max_position_embeddings, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :].to(torch.float32), emb.sin()[None, None, :, :].to(torch.float32)

    def init_weights(self, module, device):
        """
        Re-init weights after partition
        Referred from HF transformers https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L690
        """
        # Last else should always call super().init_weights() to allow initializing
        # pre-defined layers.
        for key, nested_module in module._modules.items():
            if isinstance(nested_module, LlamaRotaryEmbedding):
                module._modules[key] = LlamaRotaryEmbedding(
                    nested_module.dim, nested_module.max_position_embeddings, nested_module.base, device
                    )
        if isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)
        else:
            super().init_weights(module, device)
