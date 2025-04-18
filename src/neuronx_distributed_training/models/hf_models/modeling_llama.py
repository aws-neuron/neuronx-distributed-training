# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

""" PyTorch LLaMA model."""
import math
import os
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch_xla.core.xla_model as xm
from packaging import version
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LLAMA_INPUTS_DOCSTRING,
    LLAMA_START_DOCSTRING,
)
from transformers.models.llama.modeling_llama import LlamaAttention as LlamaAttentionHF
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as LlamaDecoderLayerHF,
)
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM as LlamaForCausalLMHF,
)
from transformers.models.llama.modeling_llama import (
    LlamaForSequenceClassification,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding
)
from transformers.models.llama.modeling_llama import LlamaMLP as LlamaMLPHF
from transformers.models.llama.modeling_llama import LlamaModel as LlamaModelHF
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm as LlamaRMSNormHF
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as LlamaRotaryEmbeddingHF
from transformers.models.llama.modeling_llama import (
    repeat_kv,
    rotate_half,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

import neuronx_distributed.parallel_layers.utils as neuronx_dist_utils
from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func
from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear
from neuronx_distributed.parallel_layers import mappings
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_size,
)

def _init_normal(std, w):
    return nn.init.normal_(w, mean=0.0, std=std)


if version.parse(torch.__version__) >= version.parse("2.1"):
    from torch_xla.utils.checkpoint import checkpoint

    checkpoint_method = checkpoint
else:
    checkpoint_method = torch.utils.checkpoint.checkpoint

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(LlamaRMSNormHF):
    def __init__(self, hidden_size, eps=1e-6, sequence_parallel_enabled=False):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(hidden_size, eps=eps)
        setattr(self.weight, "sequence_parallel_enabled", sequence_parallel_enabled)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype

        if os.environ.get("XLA_DOWNCAST_BF16", None) == "1":
            hidden_states = hidden_states.to(torch.double)

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class ActivationMultiplyMLP(torch.nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.act_fn = ACT2FN[config.hidden_act]
        self.split_size = config.intermediate_size // get_tensor_model_parallel_size()
    
    def forward(self, x):
        gate_proj, up_proj = x.split(self.split_size, dim=2)
        intermediate_states = self.act_fn(gate_proj) * up_proj
        return intermediate_states


class LlamaMLP(LlamaMLPHF):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        init_method = partial(_init_normal, config.initializer_range)
        self.gate_up_proj = ColumnParallelLinear(
            self.hidden_size,
            2 * self.intermediate_size,
            stride=2,
            bias=False,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
            reduce_dtype = self.config.reduce_dtype,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=init_method,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
            reduce_dtype = self.config.reduce_dtype,
        )
        self.activation_multiply = ActivationMultiplyMLP(config)

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            intermediate_states = self.activation_multiply(self.gate_up_proj(x))
            down_proj = self.down_proj(intermediate_states)

        return down_proj


class CoreAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_states, key_states, value_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        kv_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        causal_mask = torch.triu(torch.ones((1, 1, q_len, kv_seq_len), device="xla"), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill_(causal_mask, -10000.0)

        dtype = torch.double if os.environ.get("XLA_DOWNCAST_BF16", None) == "1" else torch.float32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=dtype).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output


class LlamaAttention(LlamaAttentionHF):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.use_flash_attention = config.use_flash_attention
        self.lnc = config.lnc

        if not hasattr(self.config, "kv_shared_group_size"):
            self.config.kv_shared_group_size = 1

        if not hasattr(self.config, "qkv_linear"):
            self.config.qkv_linear = False

        self.config.fuse_qkv = getattr(self.config, "fuse_qkv", True)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self._init_rope()

        init_method = partial(_init_normal, config.initializer_range)
        if self.config.fuse_qkv and self.num_heads == self.num_key_value_heads and not self.config.qkv_linear:
            self.qkv_proj = ColumnParallelLinear(
                self.hidden_size,
                3 * self.num_heads * self.head_dim,
                stride=3,
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=self.config.sequence_parallel_enabled,
                reduce_dtype = self.config.reduce_dtype,
            )
            self.split_size = self.num_heads * self.head_dim // get_tensor_model_parallel_size()
        elif self.config.qkv_linear:
            self.qkv_proj = GQAQKVColumnParallelLinear(
                self.hidden_size,
                [self.num_heads * self.head_dim, self.num_key_value_heads * self.head_dim],
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=self.config.sequence_parallel_enabled,
                kv_size_multiplier=self.config.kv_shared_group_size,
                fuse_qkv=self.config.fuse_qkv,
                reduce_dtype = self.config.reduce_dtype,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=self.config.sequence_parallel_enabled,
                reduce_dtype = self.config.reduce_dtype,
            )
            self.k_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=self.config.sequence_parallel_enabled,
                reduce_dtype = self.config.reduce_dtype,
            )
            self.v_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=self.config.sequence_parallel_enabled,
                reduce_dtype = self.config.reduce_dtype,
            )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=init_method,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
            reduce_dtype = self.config.reduce_dtype,
        )
        self.num_heads = neuronx_dist_utils.divide(config.num_attention_heads, get_tensor_model_parallel_size())
        self.num_key_value_heads = neuronx_dist_utils.divide(
            config.num_key_value_heads * self.config.kv_shared_group_size, get_tensor_model_parallel_size()
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.core_attn = CoreAttention()
    
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(dim=self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    dim=self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    dim=self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert use_cache is False, "KV-Cache flow is not fully supported"
        bsz, q_len, _ = hidden_states.size()

        if self.config.sequence_parallel_enabled:
            q_len, bsz, _ = hidden_states.size()
            q_len = q_len * get_tensor_model_parallel_size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            if (
                self.config.fuse_qkv
                and self.num_heads == self.num_key_value_heads
                and self.config.kv_shared_group_size == 1
            ):
                qkv_states = self.qkv_proj(hidden_states)
                query_states, key_states, value_states = qkv_states.split(self.split_size, dim=2)
            elif self.config.qkv_linear:
                query_states, key_states, value_states = self.qkv_proj(hidden_states)
            else:
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

        if self.config.sequence_parallel_enabled:
            query_states = query_states.view(q_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            key_states = key_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(1, 2, 0, 3)
            value_states = value_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(1, 2, 0, 3)
        else:
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = (
            nki_flash_attn_func(query_states, key_states, value_states, self.lnc)
            if self.use_flash_attention
            else self.core_attn(query_states, key_states, value_states)
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        if self.config.sequence_parallel_enabled:
            attn_output = attn_output.permute(2, 0, 1, 3)
            attn_output = attn_output.reshape(q_len, bsz, self.hidden_size // get_tensor_model_parallel_size())
        else:
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size // get_tensor_model_parallel_size())

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(LlamaDecoderLayerHF):
    def __init__(self, config: LlamaConfig):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, sequence_parallel_enabled=config.sequence_parallel_enabled
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, sequence_parallel_enabled=config.sequence_parallel_enabled
        )


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaModelHF):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        init_method = partial(_init_normal, config.initializer_range)
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, init_method=init_method,
            sequence_parallel_enabled=config.sequence_parallel_enabled
        )
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, sequence_parallel_enabled=config.sequence_parallel_enabled
        )

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            pass

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        # if attention_mask is None:
        #     attention_mask = torch.ones(
        #         (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        #     )
        # attention_mask = self._prepare_decoder_attention_mask(
        #     attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        # )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = checkpoint_method(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaForCausalLMHF):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size

        init_method = partial(_init_normal, config.initializer_range)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=config.sequence_parallel_enabled,
            reduce_dtype = self.config.reduce_dtype,
        )
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
            if self.config.sequence_parallel_enabled:
                logits = logits.transpose(0, 1).contiguous()

        if os.environ.get("XLA_DOWNCAST_BF16", None) == "1":
            logits = logits.double()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = parallel_cross_entropy
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))

            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            loss = torch.mean(loss)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class LlamaRotaryEmbedding(LlamaRotaryEmbeddingHF):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        nn.Module.__init__(self)

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed