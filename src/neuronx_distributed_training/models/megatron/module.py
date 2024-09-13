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

"""Megatron Module"""

import torch
from nemo.utils import logging
from neuronx_distributed.parallel_layers import parallel_state
from torch.autograd import Variable
from torch.nn.parameter import Parameter

_FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor, torch.cuda.BFloat16Tensor)


def param_is_not_shared(param):
    return not hasattr(param, "shared") or not param.shared


class MegatronModule(torch.nn.Module):
    """Megatron specific extensions of torch Module with support
    for pipelining."""

    def __init__(self, share_token_embeddings=True):
        super(MegatronModule, self).__init__()

        self.share_token_embeddings = share_token_embeddings

    def word_embeddings_weight(self):
        # if self.pre_process:
        if hasattr(self, "language_model"):
            return self.language_model.embedding.word_embeddings.weight
        elif hasattr(self, "encoder_embedding"):
            return self.encoder_embedding.word_embeddings.weight
        elif hasattr(self, "decoder_embedding"):
            return self.decoder_embedding.word_embeddings.weight
        else:
            raise ValueError(
                "Pre_process is True, but no embedding is found on this rank. Looked for language_model.embedding, encoder_embedding, and decoder_embedding"
            )

    def position_embeddings_weight(self):
        # if self.pre_process:
        if hasattr(self, "language_model"):
            return self.language_model.embedding.position_embeddings.weight
        elif hasattr(self, "encoder_embedding"):
            return self.encoder_embedding.position_embeddings.weight
        elif hasattr(self, "decoder_embedding"):
            return self.decoder_embedding.position_embeddings.weight
        else:
            raise ValueError(
                "Pre_process is True, but no embedding is found on this rank. "
                "Looked for language_model.embedding, encoder_embedding, and decoder_embedding"
            )

    def encoder_relative_position_embeddings_weight(self):
        if hasattr(self, "encoder_relative_position_embedding"):
            return self.encoder_relative_position_embedding.relative_position_embedding.weight
        else:
            raise ValueError(
                "No encoder_relative_position_embedding found on this rank. "
                "Looking for encoder_relative_position_embedding.relative_position_embedding.weight"
            )

    def initialize_word_embeddings(self, init_method, vocab_size, hidden_size):
        return

    def sync_initial_word_embeddings(self):
        if torch.distributed.is_initialized():
            if parallel_state.is_rank_in_embedding_group():
                torch.distributed.all_reduce(
                    self.word_embeddings_weight().data, group=parallel_state.get_embedding_group()
                )
        else:
            logging.warning(
                "WARNING! Distributed processes aren't initialized, so "
                "word embeddings in the last layer are not synchronized. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )

    def sync_initial_position_embeddings(self):
        # Ensure that the encoder first stage and decoder first have the same
        # initial position embedding parameter values.
        # NOTE: We don't currently support T5 with the interleaved schedule.
        if (
            parallel_state.is_rank_in_position_embedding_group()
            and parallel_state.get_pipeline_model_parallel_split_rank() is not None
        ):
            # TODO: Support tokentype embedding.
            # self.language_model.embedding.cuda()
            position_embeddings = self.position_embeddings_weight()
            torch.distributed.all_reduce(position_embeddings.data, group=parallel_state.get_position_embedding_group())

    def state_dict_for_save_checkpoint(self, destination=None, prefix="", keep_vars=False):
        """Use this function to override the state dict for
        saving checkpoints."""
        return self.state_dict(destination, prefix, keep_vars)

    def sync_initial_encoder_relative_position_embeddings(self):
        # Ensure that all encoder RPE stages have the same weights.
        if parallel_state.is_rank_in_encoder_relative_position_embedding_group():
            position_embeddings = self.encoder_relative_position_embeddings_weight()
            torch.distributed.all_reduce(
                position_embeddings.data, group=parallel_state.get_encoder_relative_position_embedding_group()
            )


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val`
    #is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_float16(val, float16_converter):
    """Convert fp32 `val` to fp16/bf16"""

    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _FLOAT_TYPES):
            val = float16_converter(val)
        return val

    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""

    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)