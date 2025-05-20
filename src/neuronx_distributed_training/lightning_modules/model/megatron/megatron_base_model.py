# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.utils import logging
from omegaconf import DictConfig, open_dict
from lightning.pytorch.trainer.trainer import Trainer
import sys
from neuronx_distributed_training.models.megatron.transformer import CoreAttention, NeuronSwitchMLP

from ..base import BaseModelModule

__all__ = ["MegatronBaseModel"]


class MegatronBaseModel(BaseModelModule):
    """
    Megatron base class
    It does the following things:
    1. Initialize the model parallel for nemo given the model parallel parameters.
    2. Turn on all the nvidia optimizations.
    3. If `cfg.tokenizer` is available, it loads the tokenizer and pad the vocab to the correct size for tensor model parallelism.
    4. If using distributed optimizer, configure to be compatible with
       O2-level optimizations and/or model parallelism.
    5. Perform gradient clipping: `grad_clip_pl_default` triggers the
       PyTorch Lightning default implementation, `with_distributed_adam`
       triggers the distributed optimizer's implementation,
       `megatron_amp_o2` triggers gradient clipping on the main grads,
       and otherwise gradient clipping is performed on the model grads.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer, no_lm_init=True):
        if trainer is None:
            raise ValueError("Trainer cannot be None for Megatron-based models. Please provide a PTL trainer object.")
        # this prevents base constructor from initializing tokenizer
        self.tokenizer = None
        self.with_distributed_adam = cfg.model.optim.get("name") == "distributed_fused_adam"

        super().__init__(cfg, trainer=trainer, no_lm_init=no_lm_init)

        # We only support selective and full checkpointing. Even in selective,
        # we support only attention checkpointing. Other checkpointing mechanisms
        # would be added on need basis
        activation_recompute_modules = []
        recompute_modules = self.config.model.get("activations_checkpoint_recompute", [])
        granularity = self.config.model.get("activations_checkpoint_granularity", None)

        if granularity == "selective":
            for module in recompute_modules:
                module_obj = getattr(sys.modules[__name__], module, None)
                if module_obj is not None:
                    activation_recompute_modules.append(module_obj)
        elif granularity == "full":
            activation_recompute_modules = "full"
        else:
            activation_recompute_modules = None
        self.nxd_config["activation_checkpoint_config"] = activation_recompute_modules

    def _validate_and_override_config(self):
        """Certain configurations might be incompatible or discouraged.
        We can check for them here and override if necessary.
        """

        if self.config.distributed_strategy.get("sequence_parallel", False) and self.config.distributed_strategy.get("tensor_model_parallel_size", 1) == 1:
            logging.info(
                "Sequence parallel should only be used with tensor parallel size > 1. Setting sequence parallel to False"
            )
            self.config.distributed_strategy.sequence_parallel = False

        if self.with_distributed_adam:
            raise ValueError("Distributed fused adam is not supported, please consider using Zero1")

        if not self.config.distributed_strategy.get("zero1", None):
            raise ValueError("We see accuracy issues without zero1, please consider enabling zero1")

        if self.config.model.get("megatron_amp_o2", None):
            raise ValueError(
                "Currently we don't support megatron_amp_o2, please consider using zero1 with master weights"
            )

        if self.config.model.get("gradient_accumulation_fusion", False):
            raise ValueError("gradient_accumulation_fusion is not yet supported please set to False")

        if self.config.model.get("use_emha", False):
            raise ValueError("use_emha is not yet supported please set to False")

        if self.config.distributed_strategy.get("virtual_pipeline_model_parallel_size", 1) > 1:
            assert (
                self.config.model.num_layers // self.config.distributed_strategy.pipeline_model_parallel_size
            ) % self.config.distributed_strategy.virtual_pipeline_model_parallel_size == 0, (
                "Make sure the number of model chunks is the same across all pipeline stages."
            )

        if self.config.model.get("activations_checkpoint_num_layers", 1) > 1:
            raise ValueError("Full checkpointing with more than 1 layer is not supported")

        if self.config.model.get("num_micro_batches_with_partial_activation_checkpoints", None):
            raise ValueError("num_micro_batches_with_partial_activation_checkpoints feature is not supported")

        if self.config.model.get("activations_checkpoint_layers_per_pipeline", None):
            raise ValueError("activations_checkpoint_layers_per_pipeline feature is not supported")

        if not self.config.model.get("disable_layer_norm_checkpointing", True):
            raise ValueError(
                "disable_layer_norm_checkpointing is not valid, since we don't perform checkpointing for layernorm"
            )
        if self.config.model.get("activations_checkpoint_method", None) == "block":
            raise ValueError("activations_checkpoint_method can be only uniform or null")
        if self.config.model.get("bias_dropout_add_fusion", False):
            raise ValueError("bias_dropout_add_fusion is not supported")
        if self.config.model.get("bias_activation_fusion", False):
            raise ValueError("bias_activation_fusion is not supported")
        
        moe_config = self.config.model.get("moe", {})
        if moe_config.get("moe_dropout", False):
            raise ValueError("MoE dropout is not supported yet.")
        super()._validate_and_override_config()
