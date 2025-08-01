# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from neuronx_distributed_training.utils.model_utils import get_param_groups_by_weight_decay
from neuronx_distributed_training.utils import get_attribute_from_cfg
from ..base import BaseModelModule
from ..base_dpo import DPOBaseModel
from ..base_orpo import ORPOBaseModel
from omegaconf import DictConfig, open_dict
from lightning.pytorch.trainer.trainer import Trainer

class BaseHfModel(BaseModelModule):
    def on_train_start(self):
        if get_attribute_from_cfg(self.config, "dpo", False):
            self.dpo_module = DPOBaseModel(self.config, self.trainer, self.model)
            self.dpo_module.on_train_start(self.trainer, self.model, self.config)
        elif get_attribute_from_cfg(self.config, "orpo", False):
            self.orpo_module = ORPOBaseModel(self.config, self.trainer, self.model)
        else:
            super().on_train_start()

    def model_fwd_calc_loss(self, batch):
        if get_attribute_from_cfg(self.config, "dpo", False):
            return self.dpo_module.model_fwd_calc_loss(self.model, batch, self.config)
        elif get_attribute_from_cfg(self.config, "orpo", False):
            return self.orpo_module.model_fwd_calc_loss(self.model, batch, self.config)
        else:
            return super().model_fwd_calc_loss(batch)
    
    def build_model(self):
        return self._get_model()

    def setup_optimizer_param_groups(self):
        no_decay = ["bias"]
        if self.config.model.get("do_layer_norm_weight_decay", False):
            no_decay.append("LayerNorm")
            no_decay.append("norm")
        self._optimizer_param_groups = get_param_groups_by_weight_decay(self.model, no_decay)

    def get_batch_length(self, batch):
        # TODO: Needs override
        return len(batch["input_ids"])

    def process_global_batch(self, global_batch, global_batch_size=None):
        """Prepares the global batch for apex fwd/bwd functions.
        Global batch is a list of micro batches.
        """
        return global_batch
