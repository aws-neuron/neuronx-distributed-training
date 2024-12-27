# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from neuronx_distributed_training.utils.model_utils import get_param_groups_by_weight_decay

from ..base import BaseModelModule
from ..base_dpo import DPOBaseModel
from omegaconf import DictConfig, open_dict
from pytorch_lightning.trainer.trainer import Trainer


class BaseHfModel(BaseModelModule):
    def build_model(self):
        return self._get_model()

    def setup_optimizer_param_groups(self):
        no_decay = ["bias"]
        if self.config.model.get("do_layer_norm_weight_decay", False):
            no_decay.append("LayerNorm")
        self._optimizer_param_groups = get_param_groups_by_weight_decay(self.model, no_decay)

    def get_batch_length(self, batch):
        # TODO: Needs override
        return len(batch["input_ids"])

    def process_global_batch(self, global_batch, global_batch_size=None):
        """Prepares the global batch for apex fwd/bwd functions.
        Global batch is a list of micro batches.
        """
        return global_batch
