# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any

import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed_training.utils import get_attribute_from_cfg
from lightning.pytorch.core.datamodule import LightningDataModule
import logging
from torch_xla import runtime

class BaseDataModule(LightningDataModule):
    def __init__(self, cfg, trainer):
        super().__init__()
        self.config = cfg
        self.trainer = trainer
        self.dp_size = runtime.world_size() / (
            self.config.distributed_strategy.get("tensor_model_parallel_size") * self.config.distributed_strategy.get("pipeline_model_parallel_size") * get_attribute_from_cfg(self.config, "context_parallel_size", 1)
        )
        self.num_microbatches = int(
            self.config.data.global_batch_size / (self.config.data.micro_batch_size * self.dp_size)
        )

    def setup(self, stage=None):
        super().setup(stage)
        resume_checkpoint_path = self.trainer.ckpt_path
        self.init_consumed_samples = (
            self._extract_consumed_samples_from_ckpt(resume_checkpoint_path) if resume_checkpoint_path else 0
        )

    def compute_consumed_samples(self, steps_since_resume=0):
        consumed_samples = (
            self.init_consumed_samples
            + steps_since_resume * self.dp_size * self.config.data.micro_batch_size * self.num_microbatches
        )
        return int(consumed_samples)

    def _extract_consumed_samples_from_ckpt(self, ckpt_path):
        try:
            init_consumed_samples = int(float(re.findall(r"consumed_samples\=([0-9]+.[0-9]+)", ckpt_path)[0]))
        except (ValueError, TypeError, IndexError):
            logging.warning("Cannot parse the checkpoint file to get the consumed samples. assume it is zero.")
            init_consumed_samples = 0

        return init_consumed_samples

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
        When using pipeline parallelism, we need the global batch to remain on the CPU,
        since the memory overhead will be too high when using a large number of microbatches.
        Microbatches are transferred from CPU to Device inside the pipeline parallel wrapper.
        """
        return batch
    
    def _build_vocab(self):
        """
        Manipulate vocabulary (e.g., pad vocabulary for increased performance)/
        """
        # TODO: add config to allow to disable it?
        self.padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=self.tokenizer.vocab_size,
            make_vocab_size_divisible_by=self.config.model.get("make_vocab_size_divisible_by", 128),
            tensor_model_parallel_size=self.config.distributed_strategy.get("tensor_model_parallel_size", 1),
        )

    def _vocab_size_with_padding(self, orig_vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size):
        """Pad vocab size so it is divisible by model parallel size and
        still having GPU friendly size."""

        after = orig_vocab_size
        multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
        while (after % multiple) != 0:
            after += 1
        if torch.distributed.get_rank() == 0:
            logging.info(
                f"Padded vocab_size: {after}, original vocab_size: {orig_vocab_size}, dummy tokens: {after - orig_vocab_size}."
            )
        return after
    
    def get_batch_length(self, batch):
        raise NotImplementedError
    
    def process_global_batch(self, global_batch, global_batch_size=None):
        raise NotImplementedError


