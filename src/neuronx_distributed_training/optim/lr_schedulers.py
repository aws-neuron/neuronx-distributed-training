# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from nemo.core.config.schedulers import SchedulerParams
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import _get_linear_schedule_with_warmup_lr_lambda


class LinearAnnealingWithWarmupParams(SchedulerParams):
    warmup_steps: int = 0
    max_steps: int = 0


class LinearAnnealingWithWarmUp(LambdaLR):
    def __init__(self, optimizer, warmup_steps, max_steps, last_epoch=-1):
        lr_lambda = partial(
            _get_linear_schedule_with_warmup_lr_lambda,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
        super().__init__(optimizer, lr_lambda, last_epoch)
