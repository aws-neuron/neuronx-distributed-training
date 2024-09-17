# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo.core.config import AdamWParams
from nemo.core.optim import register_optimizer
from nemo.core.optim.lr_scheduler import register_scheduler
from neuronx_distributed.utils.adamw_fp32_optim_params import AdamW_FP32OptimParams

from .lr_schedulers import LinearAnnealingWithWarmUp, LinearAnnealingWithWarmupParams

register_optimizer("adamw_fp32OptState", AdamW_FP32OptimParams, AdamWParams)
register_scheduler("LinearAnnealingWithWarmUp", LinearAnnealingWithWarmUp, LinearAnnealingWithWarmupParams)
