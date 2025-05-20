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

import os
import random

import numpy as np
import torch
from nemo.utils import AppState, logging
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.random import model_parallel_xla_manual_seed

torch.cuda.set_device = lambda x: None
torch.cuda.is_available = lambda: False


def initialize_model_parallel_for_nemo(
    world_size,
    global_rank,
    local_rank,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=1,
    pipeline_model_parallel_split_rank=None,
    micro_batch_size=None,
    global_batch_size=None,
    seed=None,
):
    # updating NeMo globals
    app_state = AppState()
    app_state.global_rank = global_rank
    app_state.world_size = world_size
    app_state.local_rank = local_rank
    app_state.tensor_model_parallel_size = tensor_model_parallel_size
    app_state.context_parallel_size = context_parallel_size
    app_state.pipeline_model_parallel_size = pipeline_model_parallel_size
    app_state.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size

    app_state.tensor_model_parallel_rank = parallel_state.get_tensor_model_parallel_rank()
    app_state.pipeline_model_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    app_state.model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    app_state.data_parallel_size = parallel_state.get_data_parallel_size()
    app_state.pipeline_model_parallel_split_rank = pipeline_model_parallel_split_rank 
    app_state.virtual_pipeline_model_parallel_rank = None

    _set_random_seed(seed)

    app_state._is_megatron_initialized = True


def _set_random_seed(seed):
    """Set random seed for reproducability."""
    seed = 1234 if seed is None else seed
    if seed > 0:
        # Ensure that different pipeline stages get different seeds. Assuming 100 is the maximum
        # number of pp stages you would have.
        seed = seed + (100 * parallel_state.get_pipeline_model_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model_parallel_xla_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))
