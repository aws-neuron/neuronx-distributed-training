# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import queue
import time
from neuronx_distributed.utils.utils import hardware
from torch_neuronx.utils import get_platform_target
import torch
from typing import Dict, Any


# Mapping of string representations to PyTorch data types
DTYPE_MAP = {
    'fp16': torch.float16,
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
}

def get_lnc_size(lnc):
    hardware_type = hardware(get_platform_target())
    if hardware_type == hardware.TRN2:
        if lnc is None:
            lnc = 2
    else:
        lnc = 1
    return lnc

def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string representation to PyTorch dtype."""
    return DTYPE_MAP.get(dtype_str.lower(), torch.bfloat16)

class Throughput:
    def __init__(self, moving_avg_window_size):
        self.seqs_per_iteration = None  # batch_size * world_size * grad_accum_usteps
        self.moving_avg_window_size = moving_avg_window_size
        self.moving_avg_window = queue.Queue()
        self.window_time = 0
        self.start_time = time.time()
        self.throughput_peak = 0
        self.throughput_sum = 0
        self.throughputs = []

    def set_seqs_per_iteration(self, batch_size, world_size, grad_accum_usteps):
        self.seqs_per_iteration = batch_size * world_size * grad_accum_usteps

    def get_throughput(self):
        step_time = time.time() - self.start_time
        self.start_time += step_time
        self.window_time += step_time
        self.moving_avg_window.put(step_time)
        window_size = self.moving_avg_window.qsize()
        if window_size > self.moving_avg_window_size:
            self.window_time -= self.moving_avg_window.get()
            window_size -= 1
        throughput = window_size * self.seqs_per_iteration / self.window_time
        self.throughputs.append(throughput)
        return throughput
