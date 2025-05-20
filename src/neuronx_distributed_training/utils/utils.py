# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import queue
import time
from neuronx_distributed.utils.utils import hardware
from torch_neuronx.utils import get_platform_target
import torch
from typing import Dict, Any
from omegaconf import OmegaConf, DictConfig, ListConfig


# Mapping of string representations to PyTorch data types
DTYPE_MAP = {
    'fp16': torch.float16,
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
}

def _distributed_available():
    from lightning.fabric.accelerators.xla import XLAAccelerator
    import torch_xla.runtime as xr

    xla_available = xr.world_size() > 1 and XLAAccelerator.is_available()
    return (
        xla_available or
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )

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

def get_cast_dtype() -> torch.dtype:
    """Returns the datatype to be used for casting depending on environment variables"""

    if os.environ.get("XLA_DOWNCAST_BF16", None) == "1":
        return torch.float64
    return torch.float32

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

def get_attribute_from_cfg(cfg: Any, attr_path: str, default: Any) -> Any:
    """
    Retrieve an attribute from a nested configuration structure.

    This function searches for an attribute in a nested configuration,
    first by trying the exact path, then by searching everywhere if not found.

    Args:
        cfg (Any): The configuration object to search in.
        attr_path (str): The attribute path to search for, using dot notation.
        default (Any): The default value to return if the attribute is not found.

    Returns:
        Any: The found attribute value, or the default if not found.
    """
    def search(obj, keys, everywhere=False):
        if not keys:
            return obj

        if OmegaConf.is_config(obj):
            if isinstance(obj, DictConfig):
                if everywhere:
                    for key, value in obj.items():
                        if key == keys[-1]:
                            return value
                        result = search(value, keys, everywhere)
                        if result is not None:
                            return result
                elif keys[0] in obj:
                    return search(obj[keys[0]], keys[1:], everywhere)
            elif isinstance(obj, ListConfig):
                if everywhere:
                    for item in obj:
                        result = search(item, keys, everywhere)
                        if result is not None:
                            return result
                else:
                    try:
                        index = int(keys[0])
                        if 0 <= index < len(obj):
                            return search(obj[index], keys[1:], everywhere)
                    except ValueError:
                        pass
        elif isinstance(obj, dict):
            if everywhere:
                for key, value in obj.items():
                    if key == keys[-1]:
                        return value
                    result = search(value, keys, everywhere)
                    if result is not None:
                        return result
            elif keys[0] in obj:
                return search(obj[keys[0]], keys[1:], everywhere)
        elif isinstance(obj, (list, tuple)):
            if everywhere:
                for item in obj:
                    result = search(item, keys, everywhere)
                    if result is not None:
                        return result
            else:
                try:
                    index = int(keys[0])
                    if 0 <= index < len(obj):
                        return search(obj[index], keys[1:], everywhere)
                except ValueError:
                    pass
        return None

    keys = attr_path.split('.')
    result = search(cfg, keys) or search(cfg, keys, everywhere=True)
    return result if result is not None else default
