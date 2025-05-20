# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any, List, Optional, Union

# Update import
import lightning.fabric.accelerators.xla as xla
from lightning.fabric.utilities.device_parser import _check_data_type
from lightning_utilities.core.imports import RequirementCache


def _auto_device_count_patched() -> int:
    """Get the devices when set to auto."""
    return 2


_XLA_AVAILABLE = RequirementCache("torch_xla")


@functools.lru_cache(maxsize=1)
def _is_available_patched() -> bool:
    try:
        return _auto_device_count_patched() > 0 and _XLA_AVAILABLE
    except (ValueError, AssertionError, OSError):
        # XLA may raise these exceptions if it's not properly configured. This needs to be avoided for the cases
        # when `torch_xla` is imported but not used
        return False


def _parse_tpu_devices_str_patched(tpu_cores: str) -> Union[int, List[int]]:
    if tpu_cores in ("1", "2", "8", "32", "64", "128"):
        return int(tpu_cores)
    return [int(x.strip()) for x in tpu_cores.split(",") if len(x) > 0]


def _check_tpu_devices_valid_patched(tpu_cores: Any) -> bool:
    # allow 1 or 8 cores
    ### NEURON: This is the allowed config on Neuron
    if tpu_cores in (1, 2, 8, 32, 64, 128, None):
        return True

    # allow picking 1 of 8 indexes
    if isinstance(tpu_cores, (list, tuple, set)):
        has_1_tpu_idx = len(tpu_cores) == 1
        is_valid_tpu_idx = 1 <= list(tpu_cores)[0] <= 128

        is_valid_tpu_core_choice = has_1_tpu_idx and is_valid_tpu_idx
        return is_valid_tpu_core_choice

    return False


def _parse_tpu_devices_patched(tpu_cores: Optional[Union[int, str, List[int]]]) -> Optional[Union[int, List[int]]]:
    """
    Parses the tpu_cores given in the format as accepted by the
    :class:`~pytorch_lightning.trainer.Trainer`.

    Args:
        tpu_cores: An int of 1 or string '1' indicates that 1 core with multi-processing should be used
            An int 8 or string '8' indicates that all 8 cores with multi-processing should be used
            A list of ints or a strings containing a list of comma separated integers
            indicates the specific TPU core to use.

    Returns:
        A list of tpu_cores to be used or ``None`` if no TPU cores were requested

    Raises:
        MisconfigurationException:
            If TPU cores aren't 1, 8 or [<1-8>]
    """
    _check_data_type(tpu_cores)

    if isinstance(tpu_cores, str):
        tpu_cores = _parse_tpu_devices_str_patched(tpu_cores.strip())

    if not _check_tpu_devices_valid_patched(tpu_cores):
        raise TypeError("`tpu_cores` can only be 1, 2, 8, 32, 64, 128 or [<1-8>]")

    return tpu_cores

#update to XLAAccelerator
xla._parse_tpu_devices = _parse_tpu_devices_patched
xla._parse_tpu_devices_str = _parse_tpu_devices_str_patched
xla._check_tpu_devices_valid = _check_tpu_devices_valid_patched
xla.XLAAccelerator._auto_device_count = staticmethod(_auto_device_count_patched)
xla.XLAAccelerator.is_available = staticmethod(_is_available_patched)