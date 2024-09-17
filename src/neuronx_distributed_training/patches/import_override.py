# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types

import torch

if torch.__version__.startswith("2"):
    string_classes = str
    inf = torch.inf
else:
    string_classes = None
    inf = None


# conditionally modify the import
def modify_torch_six_import():
    if string_classes is not None:
        try:
            if "torch._six" not in sys.modules:
                # Create and add dummy module to sys.modules
                six_module = types.ModuleType("torch._six")
                six_module.string_classes = string_classes
                six_module.inf = inf
                sys.modules["torch._six"] = six_module
        except Exception as e:
            raise RuntimeError(f"Failed to override torch._six import: {e}")
