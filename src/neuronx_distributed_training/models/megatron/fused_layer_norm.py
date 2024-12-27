# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numbers

import torch
from apex.normalization import MixedFusedRMSNorm as ApexMixedFusedRMSNorm
from neuronx_distributed.parallel_layers.layer_norm import (
    LayerNorm as MixedFusedLayerNorm,
)
from torch.nn.parameter import Parameter


def get_layer_norm(hidden_size, eps=1e-5, persist_layer_norm=False, sequence_parallel=False):
    return MixedFusedLayerNorm(hidden_size, eps, sequence_parallel_enabled=sequence_parallel)


class MixedFusedRMSNorm(ApexMixedFusedRMSNorm):
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        torch.nn.Module.__init__(self)
        if "elementwise_affine" in kwargs:
            import warnings

            warnings.warn("MixedFusedRMSNorm does not support `elementwise_affine` argument")
            elementwise_affine = kwargs.pop("elementwise_affine")
            if not elementwise_affine:
                raise RuntimeError("MixedFusedRMSNorm does not support `elementwise_affine = False`")
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = True
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()
