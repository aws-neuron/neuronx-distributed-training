# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ._version import __version__
from .optim import * # noqa F403
from .patches import * # noqa F403
import logging

logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("torch_xla").setLevel(logging.ERROR)