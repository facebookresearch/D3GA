# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F


def to_device(values, device=None, non_blocking=True):
    """Transfer a set of values to the device.
    Args:
        values: a nested dict/list/tuple of tensors
        device: argument to `to()` for the underlying vector
    NOTE:
        if the device is not specified, using `th.cuda()`
    """
    if device is None:
        device = th.device("cuda")

    if isinstance(values, dict):
        return {k: to_device(v, device=device) for k, v in values.items()}
    elif isinstance(values, tuple):
        return tuple(to_device(v, device=device) for v in values)
    elif isinstance(values, list):
        return [to_device(v, device=device) for v in values]
    elif isinstance(values, th.Tensor):
        return values.to(device, non_blocking=non_blocking)
    elif isinstance(values, nn.Module):
        return values.to(device)
    else:
        return values
