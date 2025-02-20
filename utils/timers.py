# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import time
import logging

from loguru import logger


class cuda_timer:
    def __init__(self, message, active):
        self.active = active
        self.message = message
        self.start = None
        self.end = None

    def __enter__(self):
        if not self.active:
            return
        torch.cuda.synchronize()
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, tracebac):
        if not self.active:
            return
        torch.cuda.synchronize()
        logger.info(f'CUDA TIMER {time.perf_counter() - self.start:.3f}s {self.message.upper()}')


class cpu_timer:
    def __init__(self, message, active=True):
        self.active = active
        self.message = message
        self.start = None
        self.end = None

    def __enter__(self):
        if not self.active:
            return
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, tracebac):
        if not self.active:
            return
        logger.info(f'CPU  TIMER {time.perf_counter() - self.start:.3f}s {self.message.upper()}')
