# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib


def load_module(module_name, class_name=None, silent: bool = False):
    module = importlib.import_module(module_name)
    return getattr(module, class_name) if class_name else module


def load_class(class_name):
    return load_module(*class_name.rsplit(".", 1))


def instantiate(config, **kwargs):
    config = copy.deepcopy(config)
    class_name = config.pop('class_name')
    object_class = load_class(class_name)
    instance = object_class(**config, **kwargs)

    return instance
