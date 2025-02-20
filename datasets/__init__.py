# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import cv2
import numpy as np
import copy
import importlib
from typing import Any, Dict


class AttrDict:
    def __init__(self, entries):
        self.add_entries_(entries)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        return self.__dict__.__delitem__(key)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def __getattr__(self, attr):
        if attr.startswith("__"):
            return self.__getattribute__(attr)
        return self.__dict__[attr]

    def items(self):
        return self.__dict__.items()

    def __iter__(self):
        return iter(self.items())

    def add_entries_(self, entries, overwrite=True):
        for key, value in entries.items():
            if key not in self.__dict__:
                if isinstance(value, dict):
                    self.__dict__[key] = AttrDict(value)
                else:
                    self.__dict__[key] = value
            else:
                if isinstance(value, dict):
                    self.__dict__[key].add_entries_(entries=value, overwrite=overwrite)
                elif overwrite or self.__dict__[key] is None:
                    self.__dict__[key] = value

    def serialize(self):
        return json.dumps(self, default=self.obj_to_dict, indent=4)

    def obj_to_dict(self, obj):
        return obj.__dict__

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


def load_module(module_name, class_name=None, silent: bool = False):
    module = importlib.import_module(module_name)
    return getattr(module, class_name) if class_name else module


def load_class(class_name):
    return load_module(*class_name.rsplit(".", 1))


def load_from_config(config, **kwargs):
    """Instantiate an object given a config and arguments."""
    assert "class_name" in config and "module_name" not in config
    config = copy.deepcopy(config)
    class_name = config.pop("class_name")
    object_class = load_class(class_name)
    return object_class(**config, **kwargs)


def load_opencv_calib(extrin_path, intrin_path):
    cameras = {}

    fse = cv2.FileStorage()
    fse.open(extrin_path, cv2.FileStorage_READ)

    fsi = cv2.FileStorage()
    fsi.open(intrin_path, cv2.FileStorage_READ)

    names = [fse.getNode("names").at(c).string() for c in range(fse.getNode("names").size())]

    for camera in names:
        rot = fse.getNode(f"R_{camera}").mat()
        R = fse.getNode(f"Rot_{camera}").mat()
        T = fse.getNode(f"T_{camera}").mat()
        R_pred = cv2.Rodrigues(rot)[0]
        # assert np.all(np.isclose(R_pred, R))
        K = fsi.getNode(f"K_{camera}").mat()
        cameras[camera] = {
            "Rt": np.concatenate([R, T], axis=1).astype(np.float32),
            "K": K.astype(np.float32),
        }
    return cameras


def load_smplx_params(params):
    return {k: np.array(v[0], dtype=np.float32) for k, v in params[0].items() if k != "id"}


def load_smplx_topology(data_struct) -> Dict[str, Any]:
    # TODO: compute_
    topology = {
        "vi": data_struct["f"].astype(np.int64),
        "vti": data_struct["ft"].astype(np.int64),
        "vt": data_struct["vt"].astype(np.float32),
        "n_verts": data_struct["v_template"].shape[0],
    }

    return {
        "topology": topology,
        "lbs_template_verts": data_struct["v_template"].astype(np.float32),
    }


def load_static_assets(config):
    data_struct = np.load(config.data.smplx_topology)

    n_verts = data_struct["v_template"].shape[0]

    topology = AttrDict(
        dict(
            vi=data_struct["f"].astype(np.int64),
            vt=data_struct["vt"].astype(np.float32),
            vti=data_struct["ft"].astype(np.int64),
            n_verts=n_verts,
        )
    )

    static_assets = AttrDict(
        dict(
            topology=topology,
            lbs_template_verts=data_struct["v_template"],
            smplx_path=config.smplx_dir,
        )
    )

    return static_assets
