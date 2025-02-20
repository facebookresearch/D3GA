# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import torch as th
import torch.nn.functional as F

logger: logging.Logger = logging.getLogger("care.models.cal")


class ParamHolder(th.nn.Module):
    def __init__(
        self,
        param_shape: Tuple[int, ...],
        key_list: Sequence[str],
        init_value: Union[None, bool, float, int, th.Tensor] = None,
    ) -> None:
        super().__init__()
        self.logger: logging.Logger = logging.getLogger("care.strict.utils.torch.ParamHolder")

        if isinstance(param_shape, int):
            param_shape = (param_shape,)
        self.key_list: Sequence[str] = sorted(key_list)
        shp = (len(self.key_list),) + param_shape
        self.params = th.nn.Parameter(th.zeros(*shp))

        if init_value is not None:
            self.params.data[:] = init_value

    def state_dict(self, *args: Any, saving: bool = False, **kwargs: Any) -> Dict[str, Any]:
        sd = super().state_dict(*args, **kwargs)
        if saving:
            assert "key_list" not in sd
            sd["key_list"] = self.key_list
        return sd

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, **kwargs: Any
    ) -> th.nn.modules.module._IncompatibleKeys:
        # Note: Mapping is immutable while Dict is mutable. According to pyre ErrorCode[14],
        # the type of state_dict must be Mapping or supertype of Mapping to keep consistent
        # with the overrided function in its superclass.
        sd = dict(state_dict)
        if "key_list" not in sd:
            self.logger.warning("Missing key list list in state dict, only checking params shape.")
            assert sd["params"].shape == self.params.shape
            sd["key_list"] = self.key_list

        matching_kl = sd["key_list"] == self.key_list
        if strict:
            self.logger.warning("Attempting to load from mismatched key lists.")
        assert sd["params"].shape[1:] == self.params.shape[1:]

        if not matching_kl:
            src_kl = sd["key_list"]
            new_kl = sorted(set(self.key_list) | set(src_kl))
            new_shp = (len(new_kl),) + tuple(self.params.shape[1:])
            new_params = th.zeros(*new_shp, device=self.params.device)
            for f in self.key_list:
                new_params[new_kl.index(f)] = self.params[self.key_list.index(f)]
            upd = 0
            new = 0
            for f in src_kl:
                new_params[new_kl.index(f)] = sd["params"][src_kl.index(f)]
                if f in self.key_list:
                    upd += 1
                else:
                    new += 1
            self.logger.info(f"Updated {upd} keys ({100*upd/len(self.key_list):0.2f}%), added {new} new keys.")

            self.key_list = new_kl
            sd["params"] = new_params
            self.params = th.nn.Parameter(new_params)
        del sd["key_list"]
        return super().load_state_dict(sd, strict=strict, **kwargs)

    def to_idx(self, *args: Any) -> th.Tensor:
        if len(args) == 1:
            keys = args[0]
        else:
            keys = zip(*args)

        return th.tensor(
            [self.key_list.index(k) for k in keys],
            dtype=th.long,
            device=self.params.device,
        )

    def from_idx(self, idxs: th.Tensor) -> List[str]:
        return [self.key_list[idx] for idx in idxs]

    def forward(self, idxs: th.Tensor) -> th.Tensor:
        return self.params[idxs]


def scale_hook(grad: Optional[th.Tensor], scale: float) -> Optional[th.Tensor]:
    if grad is not None:
        grad = grad * scale
    return grad


class CalBase(th.nn.Module):
    def name_to_idx(self, cam_names: Sequence[str]) -> th.Tensor:
        ...


class ColorCalib(CalBase):
    def __init__(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        cameras,
        # pyre-fixme[2]: Parameter must be annotated.
        identity_camera,
        gs_lrscale: float = 1e0,
        col_lrscale: float = 1e-1,
    ) -> None:
        super(CalBase, self).__init__()

        if identity_camera not in cameras:
            identity_camera = cameras[0]
            logger.warning(f"Requested color-calibration identity camera not present, defaulting to {identity_camera}.")

        # pyre-fixme[4]: Attribute must be annotated.
        self.identity_camera = identity_camera
        # pyre-fixme[4]: Attribute must be annotated.
        self.cameras = cameras
        self.gs_lrscale = gs_lrscale
        self.col_lrscale = col_lrscale
        self.holder: ParamHolder = ParamHolder(
            # pyre-fixme[6]: For 1st param expected `Tuple[int]` but got `int`.
            3 + 3,
            cameras,
            init_value=th.FloatTensor([1, 1, 1, 0, 0, 0]),
        )

        # pyre-fixme[4]: Attribute must be annotated.
        self.identity_idx = self.holder.to_idx([identity_camera]).item()
        # pyre-fixme[4]: Attribute must be annotated.
        self.grey_idxs = [self.holder.to_idx([c]).item() for c in cameras if c.startswith("41")]

        s = th.FloatTensor([0.37, 0.52, 0.52])
        self.holder.params.data[th.LongTensor(self.grey_idxs), :3] = s

    def name_to_idx(self, cam_names: Sequence[str]) -> th.Tensor:
        return self.holder.to_idx(cam_names)

    # pyre-fixme[2]: Parameter must be annotated.
    def initialize_from_texs(self, ds) -> float:
        tex_mean = ds.tex_mean.permute(1, 2, 0)
        texs = {}
        idx = 0
        while ds[idx] is None:
            idx += 1

        for cam in self.cameras:
            samp = ds[idx, cam]
            if samp is None:
                continue

            tex = samp["tex"]
            texs[cam] = tex.permute(1, 2, 0)

        stats = {}
        for cam in texs.keys():
            t = texs[cam]
            mask = (t > 0).all(dim=2)
            t = t * ds.tex_std + tex_mean
            stats[cam] = (t[mask].mean(dim=0), t[mask].std(dim=0))

        normstats = {}
        for cam in texs.keys():
            mean, std = stats[cam]
            imean, istd = stats[self.identity_camera]
            scale = istd / std
            bias = imean - scale * mean
            normstats[cam] = (scale.clamp(max=2), bias)

        for cam, nstats in normstats.items():
            cidx = self.name_to_idx([cam])[0]
            if cidx in self.grey_idxs:
                nstats = (nstats[0] / 3, nstats[1] / 3)
            self.holder.params.data[cidx, 0:3] = nstats[0]
            self.holder.params.data[cidx, 3:6] = nstats[1]
        return len(stats.keys()) / len(ds.cameras)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith("holder.")}
        return self.holder.load_state_dict(state_dict, strict=strict)

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    def state_dict(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        destination=None,
        prefix: str = "",
        keep_vars: bool = False,
        saving: bool = False,
    ):
        sd = super(CalBase, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if saving:
            sd[prefix + "holder.key_list"] = self.holder.key_list
        return sd

    def forward(self, image: th.Tensor, cam_idxs: th.Tensor) -> th.Tensor:
        params = self.holder(cam_idxs)
        outs = []
        hook_scales = []
        for i in range(cam_idxs.shape[0]):
            idx = cam_idxs[i]
            img = image[i : i + 1]
            if idx == self.identity_idx:
                outs.append(img)
                hook_scales.append(1)
                continue

            w, b = params[i, :3], params[i, 3:]
            out = img * w[None, :, None, None] + b[None, :, None, None]
            outs.append(out)
            hook_scales.append(self.gs_lrscale if idx in self.grey_idxs else self.col_lrscale)

        hook_scales = th.tensor(hook_scales, device=image.device, dtype=th.float32)
        cal_out = th.cat(outs)

        if self.training and params.requires_grad:
            params.register_hook(lambda g, hs=hook_scales: scale_hook(g, hs[:, None]))
        return cal_out


class CameraPixelBias(th.nn.Module):
    def __init__(self, image_height, image_width, ds_rate, cameras) -> None:
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.cameras = cameras
        self.n_cameras = len(cameras)

        bias = th.zeros((self.n_cameras, 1, image_width // ds_rate, image_height // ds_rate), dtype=th.float32)
        self.register_parameter("bias", th.nn.Parameter(bias))

    def forward(self, idxs: th.Tensor):
        bias_up = F.interpolate(self.bias[idxs], size=(self.image_height, self.image_width), mode='bilinear')
        return bias_up
