# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch as th
from typing import Union
from kornia.morphology import dilation, erosion
from kornia.filters import median_blur
from kornia.filters.gaussian import gaussian_blur2d
from typing import Dict, Final, List, Optional, overload


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * th.log10(1.0 / th.sqrt(mse))


def paste(img, pkg, bg_color="white", dst=None):
    size, H, W = pkg
    C = min(img.shape)

    if th.is_tensor(img):
        if dst is None:
            if bg_color.lower() == "white":
                dst = th.ones([C, H, W]).cuda().float()
            else:
                dst = th.zeros([C, H, W]).cuda().float()
        dst[:, size[0] : size[1], size[2] : size[3]] = img
        return dst

    # I know, different shapes...
    if dst is None:
        if bg_color.lower() == "white":
            dst = np.ones([H, W, C])
        else:
            dst = np.zeros([H, W, C])

    dst[size[0] : size[1], size[2] : size[3], :] = img
    return dst


def erode_mask(alpha):
    alpha = alpha.float()

    kernel = th.ones(7, 7).cuda().float()
    alpha = dilation(alpha, kernel, engine="convolution")

    kernel = th.ones(5, 5).cuda().float()
    alpha = erosion(alpha, kernel, engine="convolution")

    return alpha


def close_holes(alpha):
    alpha = alpha.float()

    kernel = th.ones(5, 5).cuda().float()
    alpha = dilation(alpha, kernel, engine="convolution")

    kernel = th.ones(5, 5).cuda().float()
    alpha = erosion(alpha, kernel, engine="convolution")

    return alpha


def linear2color_corr_inv(img: th.Tensor, dim: int) -> th.Tensor:
    """Inverse of care.strict.utils.image.linear2color_corr.
    Removes ad-hoc 'color correction' from a gamma-corrected RGB Mugsy image
    along color channel `dim` and returns the linear RGB result."""

    gamma = 2.0
    black = 3.0 / 255.0
    color_scale = [1.4, 1.1, 1.6]

    assert img.shape[dim] == 3
    if dim == -1:
        dim = len(img.shape) - 1
    scale = th.FloatTensor(color_scale).view([3 if i == dim else 1 for i in range(img.dim())])

    img = (img + 15.0 / 255.0).pow(gamma) / (0.95 / (1 - black)) + black

    return th.clamp(img / (scale.to(img) / 1.1), 0, 1)


def linear2color_corr(img: Union[th.Tensor, np.ndarray], dim: int = -1) -> Union[th.Tensor, np.ndarray]:
    """Applies ad-hoc 'color correction' to a linear RGB Mugsy image along
    color channel `dim` and returns the gamma-corrected result."""

    if dim == -1:
        dim = len(img.shape) - 1

    gamma = 2.0
    black = 3.0 / 255.0
    color_scale = [1.4, 1.1, 1.6]

    assert img.shape[dim] == 3
    if dim == -1:
        dim = len(img.shape) - 1
    if isinstance(img, th.Tensor):
        scale = th.FloatTensor(color_scale).view([3 if i == dim else 1 for i in range(img.dim())])
        img = img * scale.to(img) / 1.1
        return th.clamp(
            (((1.0 / (1 - black)) * 0.95 * th.clamp(img - black, 0, 2)).pow(1.0 / gamma)) - 15.0 / 255.0,
            0,
            2,
        )
    else:
        scale = np.array(color_scale).reshape([3 if i == dim else 1 for i in range(img.ndim)])
        img = img * scale / 1.1
        return np.clip(
            (((1.0 / (1 - black)) * 0.95 * np.clip(img - black, 0, 2)) ** (1.0 / gamma)) - 15.0 / 255.0,
            0,
            2,
        )


@overload
def linear2srgb(img: th.Tensor, gamma: float = 2.4) -> th.Tensor:
    ...


@overload
def linear2srgb(img: np.ndarray, gamma: float = 2.4) -> np.ndarray:
    ...


def linear2srgb(img: Union[th.Tensor, np.ndarray], gamma: float = 2.4) -> Union[th.Tensor, np.ndarray]:
    if isinstance(img, th.Tensor):
        # Note: The following combines the linear and exponential parts of the sRGB curve without
        # causing NaN values or gradients for negative inputs (where the curve would be linear).
        linear_part = img * 12.92  # linear part of sRGB curve
        exp_part = 1.055 * th.pow(th.clamp(img, min=0.0031308), 1 / gamma) - 0.055
        return th.where(img <= 0.0031308, linear_part, exp_part)
    else:
        linear_part = img * 12.92
        exp_part = 1.055 * (np.maximum(img, 0.0031308) ** (1 / gamma)) - 0.055
        return np.where(img <= 0.0031308, linear_part, exp_part)


def linear2displayBatch(
    val: th.Tensor,
    gamma: float = 1.5,
    wbscale: np.ndarray = np.array([1.05, 0.95, 1.45], dtype=np.float32),
    black: float = 5.0 / 255.0,
    mode: str = "srgb",
) -> th.Tensor:
    scaling = th.from_numpy(wbscale).to(val.device)
    val = val.float() / 255.0 * scaling[None, :, None, None] - black
    if mode == "srgb":
        val = linear2srgb(val, gamma=gamma)
    else:
        val = val ** (1.0 / gamma)
    return th.clamp(val, 0, 1) * 255.0


def linear2display(
    val: th.Tensor,
    gamma: float = 1.5,
    wbscale: np.ndarray = np.array([1.05, 0.95, 1.45], dtype=np.float32),
    black: float = 5.0 / 255.0,
    mode: str = "srgb",
) -> th.Tensor:
    scaling = th.from_numpy(wbscale).to(val.device)
    val = val[None].float() * scaling[None, :, None, None] - black
    if mode == "srgb":
        val = linear2srgb(val, gamma=gamma)
    else:
        val = val ** (1.0 / gamma)
    return th.clamp(val[0], 0, 1)
