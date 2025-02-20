# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lib.cameras import batch_to_camera
from utils.graphics_utils import xyz2normals
import torch.nn.functional as F
import torch as th
from kornia.geometry.depth import depth_to_normals
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

import logging

from utils.timers import cuda_timer

from loguru import logger

bg_colors = {
    "white": th.tensor([1, 1, 1], dtype=th.float32, device="cuda").float(),
    "black": th.tensor([0, 0, 0], dtype=th.float32, device="cuda").float(),
}

# We need to ad
bg_maps = {
    "white": None,
    "black": None,
}


def paste(img, crop):
    left_w, right_w, top_h, bottom_h, W, H = crop[0], crop[1], crop[2], crop[3], int(crop[4]), int(crop[5])
    if left_w > right_w:
        img = img[:, :, :W]
    else:
        img = img[:, :, -W:]
    if top_h > bottom_h:
        img = img[:, :H, :]
    else:
        img = img[:, -H:, :]

    return img


def pad_image(img, crop, h, w):
    left_w, right_w, top_h, bottom_h, W, H = crop[0], crop[1], crop[2], crop[3], crop[4], crop[5]
    left, right, up, bottom = 0, 0, 0, 0
    dx = int(abs(w - W))
    dy = int(abs(H - h))
    if left_w > right_w:
        right = dx
    else:
        left = dx
    if top_h > bottom_h:
        bottom = dy
    else:
        up = dy

    padded = F.pad(img, (left, right, up, bottom, 0, 0), "constant", 0)

    return padded


def render(batch, pkg, bg_color, colors_precomp=None, measure_time=False, solid_bg=True, fast=False, detach=[]):
    viewpoint_camera = batch_to_camera(batch)

    # Set up background color
    crop = batch["crop"]

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(batch["height"]),
        image_width=int(batch["width"]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pkg["sh_degree"] if "sh_degree" in pkg else 0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )

    cov3D_precomp, scales, rotations = None, None, None
    means3D = pkg["means3D"]
    if "cov3D_precomp" in pkg:
        cov3D_precomp = pkg["cov3D_precomp"]
    if "scales" in pkg:
        scales = pkg["scales"]
    if "rotations" in pkg:
        rotations = pkg["rotations"]
    opacities = pkg["opacities"]
    shs = pkg["shs"]
    # bg_crop = pad_image(bg_map, crop, int(batch["height"]), int(batch["width"]))

    if len(detach) > 0:
        if "position" in detach:
            means3D = means3D.detach()
        if "covariance" in detach:
            cov3D_precomp = cov3D_precomp.detach()
        if "opacity" in detach:
            opacities = opacities.detach()

    if colors_precomp is None:
        colors_precomp = pkg["rgb"]
        if shs is not None:
            colors_precomp = None
    else:
        shs = None

    screenspace_points = th.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means2D = screenspace_points

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    with cuda_timer("Rastrizing", measure_time):
        rendered_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )[0]

    return {
        "render": paste(rendered_image, crop)
    }
