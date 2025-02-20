# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch as th
import torch.nn as nn
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
    PerspectiveCameras,
    PointsRenderer,
    AlphaCompositor,
)
from pytorch3d.common.compat import meshgrid_ij


class PCRenderer(nn.Module):
    def __init__(self, white_background=True):
        super().__init__()
        pc_settings = PointsRasterizationSettings(image_size=512, points_per_pixel=5, radius=0.007, bin_size=0)
        self.rasterizer = PointsRasterizer(raster_settings=pc_settings)
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        self.renderer = PointsRenderer(
            rasterizer=self.rasterizer, compositor=AlphaCompositor(background_color=bg_color)
        )

    @staticmethod
    def to_cameras(f) -> PerspectiveCameras:
        K = f["K"][None].cuda().float()
        c2w = f["c2w"]
        H = int(f["crop"][-1])
        W = int(f["crop"][-2])

        Rt = th.from_numpy(np.array(c2w))[None].cuda()
        Rt = th.linalg.inv(Rt).float()

        R = Rt[:, :3, :3]
        tvec = Rt[:, :3, 3]

        # K = th.eye(3)[None].cuda().float()

        # K[:, 0, 0] = fx
        # K[:, 1, 1] = fy
        # K[:, 0, 2] = cx
        # K[:, 1, 2] = cy

        image_size = th.tensor([[H, W]]).cuda().int()

        cameras = cameras_from_opencv_projection(R, tvec, K, image_size)

        return cameras

    def resize(self, H, W):
        self.rasterizer.raster_settings.image_size = (H, W)

    def forward(self, cameras, vertices, colors=None):
        B, P, C = vertices.shape
        if colors is None:
            colors = th.tensor([[154, 205, 50]]).expand(P, -1).cuda()[None] / 255.0
        point_cloud = Pointclouds(points=vertices, features=colors)
        images = self.renderer(point_cloud, cameras=cameras)
        return images[0, ..., :3]
