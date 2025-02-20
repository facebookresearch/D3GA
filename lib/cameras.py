# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from torch import nn
import numpy as np

from utils.graphics_utils import getProjectionMatrix, getWorld2View2


def batch_to_camera(batch):
    return Camera(
        colmap_id=batch["camera_id"],
        R=batch["R"],
        T=batch["T"],
        FoVx=batch["FoVx"],
        FoVy=batch["FoVy"],
        image_name=str(batch["frame_id"]) + "_" + str(batch["camera_id"]),
        uid=batch["frame_id"],
        data_device="cuda",
        width=batch["width"],
        height=batch["height"],
    )


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
        width=None,
        height=None,
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = th.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = th.device("cuda")

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = th.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def ndc2pix(self, v, S):
        return ((v + 1.0) * S - 1.0) * 0.5

    def pix2ndc(self, v, S):
        return v / S * 2.0 - 1.0

    def project_points(self, points, to_screen_space=False):
        P, _ = points.shape
        ones = th.ones(P, 1, dtype=points.dtype, device=points.device)
        points_hom = th.cat([points, ones], dim=1)
        points_out = th.matmul(points_hom, self.full_proj_transform.unsqueeze(0))

        denom = points_out[..., 3:] + 0.0000001
        projected = (points_out[..., :3] / denom).squeeze(dim=0)

        if not to_screen_space:
            return projected

        x = self.ndc2pix(projected[:, 0], self.image_width)
        y = self.ndc2pix(projected[:, 1], self.image_height)
        z = points[:, 2]

        return th.stack([x, y, z], dim=-1)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = th.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
