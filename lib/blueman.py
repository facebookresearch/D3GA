# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import trimesh
from lbsmodel.body_model import LBSModule, compute_root_transform_cuda
from lib.cage import CageBase
import os
from scipy.spatial.transform import Rotation
import trimesh
import torch
import numpy as np
import logging

from loguru import logger


class Blueman(CageBase):
    def __init__(self, config, assets):
        super().__init__()
        self.lbs_module = None
        self.center_mass = None
        self.assets = assets
        self.blueman_lbs_module = LBSModule(
            self.assets.lbs_model_json,
            self.assets.lbs_config_dict,
            self.assets.lbs_template_verts,
            self.assets.lbs_scale,
            self.assets.global_scaling,
        ).cuda()
        self.src = config.assets + f"/{config.capture_id}"
        self.create_body_model()
        self.initialization()
        self.get_star_pose()  # initialize center of mass

        logger.info("BLUEMAN base geometry initialized")

    def initialization(self):
        rot180 = np.eye(4)
        x = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        y = Rotation.from_euler('y', 180, degrees=True).as_matrix()
        z = Rotation.from_euler('z', 180, degrees=True).as_matrix()
        rot180[:3, :3] = z @ y
        self.rot180 = torch.from_numpy(rot180).cuda()[None].float()

    def create_body_model(self):
        self.lbs_module = LBSModule(
            self.assets.lbs_model_json,
            self.assets.lbs_config_dict,
            self.assets.lbs_template_verts,
            self.assets.lbs_scale,
            self.assets.global_scaling,
        ).cuda()

        self.simplified_index = np.load(os.path.join(os.path.dirname(__file__), "../assets/simplified_index.npy"))
        self.body_model_simplified = trimesh.load_mesh(os.path.join(os.path.dirname(__file__), "../assets/simplified.ply")).faces
        self.body_model_simplified = torch.from_numpy(self.body_model_simplified).cuda()[None]
        self.body_template_faces = self.assets.topology.vi.cuda()
        self.body_template_vertices = self.lbs_module.lbs_template_verts / 100

    def get_star_pose_lbs(self):
        motion = torch.zeros((1, 104)).cuda()
        motion[:, 41] = -0.5  # r_upleg_ry
        motion[:, 50] = -0.5  # l_upleg_ry
        motion[:, 20] = 1.0  # r_arm_ry
        motion[:, 32] = 1.0  # l_arm_ry

        return motion

    def get_star_pose(self, simplify=False, return_RT=False):
        motion = self.get_star_pose_lbs()

        v, RT = self.get(motion, return_rt=True)

        # v = self.body_template_vertices[None]

        if simplify:
            v = v[:, self.simplified_index, ...].float()
            f = self.body_model_simplified.cuda()[None]
            return v, f, RT, motion

        if return_RT:
            return v, self.body_template_faces, RT, motion

        return v, self.body_template_faces

    def to_body_model_space(self, geom, motion, RT):
        lbs_module = self.lbs_module
        v = geom.clone()
        v -= self.center_mass
        v = self.transform(v, torch.linalg.inv(RT))
        v *= 1000.0
        lbs_module.lbs_fn.global_scale = lbs_module.global_scaling.clone()
        v = lbs_module.unpose(v, motion)
        lbs_module.lbs_fn.global_scale[:] = torch.ones_like(lbs_module.lbs_fn.global_scale)
        return v

    def skinning(self, motion, delta=None):
        lbs_module = self.lbs_module
        B = motion.shape[0]
        template = lbs_module.lbs_template_verts.expand(B, -1, -1).detach()
        if delta is not None:
            template = (template / 100.0 + delta.expand(B, -1, -1)) * 100.0  # scaling from mm to dm

        lbs_module.lbs_fn.global_scale = lbs_module.global_scaling.clone()

        geom_lbs = lbs_module.pose(motion, template)

        t_root, R_root = lbs_module.lbs_fn.compute_root_rigid_transform(motion)

        lbs_module.lbs_fn.global_scale = torch.ones_like(lbs_module.lbs_fn.global_scale)

        return geom_lbs, R_root, t_root

    def from_body_model_to_canonical(self, geom):
        geom /= 100.0
        geom += self.center_mass
        return geom

    def canonical_kpt(self, lbs_motion, kpt):
        B = kpt.shape[0]
        lbs_module = self.blueman_lbs_module
        lbs_module.lbs_fn.global_scale = lbs_module.global_scaling.clone()
        geom = lbs_module.lbs_fn(
            lbs_motion,
            scales=lbs_module.lbs_scale.expand(B, -1),
            vertices=lbs_module.lbs_template_verts.expand(B, -1, -1),
        )[0]
        r_per_v, t_per_v, unposed_geom = lbs_module.lbs_fn.unpose(lbs_motion, lbs_module.lbs_scale, geom)
        ht = t_per_v[:, 81545]  # nose vertex
        rot = r_per_v[:, 81545]
        hr = torch.inverse(rot)
        canon_kpt = torch.einsum("bxy,bvy->bvx", hr, kpt / lbs_module.lbs_fn.global_scale - ht)
        lbs_module.lbs_fn.global_scale[:] = torch.ones_like(lbs_module.lbs_fn.global_scale)

        return canon_kpt, ht, rot

    def get(self, lbs_motion, kpt=None, return_rt=None, geometry=None, delta=None):
        geom, R_root, t_root = self.skinning(lbs_motion, delta)

        if geometry is not None:
            geom = geometry

        B = geom.shape[0]
        RT = torch.eye(4)[None].expand(B, -1, -1).cuda()
        RT[:, :3, :3] = R_root
        RT[:, :3, 3] = t_root / 1000.0
        RT = torch.linalg.inv(RT @ self.rot180)

        geom = geom / 1000
        geom = self.transform(geom, RT)

        if self.center_mass is None:
            self.center_mass = -1 * torch.mean(geom, axis=1, keepdim=True).detach()

        geom = geom + self.center_mass

        if kpt is not None:
            kpt, trans, rot = self.canonical_kpt(lbs_motion, kpt)
            return geom, kpt / 1000.0, rot

        if return_rt:
            return geom, RT

        return geom

    def transfrom_cameras(self, lbs, Rt):
        # Transform the cameras according to unposed mesh which is now placed at the origin
        _, R_root, t_root = self.skinning(lbs)
        B = lbs.shape[0]
        shift = self.center_mass[0].expand(B, -1)

        t_root *= 0.001

        R_C = Rt[:, :3, :3]
        t_C = Rt[:, :3, 3] * 0.001

        A = self.homogenization(R_C, t_C)
        B = self.homogenization(R_root, t_root)
        w2c = A @ B

        w2c = w2c @ self.rot180

        c2w = torch.linalg.inv(w2c)
        c2w[:, :3, 3] += shift

        return c2w
