# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from lib.tetgen import TetGen
from lbsmodel.body_model import LBSModule
from lib.tet_mesh import TetMesh
from .blueman import Blueman
import trimesh
import torch
import numpy as np
import matplotlib.colors as mcolors
import random
import logging

from loguru import logger


def random_color_generator():
    color = mcolors.hex2color(random.choice(list(mcolors.CSS4_COLORS.values())))
    return color


class CageBlue(Blueman):
    def __init__(self, cage_config, config, assets):
        super().__init__(config, assets)

        self.inflate_cage = cage_config.inflate
        self.config = config
        self.cage_config = cage_config
        if "face_mask" in config.train:
            self.face_mask = trimesh.load(config.train.face_mask, process=False)
        self.face_to_label = np.load(os.path.join(self.src, "face_to_label.npy"))
        self.src = os.path.join(self.src, "cages", self.cage_config.name)
        self.use_lbs_only = self.config.train.get("use_lbs_only", False)
        Path(self.src).mkdir(exist_ok=True, parents=True)

        self.tetgen = TetGen(self.src)

        self.create_cage()
        self.create_tet_mesh()
        self.create_cage_model()

        # Prepare geoemtry for training
        if self.use_lbs_only:
            self.load_mesh()
            # Update points to the sampled ones and use them to recreate LBS model
            self.create_cage_model()
        else:
            self.load_tetra()
            # self.load_lbs()

    def n_garments(self):
        return len(self.config.cages.keys())

    def create_tet_mesh(self):
        self.tet_mesh = TetMesh(f"{self.src}/cage.mesh")

    def restore(self):
        if self.use_lbs_only:
            self.tet_mesh.points = self.init_points[None].clone()
            self.cage.points = self.init_points.clone()
            self.create_cage_model()

    def create_cage_model(self):
        self.lbs_module = LBSModule(
            self.assets.lbs_model_json,
            self.assets.lbs_config_dict,
            self.assets.lbs_template_verts,
            self.assets.lbs_scale,
            self.assets.global_scaling,
        ).cuda()

        self.tetra_cage = self.tet_mesh.tetra_to_trimesh(fix_normals=not self.use_lbs_only)

        cage_weights, cage_indecies, cage_to_body_vertex = self.create_cage_skin_weights(
            self.template_trimesh, self.tetra_cage
        )

        self.cage_to_body_vertex = cage_to_body_vertex
        self.lbs_module.lbs_template_verts = self.tet_mesh.points
        self.lbs_module.lbs_fn.skin_weights = cage_weights
        self.lbs_module.lbs_fn.skin_indices = cage_indecies
        self.lbs_module.lbs_fn.nr_vertices = self.tet_mesh.points.shape[1]
        self.lbs_module.lbs_template_verts = self.invert_cage_transformation(self.tet_mesh.points)

        self.lbs_weights = self.lbs_module.lbs_fn.skin_weights
        self.lbs_indices = self.lbs_module.lbs_fn.skin_indices

        # np.save(f"{self.src}/skin_weights.npy", self.lbs_weights.cpu().numpy())
        # np.save(f"{self.src}/skin_indices.npy", self.lbs_indices.cpu().numpy())

    def invert_cage_transformation(self, vertices):
        source, faces, RT, motion = self.get_star_pose()
        v = self.to_body_model_space(vertices, motion, RT, use_cage_lbs=True)
        return v

    def create_cage_skin_weights(self, template, cage):
        template_weights = self.lbs_module.lbs_fn.skin_weights
        template_indecies = self.lbs_module.lbs_fn.skin_indices
        template.vertices += template.vertex_normals * self.inflate_cage
        vertex_id = template.kdtree.query(cage.vertices)[1]
        return template_weights[vertex_id, :], template_indecies[vertex_id, :], vertex_id

    def get_canonical(self):
        motion = torch.zeros((1, 104)).cuda()
        motion[:, 41] = -0.5  # r_upleg_ry
        motion[:, 50] = -0.5  # l_upleg_ry
        motion[:, 20] = 1.0  # r_arm_ry
        motion[:, 32] = 1.0  # l_arm_ry

        return self.get(motion)

    def invert_cage_transformation(self, vertices):
        source, faces, RT, motion = self.get_star_pose(return_RT=True)
        return self.to_body_model_space(vertices, motion, RT)

    def load_lbs(self):
        B = self.lbs_indices.max().item() + 1
        P = self.lbs_indices.size(0)
        # self.rgb = torch.from_numpy(np.array([random_color_generator() for _ in range(B)])).cuda()
        np.random.seed(33)
        self.rgb = torch.from_numpy(np.array([np.random.choice(range(255), size=3) / 255 for _ in range(B)])).cuda()
        self.rgb = self.rgb[None].expand(P, -1, -1).float()
        self.rgb = torch.gather(self.rgb, 1, self.lbs_indices[..., None].expand(-1, -1, 3))

    def lbs_to_color(self, lbs):
        rgb = torch.einsum("ikj,ik->ij", self.rgb, lbs.clone().detach())
        # trimesh.PointCloud(vertices=self.canonical_vertices.cpu().numpy(), colors=(rgb * 255).cpu().numpy().astype(np.int8)).export("test.ply")
        rgb = torch.einsum("ikj,ik->ij", rgb[self.cage.tetras][self.tetra_id], self.barys)
        return rgb[None]
