# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
import trimesh
from lib.tetgen import TetGen
from lib.tet_mesh import TetMesh
from .smplman import Smplman
import trimesh
import torch as th
import numpy as np
import matplotlib.colors as mcolors
import random
import logging

from loguru import logger


def random_color_generator():
    color = mcolors.hex2color(random.choice(list(mcolors.CSS4_COLORS.values())))
    return color


class CageSmpl(Smplman):
    def __init__(self, cage_config, config, assets):
        super().__init__(config, assets)

        self.inflate_cage = cage_config.inflate
        self.config = config
        self.cage_config = cage_config
        self.face_to_label = np.load(os.path.join(self.src, "face_to_label.npy"))
        self.src = os.path.join(self.src, "cages", self.cage_config.name)
        Path(self.src).mkdir(exist_ok=True, parents=True)

        self.tetgen = TetGen(self.src)

        self.create_cage()
        self.create_tet_mesh()
        self.create_cage_model()

        # Prepare geoemtry for training
        self.load_tetra()

    def n_garments(self):
        return len(self.config.cages.keys())

    def create_tet_mesh(self):
        self.tet_mesh = TetMesh(f"{self.src}/cage.mesh")

    def create_cage_model(self, use_mc_source=False):
        super().create_body_model()

        tetra_cage = self.tet_mesh.tetra_to_trimesh()
        if use_mc_source:
            tetra_cage = self.mc_source.copy()

        # Get star pose for better NN search
        f = self.lbs_module.faces_tensor.cpu().numpy()
        v = self.get_star_pose(return_Tbs=True)[0][0].cpu().numpy()
        template = trimesh.Trimesh(v, f, process=False)

        weights, self.nn_ids = self.find_nn(template, tetra_cage)

        _, T, A, bs = self.get_star_pose(return_Tbs=True)
        num_joints = self.lbs_module.J_regressor.shape[0]
        T = th.matmul(self.skin_weights, A.view(1, num_joints, 16)).view(1, -1, 4, 4)

        # Unpose the star pose to t pose
        vtn = self.unpose(th.from_numpy(tetra_cage.vertices)[None].cuda().float(), T, bs)

        self.body_template_faces = th.from_numpy(template.faces).cuda()
        self.body_template_vertices = vtn

    def get_canonical(self):
        v, f = self.get_star_pose()
        return v[0], f
