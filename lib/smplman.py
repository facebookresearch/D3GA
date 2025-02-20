# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from lib.cage import CageBase
from tetra_sampler.body_model import SMPLlayer

from pathlib import Path

import trimesh
import torch as th
import numpy as np
from tetra_sampler.lbs import batch_rodrigues

from utils.mesh_utils import save_mesh, subdivide_loop

from loguru import logger


class Smplman(CageBase):
    def __init__(self, config, assets, densify_template=True):
        super().__init__()

        # We need to change the global rotation and translation order depending on the Mocup body_model or original from
        # https://github.com/vchoutas/smplx, where rotation is in the pose, comapred to post LBS multiplication
        self.mocap = not config.dataset_name == "thuman4"
        self.lbs_module = None
        self.assets = assets
        self.config = config
        self.inflate_cage = 0
        self.src = f"{self.config.assets}/{config.capture_id}"
        self.rgb = None
        self.densify_template=densify_template

        Path(self.src).mkdir(exist_ok=True, parents=True)

        self.create_body_model()

        logger.info(f"[ {self.config.model_type.upper()} ] model geometry initialized")

    def lbs_to_color(self, vertices, faces, weights, file_name):
        if self.rgb is None:
            self.rgb = np.array([np.random.choice(range(255), size=3) / 255 for _ in range(weights.shape[1])])
        colors = np.einsum("ikj,ik->ij", self.rgb[None].repeat(vertices.shape[0], 0), weights.cpu().numpy())
        save_mesh(vertices, faces, colors, file_name)

    def subdivide_mesh(self, vertices, faces, weights, iters=1):
        nv, nf, nw = subdivide_loop(vertices, faces, weights, iterations=iters)

        return nv, nf, nw

    def unpose(self, vertices, T, BS):
        deform_v = self.to_homo(vertices)
        deform_v = th.unsqueeze(deform_v, dim=-1)
        vtn = th.matmul(th.inverse(T)[:, self.nn_ids], deform_v)[:, :, :3, 0] - BS[:, self.nn_ids]
        return vtn

    def to_homo(self, verts):
        B, N, C = verts.shape
        homogen_coord = th.ones([B, N, 1]).cuda()
        homo = th.cat([verts, homogen_coord], dim=2)
        return homo

    def create_body_model(self):
        self.lbs_module = SMPLlayer(
            self.config.data.smplx_model,
            model_type=self.config.model_type,
            gender="neutral",
            use_joints=True,
            regressor_path=self.config.data.joint_regressor
        ).cuda()

        self.body_template_faces = self.lbs_module.faces_tensor
        self.body_template_vertices = self.lbs_module.v_template

        # Get star pose for better NN search
        f = self.lbs_module.faces_tensor.cpu().numpy()
        v = self.get_star_pose(return_Tbs=True)[0][0].cpu().numpy()

        delta_path = self.config.data.get("smplx_offset", None)
        if delta_path is not None:
            delta = np.load(delta_path)[0]
            v += delta

        w = self.lbs_module.weights.cpu().numpy()
        if self.densify_template:
            nv, nf, nw = self.subdivide_mesh(v, f, w)
        else:
            nv, nf, nw = v, f, w
        template = trimesh.Trimesh(v, f, process=False)
        dense_template = trimesh.Trimesh(nv, nf, process=False)
        weights, self.nn_ids = self.find_nn(template, dense_template)

        # Visualize LBS weights
        # self.lbs_to_color(nv, nf, weights, "dense.ply")
        # self.lbs_to_color(v, f, self.lbs_module.weights, "template.ply")

        _, T, A, bs = self.get_star_pose(return_Tbs=True)
        num_joints = self.lbs_module.J_regressor.shape[0]
        self.skin_weights = th.from_numpy(nw).cuda().float()
        T = th.matmul(self.skin_weights, A.view(1, num_joints, 16)).view(1, -1, 4, 4)

        # Unpose the star pose to t pose
        vtn = self.unpose(th.from_numpy(nv)[None].cuda().float(), T, bs)

        self.body_template_faces = th.from_numpy(nf).cuda()
        self.body_template_vertices = vtn

        # geom = self.get(self.init_lbs_motion)
        # trimesh.Trimesh(geom[0].cpu().numpy(), nf).export("test.ply")
        # exit(0)

    def find_nn(self, template, cage):
        template_weights = self.lbs_module.weights
        template.vertices += template.vertex_normals * self.inflate_cage
        vertex_id = template.kdtree.query(cage.vertices)[1]
        return template_weights[vertex_id, :], vertex_id

    def get_star_pose_lbs(self):
        poses = th.zeros(1, self.lbs_module.NUM_POSES).cuda()
        shape = th.zeros(1, 10).cuda()

        poses[:, 5] = np.pi / 6
        poses[:, 8] = -np.pi / 6

        return poses, shape

    def get_star_pose(self, return_Tbs=False):
        poses, shapes = self.get_star_pose_lbs()
        Rh = th.zeros(1, 3).cuda().float()
        Th = th.zeros(1, 3).cuda().float()
        expression = th.zeros(1, 10).cuda().float()
        vertices = self.get(
            {"poses": poses, "shapes": shapes, "expression": expression, "Rh": Rh, "Th": Th}, return_Tbs=return_Tbs
        )

        if return_Tbs:
            return vertices

        return vertices, self.body_template_faces

    def get_t_pose(self):
        poses = th.zeros(1, 87).cuda()
        shapes = th.zeros(1, 10).cuda()
        Rh = th.zeros(1, 3).cuda().float()
        Th = th.zeros(1, 3).cuda().float()
        expression = th.zeros(1, 10).cuda().float()
        return self.get(
            {"poses": poses, "shapes": shapes, "expression": expression, "Rh": Rh, "Th": Th}, return_Tbs=True
        )

    def deform(self, A, Bs, Rh, Th, delta=None):
        num_joints = self.lbs_module.J_regressor.shape[0]
        T = th.matmul(self.skin_weights, A.view(1, num_joints, 16)).view(1, -1, 4, 4)
        vertices = self.body_template_vertices
        if delta is not None:
            vertices = vertices + delta
        geom = vertices + Bs[:, self.nn_ids]
        geom = self.to_homo(geom)
        geom = th.unsqueeze(geom, dim=-1)
        geom = th.matmul(T[:, self.nn_ids], geom)[:, :, :3, 0]

        transl = Th.unsqueeze(dim=1)
        rot = batch_rodrigues(Rh)

        geom = th.matmul(geom, rot.transpose(1, 2)) + transl

        return geom

    def get(self, batch, return_Tbs=False, delta=None, global_transform=False):
        poses = batch["poses"]
        shapes = batch["shapes"]
        expression = batch["expression"]
        apply = 1 if global_transform else 0
        Rh = batch["Rh"] * apply  # disabled, we transform cameras instead
        Th = batch["Th"] * (apply if self.mocap else 1.0) # Mocup problem look line 182

        geom, T, A, bs = self.lbs_module(poses=poses, shapes=shapes, Rh=Rh, Th=Th, expression=expression)

        if return_Tbs:
            return geom, T, A, bs

        deformed = self.deform(A, bs, Rh, Th, delta)

        return deformed

    def get_landmarks(self, batch):
        poses = batch["poses"]
        shapes = batch["shapes"]
        expression = batch["expression"]
        Rh = batch["Rh"]
        Th = batch["Th"]

        lmks, T, A, bs = self.lbs_module(poses=poses, shapes=shapes, Rh=Rh, Th=Th, expression=expression, return_verts=False)

        return lmks

    def transfrom_cameras(self, smplx, w2c):
        Rh = smplx["Rh"].cuda()
        Rh = batch_rodrigues(Rh)
        Th = smplx["Th"].cuda()

        R_C = w2c[:, :3, :3]
        t_C = w2c[:, :3, 3]

        A = self.homogenization(R_C, t_C)
        B = self.homogenization(Rh, Th)
        # Thuman is using different model and the global rotation is used in pose comapred to post multiplication in mocup body model
        # Thus, we need to move our transformaiton acordingly... :/ 
        w2c = A @ B if self.mocap else A

        c2w = th.linalg.inv(w2c)

        return c2w
