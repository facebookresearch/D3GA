# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from glob import glob
import os

from tqdm import tqdm
from utils.mesh_utils import storePly
import numpy as np
import torch
import trimesh

import matplotlib.colors as mcolors
import random
from numpy.random import default_rng
import logging

from tetra_sampler import Tetra, compute_bary

from loguru import logger


def random_color_generator():
    color = mcolors.hex2color(random.choice(list(mcolors.CSS4_COLORS.values())))
    return color


class Geometry:
    def __init__(self, config, omni_config):
        self.src = os.path.join(omni_config.assets, omni_config.capture_id)
        self.canonical_frame = omni_config.model.canonical_frame
        self.load_tetra()
        self.load_lbs()
        self.omni_config = omni_config
        self.n_sample = omni_config.train.canon_init_n_points

        logger.info(f"Tetra mesh with {self.cage.n()} tetras has been loaded...")

        self.initialize()

    def load_tetra(self):
        self.cage = Tetra(f"{self.src}/cage.mesh")
        Dn = self.cage.gradient(self.cage.points[self.cage.tetras]).detach()
        self.Dn_inv = torch.linalg.inv(Dn)
        self.canonical_source = trimesh.load(f"{self.src}/cages/{self.canonical_frame}.ply", process=False)
        self.canonical_vertices = torch.from_numpy(self.canonical_source.vertices).float().cuda()
        self.canonical_triangles = self.cage.get_triangles(self.canonical_vertices).cuda()
        self.canonical_tetras = self.canonical_vertices[self.cage.tetras].cuda()
        self.tri_to_tetra = self.cage.triangle_to_tetra.int().contiguous().cuda()

    def state_dict(self):
        return self.canonical_vertices, self.barys, self.tetra_id, self.canonical_gradient

    def load_state_dict(self, state):
        self.canonical_vertices, self.barys, self.tetra_id, self.canonical_gradient = state

    def load_lbs(self):
        self.lbs_weights = np.load(f"{self.src}/skin_weights.npy")
        self.lbs_weights = torch.from_numpy(self.lbs_weights).float().cuda()

        self.lbs_indecies = np.load(f"{self.src}/skin_indices.npy")
        self.lbs_indecies = torch.from_numpy(self.lbs_indecies).long().cuda()

        B = self.lbs_indecies.max().item() + 1
        P = self.lbs_indecies.size(0)
        # self.rgb = torch.from_numpy(np.array([random_color_generator() for _ in range(B)])).cuda()
        np.random.seed(33)
        self.rgb = torch.from_numpy(np.array([np.random.choice(range(255), size=3) / 255 for _ in range(B)])).cuda()
        self.rgb = self.rgb[None].expand(P, -1, -1).float()
        self.rgb = torch.gather(self.rgb, 1, self.lbs_indecies[..., None].expand(-1, -1, 3))

    def lbs_to_color(self, lbs):
        rgb = torch.einsum("ikj,ik->ij", self.rgb, lbs.clone().detach())
        # trimesh.PointCloud(vertices=self.canonical_vertices.cpu().numpy(), colors=(rgb * 255).cpu().numpy().astype(np.int8)).export("test.ply")
        rgb = torch.einsum("ikj,ik->ij", rgb[self.cage.tetras][self.tetra_id], self.barys)
        return rgb[None]

    def compute_def_grad(self, tetrapoints):
        deformed = self.cage.gradient(tetrapoints[self.tetra_id])
        J = deformed @ self.canonical_gradient
        return J

    def compute_grad(self, tetrapoints):
        J = self.cage.gradient(tetrapoints[self.tetra_id])
        return J

    def inject(self, xyz):
        if xyz.size(0) == 0:
            return

        barys, tetra_id, active_points = compute_bary(
            xyz,
            self.canonical_tetras,
            self.canonical_triangles,
            self.tri_to_tetra,
            self.cage,
        )

        canonical_gradient = torch.linalg.inv(self.cage.gradient(self.canonical_tetras[tetra_id]))

        # Combine
        self.canonical_vertices = torch.cat([self.canonical_vertices, xyz]).contiguous()
        self.barys = torch.cat([self.barys, barys]).contiguous()
        self.tetra_id = torch.cat([self.tetra_id, tetra_id]).contiguous()
        self.canonical_gradient = torch.cat([self.canonical_gradient, canonical_gradient]).contiguous()

    def remove(self, valid_points_mask):
        self.canonical_vertices = self.canonical_vertices[valid_points_mask]
        self.barys = self.barys[valid_points_mask]
        self.tetra_id = self.tetra_id[valid_points_mask]
        self.canonical_gradient = self.canonical_gradient[valid_points_mask]

    def fem_energy(self, points):
        Ds = self.cage.gradient(points[self.cage.tetras])
        F = Ds @ self.Dn_inv
        vol_loss = torch.pow(torch.linalg.det(F) - 1, 2)
        Ft = torch.einsum("ijk->ikj", F)
        FtF = torch.einsum("ijk,ikl->ijl", Ft, F)
        trace = torch.einsum('ijj->i', FtF)
        vol_sheer = trace - 3

        l = 0.5
        m = 0.5

        return l * vol_loss + m * vol_sheer

    def initialize(self):
        geom_init_path = f"{self.src}/geom_init.pt"
        n_gaussains = self.omni_config.train.canon_init_n_pc * self.omni_config.train.canon_init_n_points
        if os.path.exists(geom_init_path):
            pkg = torch.load(geom_init_path)
            self.canonical_vertices = pkg["canonical_vertices"].cuda()
            self.barys = pkg["barys"].cuda()
            self.tetra_id = pkg["tetra_id"].cuda()
            self.canonical_gradient = pkg["canonical_gradient"].cuda()

            if self.canonical_vertices.size(0) > n_gaussains:
                n = self.canonical_vertices.size(0)
                perm = torch.randperm(n)
                sampled_idx = perm[:n_gaussains]
                self.canonical_vertices = self.canonical_vertices[sampled_idx]
                self.barys = self.barys[sampled_idx]
                self.tetra_id = self.tetra_id[sampled_idx]
                self.canonical_gradient = self.canonical_gradient[sampled_idx]
            return

        rng = default_rng()

        positions = []
        barycentrics = []
        tetra_ids = []
        gradients = []
        normlas = []
        colors = []

        for pc_path in tqdm(sorted(glob(f"{self.src}/color_pcs/*.ply"))):
            cage_path = pc_path.replace("color_pcs", "cages")
            mvs_path = pc_path.replace("color_pcs", "mvs")
            faces_path = pc_path.replace("color_pcs", "faces_pcs").replace("ply", "npy")

            source = trimesh.load(cage_path, process=False)
            pc = trimesh.load(pc_path, process=False)
            mesh = trimesh.load(mvs_path, process=False)
            faces = np.load(faces_path)

            # Cage
            cage_vertices = torch.from_numpy(source.vertices).float().cuda()
            tetras = cage_vertices[self.cage.tetras].cuda()
            triangles = self.cage.get_triangles(cage_vertices).cuda()

            # Point cloud
            pc_vertices = torch.from_numpy(pc.vertices).float().cuda()
            pc_colors = torch.from_numpy(pc.colors).float().cuda()
            selected_points = rng.choice(pc_vertices.size(0), size=self.n_sample, replace=False)

            # Sampled points
            normal = torch.from_numpy(mesh.face_normals[faces][selected_points]).float().cuda()
            xyz = pc_vertices[selected_points]
            rgb = pc_colors[selected_points][:, :3]
            barys, tetra_id, active_points = compute_bary(xyz, tetras, triangles, self.tri_to_tetra, self.cage)

            canonical = self.canonical_vertices[self.cage.tetras][tetra_id]
            backprojected = torch.einsum("ikj,ik->ij", canonical, barys)

            canonical_gradient = torch.linalg.inv(self.cage.gradient(tetras[tetra_id]))

            positions.append(backprojected)
            barycentrics.append(barys)
            tetra_ids.append(tetra_id)
            gradients.append(canonical_gradient)
            normlas.append(normal)
            colors.append(rgb)

        # Initialize multiframe canonical space
        self.canonical_vertices = torch.cat(positions)
        self.barys = torch.cat(barycentrics)
        self.tetra_id = torch.cat(tetra_ids)
        self.canonical_gradient = torch.cat(gradients)
        n = self.canonical_vertices.size(0)

        n = torch.cat(normlas).cpu().numpy()
        rgb = torch.cat(colors).cpu().numpy()
        storePly(f"{self.src}/points3d.ply", self.canonical_vertices.cpu().numpy(), rgb, n)

        torch.save(
            {
                "canonical_vertices": self.canonical_vertices,
                "barys": self.barys,
                "tetra_id": self.tetra_id,
                "canonical_gradient": self.canonical_gradient,
            },
            geom_init_path,
        )
