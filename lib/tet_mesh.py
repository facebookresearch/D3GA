# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import meshio
import numpy as np
import torch
import torch.nn as nn
import trimesh
import logging

from loguru import logger


class TetMesh(nn.Module):
    def __init__(self, path_tet):
        super().__init__()
        self._mesh = meshio.read(path_tet)
        self.tetras = torch.from_numpy(self._mesh.cells_dict["tetra"]).cuda()
        self.mesh_path = path_tet

        tris = self._mesh.cells_dict["triangle"]
        mesh = trimesh.Trimesh(vertices=self._mesh.points, faces=tris, process=False)
        trimesh.repair.fix_normals(mesh)

        self.faces = torch.from_numpy(mesh.faces).cuda()[None]
        self.points = torch.from_numpy(mesh.vertices).float().cuda()[None]

        self.A = self.tetras[:, 0]
        self.B = self.tetras[:, 1]
        self.C = self.tetras[:, 2]
        self.D = self.tetras[:, 3]

        v0 = torch.stack([self.A, self.B, self.C], dim=1)
        v1 = torch.stack([self.A, self.B, self.D], dim=1)
        v2 = torch.stack([self.A, self.C, self.D], dim=1)
        v3 = torch.stack([self.B, self.C, self.D], dim=1)

        self.ABCD = torch.stack([v0, v1, v2, v3], dim=1)
        self.tetra_faces = self.ABCD.view(-1, 3)
        self.V = self.volume(self.points)
        Dn = self.gradient(self.points).detach()  # template mesh
        self.Dn_inv = torch.linalg.inv(Dn)

        logger.info(
            f"Initialized Tet Mesh with Volume = {self.V.sum().item():.{5}f} m^3 | Tetras = {self.tetra_faces.shape[0]} | Vertices = {self.points.shape[1]}"
        )

    def _parse_bary(self, path):
        file = np.loadtxt(path)
        bary = torch.from_numpy(file[:, 1:]).float().cuda()[None]
        ids = torch.from_numpy(file[:, 0]).int().cuda()
        return bary, ids

    def get_positions(self, points=None):
        if points is None:
            points = self.points
        return torch.stack(
            [points[:, self.A], points[:, self.B], points[:, self.C], points[:, self.D]], dim=2
        ).transpose(3, 2)

    def tetra_to_trimesh(self, fix_normals=True):
        v = self.points.cpu().numpy()[0]
        f = self.tetra_faces.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        if fix_normals:
            trimesh.repair.fix_normals(mesh)
        return mesh

    def surface_to_trimesh(self):
        v = self.points[0].cpu().numpy()
        f = self.faces[0].cpu().numpy()
        return trimesh.Trimesh(vertices=v, faces=f, process=False)

    def n_params(self):
        return self.points.shape[1]

    def volume(self, points):
        p = torch.stack([points[:, self.A], points[:, self.B], points[:, self.C], points[:, self.D]], dim=2)
        a, b, c, d = p[:, :, 0, ...], p[:, :, 1, ...], p[:, :, 2, ...], p[:, :, 3, ...]
        P = torch.linalg.cross(b - d, c - d)[..., None]
        T = (a - d)[:, :, None, :]
        V = torch.abs(torch.einsum("bvij,bvji->bv", T, P)) / 6
        return V

    def gradient(self, points):
        v0 = points[:, self.A]
        v1 = points[:, self.B]
        v2 = points[:, self.C]
        v3 = points[:, self.D]

        return torch.stack([v3 - v0, v2 - v0, v1 - v0], dim=2)

    # Base on "A Constraint-based Formulation of Stable Neo-Hookean Materials" https://mmacklin.com/neohookean.pdf
    def fem_energy(self, points):
        B = points.shape[0]
        Dn_inv = self.Dn_inv.expand(B, -1, -1, -1)
        Ds = self.gradient(points)
        F = torch.einsum("bijk,bikl->bijl", Ds, Dn_inv)
        vol_loss = torch.pow(torch.linalg.det(F) - 1, 2)
        Ft = torch.einsum("bijk->bikj", F)
        FtF = torch.einsum("bijk,bikl->bijl", Ft, F)
        trace = torch.einsum("bijj->bi", FtF)
        vol_sheer = trace - 3

        l = 0.5
        m = 0.5

        return l * vol_loss + m * vol_sheer

    def forward(self, points):
        B = points.shape[0]
        bary = self.bary.expand(B, -1, -1)

        positions = torch.stack(
            [points[:, self.A], points[:, self.B], points[:, self.C], points[:, self.D]], dim=2
        ).transpose(3, 2)
        tetras = positions[:, self.bary_ids, ...]

        shape = torch.einsum("bijk,bik->bij", tetras, bary)

        return shape, points
