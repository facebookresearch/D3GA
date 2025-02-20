# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import trimesh
from utils.mesh_utils import to_trimesh, get_neighbours
import torch.nn.functional as F
from cager.ops import cage_processor
import trimesh
import torch
from torch import nn
import numpy as np
import networkx as nx
from tetra_sampler import Tetra, compute_bary
from pytorch3d.transforms import matrix_to_quaternion
import logging

from loguru import logger


class CageBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.tetgen = None
        self.tet_mesh = None
        self.config = None
        self.cage_config = None
        self.face_mask = None

    def n_garments(self):
        pass

    def get_trimesh_template(self):
        f = self.body_template_faces
        v = self.body_template_vertices
        return to_trimesh(v, f)

    def homogenization(self, R, t):
        B = R.shape[0]
        I = torch.eye(4)[None].expand(B, -1, -1).cuda()
        I[:, :3, :3] = R
        I[:, :3, 3] = t
        return I

    def save_mesh(self, vertices, faces, file="test.ply"):
        f = faces.detach().cpu().numpy()
        v = vertices.detach().cpu().numpy()
        trimesh.Trimesh(vertices=v, faces=f).export(file)

    def transform(self, geom, RT):
        B = geom.shape[0]
        geom = torch.cat([geom, torch.ones([B, geom.shape[1], 1]).cuda()], axis=2)
        geom = torch.einsum('bji,bki->bkj', RT, geom)[:, :, 0:3]
        return geom

    def get_centorids(self, vertices):
        f = self.body_template_faces
        v0 = vertices[:, f[:, 0]]
        v1 = vertices[:, f[:, 1]]
        v2 = vertices[:, f[:, 2]]

        return (v0 + v1 + v2) / 3.0

    def filter_using_labels(self, mc_source):
        label_id = self.cage_config.label_id
        if label_id[0] == -1 and len(label_id) == 1:
            return mc_source

        mask = np.zeros_like(self.face_to_label).astype(bool)
        for id in label_id:
            mask = mask | (self.face_to_label == id)
        mc_source.update_faces(mask)

        # cc = trimesh.graph.connected_components(mc_source.face_adjacency, min_len=2)
        # mask = np.zeros(len(mc_source.faces), dtype=bool)
        # mask[np.concatenate(cc)] = True
        # mc_source.update_faces(mask)

        return mc_source

    def create_cage(self):
        logger.info(f"[ {self.cage_config.cage_name.upper()} ] | Generating cage into {self.src}")

        # In the case of fixing the whole manually in MeshLab
        if not os.path.exists(self.src + "/mc_source.ply"):
            source, faces = self.get_star_pose()
            self.mc_source = self.filter_using_labels(to_trimesh(source, faces))
            self.mc_source.export(self.src + "/mc_source.ply")
            logger.warning(f"\nIf necessary clean {self.src}/mc_source.ply by using Meshlab smoothing function with post-cleaning process.\nOpen the mesh, fix and save with the same name [MeshLab->Export Mesh]\nPress enter after finish to continute...")
            input("")
            logger.info("Continuing...")

        self.mc_source = trimesh.load(self.src + "/mc_source.ply", process=False)

        if not os.path.exists(self.src + "/cage_template.pt"):
            self.cage_template_vertices, self.cage_template_faces = cage_processor(
                self.mc_source, cage_tris_target=self.cage_config.n_target_tris, radius=120, inflate=self.inflate_cage
            )
            torch.save((self.cage_template_vertices, self.cage_template_faces), self.src + "/cage_template.pt")
        else:
            self.cage_template_vertices, self.cage_template_faces = torch.load(self.src + "/cage_template.pt", weights_only=False)

        self.cage_trimesh = to_trimesh(self.cage_template_vertices, self.cage_template_faces)
        source, faces = self.get_star_pose()
        self.template_trimesh = to_trimesh(source, faces)
        self.cage_soruce = to_trimesh(
            self.template_trimesh.vertices + (self.template_trimesh.vertex_normals * self.inflate_cage),
            self.template_trimesh.faces,
        )

        if not os.path.exists(self.src + "/cage.ply"):
            self.cage_trimesh.export(self.src + "/cage.ply")
            self.template_trimesh.export(self.src + "/object.ply")
            self.cage_soruce.export(self.src + "/source_object.ply")

        if os.path.exists(self.src + "/cage.mesh"):
            return

        logger.warning(f"\nCheck {self.src}/cage.ply and use:\nTaubin Smoothing [MeshLab->Filters->Smoothing->Taubin (default settings)].\n" +
                       "TetGen can explode in the case of self-intersecting faces (which happens often after cage generatin)." + 
                       "Meshlab will post-clean and smooth the cage.\nPress enter after finish to continute...")
        input("")
        logger.info("Continuing...")

        self.tetgen.run()

    def grow_mask_region(self, mesh, mask, n_rings=1):
        mask = mask.copy()
        n_faces = mesh.faces.shape[0]
        ids = np.arange(n_faces, dtype=int)[mask]
        region = set(ids.tolist())
        neighbours = get_neighbours(mesh)

        for _ in range(n_rings):
            grown_region = set()
            for i in range(n_faces):
                ring = neighbours[i]
                count = 0
                for n in ring:
                    if n in region:
                        count += 1

                if count > 0 and count < len(ring):
                    grown_region |= set(ring)
                    mask[ring] = True

            region |= grown_region

        return mask

    def get_face_mask(self):
        face = np.zeros_like(self.face_to_label).astype(bool)

        if self.face_mask is None:
            return face

        # Red channel is the loaded mask
        face = self.face_mask.visual.face_colors[:, 0] == 255

        return face

    def sample_body_surface(self, name):
        outside_garment_mesh = self.mc_source.copy()

        def parse(mask, key):
            cages = self.config.cages
            labels = cages[key].label_id if key in cages else []
            for label in labels:
                if label != -1:
                    mask = mask | (self.face_to_label == label)

            return mask

        inter_samples, inter_ids, inter_faces = None, None, None
        for key in self.config.cages.keys():
            if key == "upper" or key == "lower":
                mask = np.zeros_like(self.face_to_label).astype(bool)
                mask = parse(mask, key)
                outside_garment = ~mask
                n_rings = 10
                mesh = self.mc_source.copy()
                grown_faces = self.grow_mask_region(mesh, outside_garment, n_rings=n_rings)

                mask = grown_faces ^ outside_garment
                mesh.update_faces(mask)
                samples, ids = trimesh.sample.sample_surface(mesh, self.cage_config.get("n_intsec_gaussians", 15_000))
                faces = mesh.faces[ids]

                if inter_faces is None:
                    inter_samples, inter_ids, inter_faces = samples, ids, faces
                else:
                    inter_faces = np.concatenate([inter_faces, faces])
                    inter_ids = np.concatenate([inter_ids, ids])
                    inter_samples = np.concatenate([inter_samples, samples])

        face_mask = self.get_face_mask()
        mask = np.zeros_like(self.face_to_label).astype(bool)
        for key in self.config.cages.keys():
            if key != "body" and key != "face":
                mask = parse(mask, key)

        outside_garment = ~mask
        if name == "face":
            inter_faces = None  # Dont add inter faced
            outside_garment = face_mask

        # Do not intersect with the face region
        if name == "body" and "face" in self.config.cages:
            outside_garment = outside_garment & (~face_mask)

        outside_garment_mesh.update_faces(outside_garment)

        samples, ids = trimesh.sample.sample_surface(outside_garment_mesh, self.cage_config.n_gaussians)
        faces = outside_garment_mesh.faces[ids]

        # Multilayer garment
        if len(self.config.cages.keys()) > 1:
            # inject gaussians only on the garment intersection
            # if inter_faces is not None:
            #     faces = np.concatenate([inter_faces, faces])
            #     ids = np.concatenate([inter_ids, ids])
            #     samples = np.concatenate([inter_samples, samples])

            # inject gaussians under the garment
            if "n_under_gaussians" in self.cage_config and name == "body":
                mesh = self.mc_source.copy()
                if "face" in self.config.cages:
                    mesh.update_faces(~face_mask)
                under_samples, under_ids = trimesh.sample.sample_surface(mesh, self.cage_config.n_under_gaussians)
                under_faces = mesh.faces[under_ids]
                faces = np.concatenate([under_faces, faces])
                ids = np.concatenate([under_ids, ids])
                samples = np.concatenate([under_samples, samples])

        return samples, ids, faces

    def compute_tris_bary(self, p, a, b, c):
        v0 = b - a
        v1 = c - a
        v2 = p - a

        d00 = torch.einsum("ni,ni->n", v0, v0)
        d01 = torch.einsum("ni,ni->n", v0, v1)
        d11 = torch.einsum("ni,ni->n", v1, v1)
        d20 = torch.einsum("ni,ni->n", v2, v0)
        d21 = torch.einsum("ni,ni->n", v2, v1)

        denom = (d00 * d11 - d01 * d01) + 0.0000000001;

        v = (d11 * d20 - d01 * d21) / denom;
        w = (d00 * d21 - d01 * d20) / denom;
        u = 1.0 - v - w;

        bary = torch.cat([u[:, None], v[:, None], w[:, None]], dim=-1)

        return bary

    def sample_initial_points(self):
        n = self.cage_config.n_gaussians
        is_body = self.cage_config.cage_name == "body" or self.cage_config.cage_name == "face"
        mesh = self.mc_source.copy()

        # Sample non garment surface
        if is_body:
            samples, ids, faces = self.sample_body_surface(self.cage_config.cage_name)
        else:
            dv = mesh.vertex_normals * self.inflate_cage
            mesh.vertices += dv
            samples, ids = trimesh.sample.sample_surface(mesh, n)
            faces = mesh.faces[ids]

        v0 = torch.from_numpy(mesh.vertices[faces[:, 0]])
        v1 = torch.from_numpy(mesh.vertices[faces[:, 1]])
        v2 = torch.from_numpy(mesh.vertices[faces[:, 2]])

        e0 = v1 - v0
        e1 = v2 - v0
        e3 = F.normalize(torch.linalg.cross(e0, e1))  # normal

        T = F.normalize(torch.linalg.cross(e0, e3))
        B = F.normalize(torch.linalg.cross(e0, T))
        N = e3

        TBN = torch.stack([T, B, N], dim=2)

        rots = matrix_to_quaternion(TBN).cuda().float()
        points = torch.from_numpy(samples).cuda().float()
        faces = torch.from_numpy(faces).cuda().int()
        barys = self.compute_tris_bary(points.cpu(), v0, v1, v2).cuda().float()
        vertices = torch.from_numpy(mesh.vertices).cuda().float()

        return points, rots, faces, barys

    def load_mesh(self):
        self.cage = Tetra(f"{self.src}/cage.mesh")
        init_points, init_rotations, init_faces, init_barys = self.sample_initial_points()

        self.tet_mesh.points = init_points[None].clone()
        self.cage.points = init_points.clone()

        self.register_buffer('init_points', init_points)
        self.register_buffer('init_rotations', init_rotations)
        self.register_buffer('init_faces', init_faces)
        self.register_buffer('init_barys', init_barys)

    def load_tetra(self):
        self.cage = Tetra(f"{self.src}/cage.mesh")
        Dn = self.cage.gradient(self.cage.points[self.cage.tetras]).detach()
        self.Dn_inv = torch.linalg.inv(Dn)
        init_points, init_rotations, _, _ = self.sample_initial_points()

        self.canonical_vertices = self.get_canonical()[0]

        # v = self.canonical_vertices[0].cpu().numpy()
        # f = self.cage.tetra_faces.cpu().numpy()
        # trimesh.Trimesh(vertices=v, faces=f, process=False).export("test.ply")
        canonical_triangles = self.cage.get_triangles(self.canonical_vertices).cuda()
        canonical_tetras = self.canonical_vertices[self.cage.tetras].cuda()
        self.tri_to_tetra = self.cage.triangle_to_tetra.int().contiguous().cuda()

        barys, tetra_id, _ = compute_bary(
            init_points, canonical_tetras, canonical_triangles, self.tri_to_tetra, self.cage
        )

        canonical_gradient = torch.linalg.inv(self.cage.gradient(canonical_tetras[tetra_id]))

        self.register_buffer('init_points', init_points)
        self.register_buffer('init_rotations', init_rotations)
        self.register_buffer('canonical_triangles', canonical_triangles)
        self.register_buffer('canonical_tetras', canonical_tetras)
        self.register_buffer('canonical_gradient', canonical_gradient)
        self.register_buffer('barys', barys)
        self.register_buffer('tetra_id', tetra_id)

    def compute_def_grad(self, tetrapoints):
        deformed = self.cage.gradient(tetrapoints[self.tetra_id])
        J = deformed @ self.canonical_gradient
        return J

    def compute_grad(self, tetrapoints):
        J = self.cage.gradient(tetrapoints[self.tetra_id])
        return J

    # Base on "A Constraint-based Formulation of Stable Neo-Hookean Materials" https://mmacklin.com/neohookean.pdf
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
