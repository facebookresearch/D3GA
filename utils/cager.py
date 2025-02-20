# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import trimesh
from trimesh.voxel import creation
import mcubes
import numpy as np
from scipy.sparse import coo_matrix


def create_cage(mesh, radius=60, inflate=0.01, scale=0.8):
    mesh = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy(), process=False)

    matrix = np.eye(4)
    matrix[:2, :2] *= scale
    mesh.apply_transform(matrix)

    dv = mesh.vertex_normals * inflate
    mesh.vertices += dv

    v, t = mesh.vertices, mesh.faces

    center = np.mean(v, axis=0)
    v -= center

    radius, pitch = radius, 1.0 / radius
    voxel = trimesh.voxel.creation.local_voxelize(mesh, [0, 0, 0], pitch=pitch, radius=radius, fill=True)
    densitygrid = voxel.encoding.dense
    # densitygrid = mcubes.smooth(densitygrid)
    v, t = mcubes.marching_cubes(densitygrid, 0.5)
    v = v / radius - 1

    v += center
    mesh = trimesh.Trimesh(v.copy(), t.copy(), process=False)

    matrix = np.eye(4)
    matrix[:2, :2] *= 1 / scale
    mesh.apply_transform(matrix)

    return mesh


def simplify(mesh, tris_target):
    mesh_3d = mesh.as_open3d
    mesh_3d = mesh_3d.filter_smooth_taubin(number_of_iterations=20)

    if tris_target > 0:
        mesh_3d = mesh_3d.simplify_quadric_decimation(target_number_of_triangles=tris_target)

    mesh = trimesh.Trimesh(
        np.nan_to_num(np.asarray(mesh_3d.vertices).copy()), np.asarray(mesh_3d.triangles).copy(), process=True
    )

    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh, multibody=False)

    cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=70)
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[np.concatenate(cc)] = True
    mesh.update_faces(mask)

    return mesh
