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
import open3d as o3d
from loguru import logger


def laplacian_calculation(mesh, equal_weight=True):
    neighbors = mesh.vertex_neighbors
    vertices = mesh.vertices.view(np.ndarray)
    col = np.concatenate(neighbors)
    row = np.concatenate([[i] * len(n) for i, n in enumerate(neighbors)])

    if equal_weight:
        data = np.concatenate([[1.0 / len(n)] * len(n) for n in neighbors])
    else:
        ones = np.ones(3)
        norms = [
            1.0 / np.sqrt(np.dot((vertices[i] - vertices[n]) ** 2, ones))
            for i, n in enumerate(neighbors)
        ]
        data = np.concatenate([i / i.sum() for i in norms])

    matrix = coo_matrix((data, (row, col)), shape=[len(vertices)] * 2)

    return matrix


def smoothing(mesh, num_iter=10, lbd=0.5, use_improved=True):
    vertices = mesh.vertices.copy()
    for _ in range(num_iter):

        laplacian_operator = laplacian_calculation(mesh, equal_weight=False)
        dot = laplacian_operator.dot(vertices) - vertices

        if use_improved:
            delta_d = mesh.vertex_normals
            inner_product = np.sum(dot * delta_d, axis=1, keepdims=True)
            is_outer = (inner_product >= 0).astype(np.float32)
            H_abs = np.linalg.norm(dot, axis=1, keepdims=True)
            outer_vec = is_outer * H_abs * delta_d
            inner_vec = (1 - is_outer) * (dot - inner_product * delta_d)

            vertices += lbd * (outer_vec + inner_vec)

        else:
            vertices += lbd * dot

        mesh.vertices = vertices

    return mesh


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

    radius, pitch = radius, 1.0 / (radius + 0.1)
    voxel = trimesh.voxel.creation.local_voxelize(
        mesh, [0, 0, 0], pitch=pitch, radius=radius, fill=True
    )
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


def create_cage_from_mesh(mesh, inflate=0.02):
    dv = mesh.vertex_normals * inflate
    mesh.vertices += dv

    return mesh


def smooth_cage(mesh, num_iter=25, lbd=0.25):
    cage = smoothing(mesh, num_iter=num_iter, lbd=lbd, use_improved=True)
    v, t = cage.vertices.copy(), cage.faces.copy()

    return trimesh.Trimesh(v, t, process=True)


def trimesh_to_open3d(trimesh_obj):
    if not isinstance(trimesh_obj, trimesh.Trimesh):
        raise ValueError("Input must be a Trimesh object.")
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_obj.vertices.astype(np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_obj.faces.astype(np.int32))
    return o3d_mesh


def simplify(mesh, tris_target):
    mesh_3d = trimesh_to_open3d(mesh)
    mesh_3d = mesh_3d.filter_smooth_taubin(number_of_iterations=10)

    if tris_target > 0:
        mesh_3d = mesh_3d.simplify_quadric_decimation(target_number_of_triangles=tris_target)

    mesh = trimesh.Trimesh(np.nan_to_num(np.asarray(mesh_3d.vertices).copy()), np.asarray(mesh_3d.triangles).copy(), process=True)

    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh, multibody=False)

    cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=70)
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[np.concatenate(cc)] = True
    mesh.update_faces(mask)

    return mesh


def cage_processor(mesh, cage_tris_target=0, radius=60, inflate=0.01):
    cage = create_cage(mesh, radius, inflate)
    cage = smooth_cage(cage)
    cage = simplify(cage, cage_tris_target)

    return (
        torch.from_numpy(cage.vertices).float()[None].cuda(),
        torch.from_numpy(cage.faces)[None].cuda()
    )
