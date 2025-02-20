# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from plyfile import PlyData, PlyElement
import torch
import trimesh
from  utils.graphics_utils import BasicPointCloud
import numpy as np
import open3d


def storePly(path, xyz, rgb, normals):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def to_trimesh(v, f, batch=0):
    if torch.is_tensor(v):
        v = v.detach().cpu().numpy()
    if torch.is_tensor(f):
        f = f.detach().cpu().numpy()
    if len(v.shape) == 3:
        v = v[batch]
    if len(f.shape) == 3:
        f = f[batch]
    return trimesh.Trimesh(v.copy(), f.copy(), process=False)


def save_mesh(vertices, faces, colors=None, filename="test.ply"):
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(vertices)
    mesh.triangles = open3d.utility.Vector3iVector(faces)
    write_vertex_colors = False
    if colors is not None:
        write_vertex_colors = True
        mesh.vertex_colors = open3d.utility.Vector3dVector(colors)
    open3d.io.write_triangle_mesh(filename, mesh, write_vertex_colors=write_vertex_colors)


def project_points_multi(p, Rt, K, normalize=False, size=None):
    """Project a set of 3D points into multiple cameras with a pinhole model.
    Args:
        p: [B, N, 3], input 3D points in world coordinates
        Rt: [B, NC, 3, 4], extrinsics (where NC is the number of cameras to project to)
        K: [B, NC, 3, 3], intrinsics
        normalize: bool, whether to normalize coordinates to [-1.0, 1.0]
    Returns:
        tuple:
        - [B, NC, N, 2] - projected points
        - [B, NC, N] - their
    """
    B, N = p.shape[:2]
    NC = Rt.shape[1]

    Rt = Rt.reshape(B * NC, 3, 4)
    K = K.reshape(B * NC, 3, 3)

    # [B, N, 3] -> [B * NC, N, 3]
    p = p[:, np.newaxis].expand(-1, NC, -1, -1).reshape(B * NC, -1, 3)
    p_cam = p @ Rt[:, :3, :3].mT + Rt[:, :3, 3][:, np.newaxis]
    p_pix = p_cam @ K.mT
    p_depth = p_pix[:, :, 2:]
    p_pix = (p_pix[..., :2] / p_depth).reshape(B, NC, N, 2)
    p_depth = p_depth.reshape(B, NC, N)

    if normalize:
        assert size is not None
        h, w = size
        p_pix = 2.0 * p_pix / torch.as_tensor([w, h], dtype=torch.float32, device=p.device) - 1.0
    return p_pix, p_depth


def subdivide_loop(vertices, faces, weights, iterations=None):
    """
    Subdivide a mesh by dividing each triangle into four triangles
    and approximating their smoothed surface (loop subdivision).
    This function is an array-based implementation of loop subdivision,
    which avoids slow for loop and enables faster calculation.

    Overall process:
    1. Calculate odd vertices.
      Assign a new odd vertex on each edge and
      calculate the value for the boundary case and the interior case.
      The value is calculated as follows.
          v2
        / f0 \\        0
      v0--e--v1      /   \\
        \\f1 /     v0--e--v1
          v3
      - interior case : 3:1 ratio of mean(v0,v1) and mean(v2,v3)
      - boundary case : mean(v0,v1)
    2. Calculate even vertices.
      The new even vertices are calculated with the existing
      vertices and their adjacent vertices.
        1---2
       / \\/ \\      0---1
      0---v---3     / \\/ \\
       \\ /\\/    b0---v---b1
        k...4
      - interior case : (1-kB):B ratio of v and k adjacencies
      - boundary case : 3:1 ratio of v and mean(b0,b1)
    3. Compose new faces with new vertices.

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indices of vertices which make up triangles

    Returns
    ------------
    vertices : (j, 3) float
      Vertices in space
    faces : (q, 3) int
      Indices of vertices
    iterations : int
          Number of iterations to run subdivision
    """
    try:
        from itertools import zip_longest
    except BaseException:
        # python2
        from itertools import izip_longest as zip_longest

    if iterations is None:
        iterations = 1

    def _subdivide(vertices, faces, weights):
        # find the unique edges of our faces
        edges, edges_face = trimesh.geometry.faces_to_edges(faces, return_index=True)
        edges.sort(axis=1)
        unique, inverse = trimesh.grouping.unique_rows(edges)

        # set interior edges if there are two edges and boundary if there is
        # one.
        edge_inter = np.sort(trimesh.grouping.group_rows(edges, require_count=2), axis=1)
        edge_bound = trimesh.grouping.group_rows(edges, require_count=1)
        # make sure that one edge is shared by only one or two faces.
        if not len(edge_inter) * 2 + len(edge_bound) == len(edges):
            # we have multiple bodies it's a party!
            # edges shared by 2 faces are "connected"
            # so this connected components operation is
            # essentially identical to `face_adjacency`
            faces_group = trimesh.graph.connected_components(edges_face[edge_inter])

            if len(faces_group) == 1:
                raise ValueError('Some edges are shared by more than 2 faces')

            # collect a subdivided copy of each body
            seq_verts = []
            seq_faces = []
            # keep track of vertex count as we go so
            # we can do a single vstack at the end
            count = 0
            # loop through original face indexes
            for f in faces_group:
                # a lot of the complexity in this operation
                # is computing vertex neighbors so we only
                # want to pass forward the referenced vertices
                # for this particular group of connected faces
                unique, inverse = trimesh.grouping.unique_bincount(faces[f].reshape(-1), return_inverse=True)

                # subdivide this subset of faces
                cur_verts, cur_faces = _subdivide(vertices=vertices[unique], faces=inverse.reshape((-1, 3)))

                # increment the face references to match
                # the vertices when we stack them later
                cur_faces += count
                # increment the total vertex count
                count += len(cur_verts)
                # append to the sequence
                seq_verts.append(cur_verts)
                seq_faces.append(cur_faces)

            # return results as clean (n, 3) arrays
            return np.vstack(seq_verts), np.vstack(seq_faces)

        # set interior, boundary mask for unique edges
        edge_bound_mask = np.zeros(len(edges), dtype=bool)
        edge_bound_mask[edge_bound] = True
        edge_bound_mask = edge_bound_mask[unique]
        edge_inter_mask = ~edge_bound_mask

        # find the opposite face for each edge
        edge_pair = np.zeros(len(edges)).astype(int)
        edge_pair[edge_inter[:, 0]] = edge_inter[:, 1]
        edge_pair[edge_inter[:, 1]] = edge_inter[:, 0]
        opposite_face1 = edges_face[unique]
        opposite_face2 = edges_face[edge_pair[unique]]

        # set odd vertices to the middle of each edge (default as boundary
        # case).
        odd = vertices[edges[unique]].mean(axis=1)
        odd_weigths = weights[edges[unique]].mean(axis=1)
        # modify the odd vertices for the interior case
        e = edges[unique[edge_inter_mask]]
        e_v0 = vertices[e][:, 0]
        e_v1 = vertices[e][:, 1]
        e_f0 = faces[opposite_face1[edge_inter_mask]]
        e_f1 = faces[opposite_face2[edge_inter_mask]]
        e_v2_idx = e_f0[~(e_f0[:, :, None] == e[:, None, :]).any(-1)]
        e_v3_idx = e_f1[~(e_f1[:, :, None] == e[:, None, :]).any(-1)]
        e_v2 = vertices[e_v2_idx]
        e_v3 = vertices[e_v3_idx]

        # simplified from:
        # # 3 / 8 * (e_v0 + e_v1) + 1 / 8 * (e_v2 + e_v3)
        odd[edge_inter_mask] = 0.375 * e_v0 + 0.375 * e_v1 + e_v2 / 8.0 + e_v3 / 8.0

        # weights
        e = edges[unique[edge_inter_mask]]
        e_v0 = weights[e][:, 0]
        e_v1 = weights[e][:, 1]
        e_f0 = faces[opposite_face1[edge_inter_mask]]
        e_f1 = faces[opposite_face2[edge_inter_mask]]
        e_v2_idx = e_f0[~(e_f0[:, :, None] == e[:, None, :]).any(-1)]
        e_v3_idx = e_f1[~(e_f1[:, :, None] == e[:, None, :]).any(-1)]
        e_v2 = weights[e_v2_idx]
        e_v3 = weights[e_v3_idx]

        odd_weigths[edge_inter_mask] = 0.375 * e_v0 + 0.375 * e_v1 + e_v2 / 8.0 + e_v3 / 8.0

        # find vertex neighbors of each vertex
        neighbors = trimesh.graph.neighbors(edges=edges[unique], max_index=len(vertices))
        # convert list type of array into a fixed-shaped numpy array (set -1 to
        # empties)
        neighbors = np.array(list(zip_longest(*neighbors, fillvalue=-1))).T
        # if the neighbor has -1 index, its point is (0, 0, 0), so that
        # it is not included in the summation of neighbors when calculating the
        # even
        vertices_ = np.vstack([vertices, [0.0, 0.0, 0.0]])
        weights_ = np.vstack([weights, [0.0] * weights.shape[1]])
        # number of neighbors
        k = (neighbors + 1).astype(bool).sum(axis=1)

        # calculate even vertices for the interior case
        even = np.zeros_like(vertices)
        even_weights = np.zeros_like(weights)

        # beta = 1 / k * (5 / 8 - (3 / 8 + 1 / 4 * np.cos(2 * np.pi / k)) ** 2)
        # simplified with sympy.parse_expr('...').simplify()
        beta = (40.0 - (2.0 * np.cos(2 * np.pi / k) + 3) ** 2) / (64 * k)
        even = beta[:, None] * vertices_[neighbors].sum(1) + (1 - k[:, None] * beta[:, None]) * vertices
        even_weights = beta[:, None] * weights_[neighbors].sum(1) + (1 - k[:, None] * beta[:, None]) * weights

        # calculate even vertices for the boundary case
        if edge_bound_mask.any():
            # boundary vertices from boundary edges
            vrt_bound_mask = np.zeros(len(vertices), dtype=bool)
            vrt_bound_mask[np.unique(edges[unique][~edge_inter_mask])] = True
            # one boundary vertex has two neighbor boundary vertices (set
            # others as -1)
            boundary_neighbors = neighbors[vrt_bound_mask]
            boundary_neighbors[~vrt_bound_mask[neighbors[vrt_bound_mask]]] = -1

            even[vrt_bound_mask] = (
                vertices_[boundary_neighbors].sum(axis=1) / 8.0 + (3.0 / 4.0) * vertices[vrt_bound_mask]
            )

            even_weights[vrt_bound_mask] = (
                weights_[boundary_neighbors].sum(axis=1) / 8.0 + (3.0 / 4.0) * weights[vrt_bound_mask]
            )

        # the new faces with odd vertices
        odd_idx = inverse.reshape((-1, 3)) + len(vertices)
        new_faces = np.column_stack(
            [
                faces[:, 0],
                odd_idx[:, 0],
                odd_idx[:, 2],
                odd_idx[:, 0],
                faces[:, 1],
                odd_idx[:, 1],
                odd_idx[:, 2],
                odd_idx[:, 1],
                faces[:, 2],
                odd_idx[:, 0],
                odd_idx[:, 1],
                odd_idx[:, 2],
            ]
        ).reshape((-1, 3))

        # stack the new even vertices and odd vertices
        new_vertices = np.vstack((even, odd))
        new_weights = np.vstack((even_weights, odd_weigths))

        return new_vertices, new_faces, new_weights

    for _ in range(iterations):
        vertices, faces, weights = _subdivide(vertices, faces, weights)

    return vertices, faces, weights


def get_neighbours(mesh):
    adjacent = mesh.face_adjacency
    neighbours = {}
    for a, b in adjacent:
        if a in neighbours:
            neighbours[a].append(b)
        else:
            neighbours[a] = [b]

        if b in neighbours:
            neighbours[b].append(a)
        else:
            neighbours[b] = [a]

    return neighbours


def median_filter_mesh(mesh, lables):
    n_faces = mesh.faces.shape[0]
    neighbours = get_neighbours(mesh)
    filtered_lables = []
    for i in range(n_faces):
        ring = neighbours[i]

        ring_lables = []
        for r in ring:
            ring_lables.append(lables[r])

        ring_lables = np.array(ring_lables)
        filtered = np.median(ring_lables, axis=0)
        filtered_lables.append(int(filtered))

    return np.array(filtered_lables)
