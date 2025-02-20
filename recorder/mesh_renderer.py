# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytorch3d
import torch as th
import torch.nn.functional as F
import torch.nn as nn
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader,
    TexturesVertex,
    PointLights,
    PerspectiveCameras,
    BlendParams,
)

eps: float = 1e-8


class Renderer(nn.Module):
    def __init__(self, white_background=True):
        super().__init__()

        raster_settings = RasterizationSettings(
            blur_radius=0.0, faces_per_pixel=1, perspective_correct=True, max_faces_per_bin=262144
        )

        lights = PointLights(
            device='cuda:0',
            location=((0, 0, 1),),
            ambient_color=((0.45, 0.45, 0.45),),
            diffuse_color=((0.35, 0.35, 0.35),),
            specular_color=((0.05, 0.05, 0.05),),
        )

        bg_color = [1, 1, 1] if white_background else [0, 0, 0]

        blend = BlendParams(background_color=bg_color)
        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.renderer = MeshRenderer(
            self.rasterizer,
            shader=HardFlatShader(device='cuda:0', lights=lights, blend_params=blend),
        )

    def resize(self, H, W):
        self.renderer.rasterizer.raster_settings.image_size = (H, W)
        self.rasterizer.raster_settings.image_size = (H, W)

    def forward(self, cameras, vertices, faces, verts_rgb=None, meshes=None):
        if meshes is None:
            B, N, V = vertices.shape
            if verts_rgb is None:
                verts_rgb = th.ones(1, N, V)
            textures = TexturesVertex(verts_features=verts_rgb.cuda())
            meshes = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)

        P = cameras.get_world_to_view_transform().inverse().get_matrix().transpose(1, 2)[:, :3, 3]
        self.renderer.shader.lights.location = P

        rendering = self.renderer(meshes, cameras=cameras)
        return rendering[0, :, :, 0:3]

    def map(self, cameras, vertices, faces):
        meshes = pytorch3d.structures.Meshes(verts=vertices, faces=faces)
        postion = meshes.verts_packed()  # (V, 3)
        faces_postions = postion[faces][0]
        faces_postions_view = cameras.get_world_to_view_transform().transform_points(postion)[faces][0]
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces][0]

        fragments = self.rasterizer(meshes, cameras=cameras)
        mask = (fragments.pix_to_face > 0).float()[0, :, :, 0:1]  # [n, y, x, k]

        maps = interpolate_face_attributes(
            fragments.pix_to_face,
            fragments.bary_coords,
            th.cat([faces_postions, faces_postions_view], dim=-1),
        )

        position_map = maps[0, :, :, 0, 0:3]
        depth_map = maps[0, :, :, 0, 3:6]

        maps = interpolate_face_attributes(
            fragments.pix_to_face,
            th.ones_like(fragments.bary_coords),
            faces_normals,
        )

        normal_map = maps[0, :, :, 0, 0:3]

        veclen = th.norm(normal_map, dim=2, keepdim=True).clamp(min=eps)
        normal_map = normal_map / veclen

        return position_map, normal_map, depth_map[..., 2:3], mask
