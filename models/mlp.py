# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from utils.pos_encoder import get_embedder

import torch as th
import torch.nn.functional as F
import torch.nn as nn
import tinycudann as tcnn
from loguru import logger


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        th.nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")


def get_cond_size(config, cage_config, name):
    if 'is_smpl_body' in config and config.is_smpl_body:
        if config.model_type == "smplx":
            return 66 + 12 + 9
        if config.model_type == "smpl":
            return 72
    if 'face' == cage_config.cage_name:
        if name == "COLOR":
            use_face_rot = config.train.get("use_face_rot", True)
            return config.face_mlp.n_output + (4 if use_face_rot else 0)
        else:
            return config.face_mlp.n_output

    return 98


class DeformationField(th.nn.Module):
    def __init__(self, config, cage_config) -> None:
        super().__init__()
        self.name = "DEFORMATION"
        self.tanh = th.nn.Tanh()
        self.sigmoid = th.nn.Sigmoid()
        self.config = config.deform_mlp
        self.embedder, self.dim = get_embedder(7)
        self.scaling = cage_config.get("node_scale", self.config.scale)
        self.n_input = get_cond_size(config, cage_config, self.name) + self.dim

        self.network = nn.ModuleList(
            [nn.Linear(self.n_input, self.config.n_nodes)]
            + [nn.Linear(self.config.n_nodes, self.config.n_nodes) for i in range(self.config.n_layers)]
        )

        self.output = nn.Linear(self.config.n_nodes, 3)
        with th.no_grad():
            self.output.weight *= 0.33
        self.network.apply(kaiming_leaky_init)

    def forward(self, canonical, pose):
        P, C = canonical.shape

        z = th.cat([pose.expand(P, -1), self.embedder(canonical)], dim=1)

        h = z
        for i, l in enumerate(self.network):
            h = self.network[i](h)
            h = F.leaky_relu(h, negative_slope=0.1)
        pred = self.output(h)

        return self.tanh(pred) * self.scaling


class CanonicalField(th.nn.Module):
    def __init__(self, config, cage_config, bary_size=4) -> None:
        super().__init__()
        self.name = "CANONICAL"
        self.tanh = th.nn.Tanh()
        self.sigmoid = th.nn.Sigmoid()
        self.config = config.canon_mlp
        self.bary_size = bary_size
        self.n_input = get_cond_size(config, cage_config, self.name) + 4 + 3 + bary_size

        self.network = nn.ModuleList(
            [nn.Linear(self.n_input, self.config.n_nodes)]
            + [nn.Linear(self.config.n_nodes, self.config.n_nodes) for i in range(self.config.n_layers)]
        )

        self.output = nn.Linear(self.config.n_nodes, 4 + 3 + self.bary_size)
        with th.no_grad():
            self.output.weight *= 0.33
        self.network.apply(kaiming_leaky_init)

    def forward(self, barys, rots, scales, pose):
        P, C = barys.shape

        z = th.cat([pose.expand(P, -1), rots, scales, barys], dim=1)

        h = z
        for i, l in enumerate(self.network):
            h = self.network[i](h)
            h = F.leaky_relu(h, negative_slope=0.1)

        pred = self.output(h)

        return (
            self.tanh(pred[:, 0:self.bary_size]) * self.config.scale_bary,
            pred[:, self.bary_size:self.bary_size + 4],
            pred[:, self.bary_size + 4:],
        )


class CanonicalLBSField(th.nn.Module):
    def __init__(self, config, cage_config) -> None:
        super().__init__()
        self.name = "CANONICAL"
        self.tanh = th.nn.Tanh()
        self.sigmoid = th.nn.Sigmoid()
        self.embedder, self.dim = get_embedder(7)
        self.config = config.canon_mlp
        self.n_input = get_cond_size(config, cage_config, self.name) + self.dim + 3 + 4

        self.network = nn.ModuleList(
            [nn.Linear(self.n_input, self.config.n_nodes)]
            + [nn.Linear(self.config.n_nodes, self.config.n_nodes) for i in range(self.config.n_layers)]
        )

        self.output = nn.Linear(self.config.n_nodes, 3 + 3 + 4)
        with th.no_grad():
            self.output.weight *= 0.33
        self.network.apply(kaiming_leaky_init)

    def forward(self, canonical, rots, scales, pose):
        P, C = canonical.shape

        z = th.cat([pose.expand(P, -1), rots, scales, self.embedder(canonical)], dim=1)

        h = z
        for i, l in enumerate(self.network):
            h = self.network[i](h)
            h = F.leaky_relu(h, negative_slope=0.1)

        pred = self.output(h)

        return (
            self.tanh(pred[:, 0:3]) * self.config.scale_bary,
            self.tanh(pred[:, 3:7]) * self.config.scale_rot,
            self.tanh(pred[:, 7:10]) * self.config.scale_scale,
        )


class ColorField(th.nn.Module):
    def __init__(self, config, cage_config) -> None:
        super().__init__()
        self.name = "COLOR"
        self.tanh = th.nn.Tanh()
        self.sigmoid = th.nn.Sigmoid()
        self.config = config.color_mlp
        self.frame_config = config.get("frame_embedder", None)
        self.camera_config = config.get("camera_embedder", None)
        self.use_only_rgb = self.config.get("use_only_rgb", False)
        self.use_pose = self.config.use_pose
        self.use_view_enc = self.config.use_view_enc
        # self.num_latent_codes = config.train.num_latent_codes

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": "SphericalHarmonics",
                        "degree": 4,
                    },
                    {"otype": "Identity"},
                ],
            },
        )

        sh = config.color_mlp.n_features
        n_cond = get_cond_size(config, cage_config, self.name) if self.use_pose else 0
        n_view_enc = 0
        if self.use_view_enc:
            n_view_enc = self.direction_encoding.n_output_dims

        frame_n_dim = self.frame_config.n_dims if self.frame_config is not None else 0
        camera_n_dim = self.camera_config.n_dims if self.camera_config is not None else 0

        self.n_input = n_cond + sh + n_view_enc + frame_n_dim + camera_n_dim
        if self.use_only_rgb:
            self.n_input = sh

        self.network = nn.ModuleList(
            [nn.Linear(self.n_input, self.config.n_nodes)]
            + [nn.Linear(self.config.n_nodes, self.config.n_nodes) for i in range(self.config.n_layers)]
        )

        self.output = nn.Linear(self.config.n_nodes, 3 + 1)
        with th.no_grad():
            self.output.weight *= 0.33
        self.network.apply(kaiming_leaky_init)

    def forward(self, shs, pose, view_dir, frame_encoding=None, camera_encoding=None, shadow=None):
        P, C = shs.shape

        z = shs

        if not self.use_only_rgb:
            if frame_encoding is not None:
                z = th.cat([frame_encoding.expand(P, -1), z], dim=1)

            if camera_encoding is not None:
                z = th.cat([camera_encoding.expand(P, -1), z], dim=1)

            if shadow is not None:
                z = th.cat([shadow, z], dim=1)

            if self.use_pose:
                z = th.cat([pose.expand(P, -1), z], dim=1)

            if self.use_view_enc:
                z = th.cat([self.direction_encoding(view_dir), z], dim=1)

        h = z
        for i, l in enumerate(self.network):
            h = self.network[i](h)
            h = F.leaky_relu(h, negative_slope=0.1)

        pkg = self.output(h)

        return self.sigmoid(pkg[:, 0:3]), self.sigmoid(0.1 + pkg[:, 3:4])


class FaceDecoder(th.nn.Module):
    def __init__(self, config, n_valid_kpts) -> None:
        super().__init__()
        self.name = "FACE_DECODER"
        self.config = config.face_mlp
        self.n_input = n_valid_kpts
        self.n_dim = 3

        self.network = nn.ModuleList(
            [nn.Linear(self.n_input * self.n_dim, self.config.n_nodes)]
            + [nn.Linear(self.config.n_nodes, self.config.n_nodes) for i in range(self.config.n_layers)]
        )

        self.output = nn.Linear(self.config.n_nodes, self.config.n_output)
        with th.no_grad():
            self.output.weight *= 0.33
        self.network.apply(kaiming_leaky_init)

    def forward(self, kpt):
        h = kpt.reshape(-1)
        for i, l in enumerate(self.network):
            h = self.network[i](h)
            h = F.leaky_relu(h, negative_slope=0.1)

        pred = self.output(h)

        return pred


class ShadowDecoder(th.nn.Module):
    def __init__(self, config, template) -> None:
        super().__init__()
        self.name = "SHADOW_DECODER"
        self.sigmoid = th.nn.Sigmoid()
        self.config = config.shadow_mlp
        self.canonical = template
        self.embedder, self.dim = get_embedder(7)

        self.n_input = 98 + self.dim

        self.network = nn.ModuleList(
            [nn.Linear(self.n_input, self.config.n_nodes)]
            + [nn.Linear(self.config.n_nodes, self.config.n_nodes) for i in range(self.config.n_layers)]
        )

        self.output = nn.Linear(self.config.n_nodes, 1)
        with th.no_grad():
            self.output.weight *= 0.33
        self.network.apply(kaiming_leaky_init)
        self.embedded_template = self.embedder(template)

    def forward(self, pose):
        P, C = self.embedded_template.shape

        z = th.cat([pose[6:].expand(P, -1), self.embedded_template], dim=1)

        h = z
        for i, l in enumerate(self.network):
            h = self.network[i](h)
            h = F.leaky_relu(h, negative_slope=0.1)
        pred = self.output(h)

        return self.sigmoid(pred)
