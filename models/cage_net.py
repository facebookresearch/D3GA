# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch as th

from lib.cage_smplman import CageSmpl
from lib.cage_blueman import CageBlue
from lib.cameras import batch_to_camera
from torch import nn

from utils.general_utils import build_scaling_rotation
from utils.general_utils import strip_symmetric
import torch.nn.functional as F
from .mlp import DeformationField, CanonicalField, ColorField
from utils.general_utils import inverse_sigmoid
from pytorch3d.ops import knn_points

from loguru import logger

mask_colors = {
    "red": th.tensor([1.0, 0.0, 0.0]).cuda(),
    "green": th.tensor([0.0, 1.0, 0.0]).cuda(),
    "blue": th.tensor([0.0, 0.0, 1.0]).cuda(),
    "gray": th.tensor([0.5, 0.5, 0.5]).cuda(),
}


class CageNet(nn.Module):
    def __init__(self, cage_config, config, assets):
        super().__init__()

        self.config = config
        self.cage_config = cage_config
        self.name = cage_config.cage_name
        if 'is_smpl_body' in config and config.is_smpl_body:
            logger.info(f"Using {cage_config.cage_name} cage SMPLEMAN")
            self.geometry = CageSmpl(cage_config, config, assets)
        else:
            logger.info(f"Using {cage_config.cage_name} cage BLUEMAN")
            self.geometry = CageBlue(cage_config, config, assets)

        self.deformation_field = DeformationField(config, cage_config).cuda()
        self.canonical_field = CanonicalField(config, cage_config).cuda()
        self.color_field = ColorField(config, cage_config).cuda()

        self.tet_offset_pre_lbs = config.train.tet_offset_pre_lbs
        n_color_features = config.color_mlp.n_features
        self.max_sh_degree = config.train.max_sh_degree

        self.silhouette_color = mask_colors[cage_config.color]

        rots = self.geometry.init_rotations
        points = self.geometry.init_points
        colors = th.rand(points.size(0), n_color_features).float().cuda() * 0.33

        # SH colors
        shs = th.rand((points.size(0), 3)).cuda() / 255
        features = th.zeros((shs.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = shs
        opacities = inverse_sigmoid(0.2 * th.ones((points.shape[0], 1), dtype=th.float, device="cuda"))
        dist2 = th.clamp_min(knn_points(points[None], points[None], K = 4)[0][0, :, 1:].mean(-1), 0.0000001)
        scales = th.log(th.sqrt(dist2))[..., None].repeat(1, 3)

        self.colors_feat = nn.Parameter(colors.contiguous().requires_grad_(True))
        self.rotation = nn.Parameter(rots.contiguous().requires_grad_(True))
        self.scaling = nn.Parameter(scales.contiguous().requires_grad_(True))

        # SH part
        if self.use_SHS():  # register and add to optimizer
            self.opacities = nn.Parameter(opacities.contiguous().requires_grad_(True))
            self.features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self.features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        self.scaling_activation = th.exp
        self.scaling_inverse_activation = th.log
        self.rotation_activation = th.nn.functional.normalize
        self.opacity_activation = th.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

    def restore(self):
        pass

    def get_lr(self, name):
        default_lr = self.config.train.get(name, 0.001)
        if name in self.cage_config:
            default_lr = self.cage_config.get(name, 0.001)
        return default_lr

    def get_parameters(self):
        params = [
            {"params": self.colors_feat, "lr": self.get_lr("feature_lr")},
            {"params": self.deformation_field.parameters(), "lr": self.get_lr("deform_mlp_lr")},
            {"params": self.canonical_field.parameters(), "lr": self.get_lr("canon_mlp_lr")},
            {"params": self.color_field.parameters(), "lr": self.get_lr("color_mlp_lr")},
        ]

        if self.use_SHS():
            params.append({"params": self.opacities, "lr": self.config.train.opacity_lr})
            params.append({"params": self.features_dc, "lr": self.config.train.feature_lr})
            params.append({"params": self.features_rest, "lr": self.config.train.feature_lr / 20.0})
            params.append({"params": self.rotation, "lr": self.config.train.rotation_lr})
            params.append({"params": self.scaling, "lr": self.config.train.scaling_lr})

        return params

    def describe_ply(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self.features_dc.shape[1] * self.features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self.features_rest.shape[1] * self.features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def get_ply(self):
        f_dc = self.features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.opacities.detach().cpu().numpy()
        scale = self.scaling.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        return (f_dc, f_rest, opacities, scale, rotation)

    def use_SHS(self):
        if "use_shs" in self.config.train:
            return self.config.train.use_shs
        return False

    @property
    def get_scales(self):
        return self.scaling_activation(self.scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotation)

    @property
    def get_colors_feat(self):
        return self.colors_feat

    @property
    def get_opacity(self):
        return self.opacity_activation(self.opacities)

    @property
    def get_features(self):
        features_dc = self.features_dc
        features_rest = self.features_rest
        return th.cat((features_dc, features_rest), dim=1)

    def build_covariance_from_scaling_rotation(self, scaling, rotation):
        L = build_scaling_rotation(scaling, rotation)
        cov3D = L @ L.transpose(1, 2)
        return cov3D

    def get_cond(self, batch, name):
        if batch["lbs"].shape[0] == 104:  # socio
            lbs = batch["lbs"][6:]
        else:
            if not self.training:
                lbs = batch["lbs"]
            else:
                lbs = batch["smplx"]["poses"][0, :87]

        if "face" == self.cage_config.cage_name and "face_embs" in batch:
            embs = batch["face_embs"]
            q = batch["face_rot"]
            use_face_rot = self.config.train.get("use_face_rot", True)

            if name == "COLOR" and use_face_rot:
                return th.cat([embs, q])

            return embs

        return lbs

    def forward(self, batch):
        viewpoint_camera = batch_to_camera(batch)
        lbs = batch["lbs"]
        lbs = batch["smplx"] if "smplx" in batch else lbs[None]

        input_tetpoints = self.geometry.get(lbs)[0]
        canon_tetpoints = self.geometry.cage.points
        tetra_id = self.geometry.tetra_id
        tetra_faces = self.geometry.cage.tetras

        delta_node = self.deformation_field(canon_tetpoints, self.get_cond(batch, "DEFORMATION"))

        delta_bary, delta_rot, delta_scale = self.canonical_field(
            self.get_rotation,
            self.get_scales,
            self.geometry.barys,
            self.get_cond(batch, "CANONICAL"),
        )

        # Apply offset to the template mesh before LBS deformation
        if self.tet_offset_pre_lbs:
            tetpoints = self.geometry.get(lbs, delta=delta_node)[0]
        else:
            # Calcualte the offset in the posed space
            tetpoints = input_tetpoints + delta_node

        canon_barys = self.geometry.barys + delta_bary
        scales = self.scaling_activation(self.scaling + delta_scale) 
        rotation = self.rotation_activation(self.rotation + delta_rot)
        color_feat = self.get_colors_feat

        deformed_tetras = tetpoints[tetra_faces]

        J = self.geometry.compute_def_grad(deformed_tetras)
        actual_covariance = self.build_covariance_from_scaling_rotation(scales, rotation)
        cov3D = J @ actual_covariance @ J.transpose(1, 2)
        cov3D_precomp = strip_symmetric(cov3D)

        fm_energy = self.geometry.fem_energy(tetpoints).mean()[None]
        scale_energy = th.mean(th.mean(scales**2, axis=1))[None]

        deformed = deformed_tetras[tetra_id]

        means3D = th.einsum("ikj,ik->ij", deformed, canon_barys)

        n = means3D.size(0)
        cam_pos = viewpoint_camera.camera_center[None].cuda().expand(n, -1).detach()
        directions = means3D - cam_pos
        viewdirs = directions / th.linalg.norm(directions, dim=-1, keepdims=True)

        shadow = None
        if "pred_ao" in batch:
            ao = batch["pred_ao"][self.geometry.cage_to_body_vertex][..., None][tetra_faces]
            ao = ao[tetra_id]
            shadow = th.einsum("ikj,ik->ij", ao, canon_barys)

        silhouette_rgb = self.silhouette_color[None].expand(n, -1)

        shs, rgb, opacities = None, None, None

        if self.use_SHS():
            shs = self.get_features
            opacities = self.get_opacity
        else:
            rgb, opacities = self.color_field(
                color_feat,
                self.get_cond(batch, "COLOR"),
                viewdirs,
                batch["frame_encoding"],
                batch["camera_encoding"],
                shadow,
            )

            # if shadow is not None:
            #     rgb = rgb * shadow

        garment_pkg = {
            "shs": shs,
            "rgb": rgb,
            "opacities": opacities,
            "cov3D_precomp": cov3D_precomp,
            "silhouette_rgb": silhouette_rgb,
            "means3D": means3D,
            "canonical_means3D": th.einsum("ikj,ik->ij", canon_tetpoints[tetra_faces][tetra_id], self.geometry.barys),
            "color_feat": color_feat,
            "fm_energy": fm_energy,
            "scale_energy": scale_energy,
            "geometry": {
                "name": self.cage_config.cage_name,
                "input_tetpoints": input_tetpoints,
                "canon_tetpoints": canon_tetpoints,
                "delta_node": delta_node.detach(),
                "deformed_tetpoints": tetpoints,
                "faces": self.geometry.cage.triangles[None],
            },
        }

        return garment_pkg
