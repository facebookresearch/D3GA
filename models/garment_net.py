# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
import torch as th
import numpy as np
import logging
import pytorch3d
from lib.calibration import CameraCalibration
from lib.cameras import batch_to_camera

from models.cage_net import CageNet
from models.mesh_net import MeshNet
from models.color_calib import CameraPixelBias
from models.embeddings import Embedding

from models.learnable_blur import LearnableBlur
from models.mlp import FaceDecoder, ShadowDecoder
from utils.pca_utils import build_pca

from torchvision.transforms.functional import gaussian_blur

from loguru import logger


class GarmentNet(nn.Module):
    def __init__(self, config, assets, n_frames):
        super().__init__()
        self.use_frame_embedder = "frame_embedder" in config
        self.use_cam_embeddings = "camera_embedder" in config
        self.image_width = config.get("image_width", 667)
        self.image_height = config.get("image_height", 1024)
        self.primitive_type = config.get("primitive_type", "cage")

        if self.use_frame_embedder:
            self.embeddings = Embedding(n_frames=n_frames, **config.frame_embedder).cuda()
        if self.use_cam_embeddings:
            self.cam_embeddings = Embedding(n_frames=len(assets.camera_ids), **config.camera_embedder).cuda()

        # self.bg_net = BGModel(allcameras=assets.camera_ids, img_res=(self.image_height, self.image_width)).cuda()
        self.learnable_calib = CameraCalibration(assets.camera_ids, config.train.test_camera).cuda()
        self.learnable_blur = LearnableBlur(assets.camera_ids).cuda()
        self.pixel_cal = CameraPixelBias(image_height=self.image_height, image_width=self.image_width, ds_rate=8, cameras=assets.camera_ids)

        self.garments = nn.ModuleList()
        for name in config.cages.keys():
            cage_config = config.cages[name]
            cage_config.name = "body" if name == "face" else name  # for face use the same cage as body
            cage_config.cage_name = name
            if self.primitive_type == "cage":
                self.garments.append(CageNet(cage_config, config, assets))
            if self.primitive_type == "mesh":
                self.garments.append(MeshNet(cage_config, config, assets))

        self.n_frames = n_frames
        self.inference_encoding = None
        self.test_embedding = None
        self.config = config
        self.use_background = config.train.get("use_background", False)
        self.use_bg_network = config.train.get("use_bg_network", False)
        self.use_color_calib = config.train.get("use_color_calib", True)
        self.use_random_bg = config.train.get("use_random_bg", False)
        self.use_ao = config.train.get("use_ao", False)
        self.use_face_emb = "face" in config.cages.keys()

        self.use_opt_smplx = config.train.get("use_opt_smplx", False) and "smplx" in assets
        if self.use_opt_smplx:
            self.prepare_opt_tensors(assets["smplx"])
            logger.warning("Using optimizable SMPLX parameters!")

        if self.use_ao:
            template = (assets.lbs_template_verts / 1000.0).cuda().float()
            self.shadow_decoder = ShadowDecoder(config, template).cuda()

        if self.use_face_emb:
            self.face_kpt_std = th.from_numpy(assets.face_kpt_std).float().cuda()
            self.face_kpt_mean = th.from_numpy(assets.face_kpt_mean).float().cuda()
            self.face_kpt_mask = th.from_numpy(assets.face_kpt_mask).bool().cuda()
            self.use_mean_kpt = False

            n_valid_kpts = self.face_kpt_mask.int().sum().item()
            self.face_decoder = FaceDecoder(config, n_valid_kpts).cuda()

    def prepare_opt_tensors(self, smplx):
        rotations = {}
        translation = {}
        poses = {}
        expressions = {}
        for seq in smplx.keys():
            rotations = {}
            translation = {}
            poses = {}
            for key, values in smplx[seq].items():
                k = str(key).zfill(5)
                k = f"{seq}_{k}"
                rotations[k] = th.as_tensor(values["Rh"])[None].cuda()
                translation[k] = th.as_tensor(values["Th"])[None].cuda()
                expressions[k] = th.as_tensor(values["expression"])[None].cuda()
                poses[k] = th.as_tensor(values["poses"])[None].cuda()

        self.optimizable_rotations = th.nn.ParameterDict(rotations).cuda()
        self.optimizable_translations = th.nn.ParameterDict(translation).cuda()
        self.optimizable_poses = th.nn.ParameterDict(poses).cuda()
        # self.optimizable_expressions = th.nn.ParameterDict(expressions).cuda()

    def restore(self):
        for grament in self.garments:
            grament.restore()

    def get_parameters(self):
        params = []

        if self.use_opt_smplx:
            params.append({"params": self.optimizable_rotations.parameters(), "lr": 0.001})
            params.append({"params": self.optimizable_translations.parameters(), "lr": 0.0001})
            params.append({"params": self.optimizable_poses.parameters(), "lr": 0.001})
           #  params.append({"params": self.optimizable_expressions.parameters(), "lr": 0.001})

        params.append({"params": self.learnable_calib.parameters(), "lr": 0.0001})
        params.append({"params": self.learnable_blur.parameters(), "lr": 0.001})
        params.append({"params": self.pixel_cal.parameters(), "lr": 0.000005})

        if self.use_ao:
            params.append({"params": self.shadow_decoder.parameters(), "lr": self.config.train.lr})

        if self.use_frame_embedder:
            params.append({"params": self.embeddings.parameters(), "lr": self.config.train.lr})

        if self.use_cam_embeddings:
            params.append({"params": self.cam_embeddings.parameters(), "lr": self.config.train.lr})

        if self.use_bg_network:
            params.append({"params": self.bg_net.parameters(), "lr": 0.01})

        if self.use_face_emb:
            params.append({"params": self.face_decoder.parameters(), "lr": self.get_lr("face_mlp_lr")})

        for garment in self.garments:
            params += garment.get_parameters()

        return params

    def get_lr(self, name):
        return self.config.train.get(name, 0.001)

    def merge(self, pkg, pred):
        if pkg is None:
            if "geometry" in pred:
                pred["geometry"] = [pred["geometry"]]
            return pred
        for key in pkg.keys():
            if pred[key] is None:
                continue
            if key == "geometry":
                pkg["geometry"].append(pred[key])
                continue
            pkg[key] = th.cat([pkg[key], pred[key]])
        return pkg

    def get_additional_embedding(self, batch):
        batch["frame_encoding"] = None
        batch["camera_encoding"] = None

        if self.training:
            frame_id = th.tensor([batch["frame_id"]]).cuda()
            camera_id = th.tensor([batch["order_cam_idx"]]).cuda()
            if self.use_frame_embedder:
                batch["frame_encoding"] = self.embeddings(frame_id)[0, :, 0, 0]
            if self.use_cam_embeddings:
                batch["camera_encoding"] = self.cam_embeddings(camera_id)[0, :, 0, 0]
        else:
            if self.use_frame_embedder:
                batch["frame_encoding"] = self.embeddings.average()[0]
            if self.use_cam_embeddings:
                batch["camera_encoding"] = self.cam_embeddings.average()[0]

    def get_face_embedding(self, batch):
        if self.use_face_emb:
            kpt = batch["face_kpt"][:, 0:3]
            conf = batch["face_kpt"][:, 3:4] / 100.0
            _, kpt, rot = self.garments[0].geometry.get(batch["lbs"][None], kpt=kpt[None])
            kpt = kpt[0][self.face_kpt_mask]
            kpt = (kpt - self.face_kpt_mean) / self.face_kpt_std
            q = pytorch3d.transforms.matrix_to_quaternion(rot)[0]
            if self.use_mean_kpt:
                kpt = th.zeros_like(self.face_kpt_mean)

            embs = self.face_decoder(kpt)

            batch["face_embs"] = embs
            batch["face_rot"] = q

    def get_background(self, batch, pkg):
        if self.use_bg_network:
            bgs = self.bg_net(camindex=batch["camera_id"])[0]
            pkg["bg_map"] = bgs

        if (self.use_bg_network and batch["iteration"] < self.config.train.enable_bg) or self.use_random_bg:
            noise = gaussian_blur(th.rand_like(batch["orig_image"]), [7, 7])
            pkg["bg_noise"] = noise

    def get_shadow(self, batch):
        if self.use_ao:
            lbs = batch["lbs"]
            ao = self.shadow_decoder(lbs)
            batch["pred_ao"] = ao.squeeze(1)

    def update_batch(self, batch):
        poses = None
        if not self.training:
            return poses
        if self.use_opt_smplx:
            seq_id = batch["seq_id"]
            frame_id = str(batch["order_frame_idx"]).zfill(5)
            k = f"{seq_id}_{frame_id}"

            poses = self.optimizable_poses[k]
            rh = self.optimizable_rotations[k]
            th = self.optimizable_translations[k]
            iteration = batch["iteration"]

            if iteration > 400_000:
                rh = rh.detach()
                th = th.detach()
                poses = poses.detach()

            batch["smplx"]["Rh"] = rh
            batch["smplx"]["Th"] = th
            batch["smplx"]["poses"] = poses
            # batch["smplx"]["expressions"] = self.optimizable_expressions[k]

        return poses

    def eval_layer(self, batch, names):
        self.get_additional_embedding(batch)
        self.get_face_embedding(batch)
        self.get_shadow(batch)

        pkg = None
        for garment in self.garments:
            if garment.name in names:
                output = garment(batch)
                pkg = self.merge(pkg, output)

        return pkg

    def forward(self, batch):
        poses = self.update_batch(batch)
        self.get_additional_embedding(batch)
        self.get_face_embedding(batch)
        self.get_shadow(batch)

        pkg = None
        for garment in self.garments:
            output = garment(batch)
            pkg = self.merge(pkg, output)

        self.get_background(batch, pkg)

        pkg["optimizable_poses"] = poses
        pkg["frame_encoding"] = batch["frame_encoding"]
        if self.use_color_calib:
            pkg["rgb"] = self.learnable_calib(pkg["rgb"], batch["camera_id"])

        return pkg
