# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
from pathlib import Path

from tqdm import tqdm
import trimesh

from datasets import load_static_assets
from datasets.actorshq_dataset import ActorsHQDataset

import numpy as np
import torch as th
import torch.nn as nn
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.structures import Meshes
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, PerspectiveCameras

from omegaconf import OmegaConf
import logging
from torch.utils.data import DataLoader
from utils.data_utils import to_device

from utils.mesh_utils import median_filter_mesh

from loguru import logger


class Renderer(nn.Module):
    def __init__(self, pc_settings: RasterizationSettings):
        super().__init__()
        self.rasterizer = MeshRasterizer(raster_settings=pc_settings)

    @staticmethod
    def to_cameras(Rt, K, image_size) -> PerspectiveCameras:
        R = Rt[:, :3, :3]
        tvec = Rt[:, :3, 3]

        cameras = cameras_from_opencv_projection(R, tvec, K, image_size)

        return cameras

    def resize(self, H, W):
        self.rasterizer.raster_settings.image_size = (H, W)

    def forward(self, cameras, vertices, faces, images):
        images = images.permute(0, 2, 3, 1).cuda()
        B, H, W, C = images.shape
        colors = th.ones_like(vertices)
        meshes = Meshes(verts=vertices.float(), faces=faces.long())
        fragmetns = self.rasterizer(meshes, cameras=cameras)

        take = fragmetns.pix_to_face >= 0
        idx = fragmetns.pix_to_face

        colors = th.zeros_like(faces[0])

        n_face = faces.size(1)

        for i in range(B):
            to_take = take[i, ..., 0]
            ids = idx[i, ..., 0][to_take]
            rgb = images[i][to_take]

            if i > 0:
                ids -= i * n_face

            colors[ids] += rgb

        return colors


class Segmenter:
    def __init__(self, config, assets, body_model):
        self.config = config
        self.assets = assets
        self.body_model = body_model
        self.raster_settings = RasterizationSettings(max_faces_per_bin=131072 * 2)
        self.renderer = Renderer(self.raster_settings)
        self.dst = os.path.join(config.assets, config.capture_id)
        Path(self.dst).mkdir(parents=True, exist_ok=True)

    def id_to_rgb(self, ids):
        if self.config.dataset_name == "socio":
            with open("/mnt/home/hewen/data/sociopticon/segmentation/consolidated/profile.json") as f:
                color_dict = {}
                for label in json.load(f)["label_definition"]:
                    color_dict[label["id"]] = np.array(label["color"])

            colors = []
            for i in ids:
                rgb = color_dict[i]
                colors.append(rgb)
            return np.array(colors)

        # if self.config.dataset_name == "zju" or self.config.dataset_name == "h36m":
        rgb_ids = np.array([np.random.choice(range(255), size=3) for _ in range(20)])
        colors = []
        for i in ids:
            rgb = rgb_ids[i]
            if i == 0:
                rgb *= 0
            colors.append(rgb)
        return np.array(colors)

    def majority_vote(self, part_per_face):
        segmetns = []
        for part in tqdm(part_per_face):
            mask = part > 0
            if np.sum(mask) == 0:
                segment = 0
            else:
                unique, counts = np.unique(part[mask], return_counts=True)
                segment = unique[counts.argmax()]
            segmetns.append(segment)

        return np.array(segmetns)

    def get_loader(self):
        N_FRAMES = 200

        if self.config.dataset_name == "actorshq":
            dataset = ActorsHQDataset(test_camera=[], **self.config.data)
        else:
            raise NotImplementedError("Dataset not implemented")

        dataset.random_frames(n=N_FRAMES * dataset.n_cameras())
        return DataLoader(
            dataset,
            batch_size=self.config.train.get("batch_size", 1),
            pin_memory=True,
            num_workers=self.config.train.get("num_workers", 8),
            drop_last=False,
            worker_init_fn=lambda _: np.random.seed(),
        )

    def get_cameras(self, motion, Rt, image_size, K):
        c2w = self.body_model.transfrom_cameras(motion, Rt)
        w2c = th.linalg.inv(c2w)
        return Renderer.to_cameras(w2c, K, image_size)

    def run(self):
        face_to_label_path = f"{self.dst}/face_to_label.npy"
        segment_pc_path = f"{self.dst}/segmented.ply"

        if os.path.exists(face_to_label_path):
            return

        loader = self.get_loader()

        view_per_camera = []
        faces = self.body_model.body_template_faces[None]
        vertices = self.body_model.get_star_pose()
        if type(vertices) is tuple:
            vertices = vertices[0]
        centroids = self.body_model.get_centorids(vertices)
        mesh_template = self.body_model.get_trimesh_template()

        MAX_SAMPLES = 512

        logger.info(f"Transfering segmentation onto body model mesh")

        for index, batch in tqdm(enumerate(loader)):
            if index > MAX_SAMPLES:
                break
            if batch is None:
                continue

            seg_part = batch["seg_part"].cuda()
            Rt = batch["Rt"].cuda()
            K = batch["K"].cuda()

            lbs = to_device(batch["lbs_motion"])
            lbs = batch["smplx"] if "smplx" in batch else lbs
            lbs = to_device(lbs)

            vertices = self.body_model.get(lbs)

            B, _, H, W = seg_part.shape
            self.renderer.resize(H, W)
            image_size = th.tensor([[H, W]]).expand(B, -1).cuda()
            cameras = self.get_cameras(lbs, Rt, image_size, K)

            parts = self.renderer(cameras, vertices, faces, seg_part)

            view_per_camera.append(parts)

        part_per_face = th.cat(view_per_camera, dim=-1).int()
        segment = self.majority_vote(part_per_face.cpu().numpy())
        segment = median_filter_mesh(mesh_template, segment)

        rgb = self.id_to_rgb(segment)
        points = centroids[0].cpu().numpy()
        trimesh.PointCloud(vertices=points, colors=rgb).export(segment_pc_path)
        np.save(face_to_label_path, segment)
