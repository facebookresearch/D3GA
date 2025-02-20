# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from glob import glob
import os
import numpy as np
from datasets.actorshq_dataset import SEQUENCES
import torch as th
from models.garment_net import GarmentNet
import logging
from pathlib import Path
import re
import pandas as pd

from renderer import render
from utils.load_module import instantiate

from loguru import logger


class Trainer:
    def __init__(self, config, static_assets, is_eval=False) -> None:
        self.is_eval = is_eval
        n_frames = self.get_n_frames(config)
        self.config = config
        self.model = GarmentNet(config, static_assets, n_frames).cuda()
        self.assets = static_assets
        self.model_path = config.train.run_dir
        self.max_sh_degree = config.train.max_sh_degree
        self.active_sh_degree = 0
        self.use_background = config.train.get("use_background", False)
        self.use_blur = config.train.get("use_blur", False)
        self.use_pixel_cal = config.train.get("use_pixel_cal", False)
        self.use_color_calib = config.train.get("use_color_calib", True)
        self.bg_color = "white" if "background" not in config.train else config.train.background
        
        logger.info(f"Total number of parameters {sum(p.numel() for p in self.model.parameters())}")

        self.initialize()

    def get_n_frames(self, config):
        n_frames = 0
        if config.dataset_name == "actorshq":
            for seq in SEQUENCES:
                fmts = dict(sequence=seq)
                src = config.data.smplx_poses.format(**fmts)
                n_frames = len(glob(f"{src}/*.json")) - config.data.n_testing_frames
        else:
            df = pd.read_csv(config.data.root_path + f"/frame_splits_list.csv")
            frame_list = df[df.split == "train"].frame.tolist()
            n_frames = len(frame_list)

        logger.info(f"Trainer initialized with {n_frames}")

        return n_frames

    def initialize(self):
        # Build optimizer
        params = self.model.get_parameters()

        self.optimizer = instantiate(self.config.train.optimizer, params=params)
        self.scheduler = instantiate(self.config.train.lr_scheduler, optimizer=self.optimizer)

    def eval(self):
        self.model.eval()
        self.is_eval = True

    def set_mean_kpt(self):
        self.model.use_mean_kpt = True

    def has_layers(self, names):
        layers = [g.name for g in self.model.garments]
        contains = True
        for name in names:
            if name not in layers:
                contains = False
        return contains

    def eval_layer(self, frame, names, fast=True):
        garment_pkg = self.model.eval_layer(frame, names)
        rasterized = render(frame, garment_pkg, bg_color=self.bg_color, solid_bg=not self.use_background, fast=fast)
        pred_image = rasterized["render"]

        return {
            "pred_image": pred_image,
        }

    def fit(self, frame):
        garment_pkg = self.model(frame)
        garment_pkg["sh_degree"] = self.active_sh_degree
        
        if not self.is_eval:
            col = np.random.rand(3)
        else:
            col = np.array([1, 1, 1]) if self.bg_color == "white" else np.array([0, 0, 0])

        bg_color_cuda = th.from_numpy(col).to(th.float32).cuda()

        rasterized = render(frame, garment_pkg, bg_color=bg_color_cuda, solid_bg=not self.use_background)

        pred_silhouette = render(
            frame,
            garment_pkg,
            colors_precomp=garment_pkg["silhouette_rgb"],
            bg_color=th.zeros_like(bg_color_cuda),
            detach=self.config.get("detach_silhouette", []),
        )["render"]

        gt_image = frame["orig_image"] if self.use_background else frame["image"]
        if "bg_noise" in garment_pkg:
            with th.no_grad():
                noise = garment_pkg["bg_noise"]
                # To have zero loss in the background region
                if self.use_blur:
                    noise = self.model.learnable_blur(noise[None], frame['camera_id'])[0].detach()
                gt_image = (1 - frame['alpha']) * noise + frame['alpha'] * frame["orig_image"]

        pred_image = rasterized["render"]

        blur_weights = None
        if self.use_blur:
            pred_image = self.model.learnable_blur(pred_image[None], frame["camera_id"])[0]
            blur_weights = self.model.learnable_blur.reg(frame["camera_id"])

        if self.use_pixel_cal:
            cam_idx = th.tensor([frame["order_cam_idx"]]).cuda()
            pixel_bias = self.model.pixel_cal(cam_idx)[0]
            pred_image = pred_image + pixel_bias

        return {
            "gt_image": gt_image,
            "garment_pkg": garment_pkg,
            "blur_weights": blur_weights,
            "pred_silhouette": pred_silhouette,
            "pred_image": pred_image,
            "bg_color": bg_color_cuda
        }

    def get_optical_flow(self, image1, image2, scale_image=2):
        return self.model.flow_network(image1, image2, scale_image)

    def restore(self, iteration=None, strict=True, return_state=False):
        path = os.path.join(self.model_path, "checkpoints")
        if os.path.exists(path):
            checkpoints = sorted(glob(path + "/*.pth"))
            if len(checkpoints) > 0:
                path = checkpoints[-1]
                if iteration is not None:
                    logger.info(f"Trying to find checkpoint with iter={iteration}")
                    for checkpoint in checkpoints:
                        if int(iteration) == int(re.findall(r'\d+\.?\d*', Path(checkpoint).stem)[0]):
                            path = checkpoint
                            logger.info(f"Found {path}!")
                            break

                (model_params, first_iter) = th.load(path, weights_only=False)

                model_dict, opt_dict, scheuler_dict = model_params

                logger.info(f"Initialized from {first_iter}th step!")

                self.model.load_state_dict(model_dict, strict=strict)
                if not self.is_eval:
                    self.optimizer.load_state_dict(opt_dict)
                    self.scheduler.load_state_dict(scheuler_dict)

                self.model.restore()

                if return_state:
                    return model_dict, first_iter
                return first_iter

        if return_state:
            None, 0
        return 0

    def oneupSHdegree(self, iteration):
        if iteration % 1000 != 0 or not self.config.train.use_shs:
            return
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def step(self, loss, iteration):
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.model.parameters(), 2.5, foreach=True)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.oneupSHdegree(iteration)

    def save(self, iteration=-1, name=None):
        if (iteration % self.config.train.checkpoint_n_steps != 0) and name is None:
            return

        if name is not None:
            path = name
        else:
            path = "/checkpoints/chkpnt" + str(iteration).zfill(6) + ".pth"

        path = self.model_path + path

        model_params = (self.model.state_dict(), self.optimizer.state_dict(), self.scheduler.state_dict())

        th.save((model_params, iteration), path)

        logger.info(f"\n[ITER {iteration}] Saving Checkpoint to {path}")
