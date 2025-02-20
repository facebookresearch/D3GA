# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import torch as th
import torch.nn.functional as F
import numpy as np
from kornia.filters.gaussian import gaussian_blur2d
from kornia.filters import median_blur
from kornia.morphology import dilation, erosion
import colour
import logging
from pytorch3d.transforms import matrix_to_quaternion
from kornia.color.xyz import xyz_to_rgb
from kornia.color.rgb import linear_rgb_to_rgb
import cv2
from utils.data_utils import to_device
from utils.image_utils import (
    erode_mask,
    close_holes,
    linear2color_corr,
    linear2color_corr_inv,
    paste,
)
from recorder.mesh_renderer import Renderer
from recorder.pc_renderer import PCRenderer

from loguru import logger

red = th.tensor([1.0, 0.0, 0.0]).cuda()
green = th.tensor([0.0, 1.0, 0.0]).cuda()
blue = th.tensor([0.0, 0.0, 1.0]).cuda()
gray = th.tensor([0.5, 0.5, 0.5]).cuda()


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


class Batcher:
    def __init__(self, config, body_model) -> None:
        self.config = config
        self.body_model = body_model
        self.col_calib = None
        self.mesh_renderer = Renderer().cuda()
        self.use_erode_mask = config.train.get("erode_mask", False)
        self.use_close_holes = config.train.get("use_close_holes", False)
        self.use_bg_network = config.train.get("use_bg_network", False)
        self.bg_color = "white" if "background" not in config.train else config.train.background.lower()
        # if os.path.exists(config.train.color_calibration):
        #     self.col_calib = self.load_calibration_params(config.train.color_calibration)


    def load_calibration_params(self, params_json):
        with open(params_json, "r") as json_file:
            json_filedata = json.load(json_file)
        gpu = {}
        for key in json_filedata.keys():
            gpu[key] = th.from_numpy(np.asarray(json_filedata[key]["ccm"])).cuda().float()
        return gpu

    def process_color(self, camera_id, image):
        cal_mat = self.col_calib[camera_id]
        im_xyz = ((image - 2.0) / 255.0).permute(1, 2, 0) @ cal_mat
        im_rgb = xyz_to_rgb(im_xyz.permute(2, 0, 1)[None])
        im_srgb = linear_rgb_to_rgb(im_rgb)
        return im_srgb[0]

    def calibrate_color(self, camera_id, images):
        if self.col_calib is not None:
            cam_key = "camera" + camera_id
            if cam_key not in self.col_calib:
                return None
            return self.process_color(cam_key, images)

        if self.config.train.get("use_gamma_space", False):
            return linear2color_corr(images / 255.0, dim=0)

        return images / 255.0

    def dilate_alpha(self, alpha, kernel=5):
        kernel = th.ones(kernel, kernel).cuda()
        img = dilation(alpha[None], kernel)
        return img[0]

    def errode_alpha(self, alpha, kernel=5):
        kernel = th.ones(kernel, kernel).cuda()
        img = erosion(alpha[None], kernel)
        return img[0]

    def blur_alpha(self, alpha, sigma=1, kernel=3):
        return gaussian_blur2d(alpha[None], kernel_size=(kernel, kernel), sigma=(sigma, sigma))[0]

    def median_filter(self, image, kernel=5):
        return median_blur(image[None], (kernel, kernel))[0]

    def get_silhouette(self, seg_part):
        C, H, W = seg_part.shape
        silhouette = th.ones((H, W, 3)).float().cuda() * float(self.bg_color == "white")
        cages = self.config.cages

        def get_mask(labels, seg_part):
            mask = th.zeros_like(seg_part).bool()
            for label in labels:
                if label != -1:
                    mask = mask | (seg_part == label)

            return mask

        keys = self.config.cages.keys()
        if ("body" in keys and len(keys) == 1) or ("body" in keys and "face" in keys and len(keys) == 2):
            face = get_mask(cages["face"].label_id if "face" in cages else [-1], seg_part)
            body = ~(seg_part == 0) & ~face
        else:
            upper = get_mask(cages["upper"].label_id if "upper" in cages else [-1], seg_part)
            lower = get_mask(cages["lower"].label_id if "lower" in cages else [-1], seg_part)
            face = get_mask(cages["face"].label_id if "face" in cages else [-1], seg_part)
            body = ~(seg_part == 0) & ~upper & ~lower & ~face

            silhouette[upper[0]] = red
            silhouette[lower[0]] = green
            silhouette[face[0]] = gray

        silhouette[body[0]] = blue

        return silhouette.permute(2, 0, 1).float()

    def paste(self, dst, src, crop):
        dst[:, crop[0] : crop[1], crop[2] : crop[3]] = src[:, crop[0] : crop[1], crop[2] : crop[3]]
        return dst

    def process(self, batch):
        # try:
        if batch is None or batch["image"] is None:
            return None

        has_geometry = "geom" in batch

        K = batch["K"]
        Rt = batch["Rt"].cuda()
        images = batch["image"].cuda()
        seg_fg = batch["seg_fg"].cuda().float()
        boundary_fg = batch["boundary_fg"].cuda().float()
        seg_part = batch["seg_part"].cuda().int()
        seg_fg = ((seg_part > 0) | (seg_fg > 0)).float()
        alpha = seg_fg.clone()

        alpha = median_blur(alpha.float(), (7, 7))

        if self.use_erode_mask:
            alpha = erode_mask(alpha)

        if self.use_close_holes:
            alpha = close_holes(alpha)

        motion = batch["lbs_motion"].cuda().float()
        c2w = self.body_model.transfrom_cameras(batch["smplx"] if "smplx" in batch else motion, Rt)
        if has_geometry:
            geom_vertices = batch["geom"].cuda().float()

        frames = []
        for i in range(images.size(0)):
            B, C, H, W = images.shape
            order_frame_idx = batch["_index"]["order_frame_idx"][i].item()
            frame_id = batch["_index"]["frame"][i].item()
            seq_id = batch["_index"]["seq"][i]
            order_cam_idx = batch["_index"]["order_cam_idx"][i].item()
            camera_id = batch["camera_ids"][0][i]

            img = images[i]
            calib_img = self.calibrate_color(camera_id, img)

            # Calibartion exists but does not have the given camera
            if calib_img is None:
                return None

            fx = float(K[i][0][0])
            fy = float(K[i][1][1])
            cx = int(math.floor(K[i][0][2]))
            cy = int(math.floor(K[i][1][2]))

            left_w, right_w = cx, W - cx
            top_h, bottom_h = cy, H - cy
            cx = max(left_w, right_w)
            cy = max(top_h, bottom_h)
            w = int(2 * cx)
            h = int(2 * cy)

            crop_params = np.array([left_w, right_w, top_h, bottom_h, W, H])

            cam2world = c2w[i].cpu().numpy()
            w2c = np.linalg.inv(cam2world)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            if self.bg_color == "white":
                bg_img = calib_img * seg_fg[i] + (1.0 - seg_fg[i])
            else:  # black
                bg_img = calib_img * seg_fg[i]

            frame = {
                "iter_id": frame_id,
                "camera_id": camera_id,
                "frame_id": frame_id,
                "seq_id": seq_id,
                "order_cam_idx": order_cam_idx,
                "order_frame_idx": order_frame_idx,
                "lbs": motion[i],
                "R": R,
                "T": T,
                "K": K[i].cuda(),
                "Rt": Rt[i].cuda(),
                "c2w": cam2world,
                "fl_x": fx,
                "fl_y": fy,
                "FoVx": focal2fov(fx, w),
                "FoVy": focal2fov(fy, h),
                "width": w,
                "height": h,
                "cx": cx,
                "cy": cy,
                "crop": crop_params,
                "image": bg_img,
                "orig_image": calib_img,
                "alpha": alpha[i],
                "boundary_fg": boundary_fg[i],
                "silhouette": self.get_silhouette(seg_part[i]),
            }

            if "ao" in batch:
                frame["ao"] = batch["ao"][i].cuda()

            if "smplx" in batch:
                frame["smplx"] = to_device(batch["smplx"])

            if "face_kpt" in batch:
                frame["face_kpt"] = batch["face_kpt"][i].cuda().float()

            if has_geometry:
                self.mesh_renderer.resize(h, w)
                cameras = PCRenderer.to_cameras(frame)
                # vertices = self.blueman.get(motion)
                vertices, RT = self.body_model.get(motion[i][None], geometry=geom_vertices[i][None], return_rt=True)
                unposed = self.body_model.to_body_model_space(vertices, motion[i][None], RT)
                faces = self.body_model.body_template_faces[None]

                pos_map, normal_map, depth_map, mask = self.mesh_renderer.map(cameras, vertices, faces)

                frame["pose_vertices"] = vertices
                frame["unpose_vertices"] = self.body_model.from_body_model_to_canonical(unposed)
                frame["normal_map"] = paste(normal_map.permute(2, 0, 1), crop_params, bg_color="black")
                frame["position_map"] = paste(pos_map.permute(2, 0, 1), crop_params, bg_color="black")
                frame["mask_map"] = paste(mask.permute(2, 0, 1), crop_params, bg_color="black")
                frame["depth_map"] = paste(depth_map.permute(2, 0, 1), crop_params, bg_color="black")

            frames.append(frame)

        # except Exception as e:
        #     logger.error(str(e))
        #     return None

        return frames
