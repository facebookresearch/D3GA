# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from glob import glob
import os

from torch.utils.data.dataset import Dataset

import torch as th
import numpy as np
import cv2
import json
from pathlib import Path
from scipy import ndimage
import joblib

from . import load_opencv_calib

SEQUENCES = ["Sequence1"]

from loguru import logger

def load_smplx(src, frames):
    params = {}
    for i, file in enumerate(sorted(glob(f"{src}/*.json"))):
        f = open(file)
        data = json.load(f)
        frame_id = int(Path(file).stem)
        if frame_id in frames:
            params[frame_id] = {k: np.array(v[0], dtype=np.float32) for k, v in data.items() if k != "id"}
        # params[i] = {k: np.array(v[0], dtype=np.float32) for k, v in data[0].items() if k != "id"}

    return params

def load_thuman_smplx(src, frames):
    smpl_data = np.load(src, allow_pickle=True)
    smpl_data = dict(smpl_data)
    smpl_data = {k: v.astype(np.float32) for k, v in smpl_data.items()}

    params = {}
    for i in range(len(smpl_data["body_pose"])):
        if i not in frames:
            continue
        Rh = smpl_data["global_orient"][i]
        Th = smpl_data["transl"][i]
        poses = smpl_data["body_pose"][i]
        expression = smpl_data["expression"][i]
        shapes = smpl_data["betas"][0]
        jaw_pose = smpl_data["jaw_pose"][i]
        right_hand_pose = smpl_data["right_hand_pose"][i]
        left_hand_pose = smpl_data["left_hand_pose"][i]
        leye_pose = np.zeros([3], dtype=np.float32)
        reye_pose = np.zeros([3], dtype=np.float32)

        # 165 (Importnt the global rotation is in pose!)
        smplx_pose = np.concatenate([Rh, poses, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose])

        params[i] = {
            "id": i,
            "Rh": Rh,
            "Th": Th,
            "body_pose": poses,
            "expression": expression,
            "shapes": shapes,
            "jaw_pose": jaw_pose,
            "right_hand_pose": right_hand_pose,
            "left_hand_pose": left_hand_pose,
            "poses": smplx_pose,
        }

    return params

def transform_pca(pca, pose_conds, sigma_pca = 2.):
    pose_conds = pose_conds.reshape(1, -1)
    lowdim_pose_conds = pca.transform(pose_conds)
    std = np.sqrt(pca.explained_variance_)
    lowdim_pose_conds = np.maximum(lowdim_pose_conds, -sigma_pca * std)
    lowdim_pose_conds = np.minimum(lowdim_pose_conds, sigma_pca * std)
    new_pose_conds = pca.inverse_transform(lowdim_pose_conds)
    return new_pose_conds[0]


class ActorsHQDataset(Dataset):
    def __init__(
        self,
        smplx_poses,
        image,
        image_alpha,
        image_part_mask,
        cameras,
        extrinsics,
        intrinsics,
        test_camera,
        use_all_frames=False,
        eval=False,
        warmup=False,
        image_height=1022,
        image_width=748,
        n_testing_frames=300,
        **kwargs,
    ):
        super().__init__()

        self.image = image
        self.image_height = image_height
        self.image_width = image_width
        self.test_camera = test_camera
        self.image_path = image
        self.image_alpha = image_alpha
        self.eval = eval
        self.image_part_mask_path = image_part_mask
        self.smplx_poses_path = smplx_poses
        self.warmup = warmup
        self.frame_list = None
        self.cameras = {}
        self.cam2idx = {}
        self.smplx = {}
        self.smplx_pca = None
        self.internal_counter = 0
        self.warmup_idx = 0
        self.use_all_frames = use_all_frames
        self.n_testing_frames = n_testing_frames

        for seq in SEQUENCES:
            fmts = dict(sequence=seq)
            if eval:
                self.set_test_frame_list(fmts)
            else:
                self.set_train_frame_list(fmts)

            all_cameras = load_opencv_calib(extrinsics.format(**fmts), intrinsics.format(**fmts))
            cameras = all_cameras.keys()
            self.cameras = {c: all_cameras[c] for c in cameras}

            for i, cam in enumerate(self.cameras.keys()):
                self.cam2idx[cam] = i

            src = self.smplx_poses_path.format(**fmts)
            logger.info(f"Loaded {len(cameras)} cameras from {seq} using {src}")
            logger.info(f"Loaded {len(self.frame_list)} frames from {seq} using {src}")

    def set_test_frame_list(self, fmts):
        seq = fmts["sequence"]
        src = self.smplx_poses_path.format(**fmts)

        files = sorted(glob(f"{src}/*.json"))
        fs = [[seq, str(int(Path(file).stem)).zfill(6)] for file in files]

        logger.info(f"Loaded TEST {len(fs)} available frames!")

        testing_frames = self.n_testing_frames
        if self.use_all_frames:
           testing_frames = 1

        self.frame_list = fs[-testing_frames:]
        fr = list(range(len(fs)))[-testing_frames:]
        self.smplx[seq] = load_smplx(src, fr)

    def set_train_frame_list(self, fmts):
        seq = fmts["sequence"]
        src = self.smplx_poses_path.format(**fmts)

        files = sorted(glob(f"{src}/*.json"))
        fs = [[seq, str(int(Path(file).stem)).zfill(6)] for file in files]

        logger.info(f"Loaded TRAIN {len(fs)} available frames!")

        testing_frames = self.n_testing_frames
        if self.use_all_frames:
           testing_frames = 1

        self.frame_list = fs[:-testing_frames]
        fr = list(range(len(fs)))[:-testing_frames]
        self.smplx[seq] = load_smplx(src, fr)

    def random_frames(self, n=200):
        frames = []
        n = min(n, len(self.frame_list))
        for i in np.random.choice(range(len(self.frame_list)), n, replace=False).tolist():
            frames.append(self.frame_list[i])
        self.frame_list = np.array(frames)

    def n_cameras(self):
        return len(self.cameras)

    def n_frames(self):
        return len(self.frame_list)

    def __len__(self):
        return self.n_frames()

    def load_pca(self):
        actor_name = [part for part in self.image.split(os.sep) if "Actor" in part][0]
        self.smplx_pca = joblib.load(f'pose_pca/pca_{actor_name}.ckpt')
        logger.info(f"PCA checkpoint has been restored for {actor_name}!")

    @staticmethod
    def get_boundary_mask(mask, kernel_size=3):
        """
        :param mask: np.uint8
        :param kernel_size:
        :return:
        """
        mask_bk = mask.copy()
        thres = 128
        mask[mask < thres] = 0
        mask[mask > thres] = 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_erode = cv2.erode(mask.copy(), kernel)
        mask_dilate = cv2.dilate(mask.copy(), kernel)
        boundary_mask = (mask_dilate - mask_erode) == 1
        boundary_mask = np.logical_or(boundary_mask, np.logical_and(mask_bk > 5, mask_bk < 250))

        return boundary_mask, mask == 1

    def __getitem__(self, idx):
        if self.warmup and not self.eval:
            # We sample with low frequency frames
            if (self.internal_counter + 1) % 100 == 0:
                self.warmup_idx = np.random.choice(list(range(0, len(self))))
            idx = self.warmup_idx

        self.internal_counter += 1

        seq, frame = self.frame_list[idx]
        camera_id = np.random.choice(list(self.cameras.keys())) if not self.eval else self.test_camera[0]
        fmts = dict(frame=int(frame), sequence=seq, camera=camera_id)

        sample = {}

        smplx = self.smplx[seq][int(frame)]

        # if self.eval:
        poses = smplx["poses"]

        # if self.smplx_pca is None:
        #     self.load_pca()

        sample["smplx"] = smplx
        sample["lbs_motion"] = poses[:87]  # transform_pca(self.smplx_pca, poses, sigma_pca=1.5)[:87]
        sample["image"] = np.transpose(cv2.imread(self.image_path.format(**fmts))[..., ::-1].astype(np.float32),axes=(2, 0, 1),)
        C, H, W = sample["image"].shape

        seg_path = self.image_part_mask_path.format(**fmts)
        if not os.path.exists(seg_path):
            image_part_mask_path = self.image_part_mask_path.replace("{frame:06d}.png", "{camera}_rgb{frame:06d}.png")
            seg_path = image_part_mask_path.format(**fmts)

        sample["seg_part"] = np.transpose(cv2.imread(seg_path)[..., ::-1].astype(np.float32),axes=(2, 0, 1),)[:, :H, :W]

        mask = cv2.imread(self.image_alpha.format(**fmts))[:, :, 0]
        boundary_fg, mask = self.get_boundary_mask(mask)

        sample["seg_fg"] = mask[None]
        sample["boundary_fg"] = boundary_fg[None]

        mask = sample["seg_fg"] > 0
        parts = sample["seg_part"] * mask
        parts_mask = (parts.sum(axis=0) > 0)[None, ...]
        parts = parts + (mask * 127) * (1 - parts_mask) * mask

        red = parts[0, :, :] == 255
        green = parts[1, :, :] == 255
        blue = parts[2, :, :] == 255
        gray = parts[0, :, :] == 127

        parts = np.zeros([1, H, W])
        parts[red[None, ]] = 1
        parts[green[None, ]] = 2
        parts[blue[None, ]] = 3
        parts[gray[None, ]] = 4

        sample["seg_part"] = parts.astype(np.int32)
        sample["_index"] = dict(frame=idx, seq=seq, order_frame_idx=int(frame), order_cam_idx=self.cam2idx[camera_id])
        sample["camera_ids"] = [camera_id]

        sample.update(self.cameras[camera_id])

        return sample
