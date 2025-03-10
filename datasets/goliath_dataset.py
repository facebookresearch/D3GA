# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import zipfile
from collections import namedtuple

from enum import Enum
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np

import pandas as pd
import pillow_avif
import torch
import torch.nn.functional as F

from PIL import Image
from pytorch3d.io import load_ply, save_ply
from scipy.ndimage.morphology import binary_dilation
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import pil_to_tensor

from loguru import logger

from tqdm import tqdm

# There are a lot of frame-wise assets. Avoid re-fetching those when we
# switch cameras
CACHE_LENGTH = 160


class CaptureType(Enum):
    BODY = 1
    HEAD = 2
    HAND = 3


Polygon = namedtuple("Polygon", ["vertices", "faces"])

logger = logging.getLogger(__name__)


def get_capture_type(capture_name: str) -> CaptureType:
    if "Head" in capture_name:
        return CaptureType.HEAD
    if "Hand" in capture_name:
        return CaptureType.HAND
    if "Body" in capture_name:
        return CaptureType.BODY
    raise ValueError(
        f"Could not determine capture type from capture name: {capture_name}"
    )


class BodyDataset(Dataset):
    def __init__(
        self,
        root_path: Path,
        shared_assets_path: Path,
        split: str,
        fully_lit_only: bool = True,
        partially_lit_only: bool = False,
        cameras_subset: Optional[Iterable[str]] = None,
        frames_subset: Optional[Iterable[int]] = None,
        regex=None
    ):
        """
        Dataset of heads, hands or bodies

        root_path: The path where the data is located on disk
        shared_assets_path: pt file with shared assets such as common topology
        split: Either "train" or "test"
        fully_lit_only: Whether to use only fully lit frames
        partial_lit_only: Whether to use only partially lit frames
        cameras: Subset of cameras to use, useful for validation/testing
        frames: Subset of frames to use, useful for validation/testing
        """

        if split not in {"train", "test"}:
            raise ValueError(f"Invalid split: {split}. Options are 'train' and 'test'")
        self.root_path: Path = Path(root_path)
        self.shared_assets_path: Path = shared_assets_path
        self.split: str = split
        self.fully_lit_only: bool = fully_lit_only
        self.partially_lit_only: bool = partially_lit_only

        self.capture_type: CaptureType = get_capture_type(self.root_path.name)
        self._get_fn: Callable = {
            CaptureType.BODY: self._get_for_body,
        }.get(self.capture_type)
        assert self._get_fn is not None

        self._static_get_fn: Callable = {
            CaptureType.BODY: self._static_get_for_body,
        }.get(self.capture_type)

        # Get list of cameras after filtering
        self.cameras_subset = set(cameras_subset or {})
        self.cameras = list(self.get_camera_calibration().keys())
        if regex is not None and cameras_subset is None:
            self.cameras = [c for c in self.cameras if regex in c]

        self.frames_subset = set(frames_subset or {})
        self.frames_subset = set(map(int, self.frames_subset))        

    @lru_cache(maxsize=1)
    def load_shared_assets(self) -> Dict[str, Any]:
        return torch.load(self.shared_assets_path, map_location="cpu", weights_only=False)

    def asset_exists(self, frame: int) -> bool:
        if self.capture_type in [CaptureType.HEAD, CaptureType.HAND]:
            return frame in self.get_frame_list(fully_lit_only=self.fully_lit_only)
        return True

    @lru_cache(maxsize=1)
    def get_camera_calibration(self) -> Dict[str, Any]:
        """Loads and parses camera parameters"""
        with open(self.root_path / "camera_calibration.json", "r") as f:
            camera_calibration = json.load(f)["KRT"]
        camera_params = {str(c["cameraId"]): c for c in camera_calibration}
        logger.info(f"Found {len(camera_calibration)} cameras in the calibration file")

        # We might have images for fewer cameras than there are listed in the json file
        image_zips = set([x for x in (self.root_path / "image").iterdir() if x.is_file()])
        image_zips = set([x.name.split(".")[0][3:] for x in image_zips])
        camera_params = {cid: cparams for cid, cparams in camera_params.items() if cid in image_zips} 
        logger.info(f"Left with {len(camera_params)} cameras after filtering for zips present in image/ folder")
        
        # Filter for cameras in the passed subset
        if self.cameras_subset:
            cameras_subset = set(self.cameras_subset)  # No-op if already a set
            camera_params = {cid: cparams for cid, cparams in camera_params.items() if cid in cameras_subset}

            logger.info(f"Left with {len(camera_params)} cameras after filtering for passed camera subset")

        return camera_params

    @lru_cache(maxsize=1)
    def get_camera_parameters(self, camera: str, ds: int = 2) -> Dict[str, Any]:
        krt = self.get_camera_calibration()[camera]

        K = np.array(krt["K"], dtype=np.float32).T
        K[:2, :2] /= ds
        K[:2, 2] = (K[:2, 2] + 0.5) / ds - 0.5

        Rt = np.array(krt["T"], dtype=np.float32).T[:3, :4]
        R, t = Rt[:3, :3], Rt[:3, 3]
        focal = np.array(K[:2, :2], dtype=np.float32)
        princpt = np.array(K[:2, 2], dtype=np.float32)

        return {
            "Rt": Rt,
            "K": K,
            "campos": R.T.dot(-t),
            "camrot": R,
            "focal": focal,
            "princpt": princpt,
        }

    @lru_cache(maxsize=1)
    def get_camera_list(self) -> List[str]:
        return self.cameras

    def filter_frame_list(self, frame_list: List[int]) -> List[int]:
        frames = frame_list
        if self.frames_subset:
            frames = list(set(frame_list).intersection(self.frames_subset))
        return frames

    @lru_cache(maxsize=2)
    def get_frame_list(
        self,
        fully_lit_only: bool = False,
        partially_lit_only: bool = False,
    ) -> List[int]:
        # fully lit only and partially lit only cannot be enabled at the same time
        assert not (fully_lit_only and partially_lit_only)

        df = pd.read_csv(self.root_path / f"frame_splits_list.csv")
        frame_list = df[df.split == self.split].frame.tolist()
        if (
            not (fully_lit_only or partially_lit_only)
            or self.capture_type is CaptureType.BODY
        ):
            # All frames in Body captures are fully lit
            frame_list = list(frame_list)
            return self.filter_frame_list(frame_list) 

        if fully_lit_only:
            fully_lit = {
                frame for frame, index in self.load_light_pattern() if index == 0
            }
            frame_list = [f for f in fully_lit if f in frame_list]
            return self.filter_frame_list(frame_list)

        else:
            light_pattern = self.load_light_pattern_meta()["light_patterns"]
            # NOTE: it only filters the frames with 5 lights on
            partially_lit = {
                frame
                for frame, index in self.load_light_pattern()
                if len(light_pattern[index]["light_index_durations"]) == 5
            }
            frame_list = [f for f in partially_lit if f in frame_list]
            return self.filter_frame_list(frame_list) 

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_3d_keypoints(self, frame: int) -> Optional[Dict[str, Any]]:
        if not self.asset_exists(frame):
            # Asset only exists for fully lit frames
            return None

        zip_path = self.root_path / "keypoints_3d" / "keypoints_3d.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"{frame:06d}.json", "r") as json_file:
                return json.load(json_file)

    def load_segmentation_parts(
        self, frame: int, camera: str
    ) -> Optional[torch.Tensor]:
        if not self.asset_exists(frame):
            # Asset only exists for fully lit frames
            return None

        zip_path = self.root_path / "segmentation_parts" / f"cam{camera}.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"cam{camera}/{frame:06d}.png", "r") as png_file:
                return pil_to_tensor(Image.open(BytesIO(png_file.read())))

    def load_segmentation_fgbg(self, frame: int, camera: str) -> Optional[torch.Tensor]:
        if not self.asset_exists(frame):
            # Asset only exists for fully lit frames
            return None

        zip_path = self.root_path / "segmentation_fgbg" / f"cam{camera}.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"cam{camera}/{frame:06d}.png", "r") as png_file:
                return pil_to_tensor(Image.open(BytesIO(png_file.read())))

    def load_image(self, frame: int, camera: str) -> Image:
        zip_path = self.root_path / "image" / f"cam{camera}.zip"

        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"cam{camera}/{frame:06d}.avif", "r") as avif_file:
                return pil_to_tensor(Image.open(BytesIO(avif_file.read())))

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_registration_vertices(self, frame: int) -> Optional[torch.Tensor]:
        if not self.asset_exists(frame):
            # Asset only exists for fully lit frames
            return None

        zip_path = self.root_path / "kinematic_tracking" / "registration_vertices.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"registration_vertices/{frame:06d}.ply", "r") as ply_file:
                # No faces are included
                vertices, _ = load_ply(BytesIO(ply_file.read()))
                return vertices  # Polygon(vertices=vertices, faces=None)

    @lru_cache(maxsize=1)
    def load_registration_vertices_mean(self) -> np.ndarray:
        mean_path = (
            self.root_path / "kinematic_tracking" / "registration_vertices_mean.npy"
        )
        return np.load(mean_path)

    @lru_cache(maxsize=1)
    def load_registration_vertices_variance(self) -> float:
        verts_path = (
            self.root_path / "kinematic_tracking" / "registration_vertices_variance.txt"
        )
        with open(verts_path, "r") as f:
            return float(f.read())

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_pose(self, frame: int) -> Optional[np.ndarray]:
        if not self.asset_exists(frame):
            # Asset only exists for fully lit frames
            return None

        zip_path = self.root_path / "kinematic_tracking" / "pose.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"pose/{frame:06d}.txt", "r") as f:
                return np.array(
                    [float(i) for i in f.read().splitlines()], dtype=np.float32
                )

    @lru_cache(maxsize=1)
    def load_template_mesh(self) -> torch.Tensor:  # Polygon:
        mesh_path = self.root_path / "kinematic_tracking" / "template_mesh.ply"
        with open(mesh_path, "rb") as f:
            vertices, faces = load_ply(f)
            return vertices  # Polygon(vertices=vertices, faces=faces)

    @lru_cache(maxsize=1)

    def load_floor_transforms(self) -> np.ndarray:
        floor_transform_path = self.root_path / "floor_transformation.txt"
        cam2gp = np.loadtxt(floor_transform_path, dtype=np.float64)
        assert cam2gp.shape == (3, 4)
        cam2gp = np.vstack([cam2gp, np.array([0, 0, 0, 1])])
        gp2cam = np.linalg.inv(cam2gp)
        return {
            "floor_Rt": gp2cam,
            "floor_Rt_inv": cam2gp,
        }

    @lru_cache(maxsize=1)
    def load_template_mesh_unscaled(self) -> Polygon:
        mesh_path = self.root_path / "kinematic_tracking" / "template_mesh_unscaled.ply"
        with open(mesh_path, "rb") as f:
            vertices, faces = load_ply(f)
            return vertices  # Polygon(vertices=vertices, faces=faces)

    @lru_cache(maxsize=1)
    def load_skeleton_scales(self) -> np.ndarray:
        scales_path = self.root_path / "kinematic_tracking" / "skeleton_scales.txt"
        with open(scales_path, "r") as f:
            return np.array([float(i) for i in f.read().splitlines()], dtype=np.float32)

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_ambient_occlusion(self, frame: int) -> Optional[torch.Tensor]:
        if not self.asset_exists(frame):
            # Asset only exists for fully lit frames
            return None
        zip_path = self.root_path / "uv_image" / "ambient_occlusion.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"ambient_occlusion/{frame:06d}.png", "r") as png_file:
                return pil_to_tensor(Image.open(BytesIO(png_file.read())))

    @lru_cache(maxsize=1)
    def load_ambient_occlusion_mean(self) -> torch.Tensor:
        png_path = self.root_path / "uv_image" / "ambient_occlusion_mean.png"
        return pil_to_tensor(Image.open(png_path))

    @lru_cache(maxsize=1)
    def load_color_mean(self) -> torch.Tensor:
        png_path = self.root_path / "uv_image" / "color_mean.png"
        try:
            img = Image.open(png_path)
            return pil_to_tensor(img)
        except Exception as e:
            logger.warning(
                f"error when loading color mean at `{png_path}`, skipping"
            )
            return None

    @lru_cache(maxsize=1)
    def load_color_variance(self) -> float:
        color_var_path = self.root_path / "uv_image" / "color_variance.txt"
        with open(color_var_path, "r") as f:
            return float(f.read())

    @lru_cache(maxsize=1)
    def load_color(self, frame: int) -> Optional[torch.Tensor]:
        if not self.asset_exists(frame):
            # Asset only exists for fully lit frames
            return None

        zip_path = self.root_path / "uv_image" / "color.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"color/{frame:06d}.png", "r") as png_file:
                return pil_to_tensor(Image.open(BytesIO(png_file.read())))

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_scan_mesh(self, frame: int) -> Optional[Polygon]:
        if not self.asset_exists(frame):
            # Asset only exists for fully lit frames
            return None

        zip_path = self.root_path / "scan_mesh" / "scan_mesh.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"{frame:06d}.ply", "r") as ply_file:
                vertices, faces = load_ply(BytesIO(ply_file.read()))
                return Polygon(vertices=vertices, faces=faces)

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_head_pose(self, frame: int) -> np.ndarray:
        zip_path = self.root_path / "head_pose" / "head_pose.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"{frame:06d}.txt", "r") as txt_file:
                lines = txt_file.read().decode("utf-8").splitlines()
                rows = [line.split(" ") for line in lines]
                return np.array(
                    [[float(i) for i in row] for row in rows], dtype=np.float32
                )

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_background(self, camera: str) -> torch.Tensor:
        zip_path = self.root_path / "per_view_background" / "per_view_background.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"{camera}.png", "r") as png_file:
                return pil_to_tensor(Image.open(BytesIO(png_file.read())))

    @lru_cache(maxsize=1)
    def load_light_pattern(self) -> List[Tuple[int]]:
        light_pattern_path = self.root_path / "lights" / "light_pattern_per_frame.json"
        with open(light_pattern_path, "r") as f:
            return json.load(f)

    @lru_cache(maxsize=1)
    def load_light_pattern_meta(self) -> Dict[str, Any]:
        light_pattern_path = self.root_path / "lights" / "light_pattern_metadata.json"
        with open(light_pattern_path, "r") as f:
            return json.load(f)

    @property
    def batch_filter(self) -> Callable:
        return {
            CaptureType.BODY: self._batch_filter_for_body,
            CaptureType.HEAD: self._batch_filter_for_head,
            CaptureType.HAND: self._batch_filter_for_hand,
        }.get(self.capture_type)

    def _batch_filter_for_body(self, batch):
        pass

    @property
    def static_assets(self) -> Dict[str, Any]:
        assets = self._static_get_fn()
        shared_assets = self.load_shared_assets()
        return {
            **shared_assets,
            **assets,
        }

    def _static_get_for_body(self) -> Dict[str, Any]:
        template_mesh = self.load_template_mesh()
        skeleton_scales = self.load_skeleton_scales()
        ambient_occlusion_mean = self.load_ambient_occlusion_mean()
        color_mean = self.load_color_mean()
        krt = self.get_camera_calibration()
        floor_trasnforms = self.load_floor_transforms()
        return {
            "camera_ids": list(krt.keys()),
            "template_mesh": template_mesh,
            "skeleton_scales": skeleton_scales,
            "ambient_occlusion_mean": ambient_occlusion_mean / 255.0,
            "color_mean": color_mean,
            **floor_trasnforms,
        }

    def _get_for_body(self, frame: int, camera: str) -> Dict[str, Any]:
        template_mesh = self.load_template_mesh()
        skeleton_scales = self.load_skeleton_scales()
        pose = self.load_pose(frame)
        image = self.load_image(frame, camera)
        segmentation_parts = self.load_segmentation_parts(frame, camera)
        segmentation_fgbg = (segmentation_parts != 0.0).to(torch.float32)
        camera_parameters = self.get_camera_parameters(camera, ds=4)

        def downsample(x):
            return F.interpolate(x[None], scale_factor=0.5, mode="bilinear")[0]

        row = {
            "camera_ids": [camera],
            "frame_id": frame,
            "image": downsample(image),
            "seg_part": downsample(segmentation_parts),
            "lbs_motion": pose,
            "template_mesh": template_mesh,
            "skeleton_scales": skeleton_scales,
            "seg_fg": downsample(segmentation_fgbg),
            "boundary_fg": 1 - downsample(segmentation_fgbg),
            **camera_parameters,
        }

        row["_index"] = dict(frame=frame, seq="seq", order_frame_idx=int(frame), order_cam_idx=0)

        return row

    def get(self, frame: int, camera: str) -> Dict[str, Any]:
        sample = self._get_fn(frame, camera)
        missing_assets = [k for k, v in sample.items() if v is None]
        if len(missing_assets) != 0:
            logger.warning(
                f"sample was missing these assets: {missing_assets} with idx frame_id=`{frame}`, camera_id=`{camera}` {sample['n_lights']}"
            )
            return None
        else:
            return sample


    def __getitem__(self, idx):
        # TODO(julieta) don't filter every time (it's cached, but still bad practice here)
        frame_list = self.get_frame_list(
            fully_lit_only=self.fully_lit_only,
            partially_lit_only=self.partially_lit_only,
        )
        camera_list = self.get_camera_list()

        frame = frame_list[idx // len(camera_list)]
        camera = camera_list[idx % len(camera_list)]

        try:
            data = self.get(frame, camera)
        except Exception as e:
            logger.warning(
                f"error when loading frame_id=`{frame}`, camera_id=`{camera}`, skipping"
            )
            return None

        return data


    def __len__(self):
        return len(
            self.get_frame_list(
                fully_lit_only=self.fully_lit_only,
                partially_lit_only=self.partially_lit_only,
            )
        ) * len(self.get_camera_list())


def worker_init_fn(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)


def collate_fn(items):
    """Modified form of `torch.utils.data.dataloader.default_collate`
    that will strip samples from the batch if they are ``None``."""
    items = [item for item in items if item is not None]
    return default_collate(items) if len(items) > 0 else None
