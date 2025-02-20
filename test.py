# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from concurrent.futures import ThreadPoolExecutor
from glob import glob
import os
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm
from datasets import AttrDict
from torch.utils.data import DataLoader
from datasets import load_static_assets
from datasets.goliath_dataset import BodyDataset
from datasets.actorshq_dataset import ActorsHQDataset
from lib.smplman import Smplman
from lib.blueman import Blueman
from models.trainer import Trainer
from recorder.heatmap import compute_errors
from recorder.mesh_renderer import Renderer
from recorder.pc_renderer import PCRenderer
from lib.batch import Batcher
import torch as th
from utils.pca_utils import build_pca
import torch.nn.functional as F
import torchvision
import os

from omegaconf import OmegaConf
import ffmpeg

from globus import *
from utils.text_helpers import write_text
from omegaconf import OmegaConf


from loguru import logger

background = th.tensor([1, 1, 1], dtype=th.float32, device="cuda")
th.manual_seed(33)
noise = th.rand([3, 1024, 667]).cuda().float()
gray = th.ones([3, 1024, 667]).cuda().float() * 151.0 / 255.0


def transform_pca(pca, pose_conds, sigma_pca = 3.):
    pose_conds = pose_conds.reshape(1, -1)
    lowdim_pose_conds = pca.transform(pose_conds)
    std = np.sqrt(pca.explained_variance_)
    lowdim_pose_conds = np.maximum(lowdim_pose_conds, -sigma_pca * std)
    lowdim_pose_conds = np.minimum(lowdim_pose_conds, sigma_pca * std)
    new_pose_conds = pca.inverse_transform(lowdim_pose_conds)
    return new_pose_conds[0]


def recorder(cam_trajectory, trainer, batcher, dataset, render_path, run_name, bg_color, config, pca):
    pc_renderer = PCRenderer(bg_color == "white").cuda()
    mesh_renderer = Renderer(bg_color == "white").cuda()

    ssim, psnr, lpips, counter = 0, 0, 0, 0

    font_color = (0, 0, 0) if bg_color == "white" else (1, 1, 1)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
    )

    logger.info(f"Rendering {cam_trajectory} trajectory to {render_path}")
    progress_bar = tqdm(range(len(loader)), desc=f"Trajectory: {cam_trajectory}")

    with ThreadPoolExecutor(max_workers=10) as save_pool:
        for idx, batch in enumerate(loader):
            if batch is None:
                logger.error("Empty batch...")
                continue

            processed = batcher.process(batch)

            for frame in processed:
                # SMPLX case
                if pca is not None:
                    poses = frame["smplx"]["poses"].cpu().numpy()[0]
                    # Project conditioning poses onto PCA basis
                    projected_pose = transform_pca(pca, poses, sigma_pca=2.0)
                    frame["lbs"] = th.as_tensor(projected_pose).cuda().float()

                if "test" not in cam_trajectory:
                    frame["image"] = None

                frame["iteration"] = float("inf")
                pkg = trainer.fit(frame)

                garment_pkg = pkg["garment_pkg"]
                sil_pred = pkg["pred_silhouette"]
                rendering = pkg["pred_image"]

                C, H, W = rendering.shape
                pc_renderer.resize(H, W)
                mesh_renderer.resize(H, W)

                cameras = PCRenderer.to_cameras(frame)

                gt_image = rendering.clone()
                if "test" in cam_trajectory:
                    gt_image = frame["image"].cuda()

                crop = frame["crop"]

                inputs = []
                outputs = []
                if "geometry" in garment_pkg:
                    for geom in garment_pkg["geometry"]:
                        if geom["name"] == "face":
                            continue
                        faces = geom["faces"]
                        input_render = mesh_renderer(cameras, geom["input_tetpoints"][None], faces).permute(2, 0, 1)
                        deformed_render = mesh_renderer(cameras, geom["deformed_tetpoints"][None], faces).permute(
                            2, 0, 1
                        )
                        inputs.append(
                            write_text(
                                input_render, "Input " + geom["name"], fontColor=font_color
                            )
                        )
                        outputs.append(
                            write_text(
                                deformed_render, "Deformed " + geom["name"], fontColor=font_color
                            )
                        )

                xyz = garment_pkg["means3D"][None]
                means3D = pc_renderer(cameras, xyz).permute(2, 0, 1)
                boundary_fg = 1. - frame['boundary_fg'].float()
                alpha_gt = frame["alpha"][0:1, :, :] * boundary_fg

                heatmap = None
                if frame["image"] is None:
                    row_0 = th.cat([write_text(rendering, "Prediction", fontColor=font_color)] + inputs, dim=2)
                    row_1 = th.cat(
                        [write_text(means3D, "3D Means", fontColor=font_color)] + outputs, dim=2
                    )
                    image = th.cat([row_0, row_1], dim=1)
                else:
                    gt_image_bg = gt_image * alpha_gt + (1 - alpha_gt) * float(bg_color == "white")
                    rendering_bg = rendering
                    heatmap, _ssim, _psnr, _lpips = compute_errors(
                        gt_image_bg, rendering_bg, use_npc=False, pkg=(config, idx, cam_trajectory)
                    )
                    heatmap = write_text(heatmap, f"{_psnr:.3f} (dB)", fontColor=(1, 1, 1))
                    row_1 = th.cat(
                        [
                            write_text(gt_image, "Ground truth", fontColor=font_color),
                            write_text(
                                rendering,
                                "Ours",
                                fontColor=font_color,
                            ),
                            heatmap,
                        ],
                        dim=2,
                    )
                    image = th.cat([row_1], dim=1)

                    ssim += _ssim
                    psnr += _psnr
                    lpips += _lpips
                    counter += 1

                C, H, W = image.shape
                pad_h = H % 2 != 0
                pad_w = W % 2 != 0
                image = F.pad(image, (0, pad_w, 0, pad_h), "constant", 1)

                file_name = os.path.join(render_path, "grid", "{0:05d}".format(idx) + ".png")
                save_pool.submit(torchvision.utils.save_image, image, file_name)

                if cam_trajectory != CAM_360:
                    file_name = os.path.join(render_path, "ground_truth", "{0:05d}".format(idx) + ".png")
                    gt_image = gt_image * alpha_gt
                    gt_image = th.cat([gt_image, alpha_gt])
                    save_pool.submit(torchvision.utils.save_image, gt_image, file_name)

                file_name = os.path.join(render_path, "prediction", "{0:05d}".format(idx) + ".png")
                save_pool.submit(torchvision.utils.save_image, rendering, file_name)

                if heatmap is not None:
                    file_name = os.path.join(render_path, "heatmap", "{0:05d}".format(idx) + ".png")
                    save_pool.submit(torchvision.utils.save_image, heatmap, file_name)

                progress_bar.set_postfix({"Saved": "{0:05d}".format(idx) + ".png"})
                progress_bar.update(1)

    if ssim != 0:
        n = counter
        error_path = str(Path(render_path).parent.joinpath(f"errors_{cam_trajectory}.txt"))
        with open(error_path, "w") as f:
            f.write(f"SSIM: {(ssim / n):.5f}, ")
            f.write(f"PSNR: {(psnr / n):.5f}, ")
            f.write(f"LPIPS: {(lpips / n):.5f}\n")

    glob_path = f"{render_path}/grid/*.png"
    outputs = ffmpeg.input(glob_path, pattern_type="glob", r=DEFAULT_FPS)
    videos = [
        outputs.output(f"{render_path}_{run_name}_LQ.mp4", pix_fmt='yuv420p', crf=20)
        .overwrite_output()
        .run_async(),
    ]

    for value in videos:
        value.wait()

    progress_bar.close()


def last_checkpoint(config):
    return sorted(glob(os.path.join(config, "checkpoints") + "/*.pth"))[-1]


def create_dataset(config, cameras_subset=None):
    name = config.dataset_name
    dataset, static_assets = None, None

    if name == "actorshq":
        static_assets = load_static_assets(config)
        config.train.erode_mask = False
        dataset = ActorsHQDataset(
            test_camera=[config.train.test_camera],
            eval=True,
            **config.data,
        )
        static_assets["camera_ids"] = list(dataset.cameras.keys())
    elif name == "goliath":
        dataset = BodyDataset(
            root_path=config.data.root_path,
            split="test",
            shared_assets_path=config.data.shared_assets_path,
            regex="402",
            cameras_subset=cameras_subset
        )
        static_assets = AttrDict(dataset.load_shared_assets())
        static_assets["lbs_template_verts"] = dataset.load_template_mesh()
        static_assets["lbs_scale"] = dataset.load_skeleton_scales()
        static_assets["camera_ids"] = list(dataset.cameras)
    else:
        raise NotImplementedError(f"Dataset {name} not implemented!")

    return dataset, static_assets


def create_output(render_path):
    os.makedirs(render_path + "/grid", exist_ok=True)
    os.makedirs(render_path + "/ground_truth", exist_ok=True)
    os.makedirs(render_path + "/prediction", exist_ok=True)
    os.makedirs(render_path + "/heatmap", exist_ok=True)


def build_pca_pillow(model_state, n_components=15):
    M = []
    for key, value in model_state.items():
        if 'optimizable_poses' in key:
            M.append(value.cpu().numpy()[0])

    logger.info(f"Restored {len(M)} refined SMPLX poses for PCA building!")
    if len(M) == 0:
        return None
    pca = build_pca("", M, n_components=n_components, save=False)
    return pca


def save_optimized_elements(model_state, dst):
    poses = {}
    rots = {}
    transl = {}

    def parse(key):
        return int(key.split('.')[-1].split('_')[-1])

    for key, value in tqdm(model_state.items()):
        if 'optimizable_poses' in key:
            poses[parse(key)] = value.cpu().numpy()[0]
        if 'optimizable_rotations' in key:
            rots[parse(key)] = value.cpu().numpy()[0]
        if 'optimizable_translations' in key:
            transl[parse(key)] = value.cpu().numpy()[0]

    th.save({
        "poses": poses,
        "rotations": rots,
        "translations": transl,
    }, dst)


def test(config, run_name, iteration):
    with th.no_grad():
        _, assets = create_dataset(config)
        if config.dataset_name == "goliath":
            body_model = Blueman(config, assets)
        else:
            body_model = Smplman(config, assets)

        batcher = Batcher(config, body_model)
        trainer = Trainer(config, assets, is_eval=True)

        trainer.eval()

        model_state, iteration = trainer.restore(iteration, strict=False, return_state=True)
        pca = build_pca_pillow(model_state, n_components=30)
        save_optimized_elements(model_state, config.train.run_dir + "/optimized_poses.pth")

        bg_color = "white" if "background" not in config.train else config.train.background

        dataset, _ = create_dataset(config, cameras_subset=[config.train.test_camera])
        render_path = os.path.join(config.train.run_dir, f"cinema_{str(iteration).zfill(6)}", CAM_TEST)
        create_output(render_path)
        recorder(CAM_TEST, trainer, batcher, dataset, render_path, run_name, bg_color, config, pca)


if __name__ == "__main__":
    config_path = sys.argv[1]
    iteration = None
    if len(sys.argv) > 2:
        iteration = sys.argv[2]
    config = config = OmegaConf.load(config_path)
    run_name = Path(config_path).stem

    config.train.use_background = False
    config.train.use_blur = False
    config.train.use_pixel_cal = False
    config.train.erode_mask = False
    config.train.close_holes = False

    DEFAULT_FPS = config.train.fps

    logger.info(f"Generating video with DEFAULT_FPS={DEFAULT_FPS}")

    test(config, run_name, iteration)
