# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import cv2
import copy
from datasets import AttrDict
from omegaconf import OmegaConf
from datasets.actorshq_dataset import ActorsHQDataset
from datasets.goliath_dataset import BodyDataset
from datasets import load_static_assets
from lib.segmentation import Segmenter
from lib.smplman import Smplman
from lib.blueman import Blueman
from models.trainer import Trainer
from utils.data_utils import to_device

from utils.timers import cuda_timer, cpu_timer

from lib.batch import Batcher

from utils.timers import cuda_timer
from utils.text_helpers import write_text
from recorder.heatmap import compute_heatmap
import numpy as np
import torch as th
from torch.utils.data import DataLoader

import trimesh
from utils.loss_utils import VGGLoss, l1_loss, ssim
import sys
from utils.general_utils import safe_state
from tqdm import tqdm
from recorder.pc_renderer import PCRenderer
from recorder.mesh_renderer import Renderer

from loguru import logger

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def stringify(losses):
    strings = {}
    for k in losses.keys():
        loss = losses[k]
        if th.is_tensor(loss):
            loss = loss.item()
        if isinstance(loss, float):
            strings[k] = f"{loss:.{5}f}"
        else:
            strings[k] = loss
    return strings


def check_loss(losses, iteration):
    total_loss = losses["total_loss"]
    if th.isnan(total_loss):
        loss_str = " ".join([f"{k}={v}" for k, v in stringify(losses).items()])
        logger.error(f"iter={iteration}: {loss_str}")
        raise ValueError("loss is NaN")


def seed_everything(seed=17):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

def training(train_loader, warmup_loader, config, static_assets):
    tb_writer = prepare_output_and_logger(config)

    if config.dataset_name == "goliath":
        body_model = Blueman(config, static_assets)
    else:
        body_model = Smplman(config, static_assets)

    bg_color = "white" if "background" not in config.train else config.train.background
    Segmenter(config, static_assets, body_model).run()
    batcher = Batcher(config, body_model)
    vgg_loss = VGGLoss().cuda()
    pc_renderer = PCRenderer(bg_color == "white").cuda()
    mesh_renderer = Renderer(bg_color == "white").cuda()
    trainer = Trainer(config, static_assets)
    model = trainer.model

    rgb_weight = config.train.rgb_weight
    vgg_weight = config.train.vgg_weight
    fme_weight = config.train.fme_weight
    sil_weight = config.train.sil_weight
    shadow_weight = config.train.shadow_weight
    bg_weight = config.train.bg_weight
    blur_weight = config.train.blur_weight
    lambda_dssim = config.train.lambda_dssim

    measure_time = False

    first_iter = 0
    iterations = config.train.iterations
    first_iter = trainer.restore()
    progress_bar = tqdm(range(first_iter, iterations))
    first_iter += 1
    iteration = first_iter
    skipped_frames = 0

    logger.info(f"Dataset length = {len(train_loader)}")

    warmup_steps = 200_000
    train_iter = iter(train_loader)

    warmup_iter = None
    if warmup_loader is not None:
        warmup_iter = iter(warmup_loader)

    while iteration < iterations + 1:
        with cpu_timer("Fetch data", measure_time):
            try:
                if iteration > warmup_steps:
                    batch = next(train_iter)
                elif warmup_iter is not None:
                    batch = next(warmup_iter)
                else:
                    batch = next(train_iter)
            except Exception as e:
                logger.error(f"Iterator error {str(e)}")
                train_iter = iter(train_loader)
                if warmup_iter is not None:
                    warmup_iter = iter(warmup_loader)
                # exit(-1)

        trainer.save(iteration)

        frames = batcher.process(batch)

        if frames is None:
            logger.info("Skipping empty batch...")
            if skipped_frames > 20:
                break
            skipped_frames += 1
            continue

        skipped_frames = 0

        pkgs = []
        perceptual_loss = []
        color_loss = []
        fme_loss = []
        scale_loss = []
        sil_loss = []
        shadow_loss = []
        bg_loss = []
        blur_loss = []
        code_loss = []

        with cuda_timer("Iteration FORWARD", measure_time):
            # Batch processing
            for frame in frames:
                frame = to_device(frame)
                frame["iteration"] = iteration

                pkg = trainer.fit(frame)

                # Precidtions
                garment_pkg = pkg["garment_pkg"]
                optimizable_poses = garment_pkg["optimizable_poses"]
                pred_silhouette = pkg["pred_silhouette"]
                pred_image = pkg["pred_image"]
                blur_weights = pkg["blur_weights"]
                bg_color = pkg["bg_color"]

                gt_image = pkg["gt_image"]
                gt_alpha = frame["alpha"].expand(3, -1, -1)
                gt_silhouette = frame["silhouette"] * gt_alpha
                gt_image = gt_image * gt_alpha + (1 - gt_alpha) * bg_color[:, None, None]

                boundary_fg = 1. - frame['boundary_fg'].float()
                gt_image = gt_image * boundary_fg + (1. - boundary_fg) * bg_color[:, None, None]
                gt_silhouette = gt_silhouette * boundary_fg

                rgb_l1 = l1_loss(pred_image, gt_image)
                sil_l1 = l1_loss(pred_silhouette, gt_silhouette)
                rgb_ssim = ssim(pred_image, gt_image)
                color = (1.0 - lambda_dssim) * rgb_l1 + lambda_dssim * (1.0 - rgb_ssim)
                blur_reg = (blur_weights - 1.0).abs().mean() if blur_weights is not None else -1
                code_reg = th.mean(garment_pkg["frame_encoding"] ** 2) * 0.001
                l2_poses = 0
                if optimizable_poses is not None:
                    l2_poses = th.mean(th.mean(optimizable_poses**2, axis=-1)) * 0.0075

                color_loss.append(color)
                sil_loss.append(sil_l1)
                code_loss.append(code_reg + l2_poses)
                scale_loss.append(garment_pkg["scale_energy"] * 175.0)

                fm_energy = garment_pkg["fm_energy"]
                if fm_energy is not None:
                    fme_loss.append(fm_energy.mean(dim=0) + 3.0)  # loss is between -3, 3. Shift it to 0, 6

                if blur_reg != -1:
                    blur_loss.append(blur_reg)

                if iteration > config.train.enable_vgg_from and config.train.enable_vgg_from > 0:
                    vgg = vgg_loss(pred_image[None], gt_image[None])
                    perceptual_loss.append(vgg)

                pkgs.append((pred_image.detach(), gt_image, pred_silhouette, gt_silhouette, garment_pkg, frame))

            color_loss = th.stack(color_loss).mean() * rgb_weight
            sil_loss = th.stack(sil_loss).mean() * sil_weight
            code_loss = th.stack(code_loss).mean()
            scale_loss = th.stack(scale_loss).mean()
            bg_loss = th.stack(bg_loss).mean() * bg_weight if len(bg_loss) > 0 else 0

            loss = 0
            loss += color_loss
            loss += sil_loss
            loss += bg_loss
            loss += code_loss
            loss += scale_loss

            fm = 0
            if len(fme_loss) > 0:
                fm = th.stack(fme_loss).mean() * fme_weight
                loss += fm

            blur = 0
            if len(blur_loss) > 0:
                blur = th.stack(blur_loss).mean() * blur_weight
                loss += blur

            vgg = 0
            if len(perceptual_loss) > 0:
                vgg = th.stack(perceptual_loss).mean() * vgg_weight
                loss += vgg

            losses = {
                "color_loss": color_loss,
                "sil_loss": sil_loss,
                "scale_loss": scale_loss,
                "fme_loss": fm,
                "blur_loss": blur,
                "vgg_loss": vgg,
                "bg_loss": bg_loss,
                "codes_reg": code_loss,
                "total_loss": loss,
            }

            check_loss(losses, iteration)

            trainer.step(loss, iteration)

        with th.no_grad():
            n_log = config.train.log_n_steps
            if iteration % n_log == 0:
                allocated = th.cuda.max_memory_allocated()
                capture_id = config.capture_id.split("--")
                capture_id = capture_id[3] if len(capture_id) > 2 else config.capture_id
                memory = f"{allocated / 1024**3:.2f} GB"
                progress_bar.set_description(capture_id + " | " + memory)
                progress_bar.set_postfix(stringify(losses))
                progress_bar.update(n_log)

            if iteration == iterations:
                progress_bar.close()

            # Log and save
            training_report(losses, iteration, pc_renderer, mesh_renderer, pkgs[0], model, tb_writer, config, bg_color)

        iteration += 1

    # Final model saved
    trainer.save(name="model.pth")


def prepare_output_and_logger(config):
    model_path = config.train.run_dir

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(os.path.join(model_path, "means3D"), exist_ok=True)
    os.makedirs(os.path.join(model_path, "progress"), exist_ok=True)
    os.makedirs(os.path.join(model_path, "checkpoints"), exist_ok=True)
    if config.train.use_shs:
        os.makedirs(os.path.join(model_path, "ply"), exist_ok=True)

    return SummaryWriter(log_dir=config.train.tb_dir)


def training_report(losses, iteration, pc_renderer, mesh_renderer, pkg, model, tb_writer, config, bg_color):
    n_log = config.train.log_n_steps
    if iteration % n_log == 0:
        for key in losses.keys():
            tb_writer.add_scalar(f"loss/{key}", losses[key], iteration)

    if iteration % config.train.log_progress_n_steps == 0:
        pred_image, gt_image, pred_sil, sil_gt, garment_pkg, frame = pkg
        C, H, W = pred_image.shape
        pc_renderer.resize(H, W)
        mesh_renderer.resize(H, W)
        cameras = PCRenderer.to_cameras(frame)
        xyz = garment_pkg["means3D"].detach().clone()[None]

        rendered = pred_image.permute(1, 2, 0).detach().cpu().numpy()
        gt = gt_image.permute(1, 2, 0).cpu().numpy()
        sil_pred = pred_sil.permute(1, 2, 0).detach().cpu().numpy()
        sil_gt = sil_gt.permute(1, 2, 0).cpu().numpy()
        means3D = pc_renderer(cameras, xyz).cpu().numpy()
        heatmap, psnr = compute_heatmap(gt_image, pred_image)
        crop = frame["crop"]

        font_color = (0, 0, 0) if bg_color == "white" else (1, 1, 1)

        inputs = []
        outputs = []
        if "geometry" in garment_pkg:
            for geom in garment_pkg["geometry"]:
                if geom["name"] == "face":
                    continue
                faces = geom["faces"]
                input_render = mesh_renderer(cameras, geom["input_tetpoints"][None], faces).cpu().numpy()
                deformed_render = mesh_renderer(cameras, geom["deformed_tetpoints"][None], faces).cpu().numpy()
                inputs.append(
                    write_text(
                        input_render,
                        "Input " + geom["name"].replace("_", " ")
                    )
                )
                outputs.append(
                    write_text(
                        deformed_render,
                        "Deformed " + geom["name"].replace("_", " ")
                    )
                )

        a = np.concatenate(
            [
                write_text(gt, "Ground truth", fontColor=font_color),
                write_text(heatmap, f"Heatmap {psnr:.3f} (dB)", fontColor=(1, 1, 1)),
                write_text(sil_gt, "GT sil", fontColor=font_color),
            ]
            + inputs,
            axis=1,
        )

        b = np.concatenate(
            [
                write_text(rendered, "Prediction", fontColor=font_color),
                write_text(means3D, "3D Means"),
                write_text(sil_pred, "Pred sil", fontColor=font_color),
            ]
            + outputs,
            axis=1,
        )

        grid = np.concatenate([a, b], axis=0)
        progress = cv2.cvtColor((np.clip(grid * 255, 0, 255)).astype(np.uint8), cv2.COLOR_RGB2BGR)

        step_id = str(iteration).zfill(6) + "_" + str(frame["frame_id"]).zfill(6) + "_" + frame["camera_id"]

        model_path = config.train.run_dir

        cv2.imwrite(os.path.join(model_path, "progress", step_id + ".png"), progress)

        step_id = str(iteration).zfill(6) + "_" + str(frame["frame_id"]).zfill(6)
        # if config.train.use_shs:
        #     p = os.path.join(model_path, "ply", step_id + ".ply")
        #     model.save_ply(garment_pkg["means3D"], p)

        if garment_pkg["rgb"] is not None:
            v = garment_pkg["means3D"].detach().cpu().numpy()
            c = garment_pkg["rgb"].detach().cpu().numpy() * 255
            p = os.path.join(model_path, "means3D", step_id + ".ply")
            trimesh.PointCloud(vertices=v, colors=np.clip(np.nan_to_num(c), 0, 255)).export(p)


# Use always the same seed in the case of creating optical flow loader
def seed_worker(worker_id):
    worker_seed = th.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_loader(config):
    generator = th.Generator()
    generator.manual_seed(42)
    
    test_camera = config.train.test_camera
    name = config.dataset_name

    dataset_warmup = None

    if name == "actorshq":
        static_assets = load_static_assets(config)
        dataset = ActorsHQDataset(test_camera=[test_camera], **config.data)
        static_assets["camera_ids"] = list(dataset.cameras.keys())
        static_assets["smplx"] = copy.deepcopy(dataset.smplx)
        dataset_warmup = ActorsHQDataset(test_camera=[test_camera], warmup=True, **config.data)
    elif name == "goliath":
        dataset = BodyDataset(
            root_path=config.data.root_path,
            split="train",
            shared_assets_path=config.data.shared_assets_path,
            regex="402"
            # cameras_subset=["401152", "401195"],
            # frames_subset=["000499"]
        )
        static_assets = AttrDict(dataset.load_shared_assets())
        static_assets["lbs_template_verts"] = dataset.load_template_mesh()
        static_assets["lbs_scale"] = dataset.load_skeleton_scales()
        static_assets["camera_ids"] = list(dataset.cameras)
    else:
        raise RuntimeError("Dataset not supported!")

    batch_size = config.train.get("batch_size", 1)
    num_workers = config.train.get("num_workers", 8)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=True,
        pin_memory=True,
    )

    warmup_loader = None
    if dataset_warmup is not None:
        warmup_loader = DataLoader(dataset_warmup, batch_size=batch_size)

    return train_loader, warmup_loader, static_assets

if __name__ == "__main__":
    config = config = OmegaConf.load(sys.argv[1])

    if not os.path.isabs(config.train.run_dir):
        config.train.run_dir = os.path.join(config.train.run_dir)

    if not "tb_dir" in config.train:
        config.train.tb_dir = os.path.join(config.train.run_dir, "tb")

    logger.info(f"{config.train.run_dir=}")
    logger.info(f"{config.train.tb_dir=}")

    os.makedirs(config.train.run_dir, exist_ok=True)

    OmegaConf.save(config, f"{config.train.run_dir}/config.yml")
    os.makedirs(config.train.ckpt_dir, exist_ok=True)

    logger.info(f"Using {th.cuda.get_device_name(0)}")

    safe_state(True)
    seed_everything()

    train_loader, warmup_loader, static_assets = create_loader(config)

    training(train_loader, warmup_loader, config, static_assets)
