# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim

lpips_criterion = lpips.LPIPS("vgg").cuda()


def dist_to_rgb_jet(errors, min_dist=0.0, max_dist=1.0):
    import matplotlib as mpl
    import matplotlib.cm as cm

    norm = mpl.colors.Normalize(vmin=min_dist, vmax=max_dist)
    cmap = cm.get_cmap(name="jet")
    colormapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    return colormapper.to_rgba(errors)


def dist_to_rgb(errors):
    h, w, d = errors.shape
    scale = 1.0
    errors = np.clip(0, scale, errors)
    errors = errors.reshape(h * w)
    heat = dist_to_rgb_jet(errors, 0, scale)[:, 0:3]
    heat = heat.reshape(h, w, 3)
    heat = np.minimum(np.maximum(heat * 255, 0), 255).astype(np.uint8)
    return heat


def compute_errors(target, fake, use_npc=False, pkg=None):
    s = ssim(fake, target).mean().item()
    p = psnr(fake, target).mean().item()
    l = lpips_criterion(fake, target).mean().item()

    target = target.permute(1, 2, 0).cpu().numpy()
    fake = fake.permute(1, 2, 0).cpu().numpy()

    errors = np.linalg.norm((target - fake), axis=2, keepdims=True, ord=2)
    heat = dist_to_rgb(errors)
    heatmap = torch.from_numpy(heat).cuda().permute(2, 0, 1) / 255

    return heatmap, s, p, l


def compute_heatmap(target, fake):
    p = psnr(fake, target).mean().item()

    target = target.permute(1, 2, 0).cpu().numpy()
    fake = fake.permute(1, 2, 0).cpu().numpy()

    errors = np.linalg.norm((target - fake), axis=2, keepdims=True, ord=2)
    heat = dist_to_rgb(errors) / 255.0

    return heat.astype(np.float32), p
