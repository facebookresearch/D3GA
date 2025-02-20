# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import kornia
import torch as th
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn
from lib.cameras import batch_to_camera
from renderer import render
from torchvision import transforms, utils, models
import torch

bce = nn.BCELoss()


def bce_loss(pred, gt):
    return bce(pred[None], gt[None])


def log_loss(x):
    return th.mean(9.3 + th.log(x + 0.0001) + th.log(1.0 - x + 0.0001))


def l1_loss(network_output, gt):
    return th.abs(network_output - gt).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def cosine_loss(pred_normals, gt_normals):
    onward = th.ones_like(pred_normals)
    onward[0] = 0
    onward[1] = 0
    mask = (F.cosine_similarity(pred_normals, onward, dim=0) > 0.1).detach()
    N = mask.sum()
    cosine = F.cosine_similarity(pred_normals, gt_normals, dim=0)
    return ((mask + 0.001) * (1.0 - cosine)).sum() / N


def gaussian(window_size, sigma):
    gauss = th.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class BasePerceptualLoss(nn.Module):
    def __init__(
        self,
        net,
    ) -> None:
        super().__init__()
        self.net = net

    def forward(self, preds, targets, mask=None) -> th.Tensor:
        if mask is None:
            C, H, W = targets.shape
            mask = th.ones((1, 1, H, W)).cuda()

        return self.net(preds[None] * 255, targets[None] * 255, mask)


class VGGLoss(nn.Module):
    def __init__(self, n_layers=5):
        super().__init__()
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)  
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers)
            prev_layer = next_layer
        for param in self.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()
        
    @staticmethod
    def crop(img: th.Tensor, i: th.Tensor, j: th.Tensor, size: int) -> th.Tensor:
        _, _, h, w = img.shape
        return img[:, :, min(i, 0) : max(i + size, h), min(j, 0) : max(j + size, w)]

    @staticmethod
    def downsize(img: th.Tensor, scale: int = 2) -> th.Tensor:
        _, _, h, w = img.shape
        if h == 512 and w == 512:
            return img
        return F.interpolate(img, scale_factor=1 / scale, mode="bilinear")

    def random_crop(
        self, pred: th.Tensor, gt: th.Tensor, size=512
    ) -> tuple[th.Tensor, th.Tensor]:
        _, _, h, w = pred.shape

        if w <= size and h <= size:
            return pred, gt

        i = th.randint(0, h - size + 1, size=(1,)).item()
        j = th.randint(0, w - size + 1, size=(1,)).item()

        return self.crop(pred, i, j, size), self.crop(gt, i, j, size)

    def forward(self, pred, gt):
        loss = 0
        source, target = self.random_crop(self.downsize(pred), self.downsize(gt))
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)
        return loss


def crop_image(gt_mask, patch_size, randomly, bg_color, *args):
    mask_uv = torch.argwhere(gt_mask > 0.)
    min_v, min_u = mask_uv.min(0)[0]
    max_v, max_u = mask_uv.max(0)[0]
    len_v = max_v - min_v
    len_u = max_u - min_u
    max_size = max(len_v, len_u)

    cropped_images = []
    if randomly and max_size > patch_size:
        random_v = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
        random_u = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
    for image in args:
        cropped_image = bg_color[:, None, None] * torch.ones((3, max_size, max_size), dtype = image.dtype, device = image.device)
        if len_v > len_u:
            start_u = (max_size - len_u) // 2
            cropped_image[:, :, start_u: start_u + len_u] = image[:, min_v: max_v, min_u: max_u]
        else:
            start_v = (max_size - len_v) // 2
            cropped_image[:, start_v: start_v + len_v, :] = image[:, min_v: max_v, min_u: max_u]

        if randomly and max_size > patch_size:
            cropped_image = cropped_image[:, random_v: random_v + patch_size, random_u: random_u + patch_size]
        else:
            cropped_image = F.interpolate(cropped_image[None], size = (patch_size, patch_size), mode = 'bilinear')[0]
        cropped_images.append(cropped_image)

    if len(cropped_images) > 1:
        return cropped_images
    else:
        return cropped_images[0]
