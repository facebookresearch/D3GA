# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np
import torch


def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale / 10, thickness=1)
        new_width = textSize[0][0]
        if new_width <= width:
            return scale / 10
    return 1


def put_text(img, text, pos=(10, 30), fontScale=1, thickness=1, fontColor=(1, 1, 1)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pos, font, fontScale, fontColor, thickness, cv2.LINE_AA)
    return img


def write_text(img, text, fontColor=(0, 0, 0), thickness=2, bottom=False, X=15):
    convert_back = False
    device = 'cpu'
    if torch.is_tensor(img):
        device = img.device
        convert_back = True
        img = img.permute(1, 2, 0).cpu().numpy()
    img = np.ascontiguousarray(img).astype(np.float32)
    H, W, C = img.shape
    font_scale = get_optimal_font_scale(" " * 25, W)
    Y = int(font_scale * 30) if not bottom else H - int(font_scale * 15)
    img = put_text(
        img,
        text,
        fontScale=font_scale,
        thickness=thickness,
        fontColor=fontColor,
        pos=(X, Y),
    )

    if convert_back:
        return torch.from_numpy(img).permute(2, 0, 1).to(device)

    return img
