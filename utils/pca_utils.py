# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sklearn.decomposition import PCA
import joblib
from loguru import logger
import numpy as np
from pathlib import Path


def build_pca(actor_name, body_poses, n_components=20, save=True):
    body_poses = np.array(body_poses)
    body_poses = body_poses.reshape(body_poses.shape[0], -1)
    logger.info(f"Building PCA for {body_poses.shape} frames...")
    pca = PCA(n_components=n_components)
    pca.fit(body_poses)
    if save:
        Path("pose_pca").mkdir(exist_ok=True)
        joblib.dump(pca, f'pose_pca/pca_{actor_name}.ckpt')

    return pca
