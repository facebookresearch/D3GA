# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from loguru import logger


class TetGen:
    def __init__(self, src) -> None:
        self.switches = "pq1.414a0.05"
        self.build_path = f"submodules/tetrahedralize/build/"
        self.src = src

        if not os.path.exists(self.build_path + "/tetra"):
            logger.error(
                f"Tetgen wrapper was not built. Cmake {self.build_path} to get the binary!"
            )

    def run(self):
        cmd = []
        cmd.append(f"cd {self.build_path} &&")
        cmd.append(f"./tetra {self.switches}")  # Tetgen options
        cmd.append(f"{self.src}/cage.ply")  # Input
        cmd.append(f"{self.src}")  # Input
        cmd.append("false")  # Calculate barycetrins

        cmd = " ".join(str(x) for x in cmd)

        logger.info(f"Running tetget cmd = {cmd}")

        os.system(cmd)
