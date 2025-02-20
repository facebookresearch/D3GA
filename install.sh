# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

### Install dependencies ###

pip install -r requirements.txt
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install kornia lpips

export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

### Install submodules ###

cd submodules
pip install ./bvh
pip install ./diff-gaussian-rasterization
pip install ./simple-knn
pip install ./tetra-sampler

cd tetrahedralize
mkdir build
cd build
cmake ..
make
