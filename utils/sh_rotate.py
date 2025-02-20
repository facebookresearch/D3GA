# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th

s_c3 = 0.94617469575  # (3*sqrt(5))/(4*sqrt(pi))
s_c4 = -0.31539156525  # (-sqrt(5))/(4*sqrt(pi))
s_c5 = 0.54627421529  # (sqrt(15))/(4*sqrt(pi))

s_c_scale = 1.0 / 0.91529123286551084
s_c_scale_inv = 0.91529123286551084

s_rc2 = 1.5853309190550713 * s_c_scale
s_c4_div_c3 = s_c4 / s_c3
s_c4_div_c3_x2 = (s_c4 / s_c3) * 2.0

s_scale_dst2 = s_c3 * s_c_scale_inv
s_scale_dst4 = s_c5 * s_c_scale_inv


def rotate_band_0(dst, x, R):
    dst[:, 0] = x[:, 0]


# 9 multiplies
def rotate_band_1(dst, x, R):
    # derived from  SlowRotateBand1
    dst[:, 0] = (
        (R[:, 1, 1][..., None]) * x[:, 0] + (-R[:, 1, 2][..., None]) * x[:, 1] + (R[:, 1, 0][..., None]) * x[:, 2]
    )
    dst[:, 1] = (
        (-R[:, 2, 1][..., None]) * x[:, 0] + (R[:, 2, 2][..., None]) * x[:, 1] + (-R[:, 2, 0][..., None]) * x[:, 2]
    )
    dst[:, 2] = (
        (R[:, 0, 1][..., None]) * x[:, 0] + (-R[:, 0, 2][..., None]) * x[:, 1] + (R[:, 0, 0][..., None]) * x[:, 2]
    )


# 48 multiplies
def rotate_band_2(dst, x, R):
    # Sparse matrix multiply
    sh0 = x[:, 3] + x[:, 4] + x[:, 4] - x[:, 1]
    sh1 = x[:, 0] + s_rc2 * x[:, 2] + x[:, 3] + x[:, 4]
    sh2 = x[:, 0]
    sh3 = -x[:, 3]
    sh4 = -x[:, 1]

    # Rotations.  R0 and R1 just use the raw matrix columns
    r2x = (R[:, 0, 0] + R[:, 0, 1])[..., None]
    r2y = (R[:, 1, 0] + R[:, 1, 1])[..., None]
    r2z = (R[:, 2, 0] + R[:, 2, 1])[..., None]

    r3x = (R[:, 0, 0] + R[:, 0, 2])[..., None]
    r3y = (R[:, 1, 0] + R[:, 1, 2])[..., None]
    r3z = (R[:, 2, 0] + R[:, 2, 2])[..., None]

    r4x = (R[:, 0, 1] + R[:, 0, 2])[..., None]
    r4y = (R[:, 1, 1] + R[:, 1, 2])[..., None]
    r4z = (R[:, 2, 1] + R[:, 2, 2])[..., None]

    # dense matrix multiplication one column at a time

    # column 0
    sh0_x = sh0 * R[:, 0, 0][..., None]
    sh0_y = sh0 * R[:, 1, 0][..., None]
    d0 = sh0_x * R[:, 1, 0][..., None]
    d1 = sh0_y * R[:, 2, 0][..., None]
    d2 = sh0 * (R[:, 2, 0] * R[:, 2, 0] + s_c4_div_c3)[..., None]
    d3 = sh0_x * R[:, 2, 0][..., None]
    d4 = sh0_x * R[:, 0, 0][..., None] - sh0_y * R[:, 1, 0][..., None]

    # column 1
    sh1_x = sh1 * R[:, 0, 2][..., None]
    sh1_y = sh1 * R[:, 1, 2][..., None]
    d0 += sh1_x * R[:, 1, 2][..., None]
    d1 += sh1_y * R[:, 2, 2][..., None]
    d2 += sh1 * (R[:, 2, 2] * R[:, 2, 2] + s_c4_div_c3)[..., None]
    d3 += sh1_x * R[:, 2, 2][..., None]
    d4 += sh1_x * R[:, 0, 2][..., None] - sh1_y * R[:, 1, 2][..., None]

    # column 2
    sh2_x = sh2 * r2x
    sh2_y = sh2 * r2y
    d0 += sh2_x * r2y
    d1 += sh2_y * r2z
    d2 += sh2 * (r2z * r2z + s_c4_div_c3_x2)
    d3 += sh2_x * r2z
    d4 += sh2_x * r2x - sh2_y * r2y

    # column 3
    sh3_x = sh3 * r3x
    sh3_y = sh3 * r3y
    d0 += sh3_x * r3y
    d1 += sh3_y * r3z
    d2 += sh3 * (r3z * r3z + s_c4_div_c3_x2)
    d3 += sh3_x * r3z
    d4 += sh3_x * r3x - sh3_y * r3y

    # column 4
    sh4_x = sh4 * r4x
    sh4_y = sh4 * r4y
    d0 += sh4_x * r4y
    d1 += sh4_y * r4z
    d2 += sh4 * (r4z * r4z + s_c4_div_c3_x2)
    d3 += sh4_x * r4z
    d4 += sh4_x * r4x - sh4_y * r4y

    # extra multipliers
    dst[:, 0] = d0
    dst[:, 1] = -d1
    dst[:, 2] = d2 * s_scale_dst2
    dst[:, 3] = -d3
    dst[:, 4] = d4 * s_scale_dst4


def rotate_shs(R, shs):
    dst = th.zeros_like(shs)

    rotate_band_0(dst[:, :1], shs[:, :1], R)
    rotate_band_1(dst[:, 1:4], shs[:, 1:4], R)
    rotate_band_2(dst[:, 4:10], shs[:, 4:10], R)

    return dst
