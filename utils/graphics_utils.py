#
# This implementation is based on the original Gaussian Splatting
# codebase released by the GRAPHDECO research group (Inria, 2023).
#
# Significant modifications have been made to the core computational
# components for research purposes.
#
# The original software is distributed under the Gaussian-Splatting License:
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
#
# This repository preserves the same licensing terms.
#


import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear=0.01, zfar=100.0, left=-1.0, right=1.0, bottom=-1.0, top=1.0, type='default'):
    """
    Orthogonal projection matrix following OpenGL protocol (the camera points to -z)
    """
    P = torch.zeros(4, 4)

    if type=='cryoem':
        P[0, 0] = 2.0 / (right - left)
        P[1, 1] = 2.0 / (bottom - top)
        P[2, 2] = 2.0 / (zfar - znear)
        P[0, 3] = -(right + left) / (right - left)
        P[1, 3] = -(bottom + top) / (bottom - top)
        P[2, 3] = -(zfar + znear) / (zfar - znear)
        P[3, 3] = 1.0

        # NDC to screen space (reverse y-axis)
        P[1, 1] = -P[1, 1]
        P[1, 3] = -P[1, 3]
    else:
        z_sign = -1.0 # the camera points to -z in OpenGL

        P[0, 0] = 2.0 / (right - left)
        P[1, 1] = 2.0 / (top - bottom)
        P[2, 2] = 2.0 * z_sign / (zfar - znear)
        P[0, 3] = -(right + left) / (right - left)
        P[1, 3] = -(top + bottom) / (top - bottom)
        P[2, 3] = -(zfar + znear) / (zfar - znear)
        P[3, 3] = 1.0

        # NDC to screen space (reverse y-axis)
        P[1, 1] = -P[1, 1]
        P[1, 3] = -P[1, 3]


    return P