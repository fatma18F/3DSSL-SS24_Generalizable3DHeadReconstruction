#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple, Optional
from gaussian_splatting.utils.sh_utils import SH2RGB


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

    @classmethod
    def from_obj(cls, file_path):
        vertices = []
        normals = []
        faces = []

        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                if not parts:
                    continue

                if parts[0] == 'v':  # Vertex definition
                    # Assumes vertices are defined by 'v x y z'
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'vn':  # Normal definition
                    # Assumes normals are defined by 'vn x y z'
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])

            vertex_array = np.array(vertices, dtype=float)
            normal_array = np.array(normals, dtype=float) if normals else None
            shs = torch.randn((len(vertex_array), 3))/ 255.0
            colors=SH2RGB(shs)
            return cls(points=vertex_array, normals=normal_array, colors=colors)


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


def getProjectionMatrix(znear: float, zfar: float,
                        fovX: float, fovY: float,
                        width: int, height: int,
                        cx: Optional[float] = None, cy: Optional[float] = None):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # shift the frame window due to the non-zero principle point offsets
    # Assume principle point is center of image if not given
    cx = width / 2 if cx is None else cx
    cy = height / 2 if cy is None else cy
    focal_x = fov2focal(fovX, width)
    focal_y = fov2focal(fovY, height)
    offset_x = cx - (width / 2)
    offset_x = (offset_x / focal_x) * znear
    offset_y = cy - (height / 2)
    offset_y = (offset_y / focal_y) * znear

    top = top + offset_y
    left = left + offset_x
    right = right + offset_x
    bottom = bottom + offset_y

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))
