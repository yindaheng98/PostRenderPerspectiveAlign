from typing import NamedTuple
import torch
from .recon import reconstruction
from .proj import projection
from .warp import warp


class Camera(NamedTuple):
    K: torch.Tensor
    R: torch.Tensor
    T: torch.Tensor


class Target(Camera):
    depth: torch.Tensor

    def __new__(cls, K, R, T, depth):
        self = super(Target, cls).__new__(cls, K, R, T)
        self.depth = depth
        return self


class Reference(Camera):
    color: torch.Tensor

    def __new__(cls, K, R, T, color):
        self = super(Reference, cls).__new__(cls, K, R, T)
        self.color = color
        return self


def PRPA(target: Target, reference: Reference, bordermode='grid_sample', kernel_size=16, occluded_dilation_size=1, occlude_dilation_size=1, max_iterations=None):
    # step 1: reconstruct 3D point cloud from local rendered image (image space `uv` to 3D space `xyz`)
    K, R_c2w, T_c2w, depth = target.K, target.R, target.T, target.depth
    xyz = reconstruction(K, R_c2w, T_c2w, depth)  # xyz[uv on local rendered image] = pos in 3D space

    # step 2: project 3D point cloud to reference image (3D space `xyz` to image space `uv`)
    K_r, R_r, t_r, color_ref = reference.K, reference.R, reference.T, reference.color
    uv, z = projection(K_r, R_r, t_r, xyz)  # uv[uv on local rendered image] = uv on reference

    # step 3: render image at local viewport according to projected `uv` and reference color (get color at `uv`)
    warped = warp(
        uv, color_ref, z, bordermode=bordermode,
        kernel_size=kernel_size,
        occluded_dilation_size=occluded_dilation_size,
        occlude_dilation_size=occlude_dilation_size,
        max_iterations=max_iterations)  # wrap = render + error erosion

    return warped
