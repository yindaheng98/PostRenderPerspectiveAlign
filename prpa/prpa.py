from typing import NamedTuple
import torch
from .proj import reprojection
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
    # step 1+2: reconstruct 3D point cloud and project to reference image
    uv, z = reprojection(target, reference)
    color_ref = reference.color

    # step 3: render image at local viewport according to projected `uv` and reference color (get color at `uv`)
    warped = warp(
        uv, color_ref, z, bordermode=bordermode,
        kernel_size=kernel_size,
        occluded_dilation_size=occluded_dilation_size,
        occlude_dilation_size=occlude_dilation_size,
        max_iterations=max_iterations)  # wrap = render + error erosion

    return warped
