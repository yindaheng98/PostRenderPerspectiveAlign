from typing import NamedTuple
import torch
from .query import query
from .warp import warp


def set_backend(backend='torch', **ti_init_kwargs):
    from .warp import set_backend as set_warp_backend
    set_warp_backend(backend, **ti_init_kwargs)
    global query
    if backend == 'taichi':
        import taichi as ti
        ti.init(**ti_init_kwargs)
        from .kernel.taichi import query as _impl
        query = _impl
    else:
        from .query import query as _impl
        query = _impl


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


def PRPA(target: Target, reference: Reference, bordermode='grid_sample', kernel_size=16, occluded_dilation_size=0, occlude_dilation_size=0, max_iterations=None):
    # step 1+2+3: reprojection + color sampling + occlusion detection
    warped, mask_occluded, mask_occlude = query(target, reference, reference.color, bordermode=bordermode)

    # step 4: error erosion to fill occluded regions
    warped = warp(
        warped, mask_occluded, mask_occlude,
        kernel_size=kernel_size,
        occluded_dilation_size=occluded_dilation_size,
        occlude_dilation_size=occlude_dilation_size,
        max_iterations=max_iterations)

    return warped
