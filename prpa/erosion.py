import torch
from .morph import MorphologyDilation


def error_erosion(warped, mask_occluded, mask_occlude, mask_occlude_dilated, kernel_size=5, occluded_dilation_size=0):
    assert mask_occluded.dim() == mask_occlude.dim() == 2
    assert mask_occluded.shape == mask_occlude.shape
    height, width = mask_occluded.shape

    # get edge of occluded region
    edge = MorphologyDilation(mask_occluded) & ~mask_occluded  # xor with the occluded region to get the edge
    edge_pos = edge.nonzero()
    # warped[edge_pos[..., 0], edge_pos[..., 1], ...] = torch.tensor([0, 0, 255], dtype=torch.uint8)  # debug
    # return warped, mask_occluded  # debug

    # get kernel for fixing warping error
    kernel = torch.cartesian_prod(
        torch.arange(-kernel_size, kernel_size+1, dtype=torch.int64, device=edge_pos.device),
        torch.arange(-kernel_size, kernel_size+1, dtype=torch.int64, device=edge_pos.device))
    kernels = edge_pos.unsqueeze(1) + kernel.unsqueeze(0)  # region to be erosion
    kernels[..., 0].clamp_(0, height-1)
    kernels[..., 1].clamp_(0, width-1)
    # warped[kernels[..., 0], kernels[..., 1], ...] = torch.tensor([0, 0, 255], dtype=torch.uint8)  # debug
    # return warped, mask_occluded  # debug

    # where to assign color
    kernel_assignmask = mask_occluded[kernels[..., 0], kernels[..., 1]]  # only assign color to occluded region
    kernel_assignmask &= ~mask_occlude[kernels[..., 0], kernels[..., 1]]  # donot assign color in occlude region
    # assign_pos = kernels[kernel_assignmask, ...]  # debug
    # warped[assign_pos[..., 0], assign_pos[..., 1], ...] = torch.tensor([0, 0, 255], dtype=torch.uint8)  # debug
    # return warped, mask_occluded  # debug

    # where to collect color for avg
    mask_occluded_dilated = mask_occluded
    if occluded_dilation_size > 0:
        mask_occluded_dilated = MorphologyDilation(mask_occluded, kernel_size=occluded_dilation_size)
    kernel_avgcolormask = ~mask_occluded_dilated[kernels[..., 0], kernels[..., 1]]  # no use color in occluded region
    kernel_avgcolormask &= ~mask_occlude_dilated[kernels[..., 0], kernels[..., 1]]  # no use color in occlude region
    # avgcolor_pos = kernels[kernel_avgcolormask, ...]  # debug
    # warped[avgcolor_pos[..., 0], avgcolor_pos[..., 1], ...] = torch.tensor([0, 255, 0], dtype=torch.uint8)  # debug
    # return warped, mask_occluded  # debug

    # delete those kernel that do not have color for avg
    kernel_validcolorcount = (kernel_avgcolormask).sum(dim=1)
    kernel_valid = kernel_validcolorcount > 0
    kernels = kernels[kernel_valid, ...]
    # warped[kernels[..., 0], kernels[..., 1], ...] = torch.tensor([0, 0, 255], dtype=torch.uint8)  # debug
    # return warped, mask_occluded  # debug
    if len(kernels) <= 0:
        return warped, mask_occluded, 0
    kernel_assignmask = kernel_assignmask[kernel_valid, ...]
    # assign_pos = kernels[kernel_assignmask, ...]  # debug
    # warped[assign_pos[..., 0], assign_pos[..., 1], ...] = torch.tensor([0, 0, 255], dtype=torch.uint8)  # debug
    # return warped, mask_occluded  # debug
    kernel_avgcolormask = kernel_avgcolormask[kernel_valid, ...]
    # avgcolor_pos = kernels[kernel_avgcolormask, ...]  # debug
    # warped[avgcolor_pos[..., 0], avgcolor_pos[..., 1], ...] = torch.tensor([0, 255, 0], dtype=torch.uint8)  # debug
    # return warped, mask_occluded  # debug
    kernel_validcolorcount = kernel_validcolorcount[kernel_valid]

    # collect and compute avg color in the kernel
    kernel_colors = warped[kernels[..., 0], kernels[..., 1]].type(torch.float32)
    kernel_colors[~kernel_avgcolormask, ...] = 0.
    kernel_avgcolor = (kernel_colors.sum(dim=1) / kernel_validcolorcount.unsqueeze(-1)).type(warped.dtype)
    kernel_assigncolor = kernel_avgcolor.unsqueeze(1).expand(-1, kernels.shape[1], -1)[kernel_assignmask, ...]

    # assign avg color in the kernel
    assign_pos = kernels[kernel_assignmask, ...]
    if len(assign_pos) <= 0:
        return warped, mask_occluded, 0
    warped[assign_pos[..., 0], assign_pos[..., 1], ...] = kernel_assigncolor
    mask_occluded[assign_pos[..., 0], assign_pos[..., 1]] = False
    return warped, mask_occluded, len(assign_pos)
