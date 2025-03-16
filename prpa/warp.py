import torch
from .occlusion import is_occlusion
from .morph import MorphologyClose
from .erosion import error_erosion


def warp(uv, color_ref, depth):
    height, width = color_ref.shape[:2]
    # done by grid_sample, same result, may be faster?
    # grid = uv[..., :2] / torch.tensor([[[width, height]]]) * 2 - 1
    # warped = F.grid_sample(color_ref.permute(2, 0, 1).unsqueeze(0).type(torch.float32), grid.unsqueeze(0),
    #                        mode='bilinear', align_corners=True)[0, ...].type(torch.uint8).permute(1, 2, 0)
    uv_idx = uv[..., :2]
    uv_idx = uv_idx.round().type(torch.int64)
    # is_edge = uv_idx[..., 1] < 0
    # is_edge |= uv_idx[..., 1] >= height
    # is_edge |= uv_idx[..., 0] < 0
    # is_edge |= uv_idx[..., 1] >= width
    uv_idx[..., 1].clamp_(0, height-1)
    uv_idx[..., 0].clamp_(0, width-1)
    warped = color_ref[uv_idx[..., 1], uv_idx[..., 0], ...]
    # warped = torch.zeros_like(color_ref)  # inverse
    # warped[uv_idx[..., 1], uv_idx[..., 0], ...] = color_ref  # inverse

    mask_occluded, mask_occlude = is_occlusion(uv_idx, depth, height, width)
    mask_occluded = MorphologyClose(mask_occluded)
    mask_occlude = MorphologyClose(mask_occlude)
    # warped[mask_occluded, :] = torch.tensor([0, 0, 255], dtype=warped.dtype)  # debug
    # warped[mask_occlude, :] = torch.tensor([0, 255, 0], dtype=warped.dtype)  # debug
    # return warped

    # mask_occluded_last = mask_occluded.clone()  # debug
    kernel_size, occluded_dilation_size, occlude_dilation_size = 8, 5, 5
    warped, mask_occluded, validcount = error_erosion(
        warped, mask_occluded, mask_occlude,
        kernel_size=kernel_size,
        occluded_dilation_size=occluded_dilation_size,
        occlude_dilation_size=occlude_dilation_size)
    # print(validcount, mask_occluded.sum())  # debug
    while mask_occluded.sum() > 0 and validcount > 0:
        warped, mask_occluded, validcount = error_erosion(
            warped, mask_occluded, mask_occlude,
            kernel_size=kernel_size,
            occluded_dilation_size=occluded_dilation_size,
            occlude_dilation_size=occlude_dilation_size)
        # print(validcount, mask_occluded.sum())  # debug
        if validcount <= 0:
            occluded_dilation_size -= 1
            occlude_dilation_size -= 1
            warped, mask_occluded, validcount = error_erosion(
                warped, mask_occluded, mask_occlude,
                kernel_size=kernel_size,
                occluded_dilation_size=occluded_dilation_size,
                occlude_dilation_size=occlude_dilation_size)
        # print(validcount, mask_occluded.sum())  # debug
    # warped[mask_occluded_last, :] = torch.tensor([255, 0, 0], dtype=warped.dtype)  # debug
    # warped[mask_occluded, :] = torch.tensor([0, 255, 0], dtype=warped.dtype)  # debug
    # warped[mask_occlude, :] = torch.tensor([0, 0, 255], dtype=warped.dtype)  # debug
    # warped[is_edge, ...] = 0
    return warped
