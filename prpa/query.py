import torch
import torch.nn.functional as F
from .reproj import reprojection
from .occlusion import is_occlusion


def query(target, camera, color_ref, bordermode='grid_sample'):
    """Reprojection + color sampling + occlusion detection."""
    uv, depth = reprojection(target, camera)

    height, width = color_ref.shape[:2]
    uv_idx = uv[..., :2]
    uv_idx = uv_idx.round().type(torch.int64)
    uv_idx[..., 1].clamp_(0, height-1)
    uv_idx[..., 0].clamp_(0, width-1)
    if bordermode != 'grid_sample':
        warped = color_ref[uv_idx[..., 1], uv_idx[..., 0], ...]
    else:
        grid = uv[..., :2] / torch.tensor([[[width, height]]], device=uv.device) * 2 - 1
        warped = F.grid_sample(color_ref.float().permute(2, 0, 1).unsqueeze(0), grid.unsqueeze(0),
                               mode='bilinear', align_corners=True)[0, ...].permute(1, 2, 0).type(color_ref.dtype)

    mask_occluded, mask_occlude = is_occlusion(uv_idx, depth, height, width)
    return warped, mask_occluded, mask_occlude
