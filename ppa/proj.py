import torch


def projection(K, R_c2w, T_c2w, xyz):
    """Project point cloud to camera"""
    height, width = xyz.shape[:2]
    xyz_world = xyz.reshape(-1, 3).T
    xyz_camera = torch.inverse(R_c2w) @ (xyz_world - T_c2w.unsqueeze(1))
    uvz = K @ xyz_camera
    uv = (uvz/uvz[-1, ...]).T.reshape(height, width, 3)
    return uv, uvz[-1, ...].reshape(height, width)


def render(uv, color_ref):
    """Warp (have warping error)"""
    height, width = color_ref.shape[:2]
    # done by grid_sample, same result, may be faster?
    # grid = uv[..., :2] / torch.tensor([[[width, height]]]) * 2 - 1
    # warped = F.grid_sample(color_ref.permute(2, 0, 1).unsqueeze(0).type(torch.float32), grid.unsqueeze(0),
    #                        mode='bilinear', align_corners=True)[0, ...].type(torch.uint8).permute(1, 2, 0)
    uv_idx = uv[..., :2]
    uv_idx = uv_idx.round().type(torch.int64)
    uv_idx[..., 1].clamp_(0, height-1)
    uv_idx[..., 0].clamp_(0, width-1)
    warped = color_ref[uv_idx[..., 1], uv_idx[..., 0], ...]
    return warped
