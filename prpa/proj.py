import torch
import torch.nn.functional as F


def reconstruction(K, R_c2w, T_c2w, depth):
    """Reconstruct point cloud from camera and depth map"""
    height, width = depth.shape
    uv = torch.ones((height, width, 3), dtype=torch.float32, device=depth.device)
    uv[..., 0] = torch.arange(0, width, dtype=torch.float32).unsqueeze(0).expand(height, -1)
    uv[..., 1] = torch.arange(0, height, dtype=torch.float32).unsqueeze(1).expand(-1, width)
    xyz_camera = torch.inverse(K) @ uv.reshape(-1, 3).T * depth.reshape(-1)
    # xyz_camera = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32)).T*1000
    xyz_world = R_c2w @ xyz_camera + T_c2w.unsqueeze(1)
    return xyz_world.T.reshape(*uv.shape)


def projection(K, R_c2w, T_c2w, xyz):
    """Project point cloud to camera"""
    height, width = xyz.shape[:2]
    xyz_world = xyz.reshape(-1, 3).T
    xyz_camera = torch.inverse(R_c2w) @ (xyz_world - T_c2w.unsqueeze(1))
    uvz = K @ xyz_camera
    uv = (uvz/uvz[-1, ...]).T.reshape(height, width, 3)
    return uv, uvz[-1, ...].reshape(height, width)


def reprojection(target, camera):
    """Fused reconstruction + projection: depth map -> projected uv coordinates."""
    xyz = reconstruction(target.K, target.R, target.T, target.depth)
    return projection(camera.K, camera.R, camera.T, xyz)


def render(uv, color_ref, bordermode='grid_sample'):
    """Warp (have warping error)"""
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
    return warped
