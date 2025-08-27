import taichi as ti
import torch
from .common import clamp_index, MAX_CHANNELS


@ti.kernel
def reproject_and_scatter_kernel(
    depth: ti.types.ndarray(dtype=ti.f32, ndim=2),
    M: ti.types.ndarray(dtype=ti.f32, ndim=2),
    t: ti.types.ndarray(dtype=ti.f32, ndim=1),
    color_ref: ti.types.ndarray(dtype=ti.u8, ndim=3),
    warped: ti.types.ndarray(dtype=ti.u8, ndim=3),
    uv_idx: ti.types.ndarray(dtype=ti.i32, ndim=3),
    z_out: ti.types.ndarray(dtype=ti.f32, ndim=2),
    mindepth_onref: ti.types.ndarray(dtype=ti.f32, ndim=2),
    channels: ti.i32,
    height: ti.i32,
    width: ti.i32,
):
    # M = K_ref @ R_ref_inv @ R_c2w @ K_inv   (3x3)
    # t = K_ref @ R_ref_inv @ (T_c2w - T_ref)  (3-vector)
    #
    # Per pixel (i, j):
    #   ray = M @ [j, i, 1]
    #   uvz = ray * depth[i, j] + t
    #   uv  = uvz[:2] / uvz[2]
    for i, j in ti.ndrange(depth.shape[0], depth.shape[1]):
        d = depth[i, j]

        ray_x = M[0, 0] * j + M[0, 1] * i + M[0, 2]
        ray_y = M[1, 0] * j + M[1, 1] * i + M[1, 2]
        ray_z = M[2, 0] * j + M[2, 1] * i + M[2, 2]

        uvz_x = ray_x * d + t[0]
        uvz_y = ray_y * d + t[1]
        uvz_z = ray_z * d + t[2]

        uv_x = uvz_x / uvz_z
        uv_y = uvz_y / uvz_z

        uv_ix = clamp_index(ti.cast(ti.round(uv_x), ti.i32), width)
        uv_iy = clamp_index(ti.cast(ti.round(uv_y), ti.i32), height)

        uv_idx[i, j, 0] = uv_ix
        uv_idx[i, j, 1] = uv_iy
        z_out[i, j] = uvz_z

        # nearest-neighbor color sampling: warped = color_ref[uv_idx[..., 1], uv_idx[..., 0], ...]
        for c in ti.static(range(MAX_CHANNELS)):
            if c < channels:
                warped[i, j, c] = color_ref[uv_iy, uv_ix, c]

        # scatter min depth for occlusion detection
        ti.atomic_min(mindepth_onref[uv_iy, uv_ix], uvz_z)


@ti.kernel
def occlusion_kernel(
    uv_idx: ti.types.ndarray(dtype=ti.i32, ndim=3),
    depth: ti.types.ndarray(dtype=ti.f32, ndim=2),
    mindepth_onref: ti.types.ndarray(dtype=ti.f32, ndim=2),
    mask_occluded: ti.types.ndarray(dtype=ti.u8, ndim=2),
    mask_occluded_onref: ti.types.ndarray(dtype=ti.u8, ndim=2),
    height: ti.i32,
    width: ti.i32,
    depth_diff_thr_for_occlusion: ti.f32,
    no_gaussian_depth: ti.f32,
):
    for i, j in ti.ndrange(uv_idx.shape[0], uv_idx.shape[1]):
        uv_ix = uv_idx[i, j, 0]
        uv_iy = uv_idx[i, j, 1]

        mindepth_onloc = mindepth_onref[uv_iy, uv_ix]

        # 排除图像边缘和无Gaussian的点
        mask_valid = 1
        if uv_ix <= 0 or uv_ix >= width - 1:
            mask_valid = 0
        if uv_iy <= 0 or uv_iy >= height - 1:
            mask_valid = 0
        if depth[i, j] > no_gaussian_depth:
            mask_valid = 0

        # 判定遮挡：和最小深度之差大于阈值
        depthdiff = ti.abs(depth[i, j] - mindepth_onloc)
        is_occluded = mask_valid == 1 and depthdiff >= depth_diff_thr_for_occlusion

        mask_occluded[i, j] = ti.cast(ti.i32(is_occluded), ti.u8)
        if is_occluded:
            mask_occluded_onref[uv_iy, uv_ix] = ti.cast(1, ti.u8)


def query(target, camera, color_ref, bordermode='grid_sample'):
    """Fused reprojection + nearest-neighbor color sampling + occlusion detection."""
    from . import use_cuda
    if use_cuda:
        from . import _C
    from ...occlusion import depth_diff_thr_for_occlusion, no_gaussian_depth

    K, R_c2w, T_c2w, depth = target.K, target.R, target.T, target.depth
    K_ref, R_ref, T_ref = camera.K, camera.R, camera.T
    height, width = color_ref.shape[:2]
    channels = color_ref.shape[2]

    R_ref_inv = torch.inverse(R_ref)
    K_inv = torch.inverse(K)
    M = (K_ref @ R_ref_inv @ R_c2w @ K_inv).to(dtype=torch.float32).contiguous()
    t = (K_ref @ R_ref_inv @ (T_c2w - T_ref)).to(dtype=torch.float32).contiguous()
    depth = depth.to(dtype=torch.float32).contiguous()
    color_ref = color_ref.contiguous()

    warped = torch.empty((*depth.shape, channels), dtype=color_ref.dtype, device=depth.device)
    uv_idx = torch.empty((*depth.shape, 2), dtype=torch.int32, device=depth.device)
    z_out = torch.empty(depth.shape, dtype=torch.float32, device=depth.device)
    mindepth_onref = torch.full((height, width), float('inf'), dtype=torch.float32, device=depth.device)

    # Kernel 1: reproject + nearest-neighbor warp + scatter mindepth
    (_C.reproject_and_scatter if use_cuda else reproject_and_scatter_kernel)(
        depth, M, t, color_ref,
        warped, uv_idx, z_out,
        mindepth_onref,
        channels, height, width,
    )

    # Kernel 2: gather mindepth → compute mask_occluded → scatter mask_occluded_onref
    mask_occluded = torch.empty(depth.shape, dtype=torch.uint8, device=depth.device)
    mask_occluded_onref = torch.zeros((height, width), dtype=torch.uint8, device=depth.device)

    (_C.occlusion if use_cuda else occlusion_kernel)(
        uv_idx, z_out,
        mindepth_onref,
        mask_occluded, mask_occluded_onref,
        height, width,
        float(depth_diff_thr_for_occlusion), float(no_gaussian_depth),
    )

    # 判定哪些点遮挡了其他点：1、在reference image上和被遮挡点重合；2、未被判定为被遮挡点
    uv_idx = uv_idx.to(torch.int64)
    mask_occlude = mask_occluded_onref[uv_idx[..., 1], uv_idx[..., 0]].bool() & ~mask_occluded.bool()
    mask_occluded = mask_occluded.bool()

    return warped, mask_occluded, mask_occlude
