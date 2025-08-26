import taichi as ti
import torch


@ti.kernel
def reprojection_kernel(
    depth: ti.types.ndarray(dtype=ti.f32, ndim=2),
    M: ti.types.ndarray(dtype=ti.f32, ndim=2),
    t: ti.types.ndarray(dtype=ti.f32, ndim=1),
    uv_out: ti.types.ndarray(dtype=ti.f32, ndim=3),
    z_out: ti.types.ndarray(dtype=ti.f32, ndim=2),
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

        uv_out[i, j, 0] = uvz_x / uvz_z
        uv_out[i, j, 1] = uvz_y / uvz_z
        uv_out[i, j, 2] = 1.0
        z_out[i, j] = uvz_z


def reprojection(target, camera):
    """Fused reconstruction + projection: depth map -> projected uv coordinates.

    Combines:
        xyz = reconstruction(K, R_c2w, T_c2w, depth)
        uv, z = projection(K_ref, R_ref, T_ref, xyz)
    into a single pass with a precomputed combined transform.
    """
    K, R_c2w, T_c2w, depth = target.K, target.R, target.T, target.depth
    K_ref, R_ref, T_ref = camera.K, camera.R, camera.T
    height, width = depth.shape

    R_ref_inv = torch.inverse(R_ref)
    K_inv = torch.inverse(K)
    M = K_ref @ R_ref_inv @ R_c2w @ K_inv
    t = K_ref @ R_ref_inv @ (T_c2w - T_ref)

    M = M.to(dtype=torch.float32).contiguous()
    t = t.to(dtype=torch.float32).contiguous()
    depth = depth.to(dtype=torch.float32).contiguous()

    uv_out = torch.empty((height, width, 3), dtype=torch.float32, device=depth.device)
    z_out = torch.empty((height, width), dtype=torch.float32, device=depth.device)

    reprojection_kernel(depth, M, t, uv_out, z_out)

    return uv_out, z_out
