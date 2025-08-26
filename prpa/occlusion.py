import torch


def get_min_depth(uv, depth, height, width):
    """Count on each pixel: get min depth among all point projected to this pixel"""
    index = (uv[..., 1] * width + uv[..., 0]).reshape(-1)
    src = depth.reshape(-1)
    min_depth = torch.zeros(height*width, dtype=depth.dtype, device=depth.device).index_reduce_(0, index, src, 'amin', include_self=False)
    return min_depth.reshape(height, width)


depth_diff_thr_for_occlusion = 0.5
no_gaussian_depth = 6e4


def is_occlusion(uv, depth, height, width):
    """Detect whether the pixel is occluded by others when project to another camera"""

    # 投影到reference image上各像素处的最小深度
    mindepth_onref = get_min_depth(uv, depth, height, width)
    # mindepth_onloc: 对于local rendered image上的每个点，在投影到reference image上后，所有和它重合的点的深度的最小值是多少？
    mindepth_onloc = mindepth_onref[uv[..., 1], uv[..., 0]]

    # 排除图像边缘和无Gaussian的点
    mask_valid = (uv[..., 0] > 0) & (uv[..., 0] < width - 1) & \
                 (uv[..., 1] > 0) & (uv[..., 1] < height - 1) & \
                 (depth <= no_gaussian_depth)

    # 判定遮挡：和最小深度之差大于阈值
    depthdiff = torch.abs(depth - mindepth_onloc)
    mask_occluded = mask_valid & (depthdiff >= depth_diff_thr_for_occlusion)

    # 判定哪些点遮挡了其他点：1、在reference image上和被遮挡点重合；2、未被判定为被遮挡点
    pos_occluded_onref = uv[mask_occluded, ...]  # 所有在local rendered image上判定为被遮挡的点在reference image上的位置
    mask_occluded_onref = torch.zeros(size=(height, width), dtype=torch.uint8, device=pos_occluded_onref.device).type(torch.bool)
    mask_occluded_onref[pos_occluded_onref[..., 1], pos_occluded_onref[..., 0]] = True  # 在reference image上标记上述位置
    mask_occlude = mask_occluded_onref[uv[..., 1], uv[..., 0]]  # 将在reference image上标记的位置再投影回local rendered image上
    mask_occlude &= ~mask_occluded  # 未被判定为被遮挡点
    # mask_tmp = mask_overlap[mask_overlap]
    # mask_tmp[depthdiff >= depth_diff_thr_for_occlusion] = False  # 有些点的重合不是因为遮挡而是因为在同一个面上收缩导致的，所有这样不行
    # mask_occlude = mask_overlap.clone()
    # mask_occlude[mask_overlap] = mask_tmp
    return mask_occluded, mask_occlude
