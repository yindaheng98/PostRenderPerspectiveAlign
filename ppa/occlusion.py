import torch


def count(uv, height, width):
    """Count on each pixel on reference image: how many point projected to this pixel?"""
    index = (uv[..., 1] * width + uv[..., 0]).reshape(-1)
    src = torch.ones_like(index)
    counts = torch.zeros(height*width, dtype=src.dtype).scatter_add_(0, index, src)
    return counts.reshape(height, width)


def get_min_depth(uv, depth, height, width):
    """Count on each pixel: get min depth among all point projected to this pixel"""
    index = (uv[..., 1] * width + uv[..., 0]).reshape(-1)
    src = depth.reshape(-1)
    min_depth = torch.zeros(height*width, dtype=depth.dtype).index_reduce_(0, index, src, 'amin', include_self=False)
    return min_depth.reshape(height, width)


depth_diff_thr_for_occlusion = 0.5
no_gaussian_depth = 6e4


def is_occlusion(uv, depth, height, width):
    """Detect whether the pixel is occluded by others when project to another camera"""

    # 与其他点有重合的点
    counts_onref = count(uv, height, width)
    # counts_onloc: 对于local rendered image上的每个点，在投影到reference image上后，有多少个点和它重合？
    counts_onloc = counts_onref[uv[..., 1], uv[..., 0]]
    mask_overlap = counts_onloc > 1

    # 投影到reference image上各像素处的最小深度
    mindepth_onref = get_min_depth(uv, depth, height, width)
    # mindepth_onloc: 对于local rendered image上的每个点，在投影到reference image上后，所有和它重合的点的深度的最小值是多少？
    mindepth_onloc = mindepth_onref[uv[..., 1], uv[..., 0]]

    # 图像边缘的点不算在重合点中
    uv_tmp = uv[mask_overlap, ...]
    mask_tmp = mask_overlap[mask_overlap]
    mask_tmp[torch.logical_or(uv_tmp[..., 0] <= 0, uv_tmp[..., 0] >= width-1)] = False
    mask_tmp[torch.logical_or(uv_tmp[..., 1] <= 0, uv_tmp[..., 1] >= height-1)] = False
    mask_overlap[mask_overlap.clone()] = mask_tmp

    # 无Gaussian的点不算在重合点中
    mask_overlap[depth > no_gaussian_depth] = False

    # 计算深度差，用于区分重合点中：1、哪些点被其他点遮挡了；2、哪些点遮挡了其他点
    depthdiff = torch.abs(depth[mask_overlap] - mindepth_onloc[mask_overlap])
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(16, 12))
    # ax = fig.subplots()
    # counts, bins = np.histogram(depthdiff.clamp_max(1).cpu().numpy(), bins=100)
    # ax.hist(bins[:-1], bins, weights=counts)
    # plt.show()

    # 判定哪些点被其他点遮挡了：1、重合点；2、和最小深度之差大于阈值
    mask_tmp = mask_overlap[mask_overlap]
    mask_tmp[depthdiff < depth_diff_thr_for_occlusion] = False
    mask_occluded = mask_overlap.clone()
    mask_occluded[mask_overlap] = mask_tmp

    # 判定哪些点遮挡了其他点：1、在reference image上和被遮挡点重合；2、未被判定为被遮挡点
    pos_occluded_onref = uv[mask_occluded, ...]  # 所有在local rendered image上判定为被遮挡的点在reference image上的位置
    mask_occluded_onref = torch.zeros(size=(height, width), dtype=torch.uint8).type(torch.bool)
    mask_occluded_onref[pos_occluded_onref[..., 1], pos_occluded_onref[..., 0]] = True  # 在reference image上标记上述位置
    mask_occlude = mask_occluded_onref[uv[..., 1], uv[..., 0]]  # 将在reference image上标记的位置再投影回local rendered image上
    mask_occlude &= ~mask_occluded  # 未被判定为被遮挡点
    # mask_tmp = mask_overlap[mask_overlap]
    # mask_tmp[depthdiff >= depth_diff_thr_for_occlusion] = False  # 有些点的重合不是因为遮挡而是因为在同一个面上收缩导致的，所有这样不行
    # mask_occlude = mask_overlap.clone()
    # mask_occlude[mask_overlap] = mask_tmp
    return mask_occluded, mask_occlude
