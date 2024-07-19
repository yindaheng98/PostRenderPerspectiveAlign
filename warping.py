import cv2
import numpy as np
import json
import torch
import torch.nn.functional as F
import argparse


def fromJSON(camera):
    height, width = camera["height"], camera["width"]

    R_c2w = torch.tensor(camera["rotation"])
    T_c2w = torch.tensor(camera["position"])
    K = torch.tensor([
        [camera["fx"], 0, camera["width"]/2],
        [0, camera["fy"], camera["height"]/2],
        [0, 0, 1]
    ])
    return K, R_c2w, T_c2w, height, width


def read_camera(idx):
    with open(idx + ".camera.json", "r") as f:
        return fromJSON(json.load(f))


def read_color(idx):
    return torch.tensor(cv2.imread(idx + ".png"))


def read_camera_color(idx):
    K, R_c2w, T_c2w, height, width = read_camera(idx)
    color = read_color(idx)
    assert color.shape[0] == height and color.shape[1] == width, ValueError("Size of color image should match camera")
    return K, R_c2w, T_c2w, color


def read_depth(idx):
    return torch.tensor(np.load(idx + ".depth.npz")["depth"][0, ...])


def read_camera_depth(idx):
    K, R_c2w, T_c2w, height, width = read_camera(idx)
    depth = read_depth(idx)
    assert depth.shape[0] == height and depth.shape[1] == width, ValueError("Size of depth map should match camera")
    return K, R_c2w, T_c2w, depth


def read_camera_rgbd(idx):
    return *read_camera(idx), read_color(idx), read_depth(idx)


def reconstrucion(K, R_c2w, T_c2w, depth):
    """Reconstruct point cloud from camera and depth map"""
    height, width = depth.shape
    uv = torch.ones((height, width, 3), dtype=torch.float32)
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


def MorphologyDilation(binary, kernel_size=1):
    return F.max_pool2d(  # dilation the occluded region
        binary.type(torch.float32)[None, None, ...],
        kernel_size=kernel_size*2+1, stride=1, padding=kernel_size
    ).type(torch.bool)[0, 0, ...]


def error_erosion(warped, mask_occluded, mask_occlude, kernel_size=5, occluded_dilation_size=0, occlude_dilation_size=0):
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
        torch.arange(-kernel_size, kernel_size+1, dtype=torch.int64),
        torch.arange(-kernel_size, kernel_size+1, dtype=torch.int64))
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
    mask_occlude_dilated = mask_occlude
    if occlude_dilation_size > 0:
        mask_occlude_dilated = MorphologyDilation(mask_occlude, kernel_size=occlude_dilation_size)
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


def MorphologyErosion(binary, kernel_size=1):
    return (~F.max_pool2d(  # dilation the occluded region
        (~binary).type(torch.float32)[None, None, ...],
        kernel_size=kernel_size*2+1, stride=1, padding=kernel_size
    ).type(torch.bool))[0, 0, ...]


def MorphologyClose(binary, kernel_size=1):
    return MorphologyErosion(MorphologyDilation(binary, kernel_size=kernel_size), kernel_size=kernel_size)


def warp(uv, color_ref, depth):
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
    return warped


parser = argparse.ArgumentParser()
parser.add_argument("--local", type=str, required=True, help="Index of locally rendered image.")
parser.add_argument("--reference", type=str, required=True, help="Index of reference image.")
parser.add_argument("--debug", action="store_true")


def main(args):
    # warp a reference image to local rendered image
    idx_loc = args.local  # local rendered image
    idx_ref = args.reference  # reference image
    K, R_c2w, T_c2w, depth = read_camera_depth(idx_loc)
    xyz = reconstrucion(K, R_c2w, T_c2w, depth)  # xyz[uv on local rendered image] = pos in 3D space

    if args.debug:
        color = torch.tensor(read_color(idx_loc))  # local rendered image
        assert color.shape[:2] == depth.shape, ValueError("Size of depth map should match color image")
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        idx = torch.abs(xyz).sum(axis=-1) < 1000
        pcd.points = o3d.utility.Vector3dVector(xyz[idx, ...].cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(color[idx, ...][..., [2, 1, 0]].cpu().numpy().astype(np.float32)/255)
        o3d.visualization.draw_geometries([pcd])

    K_r, R_r, t_r, color_ref = read_camera_color(idx_ref)
    uv, z = projection(K_r, R_r, t_r, xyz)  # uv[uv on local rendered image] = uv on reference

    if args.debug:
        rendered = render(uv, color_ref)  # wrap it
        cv2.imwrite("rendered.png", rendered.cpu().numpy())  # debug
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(24, 6))
        axs = fig.subplots(ncols=3, nrows=1)
        axs[0].set_title('reference image')
        axs[0].imshow(color_ref[..., [2, 1, 0]].cpu().numpy())
        axs[1].set_title('reconstructed point cloud')
        axs[1].imshow(rendered[..., [2, 1, 0]].cpu().numpy())
        axs[2].set_title('local rendered image')
        axs[2].imshow(color[..., [2, 1, 0]].cpu().numpy())
        fig.tight_layout(pad=5)
        plt.show()

    warped = warp(uv, color_ref, z)  # wrap it

    if args.debug:
        import time
        cv2.imwrite("warped.png", warped.cpu().numpy())
        st = time.time()
        for i in range(10):
            warped = warp(uv, color_ref, z)  # wrap it
        torch.cuda.synchronize(torch.device("cuda"))
        et = time.time()
        print((et - st)/100)

    if args.debug:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(16, 12))
        axs = fig.subplots(ncols=2, nrows=2)
        axs[0, 0].set_title('reference image')
        axs[0, 0].imshow(color_ref[..., [2, 1, 0]].cpu().numpy())
        axs[0, 1].set_title('reconstructed point cloud')
        axs[0, 1].imshow(rendered[..., [2, 1, 0]].cpu().numpy())
        axs[1, 0].set_title('local rendered image')
        axs[1, 0].imshow(color[..., [2, 1, 0]].cpu().numpy())
        axs[1, 1].set_title('warped image')
        axs[1, 1].imshow(warped[..., [2, 1, 0]].cpu().numpy())
        fig.tight_layout(pad=5)
        plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    with torch.device("cuda"):
        main(args)
