import cv2
import numpy as np
import json
from itertools import product
import torch
import torch.nn.functional as F
import argparse


def read_camera(idx):
    with open(idx + ".camera.json", "r") as f:
        camera = json.load(f)
    height, width = camera["height"], camera["width"]

    R_c2w = torch.tensor(camera["rotation"])
    T_c2w = torch.tensor(camera["position"])
    K = torch.tensor([
        [camera["fx"], 0, camera["width"]/2],
        [0, camera["fy"], camera["height"]/2],
        [0, 0, 1]
    ])
    return K, R_c2w, T_c2w, height, width


def read_color(idx):
    return torch.tensor(cv2.imread(idx + ".png"))


def read_camera_color(idx):
    return *read_camera(idx), read_color(idx)


def read_depth(idx):
    return torch.tensor(np.load(idx + ".depth.npz")["depth"][0, ...])


def read_camera_depth(idx):
    return *read_camera(idx), read_depth(idx)


def read_camera_rgbd(idx):
    return *read_camera(idx), read_color(idx), read_depth(idx)


def to_pcd(K, R_c2w, T_c2w, height, width, depth):
    """Reconstruct point cloud from camera and depth map"""
    uv = torch.ones((height, width, 3), dtype=torch.float32)
    uv[..., 0] = torch.arange(0, width, dtype=torch.float32).unsqueeze(0).expand(height, -1)
    uv[..., 1] = torch.arange(0, height, dtype=torch.float32).unsqueeze(1).expand(-1, width)
    xyz_camera = torch.inverse(K) @ uv.reshape(-1, 3).T * depth.reshape(-1)
    # xyz_camera = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32)).T*1000
    xyz_world = R_c2w @ xyz_camera + T_c2w.unsqueeze(1)
    return xyz_world.T.reshape(*uv.shape)


def projection(K, R_c2w, T_c2w, height, width, xyz):
    """Project point cloud to camera"""
    xyz_world = xyz.reshape(-1, 3).T
    xyz_camera = torch.inverse(R_c2w) @ (xyz_world - T_c2w.unsqueeze(1))
    uvz = K @ xyz_camera
    uv = (uvz/uvz[-1, ...]).T.reshape(height, width, 3)
    return uv, uvz[-1, ...].reshape(height, width)


def render(uv, color_ref, height, width):
    """Warp (have warping error)"""
    grid = uv[..., :2] / torch.tensor([[[width, height]]]) * 2 - 1
    warped = F.grid_sample(color_ref.permute(2, 0, 1).unsqueeze(0).type(torch.float32), grid.unsqueeze(0),
                           mode='bilinear', align_corners=True)[0, ...].type(torch.uint8)
    return warped.permute(1, 2, 0)


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
    counts = count(uv, height, width)
    # counts_back: 对于local rendered image上的每个点，在投影到reference image上后，有多少个点和它重合？
    counts_back = counts[uv[..., 1], uv[..., 0]]
    mask_overlap = counts_back > 1

    # 图像边缘的点不算在重合点中
    uv_tmp = uv[mask_overlap, ...]
    mask_tmp = mask_overlap[mask_overlap]
    mask_tmp[torch.logical_or(uv_tmp[..., 0] <= 0, uv_tmp[..., 0] >= width-1)] = False
    mask_tmp[torch.logical_or(uv_tmp[..., 1] <= 0, uv_tmp[..., 1] >= height-1)] = False
    mask_overlap[mask_overlap.clone()] = mask_tmp
    # 重合点中与深度最低的点深度相差不大的点不算在重合点中
    min_depth = get_min_depth(uv, depth, height, width)
    # min_depth_back: 对于local rendered image上的每个点，在投影到reference image上后，所有和它重合的点的深度的最小值是多少？
    min_depth_back = min_depth[uv[..., 1], uv[..., 0]]
    depthdiff = torch.abs(depth[mask_overlap] - min_depth_back[mask_overlap])
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(16, 12))
    # ax = fig.subplots()
    # counts, bins = np.histogram(depthdiff.clamp_max(1).cpu().numpy(), bins=100)
    # ax.hist(bins[:-1], bins, weights=counts)
    # plt.show()

    # 哪些点被其他点遮挡了
    mask_tmp = mask_overlap[mask_overlap]
    mask_tmp[depthdiff < depth_diff_thr_for_occlusion] = False
    mask_occluded = mask_overlap.clone()
    mask_occluded[mask_overlap] = mask_tmp

    # 哪些点遮挡了其他点
    occluded_pos_on_ref = uv[mask_occluded, ...]  # 所有在local rendered image上判定为被遮挡的点在reference image上的位置
    occluded_mask_on_ref = torch.zeros(size=(height, width), dtype=torch.uint8).type(torch.bool)
    occluded_mask_on_ref[occluded_pos_on_ref[..., 1], occluded_pos_on_ref[..., 0]] = True  # 在reference image上标记上述位置
    mask_occlude = occluded_mask_on_ref[uv[..., 1], uv[..., 0]]  # 将在reference image上标记的位置再投影回local rendered image上
    mask_occlude &= ~mask_occluded  # 不是被遮挡的点就是遮挡别人的点
    # mask_tmp = mask_overlap[mask_overlap]
    # mask_tmp[depthdiff >= depth_diff_thr_for_occlusion] = False
    # mask_occlude = mask_overlap.clone()
    # mask_occlude[mask_overlap] = mask_tmp
    return mask_occluded, mask_occlude


error_erosion_kernel_size = 5
error_erosion_kernel = list(product(
    range(-error_erosion_kernel_size, error_erosion_kernel_size + 1),
    range(-error_erosion_kernel_size, error_erosion_kernel_size + 1)
))
mask_occlude_dilation_kernel_size = 5
mask_occlude_dilation_padding = 2


def error_erosion(warped, mask_occluded, mask_occlude):
    assert mask_occluded.dim() == mask_occlude.dim() == 2
    assert mask_occluded.shape == mask_occlude.shape
    height, width = mask_occluded.shape

    # get edge of occluded region
    edge = F.max_pool2d(  # dilation the occluded region
        mask_occluded.type(torch.float32)[None, None, ...],
        kernel_size=3, stride=1, padding=1
    ).type(torch.bool)[0, 0, ...] & ~mask_occluded  # xor with the occluded region to get the edge
    # mask_occlude_dilation = F.max_pool2d(  # dilation the occluded region
    #     mask_occlude.type(torch.float32)[None, None, ...],
    #     kernel_size=mask_occlude_dilation_kernel_size, stride=1, padding=mask_occlude_dilation_padding
    # ).type(torch.bool)[0, 0, ...]
    # edge_pos = (edge & ~mask_occlude_dilation).nonzero()  # edge in occlude region is not edge
    edge_pos = edge.nonzero()
    # warped[edge_pos[..., 0], edge_pos[..., 1], ...] = torch.tensor([0, 0, 255], dtype=torch.uint8)  # debug
    # return warped, mask_occluded  # debug

    # get kernel for fixing warping error
    kernel = torch.tensor(error_erosion_kernel)
    kernels = edge_pos.unsqueeze(1) + kernel.unsqueeze(0)  # region to be erosion
    kernels[..., 0].clamp_(0, height-1)
    kernels[..., 1].clamp_(0, width-1)
    # warped[kernels[..., 0], kernels[..., 1], ...] = torch.tensor([0, 0, 255], dtype=torch.uint8)  # debug
    # return warped, mask_occluded  # debug

    # where to assign color
    kernel_assignmask = mask_occluded[kernels[..., 0], kernels[..., 1]]  # only assign color to occluded region
    kernel_assignmask &= ~mask_occlude[kernels[..., 0], kernels[..., 1]]  # donot assign color in occlude region
    # assign_pos = kernels[kernel_assignmask, ...]
    # warped[assign_pos[..., 0], assign_pos[..., 1], ...] = torch.tensor([0, 0, 255], dtype=torch.uint8)  # debug
    # return warped, mask_occluded  # debug

    # where to collect color for avg
    mask_occluded_dilated = F.max_pool2d(  # dilation the occluded region
        mask_occluded.type(torch.float32)[None, None, ...],
        kernel_size=mask_occlude_dilation_kernel_size, stride=1, padding=mask_occlude_dilation_padding
    ).type(torch.bool)[0, 0, ...]
    kernel_avgcolormask = ~mask_occlude[kernels[..., 0], kernels[..., 1]]  # donot use color in occlude region
    kernel_avgcolormask &= ~mask_occluded_dilated[kernels[..., 0],
                                                  kernels[..., 1]]  # donot use color in occluded region
    # avgcolor_pos = kernels[kernel_avgcolormask, ...]
    # warped[avgcolor_pos[..., 0], avgcolor_pos[..., 1], ...] = torch.tensor([0, 255, 0], dtype=torch.uint8)  # debug
    # return warped, mask_occluded  # debug

    # delete those kernel that do not have color for avg
    kernel_validcolorcount = (kernel_avgcolormask).sum(dim=1)
    kernel_valid = kernel_validcolorcount > 0
    kernels = kernels[kernel_valid, ...]
    # warped[kernels[..., 0], kernels[..., 1], ...] = torch.tensor([0, 0, 255], dtype=torch.uint8)  # debug
    # return warped, mask_occluded  # debug
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
    kernel_avgcolor = (kernel_colors.sum(dim=1) / kernel_validcolorcount.unsqueeze(-1)).type(torch.uint8)
    kernel_assigncolor = kernel_avgcolor.unsqueeze(1).expand(-1, kernels.shape[1], -1)[kernel_assignmask, ...]

    # assign avg color in the kernel
    assign_pos = kernels[kernel_assignmask, ...]
    warped[assign_pos[..., 0], assign_pos[..., 1], ...] = kernel_assigncolor
    mask_occluded[assign_pos[..., 0], assign_pos[..., 1]] = False
    return warped, mask_occluded


def warp(uv, color_ref, depth, height, width):
    uv_idx = uv[..., :2].round().type(torch.int64)
    uv_idx[..., 1].clamp_(0, height-1)
    uv_idx[..., 0].clamp_(0, width-1)
    mask_occluded, mask_occlude = is_occlusion(uv_idx, depth, height, width)
    # warped = torch.zeros_like(color_ref)  # inverse
    # warped[uv_idx[..., 1], uv_idx[..., 0], ...] = color_ref  # inverse
    warped = color_ref[uv_idx[..., 1], uv_idx[..., 0], ...]
    # mask_occluded_last = mask_occluded.clone()  # debug
    # warped, mask_occluded = error_erosion(warped, mask_occluded, mask_occlude)
    # warped[mask_occluded_last, :] = torch.tensor([255, 0, 0], dtype=warped.dtype)  # debug
    # warped[mask_occluded, :] = torch.tensor([0, 255, 0], dtype=warped.dtype)  # debug
    while mask_occluded.sum() > 0:
        warped, mask_occluded = error_erosion(warped, mask_occluded, mask_occlude)
    # warped[mask_occlude, :] = torch.tensor([0, 255, 0], dtype=warped.dtype)  # debug
    # warped[edge, :] = torch.tensor([0, 0, 255], dtype=warped.dtype)
    return warped


parser = argparse.ArgumentParser()
parser.add_argument("--local", type=str, required=True, help="Index of locally rendered image.")
parser.add_argument("--reference", type=str, required=True, help="Index of reference image.")
parser.add_argument("--debug", action="store_true")


def main(args):
    # warp a reference image to local rendered image
    idx_loc = args.local  # local rendered image
    idx_ref = args.reference  # reference image
    K, R_c2w, T_c2w, height, width, depth = read_camera_depth(idx_loc)
    xyz = to_pcd(K, R_c2w, T_c2w, height, width, depth)  # xyz[uv on local rendered image] = pos in 3D space
    K_r, R_r, t_r, height_r, width_r = read_camera(idx_ref)
    uv, z = projection(K_r, R_r, t_r, height_r, width_r, xyz)  # uv[uv on local rendered image] = uv on reference
    color = torch.tensor(read_color(idx_loc))  # local rendered image
    color_ref = torch.tensor(read_color(idx_ref))  # reference image
    if args.debug:
        import time
        st = time.time()
        warped = warp(uv, color_ref, z, height, width)  # wrap it
        torch.cuda.synchronize(torch.device("cuda"))
        et = time.time()
        print(et - st)
        rendered = render(uv, color_ref, height, width)  # wrap it
        # cv2.imwrite("warped.png", warped.cpu().numpy())
        # cv2.imwrite("rendered.png", rendered.cpu().numpy())

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        idx = torch.abs(xyz).sum(axis=-1) < 1000
        pcd.points = o3d.utility.Vector3dVector(xyz[idx, ...].cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(color[idx, ...][..., [2, 1, 0]].cpu().numpy().astype(np.float32)/255)
        o3d.visualization.draw_geometries([pcd])

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
