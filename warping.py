import cv2
import numpy as np
import json
import torch
import torch.nn.functional as F


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
    uv = torch.ones((height, width, 3), dtype=torch.float32)
    uv[..., 0] = torch.arange(0, width, dtype=torch.float32).unsqueeze(0).expand(height, -1)
    uv[..., 1] = torch.arange(0, height, dtype=torch.float32).unsqueeze(1).expand(-1, width)
    xyz_camera = torch.inverse(K) @ uv.reshape(-1, 3).T * depth.reshape(-1)
    # xyz_camera = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32)).T*1000
    xyz_world = R_c2w @ xyz_camera + T_c2w.unsqueeze(1)
    return xyz_world.T.reshape(*uv.shape)


def projection(K, R_c2w, T_c2w, height, width, xyz):
    xyz_world = xyz.reshape(-1, 3).T
    xyz_camera = torch.inverse(R_c2w) @ (xyz_world - T_c2w.unsqueeze(1))
    uvz = K @ xyz_camera
    uv = (uvz/uvz[-1, ...]).T.reshape(height, width, 3)
    return uv


def render(uv, color_ref, width, height):
    grid = uv[..., :2] / torch.tensor([[[width, height]]]) * 2 - 1
    warped = F.grid_sample(color_ref.permute(2, 0, 1).unsqueeze(0).type(torch.float32), grid.unsqueeze(0),
                           mode='bilinear', align_corners=True)[0, ...].type(torch.uint8)
    return warped.permute(1, 2, 0)


def count(uv, width, height):
    index = (uv[..., 1] * width + uv[..., 0]).reshape(-1)
    src = torch.ones_like(index)
    counts = torch.zeros(height*width, dtype=src.dtype).scatter_add_(0, index, src)
    return counts.reshape(height, width)


def warp(uv, color_ref, width, height):
    uv_idx = uv[..., :2].round().type(torch.int64)
    uv_idx[..., 1].clamp_(0, height-1)
    uv_idx[..., 0].clamp_(0, width-1)
    counts = count(uv_idx, width, height)
    mask = counts.reshape(height, width) > 1
    warped = torch.zeros_like(color_ref)
    warped[uv_idx[..., 1].clamp(0, height-1), uv_idx[..., 0].clamp(0, width-1), ...] = color_ref  # inverse
    # warped = color_ref[uv_idx[..., 1].clamp(0, height-1), uv_idx[..., 0].clamp(0, width-1), ...]
    warped[mask, :] = 255
    return warped


# warp a reference image to local rendered image
with torch.device("cuda"):
    idx_loc = "output/coffee_martini/frame1-SH1/train_interp/ours_30000/renders/00001"  # local rendered image
    idx_ref = "output/coffee_martini/frame1-SH1/train_interp/ours_30000/renders/00000"  # reference image
    K, R_c2w, T_c2w, height, width, depth = read_camera_depth(idx_loc)
    xyz = to_pcd(K, R_c2w, T_c2w, height, width, depth)  # xyz[uv on local rendered image] = pos in 3D space
    K_r, R_r, t_r, height_r, width_r = read_camera(idx_ref)
    uv = projection(K_r, R_r, t_r, height_r, width_r, xyz)  # uv[uv on local rendered image] = uv on reference
    grid = uv[..., :2] / torch.tensor([[[width, height]]]) * 2 - 1
    color = torch.tensor(read_color(idx_loc))  # local rendered image
    color_ref = torch.tensor(read_color(idx_ref))  # reference image
    warped = warp(uv, color, width, height)  # wrap it
    rendered = render(uv, color_ref, width, height)  # wrap it

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
