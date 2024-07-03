import cv2
import numpy as np
import json
import torch
import torch.nn.functional as F


def read_camera(idx):
    with open(idx + ".camera.json", "r") as f:
        camera = json.load(f)
    height, width = camera["height"], camera["width"]

    R = torch.tensor(camera["rotation"])
    t = torch.tensor(camera["position"])
    K = torch.tensor([
        [camera["fx"], 0, camera["width"]/2],
        [0, camera["fy"], camera["height"]/2],
        [0, 0, 1]
    ])
    return K, R, t, height, width


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


def to_pcd(K, R, t, height, width, depth):
    uv = torch.ones((height, width, 3), dtype=torch.float32)
    uv[..., 0] = torch.arange(0, width, dtype=torch.float32).unsqueeze(0).expand(height, -1)
    uv[..., 1] = torch.arange(0, height, dtype=torch.float32).unsqueeze(1).expand(-1, width)
    xyz_camera = torch.inverse(K) @ uv.reshape(-1, 3).T * depth.reshape(-1)
    # xyz_camera = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32)).T*1000
    xyz_world = torch.inverse(R) @ (xyz_camera - t.unsqueeze(1))
    return xyz_world.T.reshape(*uv.shape)


def projection(K, R, t, height, width, xyz):
    xyz_world = xyz.reshape(-1, 3).T
    xyz_camera = R @ xyz_world + t.unsqueeze(1)
    uvz = K @ xyz_camera
    uv = (uvz/uvz[-1, ...]).T.reshape(height, width, 3)
    return uv


with torch.device("cuda"):
    idx_src = "output/coffee_martini/frame1/train_interp/ours_30000/renders/00001"
    idx_dst = "output/coffee_martini/frame1/train_interp/ours_30000/renders/00000"
    K, R, t, height, width, depth = read_camera_depth(idx_src)
    xyz = to_pcd(K, R, t, height, width, depth)
    K_r, R_r, t_r, height_r, width_r = read_camera(idx_dst)
    uv = projection(K_r, R_r, t_r, height_r, width_r, xyz)
    grid = uv[..., :2] / torch.tensor([[[width, height]]]) * 2 - 1
    color = torch.tensor(read_color(idx_src))
    warped = F.grid_sample(color.permute(2, 0, 1).unsqueeze(0).type(torch.float32), grid.unsqueeze(0),
                           mode='bilinear', align_corners=True)[0, ...].type(torch.uint8)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18, 6))
    axs = fig.subplots(ncols=3)
    axs[0].set_title('target')
    axs[0].imshow(read_color(idx_dst)[..., [2, 1, 0]].cpu().numpy())
    axs[1].set_title('warped')
    axs[1].imshow(warped.permute(1, 2, 0)[..., [2, 1, 0]].cpu().numpy())
    axs[2].set_title('raw')
    axs[2].imshow(color[..., [2, 1, 0]].cpu().numpy())
    plt.show()
