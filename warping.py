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
    return cv2.imread(idx + ".png")


def read_camera_color(idx):
    return *read_camera(idx), read_color(idx)


def read_depth(idx):
    return np.load(idx + ".depth.npz")["depth"][0, ...]


def read_camera_depth(idx):
    return *read_camera(idx), read_depth(idx)


def read_camera_rgbd(idx):
    return *read_camera(idx), read_color(idx), read_depth(idx)


def to_pcd(K, R, t, height, width, depth_raw, device=torch.device("cuda")):
    with device:
        uv = torch.ones((height, width, 3), dtype=torch.float32)
        uv[..., 0] = torch.arange(0, width, dtype=torch.float32).unsqueeze(0).expand(height, -1)
        uv[..., 1] = torch.arange(0, height, dtype=torch.float32).unsqueeze(1).expand(-1, width)
        depth = torch.from_numpy(depth_raw).to(device)
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


device = torch.device("cuda")
with device:
    idx_src = "output/coffee_martini/frame1/train_interp/ours_30000/renders/00000"
    xyz = to_pcd(*read_camera_depth(idx_src))
    color_raw = read_color(idx_src)
    idx_dst = "output/coffee_martini/frame1/train_interp/ours_30000/renders/00001"
    K, R, t, height, width = read_camera(idx_dst)
    uv = projection(K, R, t, height, width, xyz)
    grid = uv[..., :2] / torch.tensor([[[width, height]]]) - 0.5
    color = torch.from_numpy(read_color(idx_src)).to(device)
    warped = F.grid_sample(color.permute(2, 0, 1).unsqueeze(0).type(torch.float32), grid.unsqueeze(0),
                           mode='bilinear', align_corners=True)[0, ...].type(torch.uint8)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 3))
    axs = fig.subplots(ncols=3)
    axs[0].set_title('target')
    axs[0].imshow(cv2.imread("output/coffee_martini/frame1/train_interp/ours_30000/renders/00001.png"))
    axs[1].set_title('warped')
    axs[1].imshow(warped.permute(1, 2, 0).cpu().numpy())
    axs[2].set_title('raw')
    axs[2].imshow(color.cpu().numpy())
    plt.show()
