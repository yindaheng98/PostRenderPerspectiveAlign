import cv2
import numpy as np
import json
import torch
import torch.nn.functional as F

device = torch.device("cuda")

with device:
    color_raw = cv2.imread("output/coffee_martini/frame1/train_interp/ours_30000/renders/00000.png")
    depth_raw = np.load("output/coffee_martini/frame1/train_interp/ours_30000/depth/00000.npz")["depth"][0, ...]
    with open("output/coffee_martini/frame1/train_interp/ours_30000/depth/00000.camera.json", "r") as f:
        camera = json.load(f)
    height, width = camera["height"], camera["width"]

    R = torch.tensor(camera["rotation"])
    t = torch.tensor(camera["position"])
    K = torch.tensor([
        [camera["fx"], 0, camera["width"]/2],
        [0, camera["fy"], camera["height"]/2],
        [0, 0, 1]
    ])
    uv = torch.ones(color_raw.shape, dtype=torch.float32)
    uv[..., 0] = torch.arange(0, width, dtype=torch.float32).unsqueeze(0).expand(height, -1)
    uv[..., 1] = torch.arange(0, height, dtype=torch.float32).unsqueeze(1).expand(-1, width)
    depth = torch.from_numpy(depth_raw).to(device)
    xyz_camera = torch.inverse(K) @ uv.reshape(-1, 3).T * depth.reshape(-1)
    # xyz_camera = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32)).T*1000
    xyz_world = torch.inverse(R) @ (xyz_camera - t.unsqueeze(1))
    xyz = xyz_world.T.cpu().numpy()

    with open("output/coffee_martini/frame1/train_interp/ours_30000/depth/00001.camera.json", "r") as f:
        camera = json.load(f)
    height, width = camera["height"], camera["width"]

    R = torch.tensor(camera["rotation"])
    t = torch.tensor(camera["position"])
    K = torch.tensor([
        [camera["fx"], 0, camera["width"]/2],
        [0, camera["fy"], camera["height"]/2],
        [0, 0, 1]
    ])
    xyz_camera = R @ xyz_world + t.unsqueeze(1)
    uvz = K @ xyz_camera
    uv = (uvz/uvz[-1, ...]).T.reshape(height, width, 3)

    color = torch.from_numpy(color_raw).to(device).permute(2, 0, 1)
    grid = uv[..., :2] / torch.tensor([[[width, height]]]) - 0.5
    warped = F.grid_sample(color.unsqueeze(0).type(torch.float32), grid.unsqueeze(0), mode='bilinear', align_corners=True)[0, ...].type(torch.uint8)
    print(warped)
    
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 3))
    axs = fig.subplots(ncols=3)
    axs[0].set_title('target')
    axs[0].imshow(cv2.imread("output/coffee_martini/frame1/train_interp/ours_30000/renders/00001.png"))
    axs[1].set_title('warped')
    axs[1].imshow(warped.permute(1, 2, 0).cpu().numpy())
    axs[2].set_title('raw')
    axs[2].imshow(color_raw)
    plt.show()
