import cv2
import numpy as np
import torch
import argparse
import os
from ppa import reconstruction, projection, render, warp
from ppa.data import read_color, read_camera_color, read_camera_depth


parser = argparse.ArgumentParser()
parser.add_argument("--local", type=str, required=True, help="Index of locally rendered image.")
parser.add_argument("--reference", type=str, required=True, help="Index of reference image.")
parser.add_argument("--warped", type=str, required=True, help="Index to save warped image.")
parser.add_argument("--debug", action="store_true")


def main(args):
    os.makedirs(os.path.dirname(args.warped), exist_ok=True)
    # warp a reference image to local rendered image
    idx_loc = args.local  # local rendered image
    idx_ref = args.reference  # reference image
    K, R_c2w, T_c2w, depth = read_camera_depth(idx_loc)
    xyz = reconstruction(K, R_c2w, T_c2w, depth)  # xyz[uv on local rendered image] = pos in 3D space

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

    rendered = render(uv, color_ref)  # wrap it
    cv2.imwrite(args.warped + ".no_error_erosion.png", rendered.cpu().numpy())  # debug

    if args.debug:
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
    cv2.imwrite(args.warped + ".png", warped.cpu().numpy())  # debug

    if args.debug:
        import time
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
