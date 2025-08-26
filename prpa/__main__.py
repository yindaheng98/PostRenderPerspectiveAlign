import cv2
import numpy as np
import torch
import argparse
import os
from prpa import reconstruction, projection, render, warp, PRPA
from prpa.prpa import set_backend
from prpa.data import read_camera_color, read_camera_depth


parser = argparse.ArgumentParser()
parser.add_argument("--local", type=str, required=True, help="Index of locally rendered image.")
parser.add_argument("--reference", type=str, required=True, help="Index of reference image.")
parser.add_argument("--warped", type=str, required=True, help="Index to save warped image.")
parser.add_argument("--bordermode", type=str, default='grid_sample')
parser.add_argument("--backend", type=str, default='taichi', choices=['torch', 'taichi'])
parser.add_argument("--kernel-size", type=int, default=16, help="Erosion sliding window radius.")
parser.add_argument("--occluded-dilation-size", type=int, default=1, help="Dilation size for occluded mask when selecting source pixels.")
parser.add_argument("--occlude-dilation-size", type=int, default=1, help="Dilation size for occlude mask when selecting source pixels.")
parser.add_argument("--max-iterations", type=int, default=None, help="Max iterations for the error erosion loop. Default: None (unlimited).")
parser.add_argument("--debug", action="store_true")


def main(args):
    os.makedirs(os.path.dirname(args.warped), exist_ok=True)

    # warp a reference image to local rendered image

    # step 1: reconstruct 3D point cloud from local rendered image (image space `uv` to 3D space `xyz`)
    idx_loc = args.local  # local rendered image index
    target = read_camera_depth(idx_loc + ".camera.json", idx_loc + ".depth.npz")
    K, R_c2w, T_c2w, depth = target.K, target.R, target.T, target.depth
    xyz = reconstruction(K, R_c2w, T_c2w, depth)  # xyz[uv on local rendered image] = pos in 3D space

    if args.debug:  # show reconstructed point cloud
        from prpa.data import read_color
        color = torch.tensor(read_color(idx_loc + ".png"))  # local rendered image
        assert color.shape[:2] == depth.shape, ValueError("Size of depth map should match color image")
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        idx = torch.abs(xyz).sum(axis=-1) < 1000
        pcd.points = o3d.utility.Vector3dVector(xyz[idx, ...].cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(color[idx, ...][..., [2, 1, 0]].cpu().numpy().astype(np.float32)/255)
        o3d.visualization.draw_geometries([pcd])

    # step 2: project 3D point cloud to reference image (3D space `xyz` to image space `uv`)
    idx_ref = args.reference  # reference image index
    reference = read_camera_color(idx_ref + ".camera.json", idx_ref + ".png")
    K_r, R_r, t_r, color_ref = reference.K, reference.R, reference.T, reference.color
    uv, z = projection(K_r, R_r, t_r, xyz)  # uv[uv on local rendered image] = uv on reference

    # step 3: render image at local viewport according to projected `uv` and reference color (get color at `uv`)
    rendered = render(uv, color_ref, bordermode=args.bordermode)  # wrap it
    cv2.imwrite(args.warped + ".no_error_erosion.png", rendered.cpu().numpy())  # debug

    if args.debug:  # show rendered image
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(24, 6))
        axs = fig.subplots(ncols=3, nrows=1)
        axs[0].set_title('reference image')
        axs[0].imshow(color_ref[..., [2, 1, 0]].cpu().numpy())
        axs[1].set_title('reconstructed point cloud')
        axs[1].imshow(rendered[..., [2, 1, 0]].cpu().numpy())
        axs[2].set_title('local rendered image ground truth')
        axs[2].imshow(color[..., [2, 1, 0]].cpu().numpy())
        fig.tight_layout(pad=5)
        plt.show()

    # step 3: error erosion
    warped = warp(uv, color_ref, z, bordermode=args.bordermode,
                  kernel_size=args.kernel_size,
                  occluded_dilation_size=args.occluded_dilation_size,
                  occlude_dilation_size=args.occlude_dilation_size,
                  max_iterations=args.max_iterations)  # wrap = render + error erosion
    cv2.imwrite(args.warped + ".png", warped.cpu().numpy())  # debug

    if args.debug:  # show error-eroded image
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(16, 12))
        axs = fig.subplots(ncols=2, nrows=2)
        axs[0, 0].set_title('reference image')
        axs[0, 0].imshow(color_ref[..., [2, 1, 0]].cpu().numpy())
        axs[0, 1].set_title('reconstructed point cloud')
        axs[0, 1].imshow(rendered[..., [2, 1, 0]].cpu().numpy())
        axs[1, 0].set_title('local rendered image ground truth')
        axs[1, 0].imshow(color[..., [2, 1, 0]].cpu().numpy())
        axs[1, 1].set_title('warped image')
        axs[1, 1].imshow(warped[..., [2, 1, 0]].cpu().numpy())
        fig.tight_layout(pad=5)
        plt.show()

    if args.debug:  # speed test
        import time
        for _ in range(2):
            warped = PRPA(
                target, reference, bordermode=args.bordermode,
                kernel_size=args.kernel_size,
                occluded_dilation_size=args.occluded_dilation_size,
                occlude_dilation_size=args.occlude_dilation_size,
                max_iterations=args.max_iterations)
        torch.cuda.synchronize(torch.device("cuda"))

        st = time.time()
        for i in range(10):
            warped = PRPA(
                target, reference, bordermode=args.bordermode,
                kernel_size=args.kernel_size,
                occluded_dilation_size=args.occluded_dilation_size,
                occlude_dilation_size=args.occlude_dilation_size,
                max_iterations=args.max_iterations)  # complete algorithm
        torch.cuda.synchronize(torch.device("cuda"))
        et = time.time()
        print(f"Speed: {(et - st)/10}s")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.backend == 'taichi':
        import taichi as ti
        set_backend('taichi', arch=ti.cuda, offline_cache=False)
    with torch.device("cuda"):
        main(args)
