import torch
import os
import json
from tqdm import tqdm
from os import makedirs
import torchvision
import numpy as np
from matplotlib import colors
from matplotlib import colormaps
from gaussian_splatting import GaussianModel
from gaussian_splatting.camera import camera2dict, dict2camera
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.render import prepare_rendering


def convert_dataset(dataset: CameraDataset, fix_width, fix_height, fix_width_focal, fix_height_focal):
    camera_dicts = []
    for idx, camera in enumerate(dataset):
        camera_dict = camera2dict(camera, idx)
        camera_dict["width"] = fix_width
        camera_dict["height"] = fix_height
        camera_dict["fx"] = fix_width_focal
        camera_dict["fy"] = fix_height_focal
        camera_dict["device"] = camera.R.device
        camera_dict["custom_data"] = camera.custom_data
        camera_dicts.append(camera_dict)
    return sorted(camera_dicts, key=lambda camera: camera["ground_truth_image_path"])


def colorify_depth(depth: torch.Tensor) -> torch.Tensor:
    device = depth.device
    shape = depth.shape[1:]
    depth = depth.reshape(-1).cpu().numpy()
    dsort = np.sort(depth)
    dmin, dmax = dsort[int(depth.shape[0]*0.025)], dsort[int(depth.shape[0]*0.975)]
    norm = colors.Normalize(vmin=dmin, vmax=dmax, clip=True)
    cmap = colormaps.get_cmap("viridis")
    rgba = cmap(norm(depth))
    return torch.from_numpy(rgba[..., :3]).to(device).reshape(*shape, 3).permute(2, 0, 1).contiguous().float()


def rendering(dataset: CameraDataset, gaussians: GaussianModel, render_path: str, fix_width, fix_height, fix_width_focal, fix_height_focal) -> None:
    makedirs(render_path, exist_ok=True)
    camera_dicts = convert_dataset(dataset, fix_width, fix_height, fix_width_focal, fix_height_focal)
    pbar = tqdm(camera_dicts, desc="Rendering progress")
    for idx, camera_dict in enumerate(pbar):
        camera_dict['ground_truth_image_path'] = None
        camera_dict['ground_truth_depth_path'] = None
        camera_dict['ground_truth_depth_mask_path'] = None
        camera = dict2camera(camera_dict, device=camera_dict["device"], custom_data=camera_dict["custom_data"])
        out = gaussians(camera)
        rendering = out["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        depth = 1 / out["depth"]
        np.savez_compressed(os.path.join(render_path, '{0:05d}'.format(idx) + ".depth.npz"), depth=depth.type(torch.float16).cpu().numpy())
        torchvision.utils.save_image(colorify_depth(depth), os.path.join(render_path, '{0:05d}'.format(idx) + ".depth.png"))
        with open(os.path.join(render_path, '{0:05d}'.format(idx) + ".camera.json"), "w") as f:
            json.dump(camera2dict(camera, idx), f, indent=2)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-t", "--testdata", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--mode", choices=["base", "camera"], default="base")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--fix_width", default=1600, type=int)
    parser.add_argument("--fix_height", default=1200, type=int)
    parser.add_argument("--fix_width_focal", default=880.6371, type=float)
    parser.add_argument("--fix_height_focal", default=877.9232, type=float)
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    with torch.no_grad():
        dataset, gaussians = prepare_rendering(
            sh_degree=args.sh_degree, source=args.source, device=args.device, trainable_camera=args.mode == "camera",
            load_ply=load_ply, load_camera=args.load_camera, load_depth=True)
        rendering(
            dataset, gaussians, render_path=args.testdata,
            fix_width=args.fix_width,
            fix_height=args.fix_height,
            fix_width_focal=args.fix_width_focal,
            fix_height_focal=args.fix_height_focal)
