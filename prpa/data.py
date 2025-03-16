import cv2
import numpy as np
import json
import torch

from .prpa import Camera, Target, Reference


def fromJSON(camera):
    height, width = camera["height"], camera["width"]

    R_c2w = torch.tensor(camera["rotation"])
    T_c2w = torch.tensor(camera["position"])
    K = torch.tensor([
        [camera["fx"], 0, camera["width"]/2],
        [0, camera["fy"], camera["height"]/2],
        [0, 0, 1]
    ])
    return Camera(K=K, R=R_c2w, T=T_c2w), height, width


def read_camera(path):
    with open(path, "r") as f:
        return fromJSON(json.load(f))


def read_color(path):
    return torch.tensor(cv2.imread(path))


def read_camera_color(camerapath, colorpath):
    camera, height, width = read_camera(camerapath)
    color = read_color(colorpath)
    assert color.shape[0] == height and color.shape[1] == width, ValueError("Size of color image should match camera")
    return Reference(**camera._asdict(), color=color)


def read_depth(path):
    return torch.tensor(np.load(path)["depth"][0, ...])


def read_camera_depth(camerapath, depthpath):
    camera, height, width = read_camera(camerapath)
    depth = read_depth(depthpath)
    assert depth.shape[0] == height and depth.shape[1] == width, ValueError("Size of depth map should match camera")
    return Target(**camera._asdict(), depth=depth)
