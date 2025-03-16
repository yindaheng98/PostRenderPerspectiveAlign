import cv2
import numpy as np
import json
import torch


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
