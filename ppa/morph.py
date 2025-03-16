import torch
import torch.nn.functional as F


def MorphologyDilation(binary, kernel_size=1):
    return F.max_pool2d(  # dilation the occluded region
        binary.type(torch.float32)[None, None, ...],
        kernel_size=kernel_size*2+1, stride=1, padding=kernel_size
    ).type(torch.bool)[0, 0, ...]


def MorphologyErosion(binary, kernel_size=1):
    return (~F.max_pool2d(  # dilation the occluded region
        (~binary).type(torch.float32)[None, None, ...],
        kernel_size=kernel_size*2+1, stride=1, padding=kernel_size
    ).type(torch.bool))[0, 0, ...]


def MorphologyClose(binary, kernel_size=1):
    return MorphologyErosion(MorphologyDilation(binary, kernel_size=kernel_size), kernel_size=kernel_size)
