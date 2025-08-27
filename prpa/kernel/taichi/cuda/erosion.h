#pragma once
#include <torch/extension.h>

void error_erosion_cuda(
    torch::Tensor edge_pos,
    torch::Tensor warped,
    torch::Tensor mask_occluded,
    torch::Tensor mask_occlude,
    torch::Tensor mask_occluded_dilated,
    torch::Tensor mask_occlude_dilated,
    torch::Tensor mask_occluded_out,
    torch::Tensor counter,
    int height, int width, int channels, int kernel_size);
