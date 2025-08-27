#pragma once
#include <torch/extension.h>

void reproject_and_scatter_cuda(
    torch::Tensor depth,
    torch::Tensor M,
    torch::Tensor t,
    torch::Tensor color_ref,
    torch::Tensor warped,
    torch::Tensor uv_idx,
    torch::Tensor z_out,
    torch::Tensor mindepth_onref,
    int channels, int height, int width);

void occlusion_cuda(
    torch::Tensor uv_idx,
    torch::Tensor depth,
    torch::Tensor mindepth_onref,
    torch::Tensor mask_occluded,
    torch::Tensor mask_occluded_onref,
    int height, int width,
    float depth_diff_thr, float no_gaussian_depth);
