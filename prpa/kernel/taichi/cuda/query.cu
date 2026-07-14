#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include "common.cuh"
#include "query.h"

__device__ __forceinline__ void atomicMinFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= val) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
}

template <typename scalar_t>
__global__ void reproject_and_scatter_kernel(
    const float* __restrict__ depth,
    const float* __restrict__ M,
    const float* __restrict__ t,
    const scalar_t* __restrict__ color_ref,
    scalar_t* __restrict__ warped,
    int32_t* __restrict__ uv_idx,
    float* __restrict__ z_out,
    float* __restrict__ mindepth_onref,
    int channels, int ref_height, int ref_width,
    int depth_h, int depth_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= depth_h * depth_w) return;

    int i = idx / depth_w;
    int j = idx % depth_w;
    float d = depth[idx];

    float ray_x = M[0] * j + M[1] * i + M[2];
    float ray_y = M[3] * j + M[4] * i + M[5];
    float ray_z = M[6] * j + M[7] * i + M[8];

    float uvz_x = ray_x * d + t[0];
    float uvz_y = ray_y * d + t[1];
    float uvz_z = ray_z * d + t[2];

    float uv_x = uvz_x / uvz_z;
    float uv_y = uvz_y / uvz_z;

    int uv_ix = clamp_idx((int)roundf(uv_x), ref_width);
    int uv_iy = clamp_idx((int)roundf(uv_y), ref_height);

    uv_idx[idx * 2 + 0] = uv_ix;
    uv_idx[idx * 2 + 1] = uv_iy;
    z_out[idx] = uvz_z;

    for (int c = 0; c < channels && c < MAX_CHANNELS; c++) {
        warped[idx * channels + c] = color_ref[(uv_iy * ref_width + uv_ix) * channels + c];
    }

    atomicMinFloat(&mindepth_onref[uv_iy * ref_width + uv_ix], uvz_z);
}

void reproject_and_scatter_cuda(
    torch::Tensor depth,
    torch::Tensor M,
    torch::Tensor t,
    torch::Tensor color_ref,
    torch::Tensor warped,
    torch::Tensor uv_idx,
    torch::Tensor z_out,
    torch::Tensor mindepth_onref,
    int channels, int height, int width)
{
    int depth_h = depth.size(0);
    int depth_w = depth.size(1);
    int total = depth_h * depth_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, color_ref.scalar_type(), "reproject_and_scatter_cuda", ([&] {
        reproject_and_scatter_kernel<scalar_t><<<blocks, threads>>>(
            depth.data_ptr<float>(),
            M.data_ptr<float>(),
            t.data_ptr<float>(),
            color_ref.data_ptr<scalar_t>(),
            warped.data_ptr<scalar_t>(),
            uv_idx.data_ptr<int32_t>(),
            z_out.data_ptr<float>(),
            mindepth_onref.data_ptr<float>(),
            channels, height, width,
            depth_h, depth_w);
    }));
}

__global__ void occlusion_kernel(
    const int32_t* __restrict__ uv_idx,
    const float* __restrict__ depth,
    const float* __restrict__ mindepth_onref,
    uint8_t* __restrict__ mask_occluded,
    uint8_t* __restrict__ mask_occluded_onref,
    int height, int width,
    int uv_h, int uv_w,
    float depth_diff_thr, float no_gaussian_depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= uv_h * uv_w) return;

    int uv_ix = uv_idx[idx * 2 + 0];
    int uv_iy = uv_idx[idx * 2 + 1];

    float mindepth_onloc = mindepth_onref[uv_iy * width + uv_ix];

    bool mask_valid = true;
    if (uv_ix <= 0 || uv_ix >= width - 1) mask_valid = false;
    if (uv_iy <= 0 || uv_iy >= height - 1) mask_valid = false;
    if (depth[idx] > no_gaussian_depth) mask_valid = false;

    float depthdiff = fabsf(depth[idx] - mindepth_onloc);
    bool is_occluded = mask_valid && (depthdiff >= depth_diff_thr);

    mask_occluded[idx] = is_occluded ? 1 : 0;
    if (is_occluded) {
        mask_occluded_onref[uv_iy * width + uv_ix] = 1;
    }
}

void occlusion_cuda(
    torch::Tensor uv_idx,
    torch::Tensor depth,
    torch::Tensor mindepth_onref,
    torch::Tensor mask_occluded,
    torch::Tensor mask_occluded_onref,
    int height, int width,
    float depth_diff_thr, float no_gaussian_depth)
{
    int uv_h = uv_idx.size(0);
    int uv_w = uv_idx.size(1);
    int total = uv_h * uv_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    occlusion_kernel<<<blocks, threads>>>(
        uv_idx.data_ptr<int32_t>(),
        depth.data_ptr<float>(),
        mindepth_onref.data_ptr<float>(),
        mask_occluded.data_ptr<uint8_t>(),
        mask_occluded_onref.data_ptr<uint8_t>(),
        height, width, uv_h, uv_w,
        depth_diff_thr, no_gaussian_depth);
}
