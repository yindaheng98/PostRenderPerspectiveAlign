#include <torch/extension.h>
#include "common.cuh"
#include "erosion.h"

__global__ void error_erosion_kernel(
    const int32_t* __restrict__ edge_pos,
    uint8_t* __restrict__ warped,
    const uint8_t* __restrict__ mask_occluded,
    const uint8_t* __restrict__ mask_occlude,
    const uint8_t* __restrict__ mask_occluded_dilated,
    const uint8_t* __restrict__ mask_occlude_dilated,
    uint8_t* __restrict__ mask_occluded_out,
    int32_t* __restrict__ counter,
    int height, int width, int channels, int kernel_size,
    int num_edges)
{
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_idx >= num_edges) return;

    int center_y = edge_pos[edge_idx * 2 + 0];
    int center_x = edge_pos[edge_idx * 2 + 1];

    // Phase 1: compute average color from valid pixels
    int validcount = 0;
    float color_sum[MAX_CHANNELS] = {0};

    for (int dy = -kernel_size; dy <= kernel_size; dy++) {
        int y = clamp_idx(center_y + dy, height);
        for (int dx = -kernel_size; dx <= kernel_size; dx++) {
            int x = clamp_idx(center_x + dx, width);
            int pidx = y * width + x;
            if (mask_occluded_dilated[pidx] == 0 && mask_occlude_dilated[pidx] == 0) {
                validcount++;
                for (int c = 0; c < channels && c < MAX_CHANNELS; c++) {
                    color_sum[c] += (float)warped[pidx * channels + c];
                }
            }
        }
    }

    if (validcount <= 0) return;

    uint8_t color_avg[MAX_CHANNELS];
    for (int c = 0; c < channels && c < MAX_CHANNELS; c++) {
        color_avg[c] = (uint8_t)(color_sum[c] / validcount);
    }

    // Phase 2: scatter average color to occluded pixels
    for (int dy = -kernel_size; dy <= kernel_size; dy++) {
        int y = clamp_idx(center_y + dy, height);
        for (int dx = -kernel_size; dx <= kernel_size; dx++) {
            int x = clamp_idx(center_x + dx, width);
            int pidx = y * width + x;
            if (mask_occluded[pidx] != 0 && mask_occlude[pidx] == 0) {
                for (int c = 0; c < channels && c < MAX_CHANNELS; c++) {
                    warped[pidx * channels + c] = color_avg[c];
                }
                mask_occluded_out[pidx] = 0;
                atomicAdd(counter, 1);
            }
        }
    }
}

void error_erosion_cuda(
    torch::Tensor edge_pos,
    torch::Tensor warped,
    torch::Tensor mask_occluded,
    torch::Tensor mask_occlude,
    torch::Tensor mask_occluded_dilated,
    torch::Tensor mask_occlude_dilated,
    torch::Tensor mask_occluded_out,
    torch::Tensor counter,
    int height, int width, int channels, int kernel_size)
{
    int num_edges = edge_pos.size(0);
    int threads = 256;
    int blocks = (num_edges + threads - 1) / threads;

    error_erosion_kernel<<<blocks, threads>>>(
        edge_pos.data_ptr<int32_t>(),
        warped.data_ptr<uint8_t>(),
        mask_occluded.data_ptr<uint8_t>(),
        mask_occlude.data_ptr<uint8_t>(),
        mask_occluded_dilated.data_ptr<uint8_t>(),
        mask_occlude_dilated.data_ptr<uint8_t>(),
        mask_occluded_out.data_ptr<uint8_t>(),
        counter.data_ptr<int32_t>(),
        height, width, channels, kernel_size,
        num_edges);
}
