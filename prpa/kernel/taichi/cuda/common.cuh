#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_CHANNELS 4

__device__ __forceinline__ int clamp_idx(int v, int upper) {
    return min(max(v, 0), upper - 1);
}
