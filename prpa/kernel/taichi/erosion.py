import taichi as ti
import torch
from .morph import MorphologyDilation
from .common import clamp_index, MAX_CHANNELS

TORCH_TO_TI = {
    torch.uint8: ti.u8,
    torch.int8: ti.i8,
    torch.int16: ti.i16,
    torch.int32: ti.i32,
    torch.int64: ti.i64,
    torch.float16: ti.f16,
    torch.float32: ti.f32,
    torch.float64: ti.f64,
}


@ti.kernel
def error_erosion_kernel(
    edge_pos: ti.types.ndarray(dtype=ti.i32, ndim=2),
    warped: ti.types.ndarray(ndim=3),
    mask_occluded: ti.types.ndarray(dtype=ti.u8, ndim=2),
    mask_occlude: ti.types.ndarray(dtype=ti.u8, ndim=2),
    mask_occluded_dilated: ti.types.ndarray(dtype=ti.u8, ndim=2),
    mask_occlude_dilated: ti.types.ndarray(dtype=ti.u8, ndim=2),
    mask_occluded_out: ti.types.ndarray(dtype=ti.u8, ndim=2),
    counter: ti.types.ndarray(dtype=ti.i32, ndim=1),
    height: ti.i32,
    width: ti.i32,
    channels: ti.i32,
    kernel_size: ti.i32,
    color_dtype: ti.template(),
):
    ti.loop_config(block_dim=256)
    for edge_idx in range(edge_pos.shape[0]):
        center_y = edge_pos[edge_idx, 0]  # kernel center y
        center_x = edge_pos[edge_idx, 1]  # kernel center x

        # Phase 1: Reduce - compute average color from source pixels
        validcolorcount = 0
        color_sum = ti.Vector.zero(ti.f32, MAX_CHANNELS)

        for dy in range(-kernel_size, kernel_size + 1):
            y = clamp_index(center_y + dy, height)
            for dx in range(-kernel_size, kernel_size + 1):
                x = clamp_index(center_x + dx, width)

                # Now (x,y) is one of the kernels[edge_idx, ...]:
                # kernel = torch.cartesian_prod(
                #     torch.arange(-kernel_size, kernel_size+1, dtype=torch.int64, device=edge_pos.device),
                #     torch.arange(-kernel_size, kernel_size+1, dtype=torch.int64, device=edge_pos.device))
                # kernels = edge_pos.unsqueeze(1) + kernel.unsqueeze(0)  # region to be erosion
                # kernels[..., 0].clamp_(0, height-1)
                # kernels[..., 1].clamp_(0, width-1)

                if mask_occluded_dilated[y, x] == 0 and mask_occlude_dilated[y, x] == 0:
                    # Now (x,y) is one of the pixels inside the kernel_avgcolormask:
                    # kernel_avgcolormask = ~mask_occluded_dilated[kernels[..., 0], kernels[..., 1]]
                    # kernel_avgcolormask &= ~mask_occlude_dilated[kernels[..., 0], kernels[..., 1]]

                    validcolorcount += 1  # kernel_validcolorcount = kernel_avgcolormask.sum(dim=1)

                    # Now collecting and sum kernel_colors to compute kernel_avgcolor:
                    # kernel_colors = warped[kernels[..., 0], kernels[..., 1]].type(torch.float32)
                    # kernel_colors[~kernel_avgcolormask, ...] = 0.
                    # kernel_avgcolor = (kernel_colors.sum(dim=1) / kernel_validcolorcount.unsqueeze(-1)).type(warped.dtype)
                    for channel in ti.static(range(MAX_CHANNELS)):
                        if channel < channels:
                            color_sum[channel] += ti.cast(warped[y, x, channel], ti.f32)

        if validcolorcount <= 0:
            continue

        # Now (center_y, center_x) is one of the kernel centers inside the kernel_valid:
        # kernel_validcolorcount = (kernel_avgcolormask).sum(dim=1)
        # kernel_valid = kernel_validcolorcount > 0
        # kernels = kernels[kernel_valid, ...]

        # Now computing kernel_avgcolor:
        # kernel_avgcolor = (kernel_colors.sum(dim=1) / kernel_validcolorcount.unsqueeze(-1)).type(warped.dtype)
        color_avg = ti.Vector.zero(color_dtype, MAX_CHANNELS)
        for channel in ti.static(range(MAX_CHANNELS)):
            if channel < channels:
                color_avg[channel] = ti.cast(color_sum[channel] / validcolorcount, color_dtype)

        # Phase 2: Scatter - write average color to target pixels
        for dy in range(-kernel_size, kernel_size + 1):
            y = clamp_index(center_y + dy, height)
            for dx in range(-kernel_size, kernel_size + 1):
                x = clamp_index(center_x + dx, width)
                if mask_occluded[y, x] != 0 and mask_occlude[y, x] == 0:
                    # Now (x,y) is one of the pixels inside the kernel_assignmask:
                    # kernel_assignmask = mask_occluded[kernels[..., 0], kernels[..., 1]]
                    # kernel_assignmask &= ~mask_occlude[kernels[..., 0], kernels[..., 1]]
                    # kernel_assignmask = kernel_assignmask[kernel_valid, ...]
                    # also, (x,y) is one of the pos in the assign_pos:
                    # assign_pos = kernels[kernel_assignmask, ...]

                    for channel in ti.static(range(MAX_CHANNELS)):
                        if channel < channels:
                            # Now assigning color_avg to (x,y):
                            # warped[assign_pos[..., 0], assign_pos[..., 1], ...] = kernel_assigncolor
                            warped[y, x, channel] = color_avg[channel]
                    # Now clearing mask_occluded[y, x]:
                    # mask_occluded[assign_pos[..., 0], assign_pos[..., 1]] = False
                    mask_occluded_out[y, x] = ti.cast(0, ti.u8)
                    ti.atomic_add(counter[0], 1)


def error_erosion(warped, mask_occluded, mask_occlude, mask_occlude_dilated, kernel_size=5, occluded_dilation_size=0):
    from . import use_cuda
    if use_cuda:
        from . import _C

    assert warped.dim() == 3
    assert warped.shape[2] <= MAX_CHANNELS
    assert mask_occluded.dim() == mask_occlude.dim() == 2
    assert mask_occluded.shape == mask_occlude.shape
    height, width = mask_occluded.shape

    # get edge of occluded region
    edge = MorphologyDilation(mask_occluded) & ~mask_occluded  # xor with the occluded region to get the edge
    edge_pos = edge.nonzero()
    # warped[edge_pos[..., 0], edge_pos[..., 1], ...] = torch.tensor([0, 0, 255], dtype=torch.uint8)  # debug
    # return warped, mask_occluded  # debug

    if len(edge_pos) <= 0:
        return warped, mask_occluded, 0

    # where to collect color for avg
    mask_occluded_dilated = mask_occluded
    if occluded_dilation_size > 0:
        mask_occluded_dilated = MorphologyDilation(mask_occluded, kernel_size=occluded_dilation_size)

    edge_pos = edge_pos.to(torch.int32).contiguous()
    warped_out = warped if warped.is_contiguous() else warped.contiguous()
    mask_occluded_u8 = mask_occluded.to(torch.uint8).contiguous()
    mask_occluded_out = mask_occluded_u8.clone()
    mask_occlude_u8 = mask_occlude.to(torch.uint8).contiguous()
    mask_occluded_dilated_u8 = mask_occluded_dilated.to(torch.uint8).contiguous()
    mask_occlude_dilated_u8 = mask_occlude_dilated.to(torch.uint8).contiguous()
    counter = torch.zeros(1, device=warped.device, dtype=torch.int32)

    if use_cuda:
        _C.error_erosion(
            edge_pos, warped_out,
            mask_occluded_u8, mask_occlude_u8,
            mask_occluded_dilated_u8, mask_occlude_dilated_u8,
            mask_occluded_out,
            counter, height, width, warped.shape[2], kernel_size,
        )
    else:
        error_erosion_kernel(
            edge_pos, warped_out,
            mask_occluded_u8, mask_occlude_u8,
            mask_occluded_dilated_u8, mask_occlude_dilated_u8,
            mask_occluded_out,
            counter, height, width, warped.shape[2], kernel_size, TORCH_TO_TI[warped.dtype],
        )

    validcount = int(counter.item())
    if validcount > 0:
        if warped_out is not warped:
            warped.copy_(warped_out)
        mask_occluded.copy_(mask_occluded_out.bool())
    return warped, mask_occluded, validcount
