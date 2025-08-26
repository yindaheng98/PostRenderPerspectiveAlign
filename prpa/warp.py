from .morph import MorphologyClose, MorphologyDilation
from .erosion import error_erosion


def set_backend(backend='torch', **ti_init_kwargs):
    global error_erosion
    if backend == 'taichi':
        import taichi as ti
        ti.init(**ti_init_kwargs)
        from .kernel.taichi.erosion import error_erosion as _impl
        error_erosion = _impl
    else:
        from .erosion import error_erosion as _impl
        error_erosion = _impl


def warp(warped, mask_occluded, mask_occlude, kernel_size=16, occluded_dilation_size=1, occlude_dilation_size=1, max_iterations=None):
    """Error erosion loop to fill occluded regions."""
    mask_occluded = MorphologyClose(mask_occluded)
    mask_occlude = MorphologyClose(mask_occlude)
    # warped[mask_occluded, :] = torch.tensor([0, 0, 255], dtype=warped.dtype)  # debug
    # warped[mask_occlude, :] = torch.tensor([0, 255, 0], dtype=warped.dtype)  # debug
    # return warped

    # mask_occluded_last = mask_occluded.clone()  # debug
    mask_occlude_dilated = MorphologyDilation(mask_occlude, kernel_size=occlude_dilation_size)
    warped, mask_occluded, validcount = error_erosion(
        warped, mask_occluded, mask_occlude, mask_occlude_dilated,
        kernel_size=kernel_size,
        occluded_dilation_size=occluded_dilation_size)
    # print(validcount, mask_occluded.sum())  # debug
    iteration = 0
    while mask_occluded.sum() > 0 and validcount > 0 and (max_iterations is None or iteration < max_iterations):
        iteration += 1
        warped, mask_occluded, validcount = error_erosion(
            warped, mask_occluded, mask_occlude, mask_occlude_dilated,
            kernel_size=kernel_size,
            occluded_dilation_size=occluded_dilation_size)
        # print(validcount, mask_occluded.sum())  # debug
        if validcount <= 0:
            occluded_dilation_size -= 1
            occlude_dilation_size -= 1
            mask_occlude_dilated = mask_occlude
            if occlude_dilation_size > 0:
                mask_occlude_dilated = MorphologyDilation(mask_occlude, kernel_size=occlude_dilation_size)
            warped, mask_occluded, validcount = error_erosion(
                warped, mask_occluded, mask_occlude, mask_occlude_dilated,
                kernel_size=kernel_size,
                occluded_dilation_size=occluded_dilation_size)
        # print(validcount, mask_occluded.sum())  # debug
    # warped[mask_occluded_last, :] = torch.tensor([255, 0, 0], dtype=warped.dtype)  # debug
    # warped[mask_occluded, :] = torch.tensor([0, 255, 0], dtype=warped.dtype)  # debug
    # warped[mask_occlude, :] = torch.tensor([0, 0, 255], dtype=warped.dtype)  # debug
    # warped[is_edge, ...] = 0
    return warped
