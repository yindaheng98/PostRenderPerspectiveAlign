from .morph import MorphologyClose, MorphologyDilation
from .erosionall import error_erosion_all


def warp(warped, mask_occluded, mask_occlude, kernel_size=16, occluded_dilation_size=0, occlude_dilation_size=0, max_iterations=None):
    """Error erosion loop to fill occluded regions."""
    mask_occluded = MorphologyClose(mask_occluded)
    mask_occlude = MorphologyClose(mask_occlude)
    # warped[mask_occluded, :] = torch.tensor([0, 0, 255], dtype=warped.dtype)  # debug
    # warped[mask_occlude, :] = torch.tensor([0, 255, 0], dtype=warped.dtype)  # debug
    # return warped

    mask_occlude_dilated = MorphologyDilation(mask_occlude, kernel_size=occlude_dilation_size)
    warped, mask_occluded = error_erosion_all(
        warped, mask_occluded, mask_occlude, mask_occlude_dilated,
        kernel_size=kernel_size,
        occluded_dilation_size=occluded_dilation_size,
        occlude_dilation_size=occlude_dilation_size,
        max_iterations=max_iterations)
    return warped
