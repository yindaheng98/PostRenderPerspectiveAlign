import taichi as ti

MAX_CHANNELS = 4


@ti.func
def clamp_index(index: ti.i32, upper: ti.i32) -> ti.i32:
    return ti.min(ti.max(index, 0), upper - 1)
