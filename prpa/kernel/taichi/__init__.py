from .erosion import error_erosion
from .query import query

use_cuda = False


def set_backend(cuda=False):
    global use_cuda
    use_cuda = cuda
