#include <torch/extension.h>
#include "query.h"
#include "erosion.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reproject_and_scatter", &reproject_and_scatter_cuda);
    m.def("occlusion", &occlusion_cuda);
    m.def("error_erosion", &error_erosion_cuda);
}
