#include <torch/extension.h>

#include "gridcreater.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grid_creater_forward_determ", &grid_creater_forward_determ, "grid_creater_forward_determ (CUDA)");
    m.def("grid_creater_forward", &grid_creater_forward, "grid_creater_forward (CUDA)");
    m.def("grid_creater_backward", &grid_creater_backward, "grid_creater_backward (CUDA)");
}