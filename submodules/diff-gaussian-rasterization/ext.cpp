/*
 * This file is based on the original Gaussian Splatting implementation
 * released by the GRAPHDECO research group (Inria, 2023).
 *
 * Significant modifications have been made to the core computational
 * components for research purposes.
 *
 * The original software is distributed under the Gaussian-Splatting License:
 * https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
 *
 * This repository preserves the same licensing terms.
 */


#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}