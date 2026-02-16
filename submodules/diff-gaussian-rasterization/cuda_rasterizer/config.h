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


#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 1 // Default 1, only amplitudes
#define BLOCK_X 16
#define BLOCK_Y 16

#endif