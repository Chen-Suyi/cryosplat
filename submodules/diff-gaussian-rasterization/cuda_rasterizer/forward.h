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


#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P,
		const float* means3D,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* amplitudes,
		const float* cov3D_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const int W, int H,
		int* radii,
		float2* means2D,
		float* depths,
		float* cov3Ds,
		float4* conic_amplitude,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool antialiasing);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* means2D,
		const float4* conic_amplitude,
		float* out_pixel,
		float* depths);
}


#endif
