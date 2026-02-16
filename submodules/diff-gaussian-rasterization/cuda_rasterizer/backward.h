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


#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* means2D,
		const float4* conic_amplitude,
		const float* dL_dpixels,
		float3* dL_dmean2D,
		float3* dL_dconic2D,
		float* dL_damplitude);

	void preprocess(
		int P,
		int W, int H,
		const float3* means3D,
		const int* radii,
		const float* amplitudes,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* viewmatrix,
		const float* projmatrix,
		const float3* dL_dmean2D,
		const float* dL_dconic,
		float* dL_damplitude,
		glm::vec3* dL_dmean3D,
		float* dL_dcov3D,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		bool antialiasing);
}

#endif
