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

 
#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P,
			const int width, int height,
			const float* means3D,
			const float* amplitudes,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const bool prefiltered,
			float* out_pixel,
			bool antialiasing,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int R,
			const int width, int height,
			const float* means3D,
			const float* amplitudes,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* img_buffer,
			const float* dL_dpix,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_damplitude,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dscale,
			float* dL_drot,
			bool antialiasing = false,
			bool debug = false);
	};
};

#endif
