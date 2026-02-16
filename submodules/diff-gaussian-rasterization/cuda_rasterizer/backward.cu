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


#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ float sq(float x) { return x * x; }


// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	int w, int h,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float* view_matrix,
	const float* projmatrix,
	const float* amplitudes,
	const float* dL_dconics,
	float* dL_damplitude,
	// float3* dL_dmeans,
	float* dL_dcov,
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[3 * idx], dL_dconics[3 * idx + 1], dL_dconics[3 * idx + 2] };
	float3 t = transformPoint4x3(mean, view_matrix);

	glm::mat3 J = glm::mat3(
		projmatrix[0], projmatrix[4], projmatrix[8],
		projmatrix[1], projmatrix[5], projmatrix[9],
		projmatrix[2], projmatrix[6], projmatrix[10]);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float c_xx = 0.25f * w * w * cov2D[0][0];
	float c_xy = 0.25f * w * h * cov2D[0][1];
	float c_yy = 0.25f * h * h * cov2D[1][1];
	
	float det_proj = projmatrix[0] * projmatrix[5] * projmatrix[10];
	const float dL_damplitude_v = dL_damplitude[idx] * det_proj * w * h * 0.25f;
	dL_damplitude[idx] = det_proj < 0 ? -dL_damplitude_v : dL_damplitude_v; // abs(det_proj)

	constexpr float h_var = 0.3f;

	if(antialiasing)
	{
		c_xx += h_var;
		c_yy += h_var;
	}
	
	float dL_dc_xx = 0;
	float dL_dc_xy = 0;
	float dL_dc_yy = 0;
	
	float denom = c_xx * c_yy - c_xy * c_xy;

	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		
		dL_dc_xx += denom2inv * (-c_yy * c_yy * dL_dconic.x + 2 * c_xy * c_yy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.z);
		dL_dc_yy += denom2inv * (-c_xx * c_xx * dL_dconic.z + 2 * c_xx * c_xy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.x);
		dL_dc_xy += denom2inv * 2 * (c_xy * c_yy * dL_dconic.x - (denom + 2 * c_xy * c_xy) * dL_dconic.y + c_xx * c_xy * dL_dconic.z);

		// from screen-space to NDC
		dL_dc_xx *= 0.25f * w * w;
		dL_dc_yy *= 0.25f * h * h;
		dL_dc_xy *= 0.25f * w * h;
		
		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_dc_xx + T[0][0] * T[1][0] * dL_dc_xy + T[1][0] * T[1][0] * dL_dc_yy);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_dc_xx + T[0][1] * T[1][1] * dL_dc_xy + T[1][1] * T[1][1] * dL_dc_yy);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_dc_xx + T[0][2] * T[1][2] * dL_dc_xy + T[1][2] * T[1][2] * dL_dc_yy);
		
		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_dc_xx + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][1] * dL_dc_yy;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_dc_xx + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][2] * dL_dc_yy;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_dc_xx + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_dc_xy + 2 * T[1][1] * T[1][2] * dL_dc_yy;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
__global__ void preprocessCUDA(
	int P,
	const float3* means,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float3* dL_dmean2D,
	float3* dL_dmeans,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_damplitude)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	float dL_dtx = projmatrix[0] * dL_dmean2D[idx].x;
	float dL_dty = projmatrix[5] * dL_dmean2D[idx].y;
	float dL_dtz = 0;

	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, viewmatrix);
	dL_dmeans[idx] = dL_dmean;


	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_amplitude,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float3* __restrict__ dL_dconic2D,
	float* __restrict__ dL_damplitude
)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE]; // screen-space
	__shared__ float4 collected_conic_amplitude[BLOCK_SIZE]; //screen-space

	// float accum_rec = 0 ;
	float dL_dpixel;
	if (inside)
	{
		dL_dpixel = dL_dpixels[pix_id];
	}

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_amplitude[block.thread_rank()] = conic_amplitude[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_a = collected_conic_amplitude[j];
			const float power = -0.5f * (con_a.x * d.x * d.x + con_a.z * d.y * d.y) - con_a.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float det_inv = con_a.x * con_a.z - con_a.y * con_a.y; // det(conic)
			const float det = 1.f / (det_inv + 0.0000001f); // det(cov2D)
			const float norm = 0.15915494309189535 * sqrt(det_inv); // 1/(2*pi*sqrt(det))
			const float G = norm * exp(power);
			const float alpha = con_a.w * G;


			const int global_id = collected_id[j];

			// Helpful reusable temporary variables
			const float dL_dG = con_a.w * dL_dpixel;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_a.x - gdy * con_a.y;
			const float dG_ddely = -gdy * con_a.z - gdx * con_a.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, 0.5f * G * (det * con_a.z - d.x * d.x) * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, 0.5f * G * (det * -con_a.y - d.x * d.y) * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].z, 0.5f * G * (det * con_a.x - d.y * d.y) * dL_dG);

			// Update gradients w.r.t. amplitude of the Gaussian
			atomicAdd(&(dL_damplitude[global_id]), G * dL_dpixel);
		}
	}
}

void BACKWARD::preprocess(
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
	bool antialiasing)
{
	// int num_elements = 1;
	// float3* host_dL_dmean2D = new float3[num_elements];
	// cudaMemcpy(host_dL_dmean2D, dL_dmean2D, num_elements * sizeof(float3), cudaMemcpyDeviceToHost);
	// for (int i = 0; i < num_elements; ++i)
	// 	printf("dL_dmean2D[%d]: %e, %e\n", i, host_dL_dmean2D[i].x, host_dL_dmean2D[i].y);
	// delete[] host_dL_dmean2D;

	// float3* host_dL_dconic = new float3[num_elements];
	// cudaMemcpy(host_dL_dconic, dL_dconic, num_elements * sizeof(float3), cudaMemcpyDeviceToHost);
	// for (int i = 0; i < num_elements; ++i)
	// 	printf("dL_dconic2D[%d]: %e, %e, %e\n", i, host_dL_dconic[i].x, host_dL_dconic[i].y, host_dL_dconic[i].z);
	// delete[] host_dL_dconic;

	// float* host_dL_damplitude = new float[num_elements];
	// cudaMemcpy(host_dL_damplitude, dL_damplitude, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
	// for (int i = 0; i < num_elements; ++i)
	// 	printf("dL_damplitude2D[%d]: %e\n", i, host_dL_damplitude[i]);
	// delete[] host_dL_damplitude;
	
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		W, H,
		means3D,
		radii,
		cov3Ds,
		viewmatrix,
		projmatrix,
		amplitudes,
		dL_dconic,
		dL_damplitude,
		// (float3*)dL_dmean3D,
		dL_dcov3D,
		antialiasing);
	
	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA << < (P + 255) / 256, 256 >> > (
		P,
		(float3*)means3D,
		radii,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		(float3*)dL_dmean2D,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		dL_dscale,
		dL_drot,
		dL_damplitude);

	// // --- Print dL_dmean3D ---
	// glm::vec3* host_dL_dmean3D = new glm::vec3[num_elements];
	// cudaMemcpy(host_dL_dmean3D, dL_dmean3D, num_elements * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	// for (int i = 0; i < num_elements; ++i)
	// 	printf("dL_dmean3D[%d]: %e, %e, %e\n", i,
	// 		host_dL_dmean3D[i].x,
	// 		host_dL_dmean3D[i].y,
	// 		host_dL_dmean3D[i].z);
	// delete[] host_dL_dmean3D;

	// --- Print dL_dcov3D (6 elements per Gaussian: 3 variances + 3 covariances) ---
	// float* host_dL_dcov3D = new float[6 * num_elements];
	// cudaMemcpy(host_dL_dcov3D, dL_dcov3D, 6 * num_elements * sizeof(float), cudaMemcpyDeviceToHost);
	// for (int i = 0; i < num_elements; ++i)
	// 	printf("dL_dcov3D[%d]: %e, %e, %e, %e, %e, %e\n", i,
	// 		host_dL_dcov3D[6 * i + 0],
	// 		host_dL_dcov3D[6 * i + 1],
	// 		host_dL_dcov3D[6 * i + 2],
	// 		host_dL_dcov3D[6 * i + 3],
	// 		host_dL_dcov3D[6 * i + 4],
	// 		host_dL_dcov3D[6 * i + 5]);
	// delete[] host_dL_dcov3D;

	// // --- Print dL_dscale ---
	// glm::vec3* host_dL_dscale = new glm::vec3[num_elements];
	// cudaMemcpy(host_dL_dscale, dL_dscale, num_elements * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	// for (int i = 0; i < num_elements; ++i)
	// 	printf("dL_dscale[%d]: %e, %e, %e\n", i,
	// 		host_dL_dscale[i].x,
	// 		host_dL_dscale[i].y,
	// 		host_dL_dscale[i].z);
	// delete[] host_dL_dscale;

	// // --- Print dL_drot ---
	// glm::vec4* host_dL_drot = new glm::vec4[num_elements];
	// cudaMemcpy(host_dL_drot, dL_drot, num_elements * sizeof(glm::vec4), cudaMemcpyDeviceToHost);
	// for (int i = 0; i < num_elements; ++i)
	// 	printf("dL_drot[%d]: %e, %e, %e, %e\n", i,
	// 		host_dL_drot[i].x,
	// 		host_dL_drot[i].y,
	// 		host_dL_drot[i].z,
	// 		host_dL_drot[i].w);
	// delete[] host_dL_drot;

	// // --- Print dL_damplitude (again for 3D path, optional if already printed) ---
	// float* host_dL_damplitude3D = new float[num_elements];
	// cudaMemcpy(host_dL_damplitude3D, dL_damplitude, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
	// for (int i = 0; i < num_elements; ++i)
	// 	printf("dL_damplitude3D[%d]: %.9e\n", i, host_dL_damplitude3D[i]);
	// delete[] host_dL_damplitude3D;
	
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float4* conic_amplitude,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float3* dL_dconic2D,
	float* dL_damplitude)
{
	renderCUDA << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		means2D,
		conic_amplitude,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_damplitude
		);
}
