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


#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float* cov3D, const float* viewmatrix, const float* projmatrix, const int w, const int h)
{
	// This function has been modified to fit orthogonal projection.

	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.

	glm::mat3 J = glm::mat3(
		projmatrix[0], projmatrix[4], projmatrix[8],
		projmatrix[1], projmatrix[5], projmatrix[9],
		projmatrix[2], projmatrix[6], projmatrix[10]);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { 0.25f * w * w * float(cov[0][0]), 0.25f * w * h * float(cov[0][1]), 0.25f * h * h * float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessCUDA(int P,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* amplitudes,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const int W, int H,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float4* conic_amplitude,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Nothing should be culled for cryo-EM imaging

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	float3 p_proj = transformPoint4x3(p_view, projmatrix);

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(cov3D, viewmatrix, projmatrix, W, H);

	float det = cov.x * cov.z - cov.y * cov.y; // The original determinant of covariance before anti-aliasing

	// Anti-aliasing (EWA algorithm) (may not applicable for cryoEM)
	constexpr float h_var = 0.3f; // TODO: A empirical number need to be checked

	if(antialiasing)
		cov.x += h_var;
		cov.z += h_var;
		det = cov.x * cov.z - cov.y * cov.y; // The determinant after anti-aliasing

	// Invert covariance (EWA algorithm)
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;

	// Compensation for projective matrix
	float det_proj = projmatrix[0] * projmatrix[5] * projmatrix[10];

	// Inverse 2D covariance and amplitude neatly pack into one float4
	float amplitude = amplitudes[idx] * det_proj * W * H * 0.25f;
	// float amplitude = amplitudes[idx] * det_proj;
	amplitude = det_proj < 0 ? -amplitude : amplitude; // abs(det_proj)
	conic_amplitude[idx] = { conic.x, conic.y, conic.z, amplitude };


	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_amplitude,
	float* __restrict__ out_pixel,
	const float* __restrict__ depths)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_amplitude[BLOCK_SIZE];

	// Initialize helper variables
	// No transmittance ratio in cryo-EM
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float Q = 0;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank(); // Current index of the Gaussian to fetch
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_amplitude[block.thread_rank()] = conic_amplitude[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_a = collected_conic_amplitude[j];
			float power = -0.5f * (con_a.x * d.x * d.x + con_a.z * d.y * d.y) - con_a.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (8) from cryoSplat paper.
			const float norm = 0.15915494309189535 * sqrt(con_a.x * con_a.z - con_a.y * con_a.y); // 1/(2*pi*sqrt(det))
			float alpha = con_a.w * norm * exp(power);

			// Eq. (14) from cryoSplat paper.
			Q += alpha;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}

		if (toDo <= BLOCK_SIZE)
			done = true;
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		out_pixel[pix_id] = Q;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float4* conic_amplitude,
	float* out_pixel,
	float* depths)
{
	renderCUDA << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		conic_amplitude,
		out_pixel,
		depths);
}

void FORWARD::preprocess(int P,
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
	bool antialiasing)
{
	preprocessCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		scales,
		scale_modifier,
		rotations,
		amplitudes,
		cov3D_precomp,
		viewmatrix, 
		projmatrix,
		W, H,
		radii,
		means2D,
		depths,
		cov3Ds,
		conic_amplitude,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}
