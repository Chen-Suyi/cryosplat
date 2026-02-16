#
# This implementation is based on the original Gaussian Splatting
# codebase released by the GRAPHDECO research group (Inria, 2023).
#
# Significant modifications have been made to the core computational
# components for research purposes.
#
# The original software is distributed under the Gaussian-Splatting License:
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
#
# This repository preserves the same licensing terms.
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_model import GaussianModel

def render(gaussians: GaussianModel, image_height: int, image_width: int, viewmatrix: torch.Tensor, projmatrix: torch.Tensor, scaling_modifier = 1.0, compute_cov3D_python=False):
    """
    Render the scene. 

    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration

    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix.T,
        projmatrix=projmatrix.T,
        prefiltered=False,
        debug=False,
        antialiasing=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussians.get_xyz
    means2D = screenspace_points
    amplitude = gaussians.get_amplitude

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if compute_cov3D_python:
        cov3D_precomp = gaussians.get_covariance(scaling_modifier)
    else:
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        amplitudes = amplitude,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out = {
        "rendered_image": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii
        }
    
    return out
