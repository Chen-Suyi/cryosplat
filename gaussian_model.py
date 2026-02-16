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
import numpy as np
from utils.general_utils import get_expon_lr_func, build_rotation, strip_symmetric, build_scaling_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement

from tqdm import tqdm

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def inverse_softplus(y, beta=1.0, threshold=20.0):
    """
    Numerically stable inverse of PyTorch's softplus with beta and threshold.
    """
    y = torch.as_tensor(y)
    t = threshold / beta

    linear_mask = y > t
    x = torch.empty_like(y)

    # Linear region: softplus ≈ x ⇒ x = y
    x[linear_mask] = y[linear_mask]

    # Non-linear region: numerically stable version of
    # softplus^{-1}(y) = y + (1/beta) * log(1 - exp(-beta * y))
    beta_y = beta * y[~linear_mask]
    x[~linear_mask] = y[~linear_mask] + (1.0 / beta) * torch.log1p(-torch.exp(-beta_y))

    return x

def scaling_activation(x):
    return 0.02 * torch.sigmoid(x)

def inverse_scaling_activation(y):
    return inverse_sigmoid(y * 50)

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, rotation):
            L = build_scaling_rotation(scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.amplitude_activation = torch.nn.functional.softplus
        self.amplitude_inverse_activation = inverse_softplus

        self.scaling_activation = torch.nn.functional.softplus
        self.scaling_inverse_activation = inverse_softplus

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self):
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._amplitude = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.tmp_radii = None
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._amplitude,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self._xyz,
        self._scaling, 
        self._rotation, 
        self._amplitude,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def P(self):
        return self._xyz.shape[0]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling).clamp(min=1e-3, max=1e3)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_amplitude(self):
        return self.amplitude_activation(self._amplitude)
    
    def get_covariance(self):
        return self.covariance_activation(self.get_scaling, self._rotation)
    
    def get_inv_covariance(self):
        return self.covariance_activation(1 / self.get_scaling, self._rotation)
    
    def get_det_covariance(self):
        scale = self.get_scaling
        return (scale[:, 0] * scale[:, 1] * scale[:, 2])**2
    
    def get_det_inv_covariance(self):
        return 1 / self.get_det_covariance()
    
    def get_3D_intensity(self):
        inv_scale = 1 / self.get_scaling
        inv_sqrt_det = inv_scale[:, 0] * inv_scale[:, 1] * inv_scale[:, 2]
        return self.get_amplitude * inv_sqrt_det[:, None] / (2 * torch.pi)**1.5
    
    def get_max_2D_intensity(self):
        inv_scale = 1 / self.get_scaling
        _, col_idx = inv_scale.min(dim=-1)
        row_idx = torch.arange(inv_scale.shape[0], device=inv_scale.device)
        inv_scale[row_idx, col_idx] = 1.0
        max_2D_inv_sqrt_det = inv_scale[:, 0] * inv_scale[:, 1] * inv_scale[:, 2]
        return self.get_amplitude * max_2D_inv_sqrt_det[:, None] / (2 * torch.pi)

    def create_from_given_params(self, xyz, scaling, rotation, amplitude, spatial_lr_scale : float = 1.0):
        self.spatial_lr_scale = spatial_lr_scale
        self._xyz = nn.Parameter(xyz.requires_grad_(True).cuda())
        self._scaling = nn.Parameter(scaling.requires_grad_(True).cuda())
        self._rotation = nn.Parameter(rotation.requires_grad_(True).cuda())
        self._amplitude = nn.Parameter(amplitude.requires_grad_(True).cuda())
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def initialize_gaussians_random(self, P, extent=0.5):
        xyz = 0.15 * extent * torch.randn(P, 3)
        scaling = self.scaling_inverse_activation(0.015 * extent * torch.ones((P, 3)))
        rotation = torch.zeros((P, 4))
        rotation[:, 0] = 1.0
        amplitude = self.amplitude_inverse_activation(0.5 / P * torch.ones((P, 1)))

        self.create_from_given_params(xyz, scaling, rotation, amplitude)

    def training_setup(self, optimization_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': optimization_args.position_lr, "name": "xyz"},
            {'params': [self._scaling], 'lr': optimization_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': optimization_args.rotation_lr, "name": "rotation"},
            {'params': [self._amplitude], 'lr': optimization_args.amplitude_lr, "name": "amplitude"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=optimization_args.gamma)

    def update_learning_rate(self):
        self.scheduler.step()

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l.append('amplitude')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        amplitude = self._amplitude.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, amplitude, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)

        amplitude = np.asarray(plydata.elements[0]["amplitude"])[..., np.newaxis]

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._amplitude = nn.Parameter(torch.tensor(amplitude, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def compute_cov3D(self):
        covariance = self.get_covariance()
        cov3D = torch.zeros((covariance.shape[0], 3, 3), device=covariance.device)
        cov3D[:, 0, 0] = covariance[:, 0]
        cov3D[:, 0, 1] = covariance[:, 1]
        cov3D[:, 1, 0] = covariance[:, 1]
        cov3D[:, 0, 2] = covariance[:, 2]
        cov3D[:, 2, 0] = covariance[:, 2]
        cov3D[:, 1, 1] = covariance[:, 3]
        cov3D[:, 1, 2] = covariance[:, 4]
        cov3D[:, 2, 1] = covariance[:, 4]
        cov3D[:, 2, 2] = covariance[:, 5]

        return cov3D
    
    def compute_inv_cov3D(self):
        inv_covariance = self.get_inv_covariance()
        inv_cov3D = torch.zeros((inv_covariance.shape[0], 3, 3), device=inv_covariance.device)
        inv_cov3D[:, 0, 0] = inv_covariance[:, 0]
        inv_cov3D[:, 0, 1] = inv_covariance[:, 1]
        inv_cov3D[:, 1, 0] = inv_covariance[:, 1]
        inv_cov3D[:, 0, 2] = inv_covariance[:, 2]
        inv_cov3D[:, 2, 0] = inv_covariance[:, 2]
        inv_cov3D[:, 1, 1] = inv_covariance[:, 3]
        inv_cov3D[:, 1, 2] = inv_covariance[:, 4]
        inv_cov3D[:, 2, 1] = inv_covariance[:, 4]
        inv_cov3D[:, 2, 2] = inv_covariance[:, 5]

        return inv_cov3D
    
    def generate_volume(self, D=257, extent=0.5, norm=[0, 1]):
        assert D % 2 == 1
        dimension = torch.linspace(-extent, extent, D, device='cuda')[0: -1]
        z, y, x = torch.meshgrid([dimension, dimension, dimension], indexing="ij")
        coords = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)

        volume = torch.zeros((coords.shape[0]), device='cuda')
        for mu, inv_sigma, inv_det, amp in tqdm(zip(self.get_xyz, self.compute_inv_cov3D(), self.get_det_inv_covariance(), self.get_amplitude), total=self.get_xyz.shape[0]):
            mu = mu.detach()
            inv_sigma = inv_sigma.detach()
            inv_det = inv_det.detach()
            amp = amp.detach()

            d = coords - mu
            power = -0.5 * torch.sum((d @ inv_sigma) * d, dim=-1)
            inv_denom = torch.sqrt(inv_det) / ((2 * torch.pi)**1.5)
            g = inv_denom * torch.exp(power)

            if amp.isnan().any() or g.isnan().any():
                continue
            else:
                volume += amp * g

        volume = volume.reshape((D-1, D-1, D-1)) * norm[1]

        return volume