"""
Train a Gaussian Mixture Model (GMM) for homogeneous cryo-EM reconstruction.

This script performs 3D reconstruction of a homogeneous structure by optimizing
a set of Gaussian parameters (positions, scales, rotations, and amplitudes)
to match the experimental cryo-EM particle images under known poses and CTFs.
The resulting model provides a differentiable and memory-efficient representation
of the reconstructed electron density.

Optionally, the program can generate half-maps and corresponding FSC curves
for resolution estimation and validation.

Example usage
-------------
$ python train_gmm_homo.py particles.128.mrcs \
                            --ctf ctf.pkl --poses pose.pkl -o experiments/output_gmm_homo/

# Use `--lazy` for large datasets to reduce memory usage
$ python train_gmm_homo.py particles.256.mrcs --ctf ctf.pkl --poses pose.pkl \
                            --ind good_particles.pkl -o experiments/output_gmm_homo --lazy

Outputs
-------
- Reconstructed 3D volume (GMM-based density)
- Optional half-maps and FSC curve
- Training logs, learned Gaussian parameters, and intermediate checkpoints
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.functional import mse_loss
from typing import Literal, Union
import logging
from cryodrgn import ctf, dataset, fft, utils
from cryodrgn.lattice import Lattice
from cryodrgn.pose import PoseTracker
from cryodrgn.commands_utils.fsc import calculate_cryosparc_fscs
from cryodrgn.source import write_mrc
from cryodrgn.masking import spherical_window_mask

from gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)

def add_args(parser):
    parser.add_argument(
        "particles",
        type=os.path.abspath,
        help="Input particles (.mrcs, .star, .cs, or .txt)",
    )
    parser.add_argument(
        "--poses", type=os.path.abspath, required=True, help="Image poses (.pkl)"
    )
    parser.add_argument(
        "--ctf",
        metavar="pkl",
        type=os.path.abspath,
        help="CTF parameters (.pkl) for phase flipping images",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        type=os.path.abspath,
        required=True,
        help="New or existing folder in which outputs will be " "placed",
    )
    parser.add_argument(
        "--no-half-maps",
        action="store_false",
        help="Don't produce half-maps and FSCs.",
        dest="half_maps",
    )
    parser.add_argument(
        "--no-fsc-vals",
        action="store_false",
        help="Don't calculate FSCs, but still produce half-maps.",
        dest="fsc_vals",
    )
    parser.add_argument("--ctf-alg", type=str, choices=("flip", "mul"), default="mul")
    parser.add_argument(
        "--reg-weight",
        type=float,
        default=1.0,
        help="Add this value times the mean weight to the weight map to regularize the"
        "volume, reducing noise.\nAlternatively, you can set --output-sumcount, and "
        "then use `cryodrgn_utils regularize_backproject` on the"
        ".sums and .counts files to try different regularization constants post hoc.\n"
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--output-sumcount",
        action="store_true",
        help="Output voxel sums and counts so that different regularization weights "
        "can be applied post hoc, with `cryodrgn_utils regularize_backproject`.",
    )

    group = parser.add_argument_group("Dataset loading options")
    group.add_argument(
        "--uninvert-data",
        dest="invert_data",
        action="store_false",
        help="Do not invert data sign",
    )
    group.add_argument(
        "--window",
        dest="window",
        action="store_true",
        help="Turn on real space windowing of dataset",
    )
    group.add_argument(
        "--window-r",
        type=float,
        default=0.85,
        help="Windowing radius (default: %(default)s)",
    )
    group.add_argument(
        "--datadir",
        type=os.path.abspath,
        help="Path prefix to particle stack files if loading "
        "relative stack paths from a .star or .cs file",
    )
    group.add_argument(
        "--lazy",
        action="store_true",
        help="Lazy loading if full dataset is too large to fit in memory",
    )
    group.add_argument(
        "--ind",
        type=os.path.abspath,
        metavar="PKL",
        help="Filter particles by these indices before starting backprojection",
    )
    group.add_argument(
        "--first",
        type=int,
        help="Backproject the first N images (default: all images)",
    )
    

    group = parser.add_argument_group("Tilt series options")
    group.add_argument(
        "--tilt",
        action="store_true",
        help="Flag to treat data as a tilt series from cryo-ET",
    )
    group.add_argument(
        "--ntilts",
        type=int,
        default=10,
        help="Number of tilts per particle to backproject (default: %(default)s)",
    )
    group.add_argument(
        "-d",
        "--dose-per-tilt",
        type=float,
        help="Expected dose per tilt (electrons/A^2 per tilt) (default: %(default)s)",
    )
    group.add_argument(
        "-a",
        "--angle-per-tilt",
        type=float,
        default=3,
        help="Tilt angle increment per tilt in degrees (default: %(default)s)",
    )

    group = parser.add_argument_group("Training options")
    group.add_argument(
        "--num-points",
        type=int,
        default=10000,
        help="Number of Gaussians (default: %(default)s)",
    )
    group.add_argument(
        "-n",
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--log-interval",
        type=str,
        default="100",
        help="Logging interval in N_IMGS (default: %(default)s)",
    )
    group.add_argument(
        "--position-lr",
        type=float,
        default=0.001,
        help="Learning rate for xyz",
    )
    group.add_argument(
        "--scaling-lr",
        type=float,
        default=0.001,
        help="Learning rate for scaling",
    )
    group.add_argument(
        "--rotation-lr",
        type=float,
        default=0.001,
        help="Learning rate for rotation",
    )
    group.add_argument(
        "--amplitude-lr",
        type=float,
        default=0.001,
        help="Learning rate for amplitude",
    )
    group.add_argument(
        "--gamma",
        type=float,
        default=0.2,
        help="The learning rate decays by a factor of gamma every epoch",
    )
    group.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset at the end of every epoch",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_ctf_centered(proj_spatial_centered, ctf_freq_centered):
    proj_shifted = torch.fft.ifftshift(proj_spatial_centered) # move the origin from the center to top-left
    proj_freq = torch.fft.fft2(proj_shifted)
    ctf_freq_shifted = torch.fft.ifftshift(ctf_freq_centered)
    proj_freq_ctf = proj_freq * ctf_freq_shifted
    proj_spatial_shifted = torch.fft.ifft2(proj_freq_ctf).real
    proj_ctf_applied = torch.fft.fftshift(proj_spatial_shifted)

    return proj_ctf_applied

def main(args):
    set_seed(args.seed)

    if args.outdir is not None and not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    logger.addHandler(logging.FileHandler(f"{args.outdir}/run.log"))
    logger.info(" ".join(sys.argv))
    logger.info(args)

    # set the device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info("Use cuda {}".format(use_cuda))
    if not use_cuda:
        logger.warning("WARNING: No GPUs detected")

    # load the particles
    if args.ind is not None:
        if args.tilt:
            particle_ind = utils.load_pkl(args.ind).astype(int)
            pt, tp = dataset.TiltSeriesData.parse_particle_tilt(args.particles)
            tilt_ind = dataset.TiltSeriesData.particles_to_tilts(pt, particle_ind)
            indices = tilt_ind
        else:
            indices = utils.load_pkl(args.ind).astype(int)
    else:
        indices = None

    if args.tilt:
        assert (
            args.dose_per_tilt is not None
        ), "Argument --dose-per-tilt is required for backprojecting tilt series data"
        data = dataset.TiltSeriesData(
            args.particles,
            args.ntilts,
            norm=(0, 1),
            invert_data=args.invert_data,
            datadir=args.datadir,
            ind=indices,
            lazy=args.lazy,
            dose_per_tilt=args.dose_per_tilt,
            angle_per_tilt=args.angle_per_tilt,
            device=device,
        )
    else:
        data = dataset.ImageDataset(
            mrcfile=args.particles,
            norm=(0, 1),
            invert_data=args.invert_data,
            datadir=args.datadir,
            ind=indices,
            lazy=args.lazy,
            window=False,
        )

    D = data.D
    Nimg = data.N
    lattice = Lattice(D, extent=D // 2, device=device)
    posetracker = PoseTracker.load(args.poses, Nimg, D, None, indices, device=device)

    if args.ctf is not None:
        logger.info(f"Loading ctf params from {args.ctf}")
        ctf_params = ctf.load_ctf_for_training(D - 1, args.ctf)

        if indices is not None:
            ctf_params = ctf_params[indices]
        if args.tilt:
            ctf_params = np.concatenate(
                (ctf_params, data.ctfscalefactor.reshape(-1, 1)), axis=1  # type: ignore
            )
        ctf_params = torch.tensor(ctf_params, device=device)

    else:
        ctf_params = None

    Apix = float(ctf_params[0, 0]) if ctf_params is not None else 1.0
    voltage = float(ctf_params[0, 4]) if ctf_params is not None else None
    data.voltage = voltage
    lattice_mask = lattice.get_circular_mask(D // 2)
    img_iterator = list(range(min(args.first, Nimg)) if args.first else list(range(Nimg)))

    if args.tilt:
        use_tilts = set(range(args.ntilts))
        img_iterator = [
            ii for ii in img_iterator if int(data.tilt_numbers[ii].item()) in use_tilts
        ]

    if args.window:
        window = spherical_window_mask(D=D-1, in_rad=args.window_r, out_rad=0.99).to(device)
    else:
        window = None

    # Initialize gaussians
    img_count = len(img_iterator)
    gaussian_full = GaussianModel()
    gaussian_full.initialize_gaussians_random(P=args.num_points)
    gaussian_full.training_setup(args)
    accum_counter_full = 0
    if args.half_maps:
        gaussian_half1 = GaussianModel()
        gaussian_half1.initialize_gaussians_random(P=args.num_points)
        gaussian_half1.training_setup(args)
        accum_counter_half1 = 0
        gaussian_half2 = GaussianModel()
        gaussian_half2.initialize_gaussians_random(P=args.num_points)
        gaussian_half2.training_setup(args)
        accum_counter_half2 = 0

    # Figure out how often we are going to report progress w.r.t. images processed
    if args.log_interval == "auto":
        log_interval = max(round((img_count // 1000), -2), 100)
    elif args.log_interval.isnumeric():
        log_interval = int(args.log_interval)
    else:
        raise ValueError(
            f"Unrecognized argument for --log-interval: `{args.log_interval}`\n"
            f"Given value must be an integer or the label 'auto'!"
        )

    # Start training
    for epoch in range(args.num_epochs):
        t1 = time.time()
        for i, ii in enumerate(img_iterator):
            if i % log_interval == 0:
                logger.info(f"fimage {ii} — {(i / img_count * 100):.1f}% done")

            # Input data
            r, t = posetracker.get_pose(ii)
            ff = data.get_tilt(ii) if args.tilt else data[ii]
            assert isinstance(ff, tuple)
            ff = ff[0].to(device)
            ctf_mul = 1

            if ctf_params is not None:
                freqs = lattice.freqs2d / ctf_params[ii, 0]
                c = ctf.compute_ctf(freqs, *ctf_params[ii, 1:]).view((D,D))
                ctf_mul = c.view((D,D))

            if t is not None:
                ff = lattice.translate_ht(
                    ff.view(1, -1), t.view(1, 1, 2)
                ).view((D, D))

            if args.tilt:
                tilt_idxs = torch.tensor([ii]).to(device)
                dose_filters = data.get_dose_filters(tilt_idxs, lattice, ctf_params[ii, 0])[
                    0
                ]
                ctf_mul *= dose_filters[lattice_mask]

            # Move to real space
            gt_image = fft.iht2_center(ff[0:-1, 0:-1].detach())

            if args.window and window is not None:
                gt_image *= window

            # Render projections
            viewmatrix = torch.eye(4)
            viewmatrix[:3, :3] = r
            viewmatrix = viewmatrix.cuda()
            projmatrix = getProjectionMatrix(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, type="cryoem")
            projmatrix = projmatrix.cuda()
            render_pkg = render(gaussian_full,
                                D-1,
                                D-1,
                                viewmatrix,
                                projmatrix)
                    
            proj_full, viewspace_point_tensor, visibility_filter, radii = render_pkg["rendered_image"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            if i % 10000 == 0:
                plt.matshow(gt_image.detach().squeeze().cpu().numpy())
                plt.colorbar()
                plt.savefig(os.path.join(args.outdir, "gt_image.png"))
                plt.close()

                plt.matshow(proj_full.detach().squeeze().cpu().numpy())
                plt.colorbar()
                plt.savefig(os.path.join(args.outdir, "reconstructed.png"))
                plt.close()

            # Apply CTF
            proj_full = apply_ctf_centered(proj_full, ctf_mul[0:-1, 0:-1])

            # Optimization
            loss_full = mse_loss(proj_full.squeeze(), gt_image.squeeze())
            loss_full.backward()

            gaussian_full.optimizer.step()
            gaussian_full.optimizer.zero_grad(set_to_none=True)

            # Process half maps
            if args.half_maps:
                if ii % 2 == 0:
                    render_pkg = render(gaussian_half1,
                                D-1,
                                D-1,
                                viewmatrix,
                                projmatrix)
                    
                    proj_half1, viewspace_point_tensor, visibility_filter, radii = render_pkg["rendered_image"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    proj_half1 = apply_ctf_centered(proj_half1, ctf_mul[0:-1, 0:-1])
                    loss_half1 = mse_loss(proj_half1.squeeze(), gt_image.squeeze())
                    loss_half1.backward()

                    gaussian_half1.optimizer.step()
                    gaussian_half1.optimizer.zero_grad(set_to_none=True)
                else:
                    render_pkg = render(gaussian_half2,
                                D-1,
                                D-1,
                                viewmatrix,
                                projmatrix)
                    
                    proj_half2, viewspace_point_tensor, visibility_filter, radii = render_pkg["rendered_image"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    proj_half2 = apply_ctf_centered(proj_half2, ctf_mul[0:-1, 0:-1])
                    loss_half2 = mse_loss(proj_half2.squeeze(), gt_image.squeeze())
                    loss_half2.backward()

                    gaussian_half2.optimizer.step()
                    gaussian_half2.optimizer.zero_grad(set_to_none=True)

        td = time.time() - t1
        logger.info(
            f"Backprojected {img_count} images "
            f"in {td:.2f}s ({(td / img_count):4f}s per image)"
        )

        volume_full = gaussian_full.generate_volume(D=D, extent=0.5, norm=data.norm).cpu()
        vol_fl = os.path.join(args.outdir, "gaussian.{}.mrc".format(epoch))
        write_mrc(vol_fl, np.array(volume_full).astype("float32"), Apix=Apix)
        gau_fl = os.path.join(args.outdir, "gaussian.{}.ply".format(epoch))
        gaussian_full.save_ply(gau_fl)

        # Create the half-maps, calculate the FSC curve between them, and save both to file
        if args.half_maps:
            volume_half1 = gaussian_half1.generate_volume(D=D, extent=0.5, norm=data.norm).cpu()
            volume_half2 = gaussian_half2.generate_volume(D=D, extent=0.5, norm=data.norm).cpu()
            half_fl1 = os.path.join(args.outdir, "half_map_a.{}.mrc".format(epoch))
            half_fl2 = os.path.join(args.outdir, "half_map_b.{}.mrc".format(epoch))
            write_mrc(half_fl1, np.array(volume_half1).astype("float32"), Apix=Apix)
            write_mrc(half_fl2, np.array(volume_half2).astype("float32"), Apix=Apix)

            if args.fsc_vals:
                out_file = os.path.join(args.outdir, "fsc-vals.{}.txt".format(epoch))
                plot_file = os.path.join(args.outdir, "fsc-plot.{}.png".format(epoch))
                _ = calculate_cryosparc_fscs(
                    volume_full,
                    volume_half1,
                    volume_half2,
                    apix=Apix,
                    out_file=out_file,
                    plot_file=plot_file,
                )

        # Update learning rate
        gaussian_full.update_learning_rate()
        gaussian_half1.update_learning_rate()
        gaussian_half2.update_learning_rate()

        if args.shuffle:
            random.shuffle(img_iterator)

    # Final evaluation
    volume_full = gaussian_full.generate_volume(D=D, extent=0.5, norm=data.norm).cpu()
    vol_fl = os.path.join(args.outdir, "gaussian.mrc")
    write_mrc(vol_fl, np.array(volume_full).astype("float32"), Apix=Apix)

    # Create the half-maps, calculate the FSC curve between them, and save both to file
    if args.half_maps:
        volume_half1 = gaussian_half1.generate_volume(D=D, extent=0.5, norm=data.norm).cpu()
        volume_half2 = gaussian_half2.generate_volume(D=D, extent=0.5, norm=data.norm).cpu()
        half_fl1 = os.path.join(args.outdir, "half_map_a.mrc")
        half_fl2 = os.path.join(args.outdir, "half_map_b.mrc")
        write_mrc(half_fl1, np.array(volume_half1).astype("float32"), Apix=Apix)
        write_mrc(half_fl2, np.array(volume_half2).astype("float32"), Apix=Apix)

        if args.fsc_vals:
            out_file = os.path.join(args.outdir, "fsc-vals.txt")
            plot_file = os.path.join(args.outdir, "fsc-plot.png")
            _ = calculate_cryosparc_fscs(
                volume_full,
                volume_half1,
                volume_half2,
                apix=Apix,
                out_file=out_file,
                plot_file=plot_file,
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    main(args=args)
