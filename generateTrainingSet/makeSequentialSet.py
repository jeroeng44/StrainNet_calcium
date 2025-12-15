#!/usr/bin/env python3

"""
Create a sequential dataset (like datasets/synthetic/04DEF) with cumulative strain ground truth.
Run from generateTrainingSet/ directory: python makeSequentialSet.py --image ... --mask ... --output_dir ...
"""

import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm

# Use relative imports (same pattern as generateTrainingSet.py)
from core import utils
import deformations


def save_image(path, img):
    """Save float32 image as uint8 PNG."""
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img_u8)


def main():
    parser = argparse.ArgumentParser(
        description="Create a sequential dataset with cumulative deformations (04DEF-like)."
    )
    parser.add_argument('--image', required=True, help='Path to baseline image')
    parser.add_argument('--mask', required=True, help='Path to matching binary mask')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--deformation', default='tension', choices=['tension', 'compression', 'rigid'])
    parser.add_argument('--frames', type=int, default=101, help='Number of frames including baseline')
    parser.add_argument('--total_strain', type=float, default=4.0, help='Target total strain percent, e.g. 4.0')
    parser.add_argument('--nu', type=float, default=0.49, help='Poisson ratio')
    parser.add_argument('--noise', type=float, default=0.0, help='Gaussian noise sigma')

    args_cli = parser.parse_args()

    # Load baseline image and mask
    img, mask = utils.load.image_and_mask(args_cli.image, args_cli.mask)
    print(f"Loaded image shape: {img.shape}, mask shape: {mask.shape}")

    # Create output directories
    images_dir = os.path.join(args_cli.output_dir, 'images')
    strains_dir = os.path.join(args_cli.output_dir, 'strains')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(strains_dir, exist_ok=True)

    # Create minimal args object for deformation_maker
    class MinimalArgs:
        pass
    
    args_deform = MinimalArgs()
    # Use full image dimensions
    args_deform.output_height = img.shape[0]
    args_deform.output_width = img.shape[1]
    args_deform.upper_left_corner_x = 0
    args_deform.upper_left_corner_y = 0
    args_deform.noise = args_cli.noise
    args_deform.rigid_deformation_functions = 'all'
    args_deform.compression_deformation_functions = 'all'
    args_deform.tension_deformation_functions = 'all'
    # Used by randomizeParameters (we override manually)
    args_deform.min_nu = 0.25
    args_deform.max_nu = 1.5
    args_deform.min_epsilon_xx = 0.02
    args_deform.max_epsilon_xx = 0.20
    args_deform.min_num_frames = 5
    args_deform.max_num_frames = 15
    args_deform.min_rotation_angle = 0.0
    args_deform.max_rotation_angle = 2.0
    args_deform.min_displacement = 0.0
    args_deform.max_displacement = 2.0

    # Create deformation maker
    dm = deformations.deformation_maker(img, mask, args_cli.deformation, COUNT=1, args=args_deform)
    print(f"Deformation maker initialized for {args_cli.deformation}")

    # Calculate per-step parameters
    n_steps = int(args_cli.frames)
    total_eps = float(args_cli.total_strain) / 100.0
    step_eps = total_eps / max(1, n_steps - 1)

    dm.nu = float(args_cli.nu)

    # Setup deformation parameters
    if args_cli.deformation in ['tension', 'compression']:
        sign = 1.0 if args_cli.deformation == 'tension' else -1.0
        dm.epsilon_xx_center = sign * step_eps
        dm.epsilon_xx_edge = sign * step_eps
        dm.epsilon_xx_distal = sign * step_eps
        dm.epsilon_xx_proximal = sign * step_eps
        dm.a = (4.0 * (dm.epsilon_xx_edge - dm.epsilon_xx_center)) / (dm.mask_height ** 2) if dm.mask_height > 0 else 0.0
    else:  # rigid
        dm.u_x = 0.0
        dm.u_y = 0.0
        dm.rotation = step_eps * 10.0  # small rotation per step

    # Initialize sequence (don't crop - use full image)
    current_img = img
    H, W = current_img.shape
    cumulative_strain = np.zeros((H, W, 3), dtype=np.float32)

    # Save frame 0000 (baseline, zero strain)
    save_image(os.path.join(images_dir, f"{0:04d}_im.png"), current_img)
    np.save(os.path.join(strains_dir, f"{0:04d}_strain_xx.npy"), np.zeros((H, W), dtype=np.float32))
    np.save(os.path.join(strains_dir, f"{0:04d}_strain_xy.npy"), np.zeros((H, W), dtype=np.float32))
    np.save(os.path.join(strains_dir, f"{0:04d}_strain_yy.npy"), np.zeros((H, W), dtype=np.float32))

    # Generate frames 1 to N-1
    for t in tqdm(range(1, n_steps), desc=f'Generating {n_steps} frames'):
        # Reset displacement and strain fields
        dm.initialize_displacement_field_and_strain()
        # Calculate per-step displacement and strain
        dm.calculate()

        # Warp current image (no cropping - use full image)
        im1, im2 = dm.imwarp(current_img, dm.displacement_field)

        # Get per-step strain and accumulate (apply mask to zero out outside regions)
        step_strain = dm.strain.copy()
        step_strain[mask == 0] = 0  # Zero out strain outside mask
        cumulative_strain = cumulative_strain + step_strain

        # Save full image and cumulative strain
        save_image(os.path.join(images_dir, f"{t:04d}_im.png"), im2)
        np.save(os.path.join(strains_dir, f"{t:04d}_strain_xx.npy"), cumulative_strain[:, :, 0])
        np.save(os.path.join(strains_dir, f"{t:04d}_strain_xy.npy"), cumulative_strain[:, :, 2])
        np.save(os.path.join(strains_dir, f"{t:04d}_strain_yy.npy"), cumulative_strain[:, :, 1])

        # Move to next frame
        current_img = im2

    print(f"\nSequential dataset created!")
    print(f"  Output: {args_cli.output_dir}")
    print(f"  Frames: {n_steps} (0000 to {n_steps-1:04d})")
    print(f"  Deformation: {args_cli.deformation}")
    print(f"  Total strain: {args_cli.total_strain}%")


if __name__ == '__main__':
    main()
