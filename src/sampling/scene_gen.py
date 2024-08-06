import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import math

from ..diffusion.utils import (TRUNC_VAL, load_checkpoint,
                               mc_and_save, set_random_seeds)
from ..diffusion.load_model import construct_model_from_json

"""
TODO
Features:
- save each chunk individually with option to save denoising process to create a video later on
Issues:
- floorplan chunk size is hardcoded
- the way the technical height and chunk size are multiplied x2 for coarse only is awkward for calling the method from outside
"""


@torch.no_grad()
def diagonal_scene_gen(model, model_super=None, chunk_size=64, d0=3, d2=3, technical_height=71, height_cond=None, floorplan=None, batch_size=1, verbose=True):
    # Floorplan scene dimensions
    if floorplan is not None:
        floorplan_chunk_size = 64

        floorplan_d0, floorplan_d2 = floorplan.shape[1], floorplan.shape[2]
        # Set dimensions to floorplan dimensions
        d0 = math.ceil(floorplan_d0 / floorplan_chunk_size)
        d2 = math.ceil(floorplan_d2 / floorplan_chunk_size)

        # Pad floorplan to be divisible by chunk size
        d0_pad = d0 * floorplan_chunk_size - floorplan_d0
        d2_pad = d2 * floorplan_chunk_size - floorplan_d2
        floorplan = torch.nn.functional.pad(
            floorplan, (0, d2_pad, 0, d0_pad), mode='constant', value=0)

        print(
            f"Floorplan dimensions: {floorplan_d0}x{floorplan_d2}, padded to {d0*floorplan_chunk_size}x{d2*floorplan_chunk_size}")

    chunk_size = chunk_size if model_super is None else 2 * chunk_size
    extend_d0 = chunk_size * d0
    extend_d2 = chunk_size * d2
    technical_height = technical_height if model_super is None else 2 * technical_height
    print(
        f"Generating scene of size {extend_d0}x{technical_height}x{extend_d2}, chunk size: {chunk_size}, d0: {d0}, d2: {d2}, height cond: {height_cond}")

    # Create empty scene
    scene = torch.ones((extend_d0, technical_height, extend_d2),
                       dtype=torch.float32) * TRUNC_VAL

    def get_conds_for_pos(p0, p2):
        # Return the previous chunks in negative d0 and d2
        if p0 == 0:
            cond_d0 = torch.ones((chunk_size, technical_height, chunk_size),
                                 dtype=torch.float32) * TRUNC_VAL
        else:
            prev_start = (p0 - 1) * chunk_size
            prev_end = p0 * chunk_size
            cond_d0 = scene[prev_start:prev_end, :, p2 * chunk_size:(p2 + 1) *
                            chunk_size]

        if p2 == 0:
            cond_d2 = torch.ones((chunk_size, technical_height, chunk_size),
                                 dtype=torch.float32) * TRUNC_VAL
        else:
            prev_start = (p2 - 1) * chunk_size
            prev_end = p2 * chunk_size
            cond_d2 = scene[p0 * chunk_size:(p0 + 1) * chunk_size, :, prev_start:
                            prev_end]

        if model_super is not None:
            # min pooling
            cond_d0 = -nn.MaxPool3d(2, 2)(-cond_d0[None, None, :]).squeeze()
            cond_d2 = -nn.MaxPool3d(2, 2)(-cond_d2[None, None, :]).squeeze()

        return cond_d0, cond_d2

    def get_next_pos(p0, p2):
        # we set diagonals s.t. p0 + p2 = diagonal
        if p0 > 0 and p2 < d2 - 1:
            # we can move diagonally
            return p0 - 1, p2 + 1
        else:
            # we need to move to the next diagonal
            next_diagonal = p0 + p2 + 1
            if next_diagonal < d0:
                return next_diagonal, 0
            else:
                # we go down the edge
                return d0 - 1, next_diagonal - (d0 - 1)

    def is_pos_generatable(p0, p2, is_empty):
        # Check if the position is in bounds
        if p0 >= d0 or p2 >= d2:
            return False
        if p0 < 0 or p2 < 0:
            return False

        # Check if the position is generatable
        if p0 == 0 and p2 == 0:
            return True
        elif p0 == 0:
            return not is_empty[p0][p2 - 1]
        elif p2 == 0:
            return not is_empty[p0 - 1][p2]
        else:
            return not is_empty[p0 - 1][p2] and not is_empty[p0][p2 - 1]

    def generate_batch_of_positions(i0, i2, is_empty, batch_size):
        # Generate a batch of positions
        batch = []
        while len(batch) < batch_size and is_pos_generatable(i0, i2, is_empty):
            batch.append((i0, i2))
            i0, i2 = get_next_pos(i0, i2)

        assert len(batch) > 0, "No positions were generatable."
        return batch, i0, i2

    i0, i2 = 0, 0
    is_empty = [[True for _ in range(d2)] for _ in range(d0)]

    generated_chunks = 0
    while generated_chunks < d0 * d2:
        # Get the next batch of positions
        batch_pos, i0, i2 = generate_batch_of_positions(
            i0, i2, is_empty, batch_size)
        current_batch_size = len(batch_pos)

        if verbose:
            print(
                f"Generating batch of {current_batch_size} chunks, status: {generated_chunks}/{d0*d2}, batch pos: {batch_pos}")

        # Get the conditions for the batch
        conds0, conds2 = [], []
        if floorplan is not None:
            floorplans = []

        for p0, p2 in batch_pos:
            cond_d0, cond_d2 = get_conds_for_pos(p0, p2)
            conds0.append(cond_d0.unsqueeze(0))
            conds2.append(cond_d2.unsqueeze(0))

            if floorplan is not None:
                floorplans.append(floorplan[:, p0 * floorplan_chunk_size:(p0 + 1) * floorplan_chunk_size,
                                            p2 * floorplan_chunk_size:(p2 + 1) * floorplan_chunk_size])

        conds0 = torch.stack(conds0, dim=0).cuda()
        conds2 = torch.stack(conds2, dim=0).cuda()

        floorplans = torch.stack(floorplans, dim=0).cuda(
        ) if floorplan is not None else None

        # Generate batch of heights
        heights = torch.full((current_batch_size,),
                             height_cond, dtype=torch.int64).cuda() if height_cond is not None else None

        # Sample the batch
        sampling_shape = conds0.shape
        preds = model.sample(
            shape=sampling_shape,
            conditions=[conds0, conds2],
            heights=heights,
            floorplans=floorplans)

        if model_super is not None:
            sampling_shape = (sampling_shape[0], sampling_shape[1], sampling_shape[2]
                              * 2, sampling_shape[3] * 2, sampling_shape[4] * 2)

            if verbose:
                print(f"Running super resolution to shape {sampling_shape}")
            preds = model_super.sample(
                shape=sampling_shape,
                heights=heights,
                coarse=preds)

        # Add the batch to the scene
        for i, (p0, p2) in enumerate(batch_pos):
            start_p0 = p0 * chunk_size
            start_p2 = p2 * chunk_size
            scene[start_p0:start_p0 + chunk_size, :, start_p2:start_p2 +
                  chunk_size] = preds[i, :, :]

        generated_chunks += current_batch_size
        for p0, p2 in batch_pos:
            is_empty[p0][p2] = False

    return scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model_json", type=str, default=None,
                        help='Path model configuration file. ')
    parser.add_argument("--model_path", type=str, default=None,
                        help='Path to model checkpoint.')

    parser.add_argument("--model_dir", type=str, default=None,
                        help='Path to model directory. Give either this or path to model and json directly.')

    # Super resolution
    parser.add_argument("--super_json", type=str, default=None,
                        help='Path super resolution model configuration file. ')
    parser.add_argument("--super_path", type=str, default=None,
                        help='Path to super resolution model checkpoint.')

    parser.add_argument("--super_dir", type=str, default=None,
                        help='Path to super resolution model directory. Give either this or path to mdoel and json directly.')

    # Scene dimensions
    parser.add_argument("--d0", type=int, default=3,
                        help='Number of chunks in the first dimension.')
    parser.add_argument("--d2", type=int, default=3,
                        help='Number of chunks in the second dimension.')

    # Floorplan
    parser.add_argument("--floorplan", type=str, default=None,
                        help='Path to floorplan to condition on.')

    # Height
    parser.add_argument("--height_emb", type=int, default=None,
                        help='Height embedding to use.')
    parser.add_argument("--height_tech", type=int, default=36,
                        help='Height of the scene in voxels.')

    # Sampling
    parser.add_argument("--batch_size", type=int, default=1,
                        help='Batch size.')
    parser.add_argument("--cfg_weight", type=float, default=1.0)

    # Coarse
    parser.add_argument("--steps", type=int, default=1000,
                        help='Number of sampling steps.')
    parser.add_argument("--rho", type=int, default=4)
    parser.add_argument("--S_churn", type=int, default=50)
    parser.add_argument("--S_min", type=float, default=0.05)
    parser.add_argument("--S_max", type=float, default=10)
    parser.add_argument("--S_noise", type=float, default=1.000)
    parser.add_argument("--ode", type=str, default="euler",
                        choices=["euler", "heun"])

    # Super
    parser.add_argument("--steps_super", type=int, default=125,
                        help='Number of sampling steps for super resolution.')
    parser.add_argument("--rho_super", type=int, default=7)
    parser.add_argument("--S_churn_super", type=int, default=50)
    parser.add_argument("--S_min_super", type=float, default=0.05)
    parser.add_argument("--S_max_super", type=float, default=10)
    parser.add_argument("--S_noise_super", type=float, default=1.000)
    parser.add_argument("--ode_super", type=str, default="euler",
                        choices=["euler", "heun"])

    # Output
    parser.add_argument("--output_dir", type=str, default=os.path.join("..",
                        "samples"), help='Path to output directory.')
    parser.add_argument("-n", "--name", type=str,
                        required=True, help='Name of the output file.')
    parser.add_argument("--force", action="store_true",
                        help='Force overwrite of output directory if it exists.')

    # Seed
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')

    args = parser.parse_args()

    # Set random seed
    set_random_seeds(args.seed)

    # Output directory, cannot exist
    output_path = os.path.join(args.output_dir, args.name)
    if os.path.exists(output_path):
        if not args.force:
            raise ValueError(
                f"Output path {output_path} already exists. Please choose a different name.")

    os.makedirs(output_path, exist_ok=True)

    # Device
    assert torch.cuda.is_available(), "CUDA must be available."

    # Load model
    assert args.model_dir is not None or (
        args.model_json is not None and args.model_path is not None), "Give either model directory or model json and path."

    if args.model_dir is not None:
        args.model_json = os.path.join(args.model_dir, "model.json")
        args.model_path = os.path.join(args.model_dir, "state.chkpt")

    arch, model, chunk_size = construct_model_from_json(
        args.model_json)
    model.set_all_sample_parameters(args.steps, args.rho, args.S_churn,
                                    args.S_min, args.S_max, args.S_noise)
    model.ode_solver = args.ode

    device = "cuda"
    load_checkpoint(args.model_path, device, arch)
    arch, model = arch.to(device), model.to(device)
    arch.eval(), model.eval()

    # Determine which size the model expects
    adjusted_chunk_size = arch.input_size * model.vae.scale_factor
    print(f"Adjusted chunk size: {adjusted_chunk_size}")

    if args.super_dir is not None:
        args.super_json = os.path.join(args.super_dir, "model.json")
        args.super_path = os.path.join(args.super_dir, "state.chkpt")

    if args.super_json is not None and args.super_path is not None:
        arch_super, model_super, chunk_size_super = construct_model_from_json(
            args.super_json)
        model_super.set_all_sample_parameters(args.steps_super, args.rho_super, args.S_churn_super,
                                              args.S_min_super, args.S_max_super, args.S_noise_super)
        model_super.ode_solver = args.ode_super

        load_checkpoint(args.super_path, "cuda", arch_super)
        arch_super = arch_super.eval().cuda()
        model_super = model_super.eval().cuda()
    else:
        model_super = None

    # Load floorplan
    floorplan = None
    if args.floorplan is not None:
        floorplan = np.load(args.floorplan)
        floorplan = torch.from_numpy(floorplan).long()
        floorplan = floorplan.permute(2, 0, 1)

    # Generate scene
    scene = diagonal_scene_gen(model,
                               model_super=model_super,
                               chunk_size=adjusted_chunk_size,
                               d0=args.d0,
                               d2=args.d2,
                               floorplan=floorplan,
                               technical_height=args.height_tech,
                               height_cond=args.height_emb,
                               batch_size=args.batch_size)

    # Save scene
    scene = scene.cpu().numpy()
    np.save(os.path.join(output_path, "scene.npy"), scene)
    print(f"Saved scene to {os.path.join(output_path, 'scene.npy')}")

    # Save scene mesh
    mc_and_save(scene, os.path.join(output_path, "scene.obj"))
    print(f"Saved scene mesh to {os.path.join(output_path, 'scene.obj')}")

    # Save floorplan
    if floorplan is not None:
        from ..diffusion.utils import save_floorplan
        save_floorplan(floorplan, os.path.join(output_path, "floorplan.png"))
        print(
            f"Saved floorplan to {os.path.join(output_path, 'floorplan.png')}")
