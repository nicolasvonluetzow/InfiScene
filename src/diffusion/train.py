import argparse
import math
import os
import os.path as path
import random
import shutil

import matplotlib as mpl
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import torch.nn as nn

from .utils import (get_all_files, load_checkpoint, mc_and_save,
                    save_checkpoint, set_random_seeds, save_floorplan)
from .load_model import construct_model_from_json
from .scene_dataset import SceneDataset, chunk_collate_fn
from ..sampling.scene_gen import diagonal_scene_gen


def train_model(args):
    # Set up relevant paths
    run_path = path.join(args.out, args.name)
    subrun_path = path.join(run_path, "sub0")
    i = 0
    while path.exists(subrun_path):
        i += 1
        subrun_path = path.join(run_path, f"sub{i}")

    img_path = path.join(subrun_path, "imgs")
    os.makedirs(img_path, exist_ok=True)

    # model.json
    default_json_path = path.join(run_path, "model.json")
    if not path.exists(default_json_path):
        if args.model_json is None:
            raise Exception(
                "No model.json was given and no previous one exists.")
        else:
            shutil.copy2(args.model_json, default_json_path)
    args.model_json = default_json_path if args.model_json is None else args.model_json

    # Logging
    writer = SummaryWriter(
        log_dir=path.join(args.log_dir, args.name, f"sub{i}"))
    log_file = open(path.join(subrun_path, "log.txt"), "a")

    def print_and_log(*args):
        print(*args)
        print(*args, file=log_file)

    print_and_log("Subrun:", i)
    print_and_log("Args:", args)
    print_and_log("\n---\n")
    del i

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    set_random_seeds()

    # Model
    arch, model, chunk_size = construct_model_from_json(args.model_json)
    model.sampling_timesteps = args.sampling_steps
    model.cfg_weight = args.cfg_weight

    data_downsample_factor = chunk_size // (arch.input_size *
                                            model.vae.scale_factor)
    print_and_log(
        f"Effective data downsample factor: {data_downsample_factor} (chunk size: {chunk_size}, input size: {arch.input_size}, vae scale factor: {model.vae.scale_factor})")
    print_and_log(f"Is super scaling: {arch.is_super_scaling}")

    arch, model = arch.to(device), model.to(device)
    assert model.vae is not None

    # LR scheduling
    opt = torch.optim.Adam(arch.parameters(), lr=args.lr)

    def lr_schedule(step):
        if step < args.lr_warmup:
            return max(float(step / args.lr_warmup), 1e-8)
        return math.pow(0.5, int((step - args.lr_warmup) / args.lr_schedule_interval))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)

    # Checkpointing
    checkpoint_path = path.join(run_path, "state.chkpt")
    # ema_path = path.join(run_path, "ema.chkpt")
    if path.exists(checkpoint_path) and not args.overwrite:
        if args.overwrite_lr:
            load_checkpoint(checkpoint_path, device, arch)
        else:
            load_checkpoint(checkpoint_path, device, arch, opt, scheduler)
    print_and_log("Device: \t", device)  # sanity check
    writer.add_scalar("params/lr", opt.param_groups[0]["lr"], 0)

    # Count, print and log number of parameters
    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    total_params = count_params(arch)
    print_and_log(
        f"Parameters:  \t {total_params:,}")  # ({' | '.join(individual_params)})")
    del count_params, total_params  # , individual_params

    # Read and split files
    train_files = get_all_files(args.train_data)
    if args.split_files:
        # random.shuffle(train_files)
        split_factor = 0.9
        train_files, val_files = train_files[:int(
            split_factor * len(train_files))], train_files[int(split_factor * len(train_files)):]

        # add train/val split to subrun directory
        with open(path.join(subrun_path, "train_files.txt"), "w") as f:
            for file in train_files:
                f.write(file + "\n")
        with open(path.join(subrun_path, "val_files.txt"), "w") as f:
            for file in val_files:
                f.write(file + "\n")
    else:
        val_files = get_all_files(args.val_data) if not args.no_val else []

    # Train file repeating
    if args.max_samples != 0:
        # random.shuffle(train_files)
        train_files = train_files[:args.max_samples]
        val_files = val_files[:args.max_samples]
    rep = args.min_epoch_samples // len(train_files) if len(
        train_files) < args.min_epoch_samples else 1
    train_files = train_files * rep

    print_and_log(
        f"Total files: \t {len(train_files) + len(val_files)} \nTrain/Val: \t {len(train_files)}/{len(val_files)}")
    assert len(train_files) > 0 and args.no_val or len(val_files) > 0

    if rep != 1:
        print_and_log(
            f"Epoch size:  \t {len(train_files):,} ({rep} repetitions per epoch).")

    # Data
    assert not (arch.use_height_conditioning and args.height_json is None)

    dataset = SceneDataset(train_files, not args.no_augmentation,
                           n_chunks=args.n_chunks, chunk_size=chunk_size, num_conditions=arch.num_conditions,
                           forced_boundary_prob=args.forced_boundary_prob, surface_proximity_threshold=args.surface_proximity_threshold,
                           floorplan_dir=args.floorplan_dir, boundary_dir=args.boundary_dir, semantic_dir=args.semantic_dir, height_json=args.height_json,
                           force_cubic=args.force_cubic, data_downsample_factor=data_downsample_factor, is_super_scaling=arch.is_super_scaling)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True,
                            pin_memory=True, num_workers=4, collate_fn=chunk_collate_fn)

    val_dataset = SceneDataset(val_files, not args.no_augmentation,
                               n_chunks=args.n_chunks, chunk_size=chunk_size, num_conditions=arch.num_conditions,
                               floorplan_dir=args.floorplan_dir, boundary_dir=args.boundary_dir, semantic_dir=args.semantic_dir, height_json=args.height_json,
                               force_cubic=args.force_cubic, data_downsample_factor=data_downsample_factor, is_super_scaling=arch.is_super_scaling)
    val_dataloader = DataLoader(
        val_dataset, args.batch_size, pin_memory=True, num_workers=4, collate_fn=chunk_collate_fn)

    if args.sample_freq != 0:
        sample_dataset = SceneDataset(val_files if not args.no_val else train_files, False, n_chunks=1,
                                      chunk_size=chunk_size, num_conditions=arch.num_conditions,
                                      floorplan_dir=args.floorplan_dir, keep_gen_area_cond_excess=True, boundary_dir=args.boundary_dir,
                                      height_json=args.height_json, force_cubic=args.force_cubic,
                                      data_downsample_factor=data_downsample_factor, is_super_scaling=arch.is_super_scaling)

    print_and_log("\n---\n")

    def bucketize(category_str, step, losses, noise_levels):
        left_bucket_borders = [0, 0.1, 1, 5]

        for loss, noise_level in zip(losses, noise_levels):
            noise_level = noise_level.item()
            bucket_ind = len(left_bucket_borders) - 1
            while left_bucket_borders[bucket_ind] > noise_level:
                bucket_ind -= 1

            # if the same noise level is chosen twice in a single batch this will write with the same step twice
            # the issue is too minor to fix
            writer.add_scalar(
                f"{category_str}/noise_{bucket_ind}", loss, step)

    def process_batch(batch):
        data = batch["chunks"].to(device)
        heights = batch["heights"].to(
            device) if "heights" in batch else None

        conds0 = batch["conds0"].to(device) if "conds0" in batch else None
        conds2 = batch["conds2"].to(device) if "conds2" in batch else None
        condsm0 = batch["condsm0"].to(
            device) if "condsm0" in batch else None
        condsm2 = batch["condsm2"].to(
            device) if "condsm2" in batch else None

        floorplans = batch["floorplans"].to(
            device) if "floorplans" in batch else None

        boundaries = batch["boundaries"].to(
            device) if "boundaries" in batch else None

        semantic = batch["semantic"].to(
            device) if "semantic" in batch else None

        coarse_chunks = batch["coarse_chunks"].to(
            device) if "coarse_chunks" in batch else None

        # combine all existing conditions into list
        conditions = [conds0, conds2, condsm0, condsm2]
        conditions = [cond for cond in conditions if cond is not None]

        if args.train_patch_wise and model.model.is_super_scaling and len(conditions) == 0 and floorplans is None and boundaries is None and semantic is None:
            # sample random 32 x 32 x 32 chunk from 64 x 64 x 64 chunk
            p1, p2, p3 = random.randint(0, 16), random.randint(
                0, 16), random.randint(0, 16)

            coarse_chunks = coarse_chunks[:, :,
                                          p1:p1 + 16, p2:p2 + 16, p3:p3 + 16]
            p1, p2, p3 = p1*2, p2*2, p3*2
            data = data[:, :, p1:p1 + 32, p2:p2 + 32, p3:p3 + 32]

        losses, noise_levels = model(
            data, heights, conditions=conditions, floorplans=floorplans, boundaries=boundaries, semantic=semantic, coarse=coarse_chunks)

        return losses, noise_levels

    # Training
    step = 0
    val_step = 0
    for epoch in range(args.epochs):
        model.train()
        print("LR:", opt.param_groups[0]["lr"])

        # Train loop
        epoch_train_losses = []
        for batch in tqdm(dataloader):
            losses, noise_levels = process_batch(batch)
            bucketize("buckets", step, losses, noise_levels)

            loss = losses.mean()
            writer.add_scalar(f"loss/train", loss.item(), step)
            epoch_train_losses.append(loss.item())

            step += 1

            loss = loss / args.acc_steps
            loss.backward()

            if step % args.acc_steps == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)

                # LR scheduling
                scheduler.step()
                writer.add_scalar(
                    "params/lr", opt.param_groups[0]["lr"], step)

        mean_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        writer.add_scalar(f"loss/train_mean", mean_train_loss, step)

        print_and_log(
            f"Finished epoch {epoch + 1}/{args.epochs}")
        print_and_log(f"Train loss: \t {mean_train_loss:05f}")

        # Validation
        model.eval()
        if not args.no_val:
            with torch.no_grad():
                epoch_val_losses = []
                for data in val_dataloader:  # no tqdm for val
                    losses, noise_levels = process_batch(data)
                    bucketize("buckets_val", val_step, losses, noise_levels)

                    loss = losses.mean()
                    epoch_val_losses.append(loss.item())

                    val_step += 1

                mean_val_loss = sum(epoch_val_losses) / \
                    len(epoch_val_losses)
                writer.add_scalar(f"loss/val_mean", mean_val_loss, step)

                print_and_log(f"Val loss: \t {mean_val_loss:05f}")

                # Graph with validation and training loss, needs to be saved as individual runs
                writer.add_scalars("loss/epoch", {
                    "train_mean": mean_train_loss,
                    "val_mean": mean_val_loss
                }, epoch)

        # Sampling
        if args.sample_freq != 0 and epoch % args.sample_freq == args.sample_freq - 1:
            with torch.no_grad():
                for i in range(args.sample_amount):
                    sample = random.choice(sample_dataset)

                    data = sample["chunks"].to(device)
                    heights = sample["heights"].to(
                        device).unsqueeze(0) if "heights" in sample else None

                    floorplans = sample["floorplans"].to(
                        device) if "floorplans" in sample else None

                    boundaries = sample["boundaries"].to(
                        device) if "boundaries" in sample else None

                    coarse_chunks = sample["coarse_chunks"].to(
                        device) if "coarse_chunks" in sample else None

                    if arch.num_conditions == 0:
                        preds = model.sample(
                            data.shape, heights=heights, floorplans=floorplans, boundaries=boundaries, coarse=coarse_chunks, show_progress=True)

                        if arch.is_super_scaling:
                            # output GT for super scaling
                            mc_and_save(data, path.join(
                                img_path, f"{epoch}_{i}_gt.obj"))
                    else:
                        # "Conditional super resolution is not supported."
                        assert not arch.is_super_scaling

                        preds = diagonal_scene_gen(model,
                                                   chunk_size=chunk_size // data_downsample_factor,
                                                   d0=args.sample_dim, d2=args.sample_dim,
                                                   floorplan=floorplans.squeeze() if floorplans is not None else None,
                                                   technical_height=data.shape[-2],
                                                   height_cond=heights.item() if heights is not None else None,
                                                   batch_size=args.batch_size * args.n_chunks,
                                                   verbose=False)

                    # save
                    if floorplans is not None:
                        save_floorplan(floorplans, path.join(
                            img_path, f"{epoch}_{i}_floorplan.png"))

                    if boundaries is not None:
                        mc_and_save(boundaries, path.join(
                            img_path, f"{epoch}_{i}_boundary.obj"))

                    if coarse_chunks is not None:
                        mc_and_save(coarse_chunks, path.join(
                            img_path, f"{epoch}_{i}_coarse.obj"))

                    mc_and_save(preds, path.join(
                        img_path, f"{epoch}_{i}.obj"))

        model.train()

        # Saving Checkpoint
        save_checkpoint(checkpoint_path, arch, opt, scheduler)

        epoch += 1

    print("Done.")

    writer.close()
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model specification
    parser.add_argument("-n", "--name", type=str,
                        required=True, help='Name of the experiment.')
    parser.add_argument("-m", "--model_json", type=str, default=None,
                        help='Path model configuration file. If none is given, will check the experiment directory for existing configurations.')

    # Training
    parser.add_argument("--batch_size", type=int, default=1,
                        help='Batch sizes to use at each scale. Total number of elements is batch_size * n_chunks.')
    parser.add_argument("--n_chunks", type=int, default=4,
                        help='Number of chunks per sample. Total number of elements is batch_size * n_chunks.')
    parser.add_argument("--max_samples", type=int, default=0,
                        help='Limit the amount of training samples used.')
    parser.add_argument("--min_epoch_samples", type=int, default=0,
                        help='If less training files are available, will repeat training files multiple times per epoch.')
    parser.add_argument("--no_augmentation", action="store_true",
                        help='If set, do not use data augmentation.')
    parser.add_argument("--force_cubic", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help='If set, will overwrite the existing checkpoint if it already exists.')
    parser.add_argument("--overwrite_lr", action="store_true",
                        help='If set, will overwrite the existing scheduler if it already exists.')
    parser.add_argument("--train_patch_wise", action="store_true",
                        help="If set, will train the model patch wise. Only works for unconditional super scaling models.")

    # Training Noise Parameters
    parser.add_argument("--train_noise_distribution", type=str,
                        default="uniform", choices=["uniform", "lognormal"])
    parser.add_argument("--P_mean", type=float, default=-1.2,
                        help="Mean of the noise lognormal distribution, if used.")
    parser.add_argument("--P_std", type=float, default=1.2,
                        help="Standard deviation of the noise lognormal distribution, if used.")

    # Training CFG Parameters
    parser.add_argument("--cfg_weight", type=float, default=1.0,
                        help="Weight for the classifier-free guidance. Only used if cfg is set to True in the model.json.")
    parser.add_argument("--cfg_drop_prob", type=float, default=0.2,
                        help="Condition drop probability for the classifier-free guidance. Only used if cfg is set to True in the model.json.")

    # Loss
    parser.add_argument("--epochs", type=int, default=1000,
                        help='Epochs per scale.')
    parser.add_argument("--lr", type=float, default=1e-4,
                        help='Starting learning rate.')
    parser.add_argument("--lr_warmup", type=int,
                        default=0,  help='Learning rate decay interval steps.')
    parser.add_argument("--lr_schedule_interval", type=int,
                        default=250000,  help='Learning rate decay interval steps.')
    parser.add_argument("--acc_steps", type=int, default=1,
                        help="Number of steps in gradient accumulation.")

    # Directories
    parser.add_argument("--train_data", type=str, required=True,
                        help='Path to directory containing training SDFs.')
    parser.add_argument("--val_data", type=str,
                        help='Path to directory containing validation SDFs.')
    parser.add_argument("--split_files", action="store_true",
                        help='If set, will split the train files into train and validation sets.')
    parser.add_argument("--out", type=str,
                        default=path.join("..", "output"), help='Path to output directory.')
    parser.add_argument("--log_dir", type=str,
                        default=path.join("..", "logs"), help='Path to log directory.')

    # Condition directories
    parser.add_argument("--floorplan_dir", type=str, default=None,
                        help="Path to directory containing floorplans.")
    parser.add_argument("--boundary_dir", type=str, default=None,
                        help="Path to directory containing boundaries.")
    parser.add_argument("--semantic_dir", type=str, default=None,
                        help="Path to directory containing semantic DF grids.")
    parser.add_argument("--height_json", type=str, default=None,
                        help="Path to json file containing height information.")

    # Sampling
    parser.add_argument("--sample_freq", type=int, default=5,
                        help="Interval to generate sample meshes for validation. Can be zero for no sampling.")
    parser.add_argument("--sample_dim", type=int, default=3,
                        help="Will generate samples of dim * dim chunks.")
    parser.add_argument("--sample_amount", type=int, default=2,
                        help='Number of meshes to generate per sampling.')
    parser.add_argument("--sampling_steps", type=int, default=1000,
                        help="Number of steps to take during sampling.")

    # Training chunk choices
    parser.add_argument("-fbp", "--forced_boundary_prob", type=float, default=0.0,
                        help="Probability of forcing a boundary chunk to be selected in the dataloader.")
    parser.add_argument("-spt", "--surface_proximity_threshold", type=float, default=0.0,
                        help="Threshold for the surface proximity of a chunk to not be discarded in the dataloader.")

    args = parser.parse_args()
    args.no_val = args.val_data == None and not args.split_files

    train_model(args)
