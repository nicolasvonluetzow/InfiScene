import argparse
import os

import numpy as np
import torch

from ..diffusion.load_model import construct_model_from_json
from ..diffusion.utils import mc_and_save, set_random_seeds, load_checkpoint

rho, rho_super = 4, 4
S_churn, S_churn_super = 40, 0
S_min, S_min_super = 0.25, 0.25
S_max, S_max_super = 50, 50
S_noise, S_noise_super = 1.010, 1.000


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_json', type=str, required=True)

    parser.add_argument("--super_scale_path", type=str, required=True)
    parser.add_argument("--super_scale_json", type=str, required=True)

    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=1)

    parser.add_argument('--num_timesteps', type=int, default=100)
    parser.add_argument('--num_timesteps_super', type=int, default=100)
    args = parser.parse_args()

    arch, model, chunk_size = construct_model_from_json(
        args.model_json)
    model.sampling_timesteps = args.num_timesteps
    model.rho = rho
    model.S_churn = S_churn
    model.S_min = S_min
    model.S_max = S_max
    model.S_noise = S_noise

    load_checkpoint(args.model_path, device, arch)
    arch, model = arch.to(device), model.to(device)
    arch.eval(), model.eval()

    arch_super, model_super, chunk_size_super = construct_model_from_json(
        args.super_scale_json)
    model_super.sample_timesteps = args.num_timesteps_super
    model_super.rho = rho_super
    model_super.S_churn = S_churn_super
    model_super.S_min = S_min_super
    model_super.S_max = S_max_super
    model_super.S_noise = S_noise_super

    load_checkpoint(args.super_scale_path, device, arch_super)
    arch_super, model_super = arch_super.to(device), model_super.to(device)
    arch_super.eval(), model_super.eval()

    os.makedirs(args.out_path, exist_ok=True)

    with torch.no_grad():
        set_random_seeds(9001)
        coarse, noisy_coarse = model.sample(conditions=[], heights=None, num_samples=1, show_progress=True, return_noise_levels=20)

        out_path = os.path.join(args.out_path, f"coarse.obj")
        mc_and_save(coarse.squeeze().cpu().numpy() /
                    2, out_path, threshold=0.5)

        for i in range(len(noisy_coarse)):
            if i % 20 != 0 and i < args.num_timesteps - 30:
                continue
            out_path = os.path.join(args.out_path, f"noisy_coarse_{i}.obj")
            mc_and_save(noisy_coarse[i].squeeze().cpu(
            ).numpy() / 2, out_path, threshold=0.5)

        fine, noisy_fine = model_super.sample(conditions=[], heights=None, coarse=coarse, num_samples=1, show_progress=True, return_noise_levels=20)

        out_path = os.path.join(args.out_path, f"fine.obj")
        mc_and_save(fine.squeeze().cpu().numpy(), out_path)

        for i in range(len(noisy_fine)):
            if i % 2 != 0 and i < args.num_timesteps_super - 10:
                continue
            out_path = os.path.join(args.out_path, f"noisy_fine_{i}.obj")
            mc_and_save(noisy_fine[i].squeeze().cpu().numpy(), out_path)

    print('Done')
