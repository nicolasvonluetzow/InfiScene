import argparse
import os

import numpy as np
import torch

from ..diffusion.load_model import construct_model_from_json
from ..diffusion.utils import mc_and_save, set_random_seeds, load_checkpoint

rho, rho_super = 4, 4
S_churn, S_churn_super = 40, 40
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
    parser.add_argument('--vis_path', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=1)

    parser.add_argument('--batch_size_init', type=int, default=16)
    parser.add_argument('--batch_size_super', type=int, default=4)
    parser.add_argument('--starting_index', type=int, default=0)

    parser.add_argument('--num_timesteps', type=int, default=100)
    parser.add_argument('--num_timesteps_super', type=int, default=100)
    args = parser.parse_args()

    set_random_seeds(args.starting_index)

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
    if args.vis_path is not None:
        os.makedirs(args.vis_path, exist_ok=True)

    with torch.no_grad():
        num_sampled = args.starting_index
        while num_sampled < args.num_samples:
            curr_batch_size = min(args.batch_size_init,
                                  args.num_samples - num_sampled)
            sampling_shape = (curr_batch_size, 1, chunk_size,
                              chunk_size, chunk_size)
            preds = model.sample(shape=sampling_shape,
                                 show_progress=True)

            print(f'Sampled {curr_batch_size} coarse shapes')
            num_sampled_super = 0
            while num_sampled_super < curr_batch_size:
                curr_super_batch_size = min(
                    args.batch_size_super, curr_batch_size - num_sampled_super)

                super_preds = model_super.sample(conditions=[],
                                                 coarse=preds[num_sampled_super:num_sampled_super +
                                                              curr_super_batch_size],
                                                 heights=None,
                                                 num_samples=curr_super_batch_size,
                                                 show_progress=True)

                for i in range(curr_super_batch_size):
                    if args.vis_path is not None:
                        mc_and_save(super_preds[i], os.path.join(
                            args.vis_path, f'{num_sampled}.obj'))

                    pred = super_preds[i].cpu().numpy().squeeze()
                    # save as numpy
                    np.save(os.path.join(args.out_path,
                            f'{num_sampled}.npy'), pred)
                    num_sampled += 1

                num_sampled_super += curr_super_batch_size
                print(
                    f'Sampled {num_sampled_super} super shapes from {curr_batch_size} shapes')

    print('Done')
