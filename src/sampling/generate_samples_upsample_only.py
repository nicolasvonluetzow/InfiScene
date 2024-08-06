import argparse
import os

import numpy as np
import torch

from ..diffusion.load_model import construct_model_from_json
from ..diffusion.utils import mc_and_save, set_random_seeds, load_checkpoint, get_all_files, TRUNC_VAL

rho, rho_super = 4, 8
S_churn, S_churn_super = 40, 40
S_min, S_min_super = 0.25, 0.25
S_max, S_max_super = 50, 50
S_noise, S_noise_super = 1.010, 1.000


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--super_scale_path", type=str, required=True)
    parser.add_argument("--super_scale_json", type=str, required=True)

    parser.add_argument('--in_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=None)

    parser.add_argument('--max_chunk_size', type=int, default=32)

    parser.add_argument('--num_timesteps_super', type=int, default=100)
    args = parser.parse_args()

    set_random_seeds()

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

    files = get_all_files(args.in_path)
    # keep only numpy files
    files = [f for f in files if f.endswith('.npy')]
    print(f'Found {len(files)} files')

    if args.num_samples is None:
        args.num_samples = len(files)

    with torch.no_grad():
        num_sampled = 0
        while num_sampled < args.num_samples:
            coarse = np.load(files[num_sampled])
            coarse = torch.from_numpy(coarse)[None, None, :].to(
                device).clamp(0, TRUNC_VAL)

            # pad height to be multiple of 2
            height = coarse.shape[3]
            if height % 2 != 0:
                coarse = torch.cat(
                    (coarse, torch.ones((1, 1, coarse.shape[2], 1, coarse.shape[4])).to(
                        device) * TRUNC_VAL),
                    dim=3)

            d0 = coarse.shape[2]
            d2 = coarse.shape[4]

            super = torch.ones(
                (1, 1, d0 * 2, coarse.shape[3] * 2, d2 * 2)).to(device) * TRUNC_VAL

            print(files[num_sampled])
            i_d0 = 0
            while i_d0 < d0:
                print(f'Sampling {i_d0} / {d0}')
                i_d2 = 0
                while i_d2 < d2:
                    current_coarse = coarse[:, :, i_d0:i_d0 +
                                            args.max_chunk_size, :, i_d2:i_d2+args.max_chunk_size]

                    if current_coarse.min() > 2:
                        i_d2 += args.max_chunk_size
                        continue

                    current_super = model_super.sample(conditions=[],
                                                       coarse=current_coarse,
                                                       heights=None,
                                                       num_samples=1,
                                                       show_progress=False)

                    super[:, :, i_d0*2:i_d0*2+args.max_chunk_size*2,
                          :, i_d2*2:i_d2*2+args.max_chunk_size*2] = current_super

                    i_d2 += args.max_chunk_size
                i_d0 += args.max_chunk_size

            new_basename = os.path.basename(files[num_sampled])[
                :-4] + '_super.npy'
            out_path = os.path.join(args.out_path, new_basename)

            np.save(out_path, super.squeeze().cpu().numpy())

            out_path = os.path.join(args.out_path, new_basename[:-4] + '.obj')
            mc_and_save(super.squeeze().cpu().numpy(), out_path)

            num_sampled += 1

            print(f'Sampled {num_sampled} / {args.num_samples}', new_basename)

    print('Done')
