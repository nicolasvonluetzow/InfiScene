import argparse
import os

import numpy as np
import torch


from ..diffusion.load_model import construct_model_from_json
from ..diffusion.utils import mc_and_save, set_random_seeds, load_checkpoint, get_all_files

rho, rho_super = 4, 10
S_churn, S_churn_super = 40, 40
S_min, S_min_super = 0.25, 1
S_max, S_max_super = 50, 50
S_noise, S_noise_super = 1.010, 1.000


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_json", type=str, required=True)

    parser.add_argument('--in_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--vis_path', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--starting_index', type=int, default=0)

    parser.add_argument('--num_timesteps_super', type=int, default=100)
    args = parser.parse_args()

    set_random_seeds(args.starting_index)

    arch_super, model_super, chunk_size_super = construct_model_from_json(
        args.model_json)
    model_super.sample_timesteps = args.num_timesteps_super
    model_super.rho = rho_super
    model_super.S_churn = S_churn_super
    model_super.S_min = S_min_super
    model_super.S_max = S_max_super
    model_super.S_noise = S_noise_super

    load_checkpoint(args.model_path, device, arch_super)
    arch_super, model_super = arch_super.to(device), model_super.to(device)
    arch_super.eval(), model_super.eval()

    os.makedirs(args.out_path, exist_ok=True)
    if args.vis_path is not None:
        os.makedirs(args.vis_path, exist_ok=True)

    files = get_all_files(args.in_path)
    # keep only numpy files
    files = [f for f in files if f.endswith('.npy')]
    print(f'Found {len(files)} files')

    with torch.no_grad():
        num_sampled = args.starting_index
        while num_sampled < len(files):
            curr_batch_size = min(
                args.batch_size,  len(files) - num_sampled)
            # collect batch from files
            data = []
            for i in range(curr_batch_size):
                data.append(np.load(files[num_sampled + i]))
            data = np.stack(data, axis=0)
            data = torch.from_numpy(data).float().to(device).unsqueeze(1)

            super_preds = model_super.sample(conditions=[],
                                             coarse=data,
                                             heights=None,
                                             num_samples=len(data),
                                             show_progress=False)

            for i in range(curr_batch_size):
                if args.vis_path is not None:
                    mc_and_save(super_preds[i], os.path.join(
                        args.vis_path, f'{num_sampled}.obj'))

                pred = super_preds[i].cpu().numpy().squeeze()
                # save as numpy
                np.save(os.path.join(args.out_path,
                        f'{num_sampled}.npy'), pred)
                num_sampled += 1

            print(f'Sampled {num_sampled} shapes')

    print('Done')
