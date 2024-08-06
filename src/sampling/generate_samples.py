import argparse
import os
import numpy as np

from ..diffusion.load_model import construct_model_from_json
from ..diffusion.utils import mc_and_save, set_random_seeds, load_checkpoint
import torch

if __name__ == '__main__':
    set_random_seeds()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_json", type=str, required=True,
                        help='Path model configuration file. ')
    parser.add_argument("--model_path", type=str, required=True,
                        help='Path to model checkpoint.')

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--rho", type=int, default=4)
    parser.add_argument("--S_churn", type=int, default=100)
    parser.add_argument("--S_min", type=float, default=0.05)
    parser.add_argument("--S_max", type=float, default=50)
    parser.add_argument("--S_noise", type=float, default=1.000)

    # parser.add_argument("--mc_thresh", type=float, default=1.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device {device}")

    arch, model, chunk_size = construct_model_from_json(
        args.model_json)
    model.sampling_timesteps = args.num_timesteps
    model.rho = args.rho
    model.S_churn = args.S_churn
    model.S_min = args.S_min
    model.S_max = args.S_max
    model.S_noise = args.S_noise

    load_checkpoint(args.model_path, device, arch)
    arch, model = arch.to(device), model.to(device)
    arch.eval(), model.eval()

    num_generated = 0
    while num_generated < args.num_samples:
        print(f"Generated {num_generated}/{args.num_samples}")
        curr_batch_size = min(
            args.batch_size, args.num_samples - num_generated)

        sampling_shape = (curr_batch_size, 1, chunk_size,
                          chunk_size, chunk_size)
        pred = model.sample(shape=sampling_shape)

        for i in range(curr_batch_size):
            df = pred[i].squeeze().cpu().numpy()

            output_path = os.path.join(
                args.output_dir, f"{num_generated + i}.npy")
            np.save(output_path, df)

        num_generated += curr_batch_size

    print("Done!")
