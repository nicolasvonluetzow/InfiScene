import argparse
import os
from tqdm import tqdm
import numpy as np

from ..diffusion.utils import get_all_files
from ..diffusion.scene_dataset import SceneDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=100)

    parser.add_argument('--chunk_size', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    files = get_all_files(args.input_folder)[:args.num_samples]
    dataset = SceneDataset(files, False, 1,
                           args.chunk_size, num_conditions=0, force_cubic=True)

    for i in tqdm(range(args.num_samples)):
        data = dataset[i]["chunks"].squeeze().cpu().numpy()
        assert data.ndim == 3

        output_file = os.path.join(args.output_folder, f"{i}.npy")
        np.save(output_file, data)

    print("Done!")
