import argparse
import os
import os.path as path
import numpy as np

from tqdm import tqdm

from ..diffusion.utils import get_all_files, mc_and_save, load_df_as_tensor

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--sdf_dir", type=str, required=True,
                    help='Path to directory containing SDFs.')
parser.add_argument("-o", "--out_dir", type=str,
                    default=path.join("..", "sdf_meshes"), help='Desired output directory.')
parser.add_argument("-n", "--interval", type=int, default=1,
                    help='Interval of input SDFs to use. 1 will use all.')

args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

for file_path in tqdm(get_all_files(args.sdf_dir)[::args.interval]):
    # sdf = load_df_as_tensor(file_path).detach().cpu().numpy()
    if not file_path.endswith(".npy"):
        continue
    sdf = np.load(file_path)

    name = path.basename(file_path)
    # sdf_out = path.join(args.out_dir, name.replace(".df", ".ply"))
    sdf_out = path.join(args.out_dir, name.replace(".npy", ".obj"))
    mc_and_save(sdf, sdf_out)

print("Done.")
