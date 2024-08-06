import argparse

from tqdm import tqdm
import numpy as np

from ..diffusion.utils import TRUNC_VAL, get_all_files, load_df_as_tensor

MAX_HEIGHT = 74

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--max_files', type=int, default=None)

args = parser.parse_args()
input_path = args.input_path

# os.makedirs(output_path, exist_ok=True)

files = get_all_files(input_path)[:args.max_files]
i, c = 0, 0
heights = []
for file in tqdm(files):
    valid = True
    try:
        file_tens = load_df_as_tensor(file)
    except:
        valid = False
    i += 1

    if valid:
        valid = file_tens.max() <= TRUNC_VAL and file_tens.min() >= 0.0

    if not valid:
        c += 1
        # os.remove(file)

    file_height = file_tens.shape[1]
    heights.append(file_height)

    # if file_height > MAX_HEIGHT:
    #     c += 1
    # else:
    #     shutil.copy2(file, output_path)

print(i, c)
# hist, bins = np.histogram(
#     heights, bins=[0, 60, 70, 75, 80, 100, 1000])

# print(list(zip(bins[1:], hist)))
print(heights)
