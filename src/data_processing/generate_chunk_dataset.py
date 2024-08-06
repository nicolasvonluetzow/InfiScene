import os
import random

import torch
from tqdm import tqdm

from ..diffusion.utils import (get_all_files, load_df_as_tensor,
                               save_chunk_with_conds_as_np)

path = '/mnt/hdd/3D_Front/SDF_out_5cm/'
out = '/mnt/hdd/3D_Front/SDF_out_5cm_chunks_256_scenes/'
files = get_all_files(path)[:256]

chunk_size = 64
VOXELIZATION_PADDING = 5
TRUNC_VAL = 3.0
SURFACE_PROXIMITY_THRESHOLD = 0.0
forced_boundary_prob = 0.1
N_CHUNK_FACTOR = 4.0


def generate_chunks(data):
    # number of chunks is dependent on dim0 and dim2 of the data
    n_chunks = (data.shape[0] // chunk_size) * \
        (data.shape[2] // chunk_size) * N_CHUNK_FACTOR

    pad_voxels = 2 * (chunk_size - VOXELIZATION_PADDING)
    data = torch.nn.functional.pad(
        data, (pad_voxels, 0, 0, 0, pad_voxels, 0), mode='constant', value=TRUNC_VAL)

    chunks = []
    conds0, conds2 = [], []
    while len(chunks) < n_chunks:
        dim0_max = data.shape[0] - chunk_size
        dim0_min = chunk_size
        dim2_max = data.shape[2] - chunk_size
        dim2_min = chunk_size

        if random.random() < forced_boundary_prob:
            # force the selection of a boundary chunk

            boundary_min = chunk_size
            boundary_max = pad_voxels

            d0_boundary = random.random() < 0.5
            pos = (random.randint(boundary_min, boundary_max), random.randint(dim2_min, dim2_max)) if d0_boundary else (
                random.randint(dim0_min, dim0_max), random.randint(boundary_min, boundary_max))
        else:
            pos = (random.randint(dim0_min, dim0_max),
                   random.randint(dim2_min, dim2_max))

        chunk = data[pos[0]:pos[0] + chunk_size, :, pos[1]                     :pos[1] + chunk_size].clone().detach().unsqueeze(0)

        surface_proximity = torch.sum(
            chunk != TRUNC_VAL) / (chunk.shape[0] * chunk.shape[1] * chunk.shape[2])
        if surface_proximity < SURFACE_PROXIMITY_THRESHOLD:
            continue

        chunks.append(chunk)

        cond0 = data[pos[0] - chunk_size:pos[0],
                     :, pos[1]:pos[1] + chunk_size].clone().detach().unsqueeze(0)
        conds0.append(cond0)

        cond2 = data[pos[0]:pos[0] + chunk_size,
                     :, pos[1] - chunk_size:pos[1]].clone().detach().unsqueeze(0)
        conds2.append(cond2)

    return {"chunks": torch.stack(chunks),
            "conds0": torch.stack(conds0),
            "conds2": torch.stack(conds2)}


if __name__ == '__main__':
    os.makedirs(out, exist_ok=True)

    for file in tqdm(files):
        scene = load_df_as_tensor(file)
        data = generate_chunks(scene)

        for i, (chunk, cond0, cond2) in enumerate(zip(data['chunks'], data['conds0'], data['conds2'])):
            # new file name is old file name + chunk number
            path = out + file.split('/')[-1].split('.')[0] + \
                '_' + str(i) + '.chunk'
            save_chunk_with_conds_as_np(path, chunk, cond0, cond2)
