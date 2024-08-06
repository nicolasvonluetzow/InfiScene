import argparse
import os
import numpy as np
import struct

from tqdm import tqdm

from ..diffusion.utils import get_all_files, TRUNC_VAL, mc_and_save


def load_df_as_numpy(path):
    with open(path, 'rb') as f:
        dimX = int.from_bytes(f.read(4), byteorder='little')
        dimY = int.from_bytes(f.read(4), byteorder='little')
        dimZ = int.from_bytes(f.read(4), byteorder='little')

        voxelSize = struct.unpack('f', f.read(4))[0]

        voxelToWorld = []
        for i in range(4):
            for j in range(4):
                voxelToWorld.append(struct.unpack('f', f.read(4))[0])

        voxelToWorld = np.array(voxelToWorld).reshape(4, 4)

        data = np.fromfile(f, dtype=np.float32).reshape((dimZ, dimY, dimX))
        data = data.swapaxes(0, 2)

    return data, voxelToWorld


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_folder', type=str, required=True)
    parser.add_argument('--train_data', type=str, required=True)

    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--chunk_size', type=int, default=64)

    parser.add_argument('--max_samples', type=int, default=None)

    parser.add_argument('--debug_output_folder', type=str, default=None)

    args = parser.parse_args()

    df_files = get_all_files(args.df_folder)
    print(f"Found {len(df_files)} files")

    if args.max_samples is not None:
        df_files = df_files[:args.max_samples]
        print(f"Using {len(df_files)} files")

    if args.debug_output_folder is not None:
        os.makedirs(args.debug_output_folder, exist_ok=True)

    train_files = get_all_files(args.train_data)

    min_dists_finished = []

    n_finished = 0
    # for df_file in df_files:
    while n_finished < len(df_files):
        curr_batch_size = min(args.batch_size, len(df_files) - n_finished)
        # load batch
        batch = []
        for i in range(curr_batch_size):
            batch.append(np.load(df_files[n_finished + i]))
        batch = np.array(batch)

        min_dists = [1e9] * curr_batch_size
        min_dists = np.array(min_dists)
        # min_dist_chunks = [None] * curr_batch_size
        min_dist_chunks = np.zeros(
            (curr_batch_size, args.chunk_size, args.chunk_size, args.chunk_size), dtype=np.float32)
        for train_file in tqdm(train_files, total=len(train_files), desc=f'Batch {n_finished} - {n_finished + curr_batch_size}'):
            # train_df = np.load(train_file)
            train_df, _ = load_df_as_numpy(train_file)
            # truncate to [0, TRUNC_VAL]
            train_df = np.clip(train_df, 0, TRUNC_VAL)

            # force cubic height
            if train_df.shape[1] > args.chunk_size:
                train_df = train_df[:, :args.chunk_size, :]
            elif train_df.shape[1] < args.chunk_size:
                # pad on top with TRUNC_VAL
                train_df = np.pad(train_df, ((0, 0), (args.chunk_size -
                                                      train_df.shape[1], 0), (0, 0)), constant_values=TRUNC_VAL)

            # check L1 distance for chunks in regular intervals
            max_d0_offset = train_df.shape[0] - args.chunk_size
            max_d2_offset = train_df.shape[2] - args.chunk_size

            for d0_offset in range(0, max_d0_offset, args.step_size):
                for d2_offset in range(0, max_d2_offset, args.step_size):
                    curr_chunk = train_df[d0_offset:d0_offset+args.chunk_size,
                                          :, d2_offset:d2_offset+args.chunk_size]
                    curr_chunk_batch = curr_chunk[None, :].repeat(
                        curr_batch_size, axis=0)

                    # L1 distances
                    l1_dists = np.abs(
                        batch - curr_chunk_batch).mean(axis=(1, 2, 3))
                    min_dists = np.minimum(min_dists, l1_dists)

                    if args.debug_output_folder is not None:
                        # min_dist_chunks = np.where(
                        #     l1_dists == min_dists, curr_chunk_batch, min_dist_chunks)
                        for i in range(curr_batch_size):
                            if l1_dists[i] == min_dists[i]:
                                min_dist_chunks[i] = curr_chunk

        print(f"Min dists: {min_dists}")
        min_dists_finished = np.append(min_dists_finished, min_dists)

        if args.debug_output_folder is not None:
            # iterate and save over batch
            for j, (considered_df, considered_nn) in enumerate(zip(batch, min_dist_chunks)):
                df_vis_path = os.path.join(
                    args.debug_output_folder, os.path.basename(df_files[n_finished + j][:-4]) + '.obj')
                mc_and_save(considered_df, df_vis_path)

                out_path = os.path.join(
                    args.debug_output_folder, os.path.basename(df_files[n_finished + j][:-4]) + '_nn.obj')
                mc_and_save(considered_nn, out_path)

        n_finished += curr_batch_size

    print(f"Mean min dist: {np.mean(min_dists_finished)}")
    print(f"Median min dist: {np.median(min_dists_finished)}")
    print(f"Min min dist: {np.min(min_dists_finished)}")
    print(f"Max min dist: {np.max(min_dists_finished)}")
    print(f"Variance: {np.var(min_dists_finished)}")
