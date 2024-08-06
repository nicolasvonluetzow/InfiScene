import argparse
import os
import numpy as np
from skimage.measure import marching_cubes
import trimesh
from tqdm import tqdm
import torch

from ..diffusion.utils import get_all_files, TRUNC_VAL

# from .eval_chunk_mmd import nn_classification_accuracy


def lgan_mmd_cov(all_dist):
    """From https://github.com/stevenygd/PointFlow/blob/master/metrics/evaluation_metrics.py"""
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }


def distChamfer(a, b):
    """From https://github.com/ThibaultGROUEIX/AtlasNet / https://github.com/stevenygd/PointFlow/blob/master/metrics/evaluation_metrics.py"""

    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def pairwise_chamfer_dist(X, Y):
    """Compute the chamfer distance between two given torch point clouds"""
    N_sample = X.shape[0]
    N_ref = Y.shape[0]

    all_chamfer = torch.zeros(N_sample, N_ref)
    for i in tqdm(range(N_sample), total=N_sample, desc="Pairwise Chamfer Distances Outer Loop"):
        for j in range(N_ref):
            all_chamfer[i, j] = distChamfer(
                X[i].unsqueeze(0), Y[j].unsqueeze(0))[0].mean()

    return all_chamfer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_folder', type=str, required=True)
    parser.add_argument('--gt_folder', type=str, required=True)

    parser.add_argument('--chunk_size', type=int, default=64)

    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--mc_level', type=float, default=1.0)

    args = parser.parse_args()

    df_files = get_all_files(args.df_folder)
    df_files = [f for f in df_files if f.endswith(".npy")]
    print(f"Found {len(df_files)} files")

    df_files = df_files[:args.max_samples]
    print(f"Using {len(df_files)} files")

    gt_files = get_all_files(args.gt_folder)
    print(f"Found {len(gt_files)} GT files")

    gt_files = gt_files[:args.max_samples]
    print(f"Using {len(gt_files)} GT files")

    xs = []
    for df_file in tqdm(df_files, total=len(df_files), desc="Loading DF"):
        df = np.load(df_file).clip(0, TRUNC_VAL)

        try:
            verts, faces, normals, _ = marching_cubes(df, level=args.mc_level)
            mesh = trimesh.Trimesh(
                vertices=verts, faces=faces, vertex_normals=normals)

            points, _ = trimesh.sample.sample_surface(mesh, args.num_points)

            xs.append(torch.from_numpy(points))
        except:
            print(f"Failed to load {df_file}")

    ys = []
    for gt_file in tqdm(gt_files, total=len(gt_files), desc="Loading GT"):
        gt = np.load(gt_file).clip(0, TRUNC_VAL)

        try:
            verts, faces, normals, _ = marching_cubes(gt, level=args.mc_level)
            mesh = trimesh.Trimesh(
                vertices=verts, faces=faces, vertex_normals=normals)

            points, _ = trimesh.sample.sample_surface(mesh, args.num_points)

            ys.append(torch.from_numpy(points))
        except:
            print(f"Failed to load {gt_file}")

    X = torch.stack(xs)
    Y = torch.stack(ys)

    Mxy = pairwise_chamfer_dist(X, Y)

    mmd_cov = lgan_mmd_cov(Mxy)
    print(f"Minimum Matching Distance: {mmd_cov['lgan_mmd']}")
    print(f"Coverage: {mmd_cov['lgan_cov']}")

    # Could run this, but doesn't really add anything, as its just comparing additional post-processing after DF generation
    # Mxx = pairwise_chamfer_dist(X, X)
    # Myy = pairwise_chamfer_dist(Y, Y)

    # acc = nn_classification_accuracy(Mxx, Mxy, Myy, k=1)
    # print(f"Chamfer 1NN Accuracy: {acc['acc']}")
