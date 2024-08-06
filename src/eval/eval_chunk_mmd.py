import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from ..diffusion.utils import get_all_files, TRUNC_VAL


def mmd_linear(X, Y):
    """Compute the linear maximum mean discrepancy (MMD) between two samples.

    Args:
        X ([n1, dim]): Samples from distribution 1
        Y ([n2, dim]): Samples from distribution 2

    Returns:
        [float]: MMD value
    """

    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def pairwise_l1_distances(sample, ref):
    """Return distance matrices, given two pytorch tensors"""
    N_sample = sample.shape[0]
    N_ref = ref.shape[0]

    all_l1 = torch.zeros(N_sample, N_ref)
    for i in tqdm(range(N_sample), total=N_sample, desc="Pairwise L1 Distances Outer Loop"):
        for j in range(N_ref):
            all_l1[i, j] = torch.abs(sample[i] - ref[j]).mean()
    return all_l1


def nn_classification_accuracy(Mxx, Mxy, Myy, k=1, sqrt=False):
    """Credit: https://github.com/stevenygd/PointFlow/blob/master/metrics/evaluation_metrics.py
    """

    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat(
        (Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY *
                torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) *
                    torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_folder', type=str, required=True)
    parser.add_argument('--gt_folder', type=str, required=True)

    parser.add_argument('--chunk_size', type=int, default=64)

    parser.add_argument('--max_samples', type=int, default=None)

    args = parser.parse_args()

    df_files = get_all_files(args.df_folder)
    # keep only numpy files
    df_files = [f for f in df_files if f.endswith(".npy")]
    print(f"Found {len(df_files)} files")

    if args.max_samples is not None:
        df_files = df_files[:args.max_samples]
        print(f"Using {len(df_files)} files")

    gt_files = get_all_files(args.gt_folder)
    print(f"Found {len(gt_files)} GT files")

    if args.max_samples is not None:
        gt_files = gt_files[:args.max_samples]
        print(f"Using {len(gt_files)} GT files")

    xs = []
    for df_file in df_files:
        df = np.load(df_file).clip(0, TRUNC_VAL)
        xs.append(df)

    ys = []
    for gt_file in gt_files:
        gt = np.load(gt_file).clip(0, TRUNC_VAL)
        ys.append(gt)

    xs = np.concatenate(xs).reshape(-1, args.chunk_size ** 3) / TRUNC_VAL
    ys = np.concatenate(ys).reshape(-1, args.chunk_size ** 3) / TRUNC_VAL

    mmd = mmd_linear(xs, ys)
    print(f"MMD: {mmd}")

    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).float()

    Mxx = pairwise_l1_distances(xs, xs)
    Mxy = pairwise_l1_distances(xs, ys)
    Myy = pairwise_l1_distances(ys, ys)

    nn_acc = nn_classification_accuracy(Mxx, Mxy, Myy, k=1)
    print(f"NN Acc: {nn_acc['acc']}")
