"""M6 + M1 -- component ablation (label-safe bias) and paired significance.

M6: does the training-only discriminative bias actually help? We compare POLARIS
with and without it (use_disc_bias on/off), per-split over the 10 Geom-GCN
splits.

M1: is POLARIS's edge over the best specialist (FAGCN) statistically real, given
the high variance on tiny WebKB graphs? We run both on the SAME 10 splits and
report a paired Wilcoxon signed-rank test.

Usage:  python -m scripts.run_ablation_significance --epochs 120 --splits 10
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from scipy.stats import wilcoxon

from polaris.data import load_dataset
from polaris.models import POLARIS, FAGCN
from polaris.training import train_polaris, TrainConfig

HETERO = ["Texas", "Wisconsin", "Cornell", "Actor"]


def acc_per_split(make, dname, splits, epochs, beta=0.0):
    accs = []
    for sp in range(splits):
        data, nf, ncls = load_dataset(dname, split=sp)
        torch.manual_seed(sp); np.random.seed(sp)
        m = make(nf, ncls)
        r = train_polaris(m, data, ncls, TrainConfig(epochs=epochs, beta=beta))
        accs.append(r["acc"])
    return np.array(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--splits", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(4)
    H, L = args.hidden, args.depth

    print("=== M6: label-safe discriminative-bias ablation (POLARIS) ===", flush=True)
    print(f"{'dataset':<10}{'with bias':>14}{'no bias':>14}{'delta':>9}", flush=True)
    for dn in HETERO:
        with_b = acc_per_split(
            lambda nf, nc: POLARIS(nf, H, nc, L, agg="sum", signed=True, use_disc_bias=True),
            dn, args.splits, args.epochs)
        no_b = acc_per_split(
            lambda nf, nc: POLARIS(nf, H, nc, L, agg="sum", signed=True, use_disc_bias=False),
            dn, args.splits, args.epochs)
        d = with_b.mean() - no_b.mean()
        print(f"{dn:<10}{with_b.mean()*100:>9.1f}±{with_b.std()*100:.1f}"
              f"{no_b.mean()*100:>9.1f}±{no_b.std()*100:.1f}{d*100:>+9.1f}", flush=True)

    print("\n=== M1: POLARIS vs FAGCN, paired Wilcoxon over shared splits ===", flush=True)
    print(f"{'dataset':<10}{'POLARIS':>10}{'FAGCN':>10}{'mean d':>9}{'p-value':>10}", flush=True)
    for dn in HETERO:
        dcms = acc_per_split(
            lambda nf, nc: POLARIS(nf, H, nc, L, agg="sum", signed=True), dn,
            args.splits, args.epochs)
        fag = acc_per_split(
            lambda nf, nc: FAGCN(nf, H, nc, layers=L), dn, args.splits, args.epochs)
        diff = dcms - fag
        try:
            p = wilcoxon(dcms, fag).pvalue
        except ValueError:
            p = float("nan")
        print(f"{dn:<10}{dcms.mean()*100:>10.1f}{fag.mean()*100:>10.1f}"
              f"{diff.mean()*100:>+9.1f}{p:>10.4f}", flush=True)


if __name__ == "__main__":
    main()
