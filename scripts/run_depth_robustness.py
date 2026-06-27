#!/usr/bin/env python
"""M5 -- depth-robustness probe. Substantiates the non-expansiveness claim
(Thm.~1): POLARIS/POLARIS accuracy should NOT collapse as depth grows, unlike plain
GAT. We sweep depth and report test accuracy.

Usage:  python -m scripts.run_depth_robustness --dataset Cora --seeds 3 --latex
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from polaris.data import load_dataset
from polaris.models import POLARIS, GCN, GAT, GCNII, APPNP
from polaris.training import train_polaris, TrainConfig

MODELS = ["GCN", "GAT", "GCNII", "POLARIS-U", "POLARIS"]


def build(name, nf, hid, ncls, L):
    if name == "GCN":    return GCN(nf, hid, ncls, L), 0.0
    if name == "GAT":    return GAT(nf, hid, ncls, max(L, 2), heads=4), 0.0
    if name == "GCNII":  return GCNII(nf, hid, ncls, layers=L), 0.0
    if name == "POLARIS-U":    return POLARIS(nf, hid, ncls, L, agg="sum"), 0.0
    if name == "POLARIS":  return POLARIS(nf, hid, ncls, L, agg="sum", signed=True), 0.0
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Cora")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)

    data0, nf, ncls = load_dataset(args.dataset, split=0)
    print(f"Depth robustness | {args.dataset}, hidden {args.hidden}, "
          f"{args.splits} Geom-GCN splits\n", flush=True)
    print(f"{'depth':>6}" + "".join(f"{m:>9}" for m in MODELS), flush=True)

    table = {}
    for L in args.depths:
        row = {}
        for name in MODELS:
            accs = []
            for sp in range(args.splits):
                data, nf, ncls = load_dataset(args.dataset, split=sp)
                torch.manual_seed(sp); np.random.seed(sp)
                m, beta = build(name, nf, args.hidden, ncls, L)
                r = train_polaris(m, data, ncls, TrainConfig(epochs=args.epochs, beta=beta))
                accs.append(r["acc"])
            row[name] = (float(np.mean(accs)), float(np.std(accs)))
        table[L] = row
        print(f"{L:>6}" + "".join(f"{row[m][0]*100:>9.1f}" for m in MODELS), flush=True)

    if args.latex:
        print("\n% --- LaTeX depth table body ---", flush=True)
        for L in args.depths:
            row = table[L]
            cells = " & ".join(f"{row[m][0]*100:.1f}" for m in MODELS)
            print(f"{L} & {cells} \\\\", flush=True)


if __name__ == "__main__":
    main()
