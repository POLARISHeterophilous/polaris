#!/usr/bin/env python
"""Add the LINKX baseline (Lim et al., NeurIPS 2021) under the paper's exact
protocol, so it can be slotted into the main heterophily table.

Same 10 Geom-GCN splits, hidden 64, Adam, val-selected checkpoint as every
other model. Reports LINKX next to POLARIS-U / POLARIS for direct comparison.
Writes results/linkx.txt.

Usage:  PYTHONPATH=. python scripts/run_linkx_baseline.py --epochs 120
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from polaris.data import load_dataset, edge_homophily
from polaris.models import LINKX, POLARIS
from polaris.training import train_polaris, TrainConfig

HETERO = ["Texas", "Wisconsin", "Cornell", "Actor"]
MODELS = ["LINKX", "POLARIS-U", "POLARIS"]


def build(name, nf, H, ncls, L):
    if name == "LINKX":  return LINKX(nf, H, ncls, dropout=0.5), 0.0
    if name == "POLARIS-U": return POLARIS(nf, H, ncls, L, agg="sum", signed=False), 0.0
    if name == "POLARIS":   return POLARIS(nf, H, ncls, L, agg="sum", signed=True), 0.0
    raise ValueError(name)


def acc_over_splits(name, dname, splits, epochs, H, L):
    accs = []
    for sp in range(splits):
        data, nf, ncls = load_dataset(dname, split=sp)
        torch.manual_seed(sp); np.random.seed(sp)
        m, beta = build(name, nf, H, ncls, L)
        r = train_polaris(m, data, ncls, TrainConfig(epochs=epochs, beta=beta))
        accs.append(r["acc"])
    return 100.0 * np.mean(accs), 100.0 * np.std(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default="results/linkx.txt")
    args = ap.parse_args()
    torch.set_num_threads(4)

    with open(args.out, "w") as f:
        f.write(f"LINKX (NeurIPS 2021) vs POLARIS | depth {args.depth}, hidden "
                f"{args.hidden}, {args.splits} Geom-GCN splits\n\n")
        f.write(f"{'dataset':>10}{'h':>6}" + "".join(f"{m:>14}" for m in MODELS) + "\n")
        f.flush()
        for dn in HETERO:
            data, _, _ = load_dataset(dn, split=0)
            h = edge_homophily(data)
            row = f"{dn:>10}{h:>6.2f}"
            for m in MODELS:
                mean, std = acc_over_splits(m, dn, args.splits, args.epochs,
                                            args.hidden, args.depth)
                row += f"{mean:>8.1f}±{std:>3.1f}"
            f.write(row + "\n"); f.flush()
            print(row, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
