#!/usr/bin/env python
"""Fair, single-harness comparison of the strong recent baselines against POLARIS.

EVERY model -- GGCN (official), GloGNN (official), LINKX (in-house reimpl),
POLARIS-U, POLARIS -- is trained and evaluated through the SAME pipeline: our
load_dataset (identical Geom-GCN splits), our train_polaris loop (val-selected
checkpoint, label smoothing 0.1, cross-entropy on logits), shared protocol
(hidden 64, depth 4, 120 epochs, lr 0.01, wd 5e-4, dropout 0.5). The only
thing that differs between rows is the operator. This removes the
harness/normalisation/val-selection confounds of running each repo separately.

GGCN/GloGNN import the AUTHORS' OFFICIAL model classes (see
polaris/models/external_adapters.py); their structural switches are at library
defaults (no per-dataset tuning). LINKX is our faithful reimplementation.

Writes results/fair_harness.txt.
Usage:  PYTHONPATH=. python scripts/run_fair_baselines.py --epochs 120
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from polaris.data import load_dataset, edge_homophily
from polaris.models import POLARIS, LINKX, SADEGCN, SIMGA
from polaris.models.external_adapters import GGCNAdapter, GloGNNAdapter
from polaris.training import train_polaris, TrainConfig

HETERO = ["Texas", "Wisconsin", "Cornell", "Actor"]
MODELS = ["LINKX", "GGCN", "GloGNN", "SADE-GCN", "SIMGA", "POLARIS-U", "POLARIS"]


def build(name, nf, H, ncls, L, n_nodes):
    if name == "LINKX":  return LINKX(nf, H, ncls, dropout=0.5)
    if name == "GGCN":   return GGCNAdapter(nf, H, ncls, layers=L, dropout=0.5)
    if name == "GloGNN": return GloGNNAdapter(nf, H, ncls, n_nodes=n_nodes, layers=L, dropout=0.5)
    if name == "SADE-GCN": return SADEGCN(nf, H, ncls, layers=2, dropout=0.5)
    if name == "SIMGA":  return SIMGA(nf, H, ncls, layers=2, dropout=0.5)
    if name == "POLARIS-U": return POLARIS(nf, H, ncls, L, agg="sum", signed=False)
    if name == "POLARIS":   return POLARIS(nf, H, ncls, L, agg="sum", signed=True)
    raise ValueError(name)


def acc_over_splits(name, dname, splits, epochs, H, L):
    accs = []
    for sp in range(splits):
        data, nf, ncls = load_dataset(dname, split=sp)
        torch.manual_seed(sp); np.random.seed(sp)
        m = build(name, nf, H, ncls, L, data.x.size(0))
        r = train_polaris(m, data, ncls, TrainConfig(epochs=epochs))
        accs.append(r["acc"])
    return 100.0 * np.mean(accs), 100.0 * np.std(accs)


def main():
    global HETERO
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default="results/fair_harness.txt")
    ap.add_argument("--datasets", nargs="+", default=HETERO,
                    help="subset of datasets; GGCN/GloGNN use dense O(n^2) ops "
                         "and OOM on large graphs (e.g. Actor, 7600 nodes).")
    args = ap.parse_args()
    torch.set_num_threads(4)
    HETERO = args.datasets

    with open(args.out, "w") as f:
        f.write("Fair single-harness comparison (our load_dataset + train_polaris, "
                "shared protocol: hidden 64, depth 4, 120 ep, lr 0.01, wd 5e-4,\n"
                "dropout 0.5, val-selected). GGCN/GloGNN = authors' official model "
                "classes at library defaults; LINKX = in-house. Test acc (%).\n\n")
        f.write(f"{'dataset':>10}{'h':>6}" + "".join(f"{m:>14}" for m in MODELS) + "\n")
        f.flush()
        for dn in HETERO:
            d0, _, _ = load_dataset(dn, split=0)
            h = edge_homophily(d0)
            row = f"{dn:>10}{h:>6.2f}"
            for m in MODELS:
                mean, std = acc_over_splits(m, dn, args.splits, args.epochs,
                                            args.hidden, args.depth)
                row += f"{mean:>8.1f}±{std:>3.1f}"
                print(f"{dn} {m}: {mean:.1f}±{std:.1f}", flush=True)
            f.write(row + "\n"); f.flush()
    print("FAIR_DONE", flush=True)


if __name__ == "__main__":
    main()
