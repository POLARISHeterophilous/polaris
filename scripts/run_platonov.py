#!/usr/bin/env python
"""Large heterophilous benchmarks (Platonov et al. 2023) -- resolves the
small-graph variance limitation of the WebKB experiments, and reports the full
calibration suite (ECE, NLL, Brier) alongside accuracy.

These graphs have 10k-24k nodes (vs <260 for WebKB), so per-split variance is
far smaller and significance tests have real power. We compare POLARIS against the
best heterophily specialist (FAGCN) and the strong deep baseline GCNII, over the
10 standard splits, with a paired Wilcoxon signed-rank test.

Usage:  python -m scripts.run_platonov --epochs 120 --splits 10
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from scipy.stats import wilcoxon

from polaris.data import load_dataset, edge_homophily
from polaris.models import POLARIS, GCN, GCNII, FAGCN, LINKX
from polaris.training import train_polaris, TrainConfig

DATASETS = ["Roman-empire", "Amazon-ratings", "Minesweeper"]
MODELS = ["MLP", "GCNII", "FAGCN", "LINKX", "POLARIS-U", "POLARIS"]
# GGCN/GloGNN omitted: dense O(n^2) adjacency OOMs on these 10-25k-node graphs.
METRICS = ["acc", "auc", "ece", "nll", "brier"]


def build(name, nf, hid, ncls, L):
    if name == "MLP":    return GCN(nf, hid, ncls, 1), 0.0
    if name == "GCNII":  return GCNII(nf, hid, ncls, layers=L), 0.0
    if name == "FAGCN":  return FAGCN(nf, hid, ncls, layers=L), 0.0
    if name == "LINKX":  return LINKX(nf, hid, ncls, dropout=0.5), 0.0
    if name == "POLARIS-U":    return POLARIS(nf, hid, ncls, L, agg="sum"), 0.0
    if name == "POLARIS":  return POLARIS(nf, hid, ncls, L, agg="sum", signed=True), 0.0
    raise ValueError(name)


def run_one(name, dname, nf, ncls, hid, L, epochs, splits):
    """Return dict metric -> array over splits."""
    rows = {m: [] for m in METRICS}
    for sp in range(splits):
        data, nf2, ncls2 = load_dataset(dname, split=sp)
        torch.manual_seed(sp); np.random.seed(sp)
        m, beta = build(name, nf2, hid, ncls2, L)
        r = train_polaris(m, data, ncls2, TrainConfig(epochs=epochs, beta=beta))
        for k in METRICS:
            rows[k].append(r[k])
    return {k: np.array(v) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--splits", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(4)

    for dname in DATASETS:
        data, nf, ncls = load_dataset(dname, split=0)
        h = edge_homophily(data)
        print(f"\n=== {dname} | {data.num_nodes} nodes, {ncls} classes, "
              f"h={h:.3f} | {args.splits} splits ===", flush=True)
        print(f"{'model':<8}" + "".join(f"{m:>9}" for m in METRICS), flush=True)
        results = {}
        for name in MODELS:
            res = run_one(name, dname, nf, ncls, args.hidden, args.depth,
                          args.epochs, args.splits)
            results[name] = res
            cells = "".join(f"{res[k].mean():>9.3f}" for k in METRICS)
            print(f"{name:<8}{cells}", flush=True)
        # significance: POLARIS vs FAGCN on accuracy, paired over splits
        a, b = results["POLARIS"]["acc"], results["FAGCN"]["acc"]
        try:
            p = wilcoxon(a, b).pvalue
        except ValueError:
            p = float("nan")
        print(f"  -> POLARIS vs FAGCN acc: mean diff {100*(a.mean()-b.mean()):+.1f}, "
              f"Wilcoxon p={p:.4f}", flush=True)


if __name__ == "__main__":
    main()
