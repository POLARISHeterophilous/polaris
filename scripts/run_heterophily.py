#!/usr/bin/env python
"""Heterophily benchmark -- the empirical spine of the paper.

Hypothesis: on heterophilic graphs, fixed homophily-assuming propagation
(APPNP, GCNII) degrades because it averages dissimilar neighbors, whereas POLARIS's
temperature-controlled, selective attention can down-weight bad neighbors while
remaining depth-stable (unlike GAT, which collapses).

Reports mean +/- std over the 10 standard Geom-GCN splits, and (with --latex)
emits a booktabs table body. A homophilic reference (Cora) is included to show
the regime contrast.

Usage:  python -m scripts.run_heterophily --depth 4 --latex
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from polaris.data import load_dataset, edge_homophily
from polaris.models import POLARIS, GCN, GAT, GCNII, APPNP, GPRGNN, FAGCN
from polaris.training import train_polaris, TrainConfig

MODELS = ["MLP", "GAT", "APPNP", "GCNII", "GPRGNN", "FAGCN", "POLARIS-U", "POLARIS-E", "POLARIS"]
HETERO = ["Texas", "Wisconsin", "Cornell", "Actor"]


def build(name, nf, hid, ncls, L):
    if name == "MLP":    return GCN(nf, hid, ncls, 1), 0.0     # 1-hop ~ feature ref
    if name == "GAT":    return GAT(nf, hid, ncls, max(L, 2), heads=4), 0.0
    if name == "APPNP":  return APPNP(nf, hid, ncls, layers=L), 0.0
    if name == "GCNII":  return GCNII(nf, hid, ncls, layers=L), 0.0
    if name == "GPRGNN": return GPRGNN(nf, hid, ncls, layers=10), 0.0
    if name == "FAGCN":  return FAGCN(nf, hid, ncls, layers=L), 0.0
    if name == "POLARIS-U":    return POLARIS(nf, hid, ncls, L, agg="sum"), 0.0
    if name == "POLARIS-E": return POLARIS(nf, hid, ncls, L, agg="sum"), 1.0
    if name == "POLARIS":  return POLARIS(nf, hid, ncls, L, agg="sum", signed=True), 0.0
    raise ValueError(name)


def n_splits(dname, requested):
    # Planetoid (Cora) has a single public split; WebKB/Actor have 10.
    return 1 if dname == "Cora" else requested


def evaluate(dname, args):
    """Return {model: (mean, std)} of test accuracy over splits."""
    out = {}
    ns = n_splits(dname, args.splits)
    for mname in MODELS:
        accs = []
        for sp in range(ns):
            data, nf, ncls = load_dataset(dname, split=sp)
            torch.manual_seed(sp); np.random.seed(sp)
            m, beta = build(mname, nf, args.hidden, ncls, args.depth)
            r = train_polaris(m, data, ncls, TrainConfig(epochs=args.epochs, beta=beta))
            accs.append(r["acc"])
        out[mname] = (float(np.mean(accs)), float(np.std(accs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--with-cora", action="store_true",
                    help="include homophilic Cora as a regime contrast")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)

    datasets = (["Cora"] if args.with_cora else []) + HETERO
    print(f"Heterophily benchmark | depth {args.depth}, hidden {args.hidden}, "
          f"{args.splits} splits (Cora: 1 public split)\n", flush=True)

    homo = {d: edge_homophily(load_dataset(d, split=0)[0]) for d in datasets}
    for d in datasets:
        print(f"  {d:<10} edge-homophily = {homo[d]:.3f}", flush=True)
    print(flush=True)

    print(f"{'dataset':<10}{'h':>6}" + "".join(f"{m:>16}" for m in MODELS), flush=True)
    results = {}
    for dname in datasets:
        res = evaluate(dname, args)
        results[dname] = res
        cells = "".join(f"{res[m][0]:>9.3f}+-{res[m][1]:.2f}" for m in MODELS)
        print(f"{dname:<10}{homo[dname]:>6.2f}{cells}", flush=True)

    if args.latex:
        print("\n% --- LaTeX table body (booktabs) ---", flush=True)
        for dname in datasets:
            res = results[dname]
            best = max(res[m][0] for m in MODELS)
            row = []
            for m in MODELS:
                mu, sd = res[m]
                s = f"{mu*100:.1f}\\footnotesize$\\pm${sd*100:.1f}"
                row.append(f"\\textbf{{{s}}}" if abs(mu - best) < 1e-9 else s)
            print(f"{dname} & {homo[dname]:.2f} & " + " & ".join(row) + r" \\", flush=True)


if __name__ == "__main__":
    main()
