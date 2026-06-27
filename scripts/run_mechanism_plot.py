#!/usr/bin/env python
"""E3 -- the mechanism experiment: aggregation entropy vs edge homophily.

The paper's central causal claim is: heterophily rewards LOWER aggregation
entropy (more selective attention), so POLARIS's learnable temperature should settle
at sharper attention (lower measured H_i) on heterophilous graphs than on
homophilous ones. This script measures it directly.

For each dataset we train POLARIS (free temperature) and record:
  - edge homophily h
  - measured mean aggregation entropy H_i (eval mode, val-selected checkpoint)
  - the same normalised by the per-graph ceiling log|N~(i)|  (so graphs of
    different degree are comparable)
  - learned temperature tau (layer-averaged)

Outputs a table and, if matplotlib is available, a scatter plot to
paper/figures/entropy_vs_homophily.pdf.

Usage:  python -m scripts.run_mechanism_plot --splits 5
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from polaris.data import load_dataset, entropy_ceiling, edge_homophily
from polaris.models import POLARIS
from polaris.training import train_polaris, TrainConfig

DATASETS = ["Texas", "Cornell", "Wisconsin", "Actor", "Citeseer", "Pubmed", "Cora"]


def n_splits(dname, requested):
    return 1 if dname in ("Cora", "Citeseer", "Pubmed") else requested


def mean_tau(model):
    return float(np.mean([l.tau.item() for l in model.layers]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.0, help="0=POLARIS free temp")
    args = ap.parse_args()
    torch.set_num_threads(4)

    print(f"Mechanism probe | depth {args.depth}, hidden {args.hidden}, "
          f"beta {args.beta}\n", flush=True)
    print(f"{'dataset':<10}{'h':>7}{'ceil':>7}{'H_i':>9}{'H_i/ceil':>10}"
          f"{'tau':>7}{'acc':>8}", flush=True)

    rows = []
    for dname in DATASETS:
        data0, nf, ncls = load_dataset(dname, split=0)
        h = edge_homophily(data0)
        ceil = entropy_ceiling(data0)
        Hs, taus, accs = [], [], []
        for sp in range(n_splits(dname, args.splits)):
            data, nf, ncls = load_dataset(dname, split=sp)
            torch.manual_seed(sp); np.random.seed(sp)
            model = POLARIS(nf, args.hidden, ncls, args.depth, agg="sum")
            r = train_polaris(model, data, ncls,
                          TrainConfig(epochs=args.epochs, beta=args.beta))
            Hs.append(r["agg_entropy"]); accs.append(r["acc"])
            taus.append(mean_tau(model))
        H, tau, acc = np.mean(Hs), np.mean(taus), np.mean(accs)
        rows.append((dname, h, ceil, H, H / ceil, tau, acc))
        print(f"{dname:<10}{h:>7.3f}{ceil:>7.3f}{H:>9.3f}{H/ceil:>10.3f}"
              f"{tau:>7.3f}{acc:>8.3f}", flush=True)

    # correlation: does normalised entropy rise with homophily?
    hs = np.array([r[1] for r in rows])
    norm_H = np.array([r[4] for r in rows])
    if len(hs) > 2:
        c = np.corrcoef(hs, norm_H)[0, 1]
        print(f"\nPearson corr( homophily h , normalised H_i/ceil ) = {c:+.3f}",
              flush=True)
        print("(positive => more homophilous graphs use higher-entropy / less "
              "selective attention, as the theory predicts)", flush=True)

    # optional plot
    try:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("../paper/figures", exist_ok=True)
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        for (dn, h, ceil, H, nH, tau, acc) in rows:
            ax.scatter(h, nH, s=40)
            ax.annotate(dn, (h, nH), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("edge homophily $h$")
        ax.set_ylabel(r"normalised aggregation entropy $H_i/\log|\tilde N(i)|$")
        ax.set_title("Selective on heterophily, diffuse on homophily")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = "../paper/figures/entropy_vs_homophily.pdf"
        fig.savefig(out)
        print(f"\nsaved plot -> {out}", flush=True)
    except Exception as e:
        print(f"\n[plot skipped: {e}]", flush=True)


if __name__ == "__main__":
    main()
