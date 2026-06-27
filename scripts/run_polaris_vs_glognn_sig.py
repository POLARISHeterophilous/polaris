#!/usr/bin/env python
"""Paired significance test: POLARIS vs the strongest baseline (GloGNN), on the
SAME 10 Geom-GCN splits, in the SAME harness.

Both models go through our load_dataset + train_polaris (shared protocol). We
collect per-split test accuracy for each and run a paired Wilcoxon signed-rank
test on the per-split differences. This backs (or qualifies) the "POLARIS leads"
claim: a positive mean gap that is or is not significant at the WebKB sample
size. GloGNN is the official MLP_NORM class at library defaults.

Writes results/significance_glognn.txt.
Usage:  PYTHONPATH=. python scripts/run_polaris_vs_glognn_sig.py --epochs 120
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from scipy.stats import wilcoxon

from polaris.data import load_dataset
from polaris.models import POLARIS
from polaris.models.external_adapters import GloGNNAdapter
from polaris.training import train_polaris, TrainConfig

HETERO = ["Texas", "Wisconsin", "Cornell"]   # WebKB; Actor excluded (GloGNN dense OOM)


def per_split(make, dname, splits, epochs, H, L):
    accs = []
    for sp in range(splits):
        data, nf, ncls = load_dataset(dname, split=sp)
        torch.manual_seed(sp); np.random.seed(sp)
        m = make(nf, H, ncls, L, data.x.size(0))
        r = train_polaris(m, data, ncls, TrainConfig(epochs=epochs))
        accs.append(r["acc"] * 100.0)
    return np.array(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default="results/significance_glognn.txt")
    args = ap.parse_args()
    torch.set_num_threads(4)

    mk_polaris = lambda nf, H, ncls, L, n: POLARIS(nf, H, ncls, L, agg="sum", signed=True)
    mk_glo = lambda nf, H, ncls, L, n: GloGNNAdapter(nf, H, ncls, n_nodes=n, layers=L, dropout=0.5)

    with open(args.out, "w") as f:
        f.write("Paired Wilcoxon: POLARIS vs GloGNN (official), same 10 Geom-GCN "
                "splits, same harness, shared protocol.\n"
                "Positive mean diff = POLARIS higher. Per-split acc (%).\n\n")
        f.write(f"{'dataset':>10}{'POLARIS':>9}{'GloGNN':>9}{'meanΔ':>8}{'p-value':>10}\n")
        f.flush()
        for dn in HETERO:
            a_polaris = per_split(mk_polaris, dn, args.splits, args.epochs, args.hidden, args.depth)
            a_glo = per_split(mk_glo, dn, args.splits, args.epochs, args.hidden, args.depth)
            diff = a_polaris - a_glo
            try:
                p = wilcoxon(a_polaris, a_glo).pvalue
            except ValueError:
                p = float("nan")
            f.write(f"{dn:>10}{a_polaris.mean():>9.1f}{a_glo.mean():>9.1f}"
                    f"{diff.mean():>+8.1f}{p:>10.4f}\n")
            f.flush()
            print(f"{dn}: POLARIS {a_polaris.mean():.1f} vs GloGNN {a_glo.mean():.1f} "
                  f"(Δ{diff.mean():+.1f}, p={p:.4f})", flush=True)
            # also record raw per-split for transparency
            f.write(f"           POLARIS splits: {np.round(a_polaris,1).tolist()}\n")
            f.write(f"           GloGNN splits: {np.round(a_glo,1).tolist()}\n\n")
            f.flush()
    print("SIG_DONE", flush=True)


if __name__ == "__main__":
    main()
