#!/usr/bin/env python
"""Q8: is there a measurable link between the entropy target (POLARIS-E) and
calibration (ECE/NLL)?

We sweep the normalised target entropy rho in {0.2,...,0.9} with the
regulariser on (beta=1), pinning rho via fixed_rho. For each setting we train
POLARIS-E and record: the realised aggregation entropy, test ECE, NLL, and acc.
We then report Pearson correlations across the rho grid:
  corr(rho, ECE) and corr(rho, NLL).
A clear monotone link substantiates "entropy control -> calibration"; a flat
relation would mean entropy and calibration are decoupled. We report whatever
we observe. Averaged over a few splits for stability.

Writes results/entropy_calibration.txt.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from polaris.data import load_dataset
from polaris.models import POLARIS
from polaris.training import train_polaris, TrainConfig


def run_cell(dname, rho, splits, epochs, H, L):
    ent, ece, nll, acc = [], [], [], []
    for sp in range(splits):
        data, nf, ncls = load_dataset(dname, split=sp)
        torch.manual_seed(sp); np.random.seed(sp)
        # POLARIS-E: unsigned base + entropy regulariser at a PINNED target rho
        m = POLARIS(nf, H, ncls, L, agg="sum", signed=False, fixed_rho=rho)
        r = train_polaris(m, data, ncls, TrainConfig(epochs=epochs, beta=1.0))
        ece.append(r["ece"]); nll.append(r["nll"]); acc.append(r["acc"])
        ent.append(r.get("agg_entropy", float("nan")))
    return (np.mean(ent), np.mean(ece), np.mean(nll), np.mean(acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Wisconsin")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default="results/entropy_calibration.txt")
    args = ap.parse_args()
    torch.set_num_threads(4)

    rhos = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows = []
    with open(args.out, "w") as f:
        f.write(f"Q8: entropy target (POLARIS-E) vs calibration | {args.dataset}, "
                f"{args.splits} splits, beta=1, target rho pinned.\n\n")
        f.write(f"{'rho':>6}{'realisedH':>11}{'ECE':>8}{'NLL':>8}{'acc':>8}\n")
        f.flush()
        for rho in rhos:
            e, ec, nl, ac = run_cell(args.dataset, rho, args.splits,
                                     args.epochs, args.hidden, args.depth)
            rows.append((rho, e, ec, nl, ac))
            f.write(f"{rho:>6.2f}{e:>11.3f}{ec:>8.3f}{nl:>8.3f}{ac*100:>8.1f}\n")
            f.flush()
            print(f"rho={rho}: H={e:.3f} ECE={ec:.3f} NLL={nl:.3f} acc={ac*100:.1f}", flush=True)
        R = np.array(rows)
        def pear(a, b):
            if np.std(a) < 1e-9 or np.std(b) < 1e-9: return float("nan")
            return float(np.corrcoef(a, b)[0, 1])
        f.write(f"\nPearson r(rho, ECE) = {pear(R[:,0], R[:,2]):+.3f}\n")
        f.write(f"Pearson r(rho, NLL) = {pear(R[:,0], R[:,3]):+.3f}\n")
        f.write(f"Pearson r(realisedH, ECE) = {pear(R[:,1], R[:,2]):+.3f}\n")
    print("ENTROPY_CALIB_DONE", flush=True)


if __name__ == "__main__":
    main()
