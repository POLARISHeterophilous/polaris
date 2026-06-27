#!/usr/bin/env python
"""Ablate the self-mixing coefficient lambda and the gate (reviewer Q4).

Isolates where POLARIS's stability/accuracy comes from:
  full        : learned lambda + gate           (the model)
  fix-lam-0.5 : lambda fixed at 0.5 (the Prop.2 floor), gate on
  fix-lam-0.0 : lambda fixed at 0 (no self-anchor), gate on  -> pure neighbour mix
  no-gate     : learned lambda, gate removed (u_i used directly)
  no-lam-no-gate : lambda=0.5 fixed, no gate

Signed (POLARIS) operator throughout. 10 Geom-GCN splits, same protocol as the
main table. Writes results/ablation_lambda_gate.txt.

Usage:  PYTHONPATH=. python scripts/run_lambda_gate_ablation.py --epochs 120
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from polaris.data import load_dataset
from polaris.models import POLARIS
from polaris.training import train_polaris, TrainConfig

HETERO = ["Texas", "Wisconsin", "Cornell", "Actor"]
VARIANTS = {
    "full":           dict(signed=True),
    "fix-lam-0.5":    dict(signed=True, fixed_lam=0.5),
    "fix-lam-0.0":    dict(signed=True, fixed_lam=0.0),
    "no-gate":        dict(signed=True, use_gate=False),
    "no-lam-no-gate": dict(signed=True, fixed_lam=0.5, use_gate=False),
}


def acc_over_splits(kw, dname, splits, epochs, H, L):
    accs = []
    for sp in range(splits):
        data, nf, ncls = load_dataset(dname, split=sp)
        torch.manual_seed(sp); np.random.seed(sp)
        m = POLARIS(nf, H, ncls, L, agg="sum", **kw)
        r = train_polaris(m, data, ncls, TrainConfig(epochs=epochs, beta=0.0))
        accs.append(r["acc"])
    return 100.0 * np.mean(accs), 100.0 * np.std(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default="results/ablation_lambda_gate.txt")
    args = ap.parse_args()
    torch.set_num_threads(4)

    names = list(VARIANTS)
    with open(args.out, "w") as f:
        f.write(f"lambda / gate ablation (signed POLARIS) | depth {args.depth}, "
                f"hidden {args.hidden}, {args.splits} Geom-GCN splits\n\n")
        f.write(f"{'variant':>16}" + "".join(f"{d:>11}" for d in HETERO) + "\n")
        f.flush()
        for v in names:
            row = f"{v:>16}"
            for dn in HETERO:
                mean, std = acc_over_splits(VARIANTS[v], dn, args.splits,
                                            args.epochs, args.hidden, args.depth)
                row += f"{mean:>7.1f}±{std:>3.1f}"
            f.write(row + "\n"); f.flush()
            print(row, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
