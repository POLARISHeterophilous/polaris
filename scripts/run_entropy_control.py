#!/usr/bin/env python
"""Entropy-control study: show the regulariser directly moves AGGREGATION entropy,
and trace the accuracy / calibration vs entropy trade-off.

Usage:
    python -m scripts.run_entropy_control --dataset Cora --seeds 3
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from polaris.data import load_dataset, entropy_ceiling
from polaris.models import POLARIS
from polaris.training import train_polaris, TrainConfig


def ms(rows, k):
    v = [r[k] for r in rows]
    return float(np.mean(v)), float(np.std(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Cora")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    torch.set_num_threads(4)
    data, nf, ncls = load_dataset(args.dataset)
    ceil = entropy_ceiling(data)
    print(f"{args.dataset}: entropy ceiling mean log|N~(i)| = {ceil:.3f} nats\n")

    print("=== (A) Does the regulariser control AGGREGATION entropy? (learnable rho) ===")
    print(f"{'beta':>6}{'aggEntropy':>14}{'acc':>10}{'ece':>10}")
    for beta in [0.0, 0.1, 1.0, 5.0]:
        rows = []
        for s in range(args.seeds):
            torch.manual_seed(s)
            m = POLARIS(nf, args.hidden, ncls, args.layers, agg="sum")
            rows.append(train_polaris(m, data, ncls, TrainConfig(epochs=args.epochs, beta=beta)))
        ae, ac, ec = ms(rows, "agg_entropy"), ms(rows, "acc"), ms(rows, "ece")
        print(f"{beta:>6.1f}{ae[0]:>9.3f}±{ae[1]:.3f}{ac[0]:>10.3f}{ec[0]:>10.3f}")

    print("\n=== (B) Accuracy / calibration vs target entropy (beta=2, fixed rho) ===")
    print(f"{'rho':>6}{'targetH':>10}{'measH':>14}{'acc':>10}{'ece':>10}")
    for rho in [0.2, 0.4, 0.6, 0.8]:
        rows = []
        for s in range(args.seeds):
            torch.manual_seed(s)
            m = POLARIS(nf, args.hidden, ncls, args.layers, agg="sum", fixed_rho=rho)
            rows.append(train_polaris(m, data, ncls, TrainConfig(epochs=args.epochs, beta=2.0)))
        ae, ac, ec = ms(rows, "agg_entropy"), ms(rows, "acc"), ms(rows, "ece")
        print(f"{rho:>6.1f}{rho*ceil:>10.3f}{ae[0]:>9.3f}±{ae[1]:.3f}"
              f"{ac[0]:>10.3f}{ec[0]:>10.3f}")


if __name__ == "__main__":
    main()
