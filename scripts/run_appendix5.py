#!/usr/bin/env python
"""Q1: does explicit entropy control / signedness add value over sparse
attention normalisation? We compare, under the shared protocol:
  POLARIS-U (softmax)      : unsigned, temperature-controlled entropy
  POLARIS-U (sparsemax)    : unsigned, sparse attention (some weights exactly 0)
  POLARIS   (signed)       : the full operator
Sparsemax sharpens attention (lower entropy) but, like softmax, stays
non-negative; if signedness is the heterophily driver, sparsemax should track
POLARIS-U, not signed POLARIS. Reports accuracy, ECE, and realised normalised entropy.
Appends to results/appendix/appendix5_results.txt.
"""
from __future__ import annotations
import os, numpy as np, torch, torch.nn.functional as F
from polaris.data import load_dataset
from polaris.models import POLARIS
from polaris.training import train_polaris, TrainConfig

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "results", "appendix", "appendix5_results.txt")
EPOCHS, HID, DEPTH, SPLITS = 120, 64, 4, 10


def log(m=""):
    print(m, flush=True)
    with open(OUT, "a") as f:
        f.write(m + "\n")


@torch.no_grad()
def realised_H(model, data):
    model.eval(); x = data.x; vals = []
    for i, L in enumerate(model.layers):
        x2, aux = L(x, data.edge_index, return_aux=True)
        vals.append((aux["entropy"] / aux["log_deg"].clamp(min=1e-6)).mean().item())
        x = F.elu(x2) if i < len(model.layers) - 1 else x2
    return float(np.mean(vals))


def main():
    torch.set_num_threads(4)
    log("\n##### Q1: sparse attention vs entropy control / signedness #####")
    variants = {
        "POLARIS-U (softmax)":   dict(signed=False),
        "POLARIS-U (sparsemax)": dict(signed=False, attn_norm="sparsemax"),
        "POLARIS (signed)":      dict(signed=True),
    }
    log(f"{'dataset':<10}{'variant':<20}{'acc':>8}{'ECE':>8}{'realised H':>12}")
    for d in ["Texas", "Wisconsin", "Cornell", "Cora"]:
        n_sp = 1 if d == "Cora" else SPLITS
        for name, kw in variants.items():
            a, e, h = [], [], []
            for sp in range(n_sp):
                data, nf, ncls = load_dataset(d, split=sp)
                torch.manual_seed(sp); np.random.seed(sp)
                m = POLARIS(nf, HID, ncls, DEPTH, agg="sum", **kw)
                r = train_polaris(m, data, ncls, TrainConfig(epochs=EPOCHS))
                a.append(r["acc"]); e.append(r["ece"]); h.append(realised_H(m, data))
            log(f"{d:<10}{name:<20}{np.mean(a)*100:>7.1f} {np.mean(e):>7.3f}"
                f"{np.mean(h):>12.3f}")

    log("\n##### Q2: multi-head POLARIS (signed); acc%, realised entropy #####")
    log(f"{'dataset':<10}" + "".join(f"{'h='+str(k):>16}" for k in (1, 2, 4)))
    for d in ["Texas", "Wisconsin", "Cornell"]:
        cells = []
        for k in (1, 2, 4):
            a, h = [], []
            for sp in range(SPLITS):
                data, nf, ncls = load_dataset(d, split=sp)
                torch.manual_seed(sp); np.random.seed(sp)
                m = POLARIS(nf, HID, ncls, DEPTH, agg="sum", signed=True, n_heads=k)
                r = train_polaris(m, data, ncls, TrainConfig(epochs=EPOCHS))
                a.append(r["acc"]); h.append(realised_H(m, data))
            cells.append(f"{np.mean(a)*100:5.1f}|H={np.mean(h):.2f}")
        log(f"{d:<10}" + "".join(f"{c:>16}" for c in cells))
    log("\n##### DONE #####")


if __name__ == "__main__":
    main()
