#!/usr/bin/env python
"""Sign x entropy-regulariser, with a sweep over the regulariser strength beta.

The main ablation fixes beta in {0 (POLARIS), 1 (POLARIS-E)}. Here we sweep beta on
the *signed* model to characterise the second axis: does the entropy regulariser
ever buy accuracy, and what does it do to calibration (ECE) and to the realised
aggregation entropy H? beta=0 is plain POLARIS (reference).

Same protocol/seeds as the main tables (hidden 64, depth 4, 120 epochs,
10 Geom-GCN splits; Cora single split).
"""
from __future__ import annotations
import os, numpy as np, torch, torch.nn.functional as F
from polaris.data import load_dataset
from polaris.models import POLARIS
from polaris.training import train_polaris, TrainConfig

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "results", "appendix", "signed_reg_results.txt")
EPOCHS, HID, DEPTH = 120, 64, 4
BETAS = [0.0, 0.5, 1.0, 2.0, 5.0]


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


if __name__ == "__main__":
    torch.set_num_threads(8)
    log("\n##### beta sweep on SIGNED POLARIS (acc%, ECE, realised H) #####")
    log("##### 10 Geom-GCN splits (Cora 1); beta=0 is plain POLARIS      #####")
    log(f"{'dataset':<11}{'beta':>6}{'acc':>8}{'ECE':>8}{'H':>8}")
    for d in ["Texas", "Wisconsin", "Cornell", "Actor", "Cora"]:
        n_sp = 1 if d == "Cora" else 10
        for beta in BETAS:
            a, e, h = [], [], []
            for sp in range(n_sp):
                data, nf, ncls = load_dataset(d, split=sp)
                torch.manual_seed(sp); np.random.seed(sp)
                m = POLARIS(nf, HID, ncls, DEPTH, agg="sum", signed=True)
                r = train_polaris(m, data, ncls, TrainConfig(epochs=EPOCHS, beta=beta))
                a.append(r["acc"]); e.append(r["ece"]); h.append(realised_H(m, data))
            log(f"{d:<11}{beta:>6.1f}{np.mean(a)*100:>8.1f}"
                f"{np.mean(e):>8.3f}{np.mean(h):>8.3f}")
    log("\n##### DONE #####")
