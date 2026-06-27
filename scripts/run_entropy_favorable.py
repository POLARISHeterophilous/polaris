#!/usr/bin/env python
"""Where the entropy regulariser IS favorable: calibration on homophilous graphs.

The entropy regulariser does not buy accuracy (see the signed beta-sweep). Its
theoretical home is the UNSIGNED model on homophilous graphs, where controlling
the aggregation entropy should improve calibration (ECE) without costing
accuracy. We sweep beta on POLARIS-U over the citation graphs (Cora, Citeseer,
Pubmed; public split). beta=0 is plain POLARIS-U.

Same protocol as the main tables (hidden 64, depth 4, 120 epochs).
"""
from __future__ import annotations
import os, numpy as np, torch, torch.nn.functional as F
from polaris.data import load_dataset
from polaris.models import POLARIS
from polaris.training import train_polaris, TrainConfig

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "results", "appendix", "entropy_favorable_results.txt")
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
    log("\n##### beta sweep on UNSIGNED POLARIS-U, homophilous graphs #####")
    log("##### acc%, ECE, realised H; public split; beta=0 is POLARIS-U #####")
    log(f"{'dataset':<11}{'beta':>6}{'acc':>8}{'ECE':>8}{'H':>8}")
    for d in ["Cora", "Citeseer", "Pubmed"]:
        for beta in BETAS:
            a, e, h = [], [], []
            for seed in range(3):          # 3 seeds on the fixed public split
                data, nf, ncls = load_dataset(d)
                torch.manual_seed(seed); np.random.seed(seed)
                m = POLARIS(nf, HID, ncls, DEPTH, agg="sum", signed=False)
                r = train_polaris(m, data, ncls, TrainConfig(epochs=EPOCHS, beta=beta))
                a.append(r["acc"]); e.append(r["ece"]); h.append(realised_H(m, data))
            log(f"{d:<11}{beta:>6.1f}{np.mean(a)*100:>8.1f}"
                f"{np.mean(e):>8.3f}{np.mean(h):>8.3f}")
    log("\n##### DONE #####")
