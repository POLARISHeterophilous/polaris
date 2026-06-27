#!/usr/bin/env python
"""Standalone POLARIS-E (entropy-regularised) results for the supplement.

POLARIS-E = unsigned POLARIS base with the aggregation-entropy regulariser on
(beta=1.0, target rho learned), under the SAME protocol/seeds/splits as every
other table: hidden 64, depth 4, 120 epochs, Adam lr 0.01 / wd 5e-4, label
smoothing 0.1, dropout 0.5, 10 Geom-GCN splits (Cora: 1), seed = split index.

For each dataset we report POLARIS-E accuracy, ECE, NLL, and the realised
normalised aggregation entropy, with the unsigned POLARIS-U (no regulariser) as the
reference so the comparison isolates the entropy regulariser. Calibration, not
accuracy, is POLARIS-E's value, so both metrics are shown.

Nothing fabricated; every number is a real run. Appends to
results/appendix/polaris_e_results.txt.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F

from polaris.data import load_dataset
from polaris.models import POLARIS
from polaris.training import train_polaris, TrainConfig

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "results", "appendix", "polaris_e_results.txt")
EPOCHS, HID, DEPTH, SPLITS = 120, 64, 4, 10
DATASETS = ["Texas", "Wisconsin", "Cornell", "Actor", "Cora"]


def log(msg=""):
    print(msg, flush=True)
    with open(OUT, "a") as f:
        f.write(msg + "\n")


def n_splits(d):
    return 1 if d == "Cora" else SPLITS


@torch.no_grad()
def realised_norm_entropy(model, data):
    model.eval()
    x = data.x
    vals = []
    for i, layer in enumerate(model.layers):
        x2, aux = layer(x, data.edge_index, return_aux=True)
        vals.append((aux["entropy"] / aux["log_deg"].clamp(min=1e-6)).mean().item())
        x = F.elu(x2) if i < len(model.layers) - 1 else x2
    return float(np.mean(vals))


def evaluate(name, build, beta):
    accs, eces, nlls, ents = [], [], [], []
    for d in DATASETS:
        a, e, nl, en = [], [], [], []
        for sp in range(n_splits(d)):
            data, nf, ncls = load_dataset(d, split=sp)
            torch.manual_seed(sp); np.random.seed(sp)
            m = build(nf, ncls)
            r = train_polaris(m, data, ncls, TrainConfig(epochs=EPOCHS, beta=beta))
            a.append(r["acc"]); e.append(r["ece"]); nl.append(r["nll"])
            en.append(realised_norm_entropy(m, data))
        accs.append((np.mean(a) * 100, np.std(a) * 100))
        eces.append(np.mean(e)); nlls.append(np.mean(nl)); ents.append(np.mean(en))
    return accs, eces, nlls, ents


def main():
    torch.set_num_threads(4)
    log("\n##### POLARIS-E standalone results (shared protocol, seed=split) #####")
    log(f"hidden={HID} depth={DEPTH} epochs={EPOCHS} splits={SPLITS}\n")
    rows = {
        "POLARIS-U (no reg.)": (lambda nf, nc: POLARIS(nf, HID, nc, DEPTH, agg="sum"), 0.0),
        "POLARIS-E (entropy reg.)": (lambda nf, nc: POLARIS(nf, HID, nc, DEPTH, agg="sum"), 1.0),
    }
    res = {}
    for name, (build, beta) in rows.items():
        res[name] = evaluate(name, build, beta)

    log(f"{'metric':<22}" + "".join(f"{d:>12}" for d in DATASETS))
    for name in rows:
        accs, eces, nlls, ents = res[name]
        log(f"{name}  accuracy")
        log(f"{'':<22}" + "".join(f"{a:6.1f}+-{s:3.1f}" for a, s in accs))
        log(f"{name}  ECE")
        log(f"{'':<22}" + "".join(f"{e:>12.3f}" for e in eces))
        log(f"{name}  NLL")
        log(f"{'':<22}" + "".join(f"{nl:>12.3f}" for nl in nlls))
        log(f"{name}  realised norm. entropy")
        log(f"{'':<22}" + "".join(f"{en:>12.3f}" for en in ents))
        log("")
    log("##### DONE #####")


if __name__ == "__main__":
    main()
