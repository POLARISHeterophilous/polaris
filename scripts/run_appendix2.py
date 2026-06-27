#!/usr/bin/env python
"""Second batch of appendix experiments (Q1, Q2, Q3, Q5), all under the SAME
protocol, seeds, and splits as the main tables: hidden 64, depth 4, 120 epochs,
Adam lr 0.01 / wd 5e-4, label smoothing 0.1, dropout 0.5, 10 Geom-GCN splits,
seed = split index (Cora: single public split). Nothing tuned beyond this.

Groups (selectable with --only):
  q1  signed-vs-unsigned selection: per-split val/test for both variants; how
      often validation picks each, and test acc of val-selected vs always-signed
      / always-unsigned / oracle.
  q2  multi-class diagnostic: per-class same-class attention mass p_c vs the
      uniform-attention chance level h_c (Prop. 3, per class) at inference.
  q3  entropy controllability: POLARIS-E with distinct fixed entropy targets rho;
      realised normalised aggregation entropy, accuracy, ECE per target.
  q5  signedness switch granularity: signed POLARIS + a learned signed<->unsigned
      gate at edge / node / layer scope.

Every number is measured from a real run; nothing is fabricated. Results append
to results/appendix/appendix2_results.txt as produced.
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, softmax as scatter_softmax, scatter

from polaris.data import load_dataset, edge_homophily
from polaris.models import POLARIS
from polaris.training import train_polaris, TrainConfig

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "results", "appendix", "appendix2_results.txt")
EPOCHS, HID, DEPTH, SPLITS = 120, 64, 4, 10


def log(msg=""):
    print(msg, flush=True)
    with open(OUT, "a") as f:
        f.write(msg + "\n")


def n_splits(d):
    return 1 if d == "Cora" else SPLITS


# --------------------------------------------------------------------------- #
def run_q1():
    log("\n" + "=" * 70)
    log("Q1  signed vs unsigned: validation-based selection and its stability")
    log("=" * 70)
    log(f"{'dataset':<11}{'pick=sign':>10}{'always-S':>10}{'always-U':>10}"
        f"{'val-sel':>9}{'oracle':>9}")
    for d in ["Texas", "Wisconsin", "Cornell", "Actor", "Cora"]:
        n_sign = 0
        a_s, a_u, a_sel, a_or = [], [], [], []
        for sp in range(n_splits(d)):
            data, nf, ncls = load_dataset(d, split=sp)
            torch.manual_seed(sp); np.random.seed(sp)
            rs = train_polaris(POLARIS(nf, HID, ncls, DEPTH, agg="sum", signed=True),
                            data, ncls, TrainConfig(epochs=EPOCHS))
            torch.manual_seed(sp); np.random.seed(sp)
            ru = train_polaris(POLARIS(nf, HID, ncls, DEPTH, agg="sum", signed=False),
                            data, ncls, TrainConfig(epochs=EPOCHS))
            a_s.append(rs["acc"]); a_u.append(ru["acc"])
            pick_sign = rs["val_acc"] >= ru["val_acc"]
            n_sign += int(pick_sign)
            a_sel.append(rs["acc"] if pick_sign else ru["acc"])
            a_or.append(max(rs["acc"], ru["acc"]))
        ns = n_splits(d)
        log(f"{d:<11}{n_sign:>4}/{ns:<5}{np.mean(a_s)*100:>10.1f}"
            f"{np.mean(a_u)*100:>10.1f}{np.mean(a_sel)*100:>9.1f}"
            f"{np.mean(a_or)*100:>9.1f}")


# --------------------------------------------------------------------------- #
@torch.no_grad()
def _alpha_at(layer, x, edge_index, n):
    h = layer.W(x)
    ei, _ = add_self_loops(edge_index, num_nodes=n)
    src, dst = ei[0], ei[1]
    z = (h * layer.att_dst).sum(-1)[dst] + (h * layer.att_src).sum(-1)[src]
    z = F.leaky_relu(z, layer.negative_slope)
    return scatter_softmax(z / layer.tau, dst, num_nodes=n), src, dst


@torch.no_grad()
def _decompose(model, data):
    """Per layer: same-class attention mass including the self-loop (p_incl),
    the self-loop's own share (self_mass), and the same-class mass over REAL
    neighbours only (p_excl). Returns arrays of length n_layers."""
    model.eval(); x = data.x; n = x.size(0); y = data.y
    pin, sm, pex = [], [], []
    for i, L in enumerate(model.layers):
        a, src, dst = _alpha_at(L, x, data.edge_index, n)
        same = (y[src] == y[dst]).float(); ns = (src != dst).float()
        pin.append((scatter(a * same, dst, 0, dim_size=n, reduce="sum") /
                    scatter(a, dst, 0, dim_size=n, reduce="sum").clamp(min=1e-9)).mean().item())
        sm.append(scatter(a * (src == dst).float(), dst, 0, dim_size=n,
                          reduce="sum").mean().item())
        aa = a * ns
        pex.append((scatter(aa * same * ns, dst, 0, dim_size=n, reduce="sum") /
                    scatter(aa, dst, 0, dim_size=n, reduce="sum").clamp(min=1e-9)).mean().item())
        x = L(x, data.edge_index)
        if i < len(model.layers) - 1:
            x = F.elu(x)
    return np.array(pin), np.array(sm), np.array(pex)


@torch.no_grad()
def _classwise_excl(model, data, ncls):
    """Per-class (layer 0) same-class attention mass over REAL neighbours vs the
    structural same-class neighbour fraction h_c."""
    n = data.x.size(0); y = data.y
    a, src, dst = _alpha_at(model.layers[0], data.x, data.edge_index, n)
    same = (y[src] == y[dst]).float(); ns = (src != dst).float(); aa = a * ns
    p_i = (scatter(aa * same * ns, dst, 0, dim_size=n, reduce="sum") /
           scatter(aa, dst, 0, dim_size=n, reduce="sum").clamp(min=1e-9))
    h_i = (scatter(same * ns, dst, 0, dim_size=n, reduce="sum") /
           scatter(ns, dst, 0, dim_size=n, reduce="sum").clamp(min=1e-9))
    hc = np.array([h_i[y == c].mean().item() if (y == c).any() else np.nan
                   for c in range(ncls)])
    pc = np.array([p_i[y == c].mean().item() if (y == c).any() else np.nan
                   for c in range(ncls)])
    return hc, pc


def run_q2():
    log("\n" + "=" * 70)
    log("Q2  what carries the same-class attention mass (POLARIS signed, inference)")
    log("    p_incl: same-class mass incl. self-loop;  self_mass: self-loop share;")
    log("    p_excl: same-class mass over REAL neighbours;  h: edge homophily")
    log("=" * 70)
    log(f"\n{'dataset':<11}{'h':>7}{'p_incl':>9}{'self_mass':>11}"
        f"{'p_excl':>9}{'p_excl-h':>10}  (layer-averaged, 10 splits)")
    for d in ["Texas", "Wisconsin", "Cornell", "Actor"]:
        h = edge_homophily(load_dataset(d, split=0)[0])
        PIN, SM, PEX = [], [], []
        per_layer = None
        for sp in range(n_splits(d)):
            data, nf, ncls = load_dataset(d, split=sp)
            torch.manual_seed(sp); np.random.seed(sp)
            m = POLARIS(nf, HID, ncls, DEPTH, agg="sum", signed=True)
            train_polaris(m, data, ncls, TrainConfig(epochs=EPOCHS))
            pin, sm, pex = _decompose(m, data)
            PIN.append(pin); SM.append(sm); PEX.append(pex)
        pin = np.mean(PIN, 0); sm = np.mean(SM, 0); pex = np.mean(PEX, 0)
        log(f"{d:<11}{h:>7.3f}{pin.mean():>9.3f}{sm.mean():>11.3f}"
            f"{pex.mean():>9.3f}{pex.mean()-h:>10.3f}")
    # Per-layer detail + per-class on a multi-class graph (Actor, split 0)
    log("\nPer-layer detail (Actor, mean over 10 splits): p_incl | self_mass | p_excl")
    PIN, SM, PEX = [], [], []
    for sp in range(SPLITS):
        data, nf, ncls = load_dataset("Actor", split=sp)
        torch.manual_seed(sp); np.random.seed(sp)
        m = POLARIS(nf, HID, ncls, DEPTH, agg="sum", signed=True)
        train_polaris(m, data, ncls, TrainConfig(epochs=EPOCHS))
        pin, sm, pex = _decompose(m, data); PIN.append(pin); SM.append(sm); PEX.append(pex)
    pin = np.mean(PIN, 0); sm = np.mean(SM, 0); pex = np.mean(PEX, 0)
    for l in range(len(pin)):
        log(f"  layer {l}: {pin[l]:.3f} | {sm[l]:.3f} | {pex[l]:.3f}")
    log("\nPer-class (Actor, layer 0, split 0): class  h_c  p_excl_c  diff")
    data, nf, ncls = load_dataset("Actor", split=0)
    torch.manual_seed(0); np.random.seed(0)
    m = POLARIS(nf, HID, ncls, DEPTH, agg="sum", signed=True)
    train_polaris(m, data, ncls, TrainConfig(epochs=EPOCHS))
    hc, pc = _classwise_excl(m, data, ncls)
    for c in range(ncls):
        log(f"  {c:>5}  {hc[c]:.3f}   {pc[c]:.3f}   {pc[c]-hc[c]:+.3f}")


# --------------------------------------------------------------------------- #
@torch.no_grad()
def _realised_norm_entropy(model, data):
    """Mean (over layers and nodes) normalised aggregation entropy
    H_i / log|N~(i)|, computed in eval mode to match the realised operator."""
    model.eval()
    x = data.x
    vals = []
    for i, layer in enumerate(model.layers):
        x2, aux = layer(x, data.edge_index, return_aux=True)
        ne = (aux["entropy"] / aux["log_deg"].clamp(min=1e-6)).mean().item()
        vals.append(ne)
        x = F.elu(x2) if i < len(model.layers) - 1 else x2
    return float(np.mean(vals))


def run_q3():
    log("\n" + "=" * 70)
    log("Q3  entropy controllability: POLARIS-E with fixed target rho per dataset")
    log("    (realised normalised entropy, accuracy %, ECE; mean over 10 splits)")
    log("=" * 70)
    rhos = [0.3, 0.5, 0.7, 0.9]
    for d in ["Texas", "Wisconsin", "Cornell"]:
        log(f"\n{d}:")
        log(f"  {'target rho':>11}{'realised H':>12}{'acc':>8}{'ECE':>8}")
        for rho in rhos:
            accs, eces, ents = [], [], []
            for sp in range(SPLITS):
                data, nf, ncls = load_dataset(d, split=sp)
                torch.manual_seed(sp); np.random.seed(sp)
                m = POLARIS(nf, HID, ncls, DEPTH, agg="sum", fixed_rho=rho)
                r = train_polaris(m, data, ncls, TrainConfig(epochs=EPOCHS, beta=1.0))
                accs.append(r["acc"]); eces.append(r["ece"])
                ents.append(_realised_norm_entropy(m, data))
            log(f"  {rho:>11.2f}{np.mean(ents):>12.3f}"
                f"{np.mean(accs)*100:>8.1f}{np.mean(eces):>8.3f}")


# --------------------------------------------------------------------------- #
def run_q5():
    log("\n" + "=" * 70)
    log("Q5  signedness switch granularity (acc%% | ECE), mean+-std")
    log("=" * 70)
    variants = {
        "signed (no gate)": dict(signed=True),
        "gate: edge":  dict(signed=True, sign_gate=True, sign_gate_scope="edge"),
        "gate: node":  dict(signed=True, sign_gate=True, sign_gate_scope="node"),
        "gate: layer": dict(signed=True, sign_gate=True, sign_gate_scope="layer"),
    }
    log(f"{'dataset':<11}" + "".join(f"{v:>20}" for v in variants))
    for d in ["Texas", "Wisconsin", "Cornell", "Cora"]:
        cells = []
        for v, kw in variants.items():
            accs, eces = [], []
            for sp in range(n_splits(d)):
                data, nf, ncls = load_dataset(d, split=sp)
                torch.manual_seed(sp); np.random.seed(sp)
                m = POLARIS(nf, HID, ncls, DEPTH, agg="sum", **kw)
                r = train_polaris(m, data, ncls, TrainConfig(epochs=EPOCHS))
                accs.append(r["acc"]); eces.append(r["ece"])
            cells.append(f"{np.mean(accs)*100:5.1f}+-{np.std(accs)*100:3.1f}"
                         f"|{np.mean(eces):.3f}")
        log(f"{d:<11}" + "".join(f"{c:>20}" for c in cells))


GROUPS = {"q1": run_q1, "q2": run_q2, "q3": run_q3, "q5": run_q5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(GROUPS), default=list(GROUPS))
    args = ap.parse_args()
    torch.set_num_threads(4)
    log("\n##### APPENDIX EXPERIMENTS BATCH 2 (shared protocol, seed=split) #####")
    log(f"hidden={HID} depth={DEPTH} epochs={EPOCHS} splits={SPLITS}")
    for g in args.only:
        try:
            GROUPS[g]()
        except Exception as e:
            import traceback
            log(f"\n[ERROR in {g}] {e}\n{traceback.format_exc()}")
    log("\n##### DONE #####")


if __name__ == "__main__":
    main()
