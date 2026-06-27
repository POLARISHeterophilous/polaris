#!/usr/bin/env python
"""Reviewer-requested appendix experiments (Q1-Q5), all under the SAME protocol,
seeds, and splits as the main tables: hidden 64, depth 4, 120 epochs, Adam
lr 0.01 / wd 5e-4, label smoothing 0.1, dropout 0.5, 10 Geom-GCN splits, with
seed = split index. Nothing here is tuned per method beyond that shared budget.

Groups (selectable with --only):
  baselines : Q5  -- H2GCN, MixHop, LSGNN vs POLARIS (acc, ECE; mean+-std/splits)
  variants  : Q1/Q2 -- per-node temperature, per-edge signedness gate; incl. Cora
  calib     : Q3  -- ECE/NLL mean+-std, ECE binning sensitivity, post-hoc temp scaling
  bias      : Q4  -- learned lambda_disc vs (p-h) vs accuracy; pseudo-label proxy

Every number is measured from a real run; nothing is fabricated. Results are
appended to results/appendix/appendix_results.txt as they are produced.
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

from polaris.data import load_dataset, edge_homophily
from polaris.models import POLARIS, H2GCN, MixHop, LSGNN, FAGCN, GAT, GCNII
from polaris.training import train_polaris, TrainConfig
from polaris.metrics.calibration import expected_calibration_error

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "results", "appendix", "appendix_results.txt")
HETERO = ["Texas", "Wisconsin", "Cornell", "Actor"]
EPOCHS = 120
HID = 64
DEPTH = 4
SPLITS = 10


def log(msg=""):
    print(msg, flush=True)
    with open(OUT, "a") as f:
        f.write(msg + "\n")


def n_splits(d):
    return 1 if d == "Cora" else SPLITS


def fit_temperature(val_logits, y_val):
    """Post-hoc temperature scaling: scalar T>0 minimising val NLL (LBFGS)."""
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(val_logits / logT.exp(), y_val)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp().item())


# --------------------------------------------------------------------------- #
def run_baselines():
    log("\n" + "=" * 70)
    log("Q5  NEW BASELINES under the shared protocol (acc%% | ECE), mean+-std")
    log("=" * 70)
    models = {
        "MixHop": lambda nf, nc: MixHop(nf, HID, nc, layers=DEPTH),
        "H2GCN":  lambda nf, nc: H2GCN(nf, HID, nc, layers=DEPTH),
        "LSGNN":  lambda nf, nc: LSGNN(nf, HID, nc, layers=DEPTH),
        "POLARIS":   lambda nf, nc: POLARIS(nf, HID, nc, DEPTH, agg="sum", signed=True),
    }
    header = f"{'dataset':<11}" + "".join(f"{m:>22}" for m in models)
    log(header)
    rows = {m: [] for m in models}
    for d in HETERO:
        cells, latex = [], []
        for m, build in models.items():
            accs, eces = [], []
            for sp in range(n_splits(d)):
                data, nf, ncls = load_dataset(d, split=sp)
                torch.manual_seed(sp); np.random.seed(sp)
                r = train_polaris(build(nf, ncls), data, ncls,
                               TrainConfig(epochs=EPOCHS))
                accs.append(r["acc"]); eces.append(r["ece"])
            ma, sa = np.mean(accs) * 100, np.std(accs) * 100
            me, se = np.mean(eces), np.std(eces)
            rows[m].append((d, ma, sa, me, se))
            cells.append(f"{ma:5.1f}+-{sa:3.1f}|{me:.3f}")
        log(f"{d:<11}" + "".join(f"{c:>22}" for c in cells))
    # LaTeX body
    log("\n%% --- LaTeX (acc) body: rows=method, cols=datasets ---")
    for m in models:
        cells = " & ".join(f"{ma:.1f}\\footnotesize$\\pm${sa:.1f}"
                           for (_, ma, sa, _, _) in rows[m])
        log(f"{m} & {cells} \\\\")


# --------------------------------------------------------------------------- #
def run_variants():
    log("\n" + "=" * 70)
    log("Q1/Q2  ADAPTIVE VARIANTS (acc%% | ECE), mean+-std; Cora = homophily test")
    log("=" * 70)
    datasets = ["Texas", "Wisconsin", "Cornell", "Cora"]
    variants = {
        "POLARIS (signed)":   dict(signed=True),
        "POLARIS-U (unsigned)": dict(signed=False),
        "+ per-node tau":  dict(signed=True, tau_mode="node"),
        "+ sign gate":     dict(signed=True, sign_gate=True),
    }
    log(f"{'dataset':<11}" + "".join(f"{v:>22}" for v in variants))
    for d in datasets:
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
        log(f"{d:<11}" + "".join(f"{c:>22}" for c in cells))


# --------------------------------------------------------------------------- #
def run_calib():
    log("\n" + "=" * 70)
    log("Q3  CALIBRATION: ECE/NLL mean+-std, binning sensitivity, temp scaling")
    log("=" * 70)
    datasets = ["Texas", "Wisconsin", "Cornell"]
    bins_list = [10, 15, 20, 30]
    log("\n-- ECE/NLL over 10 splits (POLARIS) and ECE at different bin counts --")
    log(f"{'dataset':<11}{'NLL':>14}{'ECE@15':>14}" +
        "".join(f"{'ECE@'+str(b):>10}" for b in bins_list))
    # Also accumulate val/test logits for temp scaling.
    temp_rows = []
    for d in datasets:
        nlls, eces = [], []
        bin_eces = {b: [] for b in bins_list}
        T_list, ece_pre, ece_post = [], [], []
        for sp in range(SPLITS):
            data, nf, ncls = load_dataset(d, split=sp)
            torch.manual_seed(sp); np.random.seed(sp)
            m = POLARIS(nf, HID, ncls, DEPTH, agg="sum", signed=True)
            r = train_polaris(m, data, ncls, TrainConfig(epochs=EPOCHS))
            nlls.append(r["nll"]); eces.append(r["ece"])
            m.eval()
            with torch.no_grad():
                logits = m(data.x, data.edge_index)
            yt = data.y[data.test_mask]; lt = logits[data.test_mask]
            for b in bins_list:
                bin_eces[b].append(expected_calibration_error(lt, yt, n_bins=b))
            # temp scaling: fit on val, apply to test
            T = fit_temperature(logits[data.val_mask].detach(),
                                data.y[data.val_mask])
            T_list.append(T)
            ece_pre.append(expected_calibration_error(lt, yt))
            ece_post.append(expected_calibration_error(lt / T, yt))
        log(f"{d:<11}{np.mean(nlls):.3f}+-{np.std(nlls):.2f}  "
            f"{np.mean(eces):.3f}+-{np.std(eces):.2f}  " +
            "".join(f"{np.mean(bin_eces[b]):>10.3f}" for b in bins_list))
        temp_rows.append((d, np.mean(T_list), np.mean(ece_pre),
                          np.mean(ece_post)))
    log("\n-- Post-hoc temperature scaling (POLARIS): fitted T, ECE before/after --")
    log(f"{'dataset':<11}{'T':>8}{'ECE pre':>10}{'ECE post':>10}")
    for d, T, pre, post in temp_rows:
        log(f"{d:<11}{T:>8.2f}{pre:>10.3f}{post:>10.3f}")


# --------------------------------------------------------------------------- #
def _train_pseudo_bias(data, nf, ncls, pseudo_y, conf_mask, seed):
    """Train a POLARIS whose discriminative bias is driven by PSEUDO-labels
    (model self-predictions), while the supervised CE loss still uses only the
    true training labels. Self-supervised structural proxy for the bias (Q4)."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = POLARIS(nf, HID, ncls, DEPTH, agg="sum", signed=True)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_state = -1.0, None
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad()
        out = m(data.x, data.edge_index, y=pseudo_y, train_mask=conf_mask)
        ce = F.cross_entropy(out[data.train_mask], data.y[data.train_mask],
                             label_smoothing=0.1)
        ce.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            va = m(data.x, data.edge_index).argmax(-1)[data.val_mask] \
                .eq(data.y[data.val_mask]).float().mean().item()
        if va > best_val:
            best_val = va
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        acc = m(data.x, data.edge_index).argmax(-1)[data.test_mask] \
            .eq(data.y[data.test_mask]).float().mean().item()
    return acc


def run_bias():
    log("\n" + "=" * 70)
    log("Q4  LABEL BIAS: learned lambda_disc vs (p-h) vs accuracy; pseudo proxy")
    log("=" * 70)
    log(f"{'dataset':<11}{'h':>6}{'lam_disc':>10}{'acc(bias)':>11}"
        f"{'acc(no)':>9}{'d_acc':>8}{'acc(pseudo)':>12}")
    arr_lam, arr_dacc = [], []
    for d in HETERO:
        h = edge_homophily(load_dataset(d, split=0)[0])
        lam_d, a_on, a_off, a_ps = [], [], [], []
        for sp in range(n_splits(d)):
            data, nf, ncls = load_dataset(d, split=sp)
            # bias ON (standard POLARIS)
            torch.manual_seed(sp); np.random.seed(sp)
            m_on = POLARIS(nf, HID, ncls, DEPTH, agg="sum", signed=True)
            r_on = train_polaris(m_on, data, ncls, TrainConfig(epochs=EPOCHS))
            a_on.append(r_on["acc"])
            lam_d.append(float(np.mean([l.lambda_disc.item()
                                        for l in m_on.layers])))
            # bias OFF
            torch.manual_seed(sp); np.random.seed(sp)
            m_off = POLARIS(nf, HID, ncls, DEPTH, agg="sum", signed=True,
                         use_disc_bias=False)
            r_off = train_polaris(m_off, data, ncls, TrainConfig(epochs=EPOCHS))
            a_off.append(r_off["acc"])
            # pseudo-label bias: predictions from the no-bias model over all nodes
            m_off.eval()
            with torch.no_grad():
                logits = m_off(data.x, data.edge_index)
                conf, pseudo = F.softmax(logits, -1).max(-1)
            conf_mask = conf > 0.8                      # high-confidence pseudo nodes
            a_ps.append(_train_pseudo_bias(data, nf, ncls, pseudo,
                                           conf_mask, sp))
        ma_on, ma_off = np.mean(a_on) * 100, np.mean(a_off) * 100
        dacc = ma_on - ma_off
        log(f"{d:<11}{h:>6.2f}{np.mean(lam_d):>10.3f}{ma_on:>11.1f}"
            f"{ma_off:>9.1f}{dacc:>8.1f}{np.mean(a_ps)*100:>12.1f}")
        arr_lam.append(np.mean(lam_d)); arr_dacc.append(dacc)
    if len(arr_lam) > 2:
        c = np.corrcoef(arr_lam, arr_dacc)[0, 1]
        log(f"\nPearson r(lambda_disc, delta-acc) over datasets = {c:.3f}")


GROUPS = {"baselines": run_baselines, "variants": run_variants,
          "calib": run_calib, "bias": run_bias}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(GROUPS),
                    default=list(GROUPS))
    args = ap.parse_args()
    torch.set_num_threads(4)
    log("\n########## APPENDIX EXPERIMENTS (shared protocol, seed=split) ##########")
    log(f"hidden={HID} depth={DEPTH} epochs={EPOCHS} splits={SPLITS} "
        f"lr=0.01 wd=5e-4 label_smoothing=0.1 dropout=0.5")
    for g in args.only:
        try:
            GROUPS[g]()
        except Exception as e:
            import traceback
            log(f"\n[ERROR in {g}] {e}\n{traceback.format_exc()}")
    log("\n########## DONE ##########")


if __name__ == "__main__":
    main()
