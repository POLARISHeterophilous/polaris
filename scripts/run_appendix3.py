#!/usr/bin/env python
"""Third batch: closes the remaining reviewer asks, same protocol/seeds/splits
(hidden 64, depth 4, 120 epochs, 10 Geom-GCN splits, seed = split index).

  q6tau  : sensitivity to the temperature bounds [tau_min, tau_max]
           (signed POLARIS; acc, ECE, realised normalised entropy).
  q3temp : post-hoc temperature scaling applied to the BASELINES too, so the
           ECE comparison with POLARIS is after-scaling as well as before.
  q6rho  : per-node vs global entropy target (POLARIS-E); acc, ECE, realised entropy.

All numbers measured; appends to results/appendix/appendix3_results.txt.
"""
from __future__ import annotations
import os, numpy as np, torch, torch.nn.functional as F
from polaris.data import load_dataset
from polaris.models import POLARIS, FAGCN, GCNII, LINKX
from polaris.training import train_polaris, TrainConfig

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "results", "appendix", "appendix3_results.txt")
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


def fit_T(val_logits, y_val):
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(val_logits / logT.exp(), y_val)
        loss.backward(); return loss
    opt.step(closure)
    return float(logT.exp().item())


from polaris.metrics.calibration import expected_calibration_error as ECE


def run_q6tau():
    log("\n" + "=" * 70)
    log("q6tau  sensitivity to temperature bounds (signed POLARIS; mean/10 splits)")
    log("=" * 70)
    bounds = [(0.25, 4.0), (0.5, 2.0), (0.1, 10.0)]
    for d in ["Texas", "Wisconsin", "Cornell"]:
        log(f"\n{d}:")
        log(f"  {'[tmin,tmax]':>14}{'acc':>8}{'ECE':>8}{'realised H':>12}")
        for tmin, tmax in bounds:
            a, e, h = [], [], []
            for sp in range(SPLITS):
                data, nf, ncls = load_dataset(d, split=sp)
                torch.manual_seed(sp); np.random.seed(sp)
                m = POLARIS(nf, HID, ncls, DEPTH, agg="sum", signed=True,
                         tau_min=tmin, tau_max=tmax)
                r = train_polaris(m, data, ncls, TrainConfig(epochs=EPOCHS))
                a.append(r["acc"]); e.append(r["ece"]); h.append(realised_H(m, data))
            log(f"  {f'[{tmin},{tmax}]':>14}{np.mean(a)*100:>8.1f}"
                f"{np.mean(e):>8.3f}{np.mean(h):>12.3f}")


def run_q3temp():
    log("\n" + "=" * 70)
    log("q3temp  post-hoc temperature scaling on baselines + POLARIS (ECE)")
    log("=" * 70)
    builders = {
        "GCNII": lambda nf, nc: GCNII(nf, HID, nc, layers=DEPTH),
        "FAGCN": lambda nf, nc: FAGCN(nf, HID, nc, layers=DEPTH),
        "LINKX": lambda nf, nc: LINKX(nf, HID, nc),
        "POLARIS":  lambda nf, nc: POLARIS(nf, HID, nc, DEPTH, agg="sum", signed=True),
    }
    for d in ["Texas", "Wisconsin"]:
        log(f"\n{d}:  {'method':>8}{'T':>7}{'ECE pre':>10}{'ECE post':>10}")
        for name, build in builders.items():
            T, pre, post = [], [], []
            for sp in range(SPLITS):
                data, nf, ncls = load_dataset(d, split=sp)
                torch.manual_seed(sp); np.random.seed(sp)
                m = build(nf, ncls)
                train_polaris(m, data, ncls, TrainConfig(epochs=EPOCHS))
                m.eval()
                with torch.no_grad():
                    lo = m(data.x, data.edge_index)
                t = fit_T(lo[data.val_mask].detach(), data.y[data.val_mask])
                yt = data.y[data.test_mask]; lt = lo[data.test_mask]
                T.append(t); pre.append(ECE(lt, yt)); post.append(ECE(lt / t, yt))
            log(f"     {name:>8}{np.mean(T):>7.2f}{np.mean(pre):>10.3f}"
                f"{np.mean(post):>10.3f}")


def run_q6rho():
    log("\n" + "=" * 70)
    log("q6rho  per-node vs global entropy target (POLARIS-E; mean/10 splits)")
    log("=" * 70)
    variants = {"global rho": dict(), "per-node rho": dict(rho_mode="node")}
    for d in ["Texas", "Wisconsin", "Cornell"]:
        log(f"\n{d}:  {'target':>14}{'acc':>8}{'ECE':>8}{'realised H':>12}")
        for name, kw in variants.items():
            a, e, h = [], [], []
            for sp in range(SPLITS):
                data, nf, ncls = load_dataset(d, split=sp)
                torch.manual_seed(sp); np.random.seed(sp)
                m = POLARIS(nf, HID, ncls, DEPTH, agg="sum", **kw)
                r = train_polaris(m, data, ncls, TrainConfig(epochs=EPOCHS, beta=1.0))
                a.append(r["acc"]); e.append(r["ece"]); h.append(realised_H(m, data))
            log(f"     {name:>14}{np.mean(a)*100:>8.1f}{np.mean(e):>8.3f}"
                f"{np.mean(h):>12.3f}")


GROUPS = {"q6tau": run_q6tau, "q3temp": run_q3temp, "q6rho": run_q6rho}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(GROUPS), default=list(GROUPS))
    args = ap.parse_args()
    torch.set_num_threads(4)
    log("\n##### APPENDIX BATCH 3 (shared protocol, seed=split) #####")
    for g in args.only:
        try:
            GROUPS[g]()
        except Exception as e:
            import traceback; log(f"\n[ERROR {g}] {e}\n{traceback.format_exc()}")
    log("\n##### DONE #####")
