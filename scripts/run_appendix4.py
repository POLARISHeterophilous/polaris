#!/usr/bin/env python
"""Reviewer follow-ups, same protocol/seeds as the main tables.

  psi   : (Q5) does the signed coefficient implement repulsion? Layer-0 psi at
          inference, split by same/different-class edges, and its correlation
          with feature dissimilarity.
  depth : (Q7) depth sweep on a LARGER graph (Actor, 7.6k nodes) for POLARIS vs
          GCN/GAT, to test graceful degradation beyond tiny Wisconsin.

Appends to results/appendix/appendix4_results.txt.
"""
from __future__ import annotations
import os, numpy as np, torch, torch.nn.functional as F
from torch_geometric.utils import add_self_loops, scatter
from polaris.data import load_dataset
from polaris.models import POLARIS, GCN, GAT
from polaris.training import train_polaris, TrainConfig

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "results", "appendix", "appendix4_results.txt")
EPOCHS, HID, DEPTH = 120, 64, 4


def log(m=""):
    print(m, flush=True)
    with open(OUT, "a") as f:
        f.write(m + "\n")


@torch.no_grad()
def layer0_psi(model, data):
    """Layer-0 signed coefficients psi_ij and edge endpoints (inference)."""
    model.eval(); L = model.layers[0]; h = L.W(data.x)
    ei, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
    src, dst = ei[0], ei[1]
    psi = torch.tanh(L.sign(torch.cat([h[dst], h[src]], dim=-1)).squeeze(-1))
    return psi, src, dst


def run_psi():
    log("\n" + "=" * 66)
    log("Q5  signed coefficient psi at inference (layer 0): repulsion check")
    log("    psi_same / psi_diff over real neighbours; corr(psi, feat. dissim.)")
    log("=" * 66)
    log(f"{'dataset':<11}{'psi_same':>10}{'psi_diff':>10}{'gap':>8}{'r(psi,dissim)':>15}")
    for d in ["Texas", "Wisconsin", "Cornell", "Actor", "Cora"]:
        n_sp = 1 if d == "Cora" else 10
        ps, pd, rs = [], [], []
        for sp in range(n_sp):
            data, nf, ncls = load_dataset(d, split=sp)
            torch.manual_seed(sp); np.random.seed(sp)
            m = POLARIS(nf, HID, ncls, DEPTH, agg="sum", signed=True)
            train_polaris(m, data, ncls, TrainConfig(epochs=EPOCHS))
            psi, src, dst = layer0_psi(m, data)
            ns = (src != dst)
            same = (data.y[src] == data.y[dst]) & ns
            diff = (data.y[src] != data.y[dst]) & ns
            ps.append(psi[same].mean().item()); pd.append(psi[diff].mean().item())
            # feature dissimilarity = 1 - cosine
            xn = F.normalize(data.x.float(), dim=-1)
            cos = (xn[src] * xn[dst]).sum(-1)
            dissim = (1 - cos)[ns]
            pn = psi[ns]
            # Pearson r
            a = pn - pn.mean(); b = dissim - dissim.mean()
            r = (a * b).sum() / (a.norm() * b.norm() + 1e-9)
            rs.append(r.item())
        log(f"{d:<11}{np.mean(ps):>10.3f}{np.mean(pd):>10.3f}"
            f"{np.mean(ps)-np.mean(pd):>8.3f}{np.mean(rs):>15.3f}")


def run_depth():
    log("\n" + "=" * 66)
    log("Q7  depth sweep on Actor (7.6k nodes): test acc %, mean over 10 splits")
    log("=" * 66)
    depths = [2, 4, 8, 16]
    builders = {
        "GCN":  lambda nf, nc, L: GCN(nf, HID, nc, L),
        "GAT":  lambda nf, nc, L: GAT(nf, HID, nc, max(L, 2), heads=4),
        "POLARIS": lambda nf, nc, L: POLARIS(nf, HID, nc, L, agg="sum", signed=True),
    }
    log(f"{'depth':>6}" + "".join(f"{m:>9}" for m in builders))
    for L in depths:
        cells = []
        for name, build in builders.items():
            accs = []
            for sp in range(10):
                data, nf, ncls = load_dataset("Actor", split=sp)
                torch.manual_seed(sp); np.random.seed(sp)
                r = train_polaris(build(nf, ncls, L), data, ncls,
                               TrainConfig(epochs=EPOCHS))
                accs.append(r["acc"])
            cells.append(np.mean(accs) * 100)
        log(f"{L:>6}" + "".join(f"{c:>9.1f}" for c in cells))


GROUPS = {"psi": run_psi, "depth": run_depth}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(GROUPS), default=list(GROUPS))
    args = ap.parse_args()
    torch.set_num_threads(8)
    log("\n##### APPENDIX BATCH 4 #####")
    for g in args.only:
        try:
            GROUPS[g]()
        except Exception as e:
            import traceback; log(f"[ERROR {g}] {e}\n{traceback.format_exc()}")
    log("\n##### DONE #####")
