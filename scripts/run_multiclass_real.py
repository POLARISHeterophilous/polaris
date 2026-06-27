#!/usr/bin/env python
"""Real-data validation of the formal multi-class theory (App. A): the averaged
neighbour signal should align with the true class mean only when homophily
exceeds the chance level 1/K, and point AWAY from it (negative projection) when
h<1/K.

For each dataset we estimate class means from features (labels used only for this
diagnostic, as in the p-vs-h probe), centre them, and for every node measure the
projection of its uniform neighbour-average onto its own (centred) class mean,
  rho_i = <m_i - mubar, mu_{y_i} - mubar> / ||mu_{y_i} - mubar||^2 ,
the real-data analogue of gamma_K(h). We report the mean projection (overall and
the fraction of classes with negative mean projection) against h and 1/K.

No training; everything is measured on the real graphs. Appends to
results/appendix/multiclass_real.txt.
"""
from __future__ import annotations
import os, numpy as np, torch
from torch_geometric.utils import scatter
from polaris.data import load_dataset, edge_homophily

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "results", "appendix", "multiclass_real.txt")


def log(m=""):
    print(m, flush=True)
    with open(OUT, "a") as f:
        f.write(m + "\n")


@torch.no_grad()
def projection(data, ncls):
    x = data.x.float()
    y = data.y
    n, d = x.shape
    mubar = x.mean(0, keepdim=True)
    mu = torch.zeros(ncls, d)
    for c in range(ncls):
        m = (y == c)
        if m.any():
            mu[c] = x[m].mean(0)
    muc = mu - mubar                                  # centred class means
    # uniform neighbour average (no self-loop)
    src, dst = data.edge_index[0], data.edge_index[1]
    m_i = scatter(x[src], dst, dim=0, dim_size=n, reduce="mean") - mubar
    own = muc[y]                                       # (n,d) centred mean of own class
    num = (m_i * own).sum(-1)
    den = (own * own).sum(-1).clamp(min=1e-9)
    rho_i = num / den
    # per-class mean projection and same-class neighbour fraction h_c
    same = (y[src] == y[dst]).float()
    hc_num = scatter(same, dst, dim=0, dim_size=n, reduce="sum")
    deg = scatter(torch.ones_like(same), dst, dim=0, dim_size=n, reduce="sum").clamp(min=1)
    h_i = hc_num / deg
    per_c_rho, per_c_h = [], []
    for c in range(ncls):
        m = (y == c)
        if m.any():
            per_c_rho.append(rho_i[m].mean().item())
            per_c_h.append(h_i[m].mean().item())
    per_c_rho = np.array(per_c_rho); per_c_h = np.array(per_c_h)
    return rho_i.mean().item(), per_c_rho, per_c_h


def main():
    torch.set_num_threads(4)
    log("\n##### MULTI-CLASS THEORY: REAL-DATA VALIDATION #####")
    log("rho = projection of uniform neighbour-average onto the true (centred)")
    log("class mean; theory: rho<0 (misleading) when h<1/K, rho>0 when h>1/K.\n")
    log(f"{'dataset':<15}{'K':>4}{'h':>7}{'1/K':>7}{'mean rho':>10}"
        f"{'%cls rho<0':>11}")
    for d in ["Roman-empire", "Texas", "Cornell", "Wisconsin", "Actor",
              "Amazon-ratings", "Cora"]:
        try:
            data, nf, ncls = load_dataset(d, split=0)
            h = edge_homophily(data)
            mean_rho, pc_rho, pc_h = projection(data, ncls)
            frac_neg = float((pc_rho < 0).mean()) * 100
            log(f"{d:<15}{ncls:>4}{h:>7.3f}{1.0/ncls:>7.3f}{mean_rho:>10.3f}"
                f"{frac_neg:>10.0f}%")
        except Exception as e:
            log(f"{d:<15} [error: {e}]")
    log("\n##### DONE #####")


if __name__ == "__main__":
    main()
