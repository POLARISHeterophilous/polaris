#!/usr/bin/env python
"""Revision experiments, using the SAME API as run_ablation_significance.py.

  polarisse : POLARIS-SE (signed + entropy regulariser) ablation on the 4 heterophily
           graphs, alongside POLARIS-U / POLARIS-E / POLARIS  -> results_polarisse.txt
  timing : runtime / #params overhead vs GAT and FAGCN -> results_timing.txt
"""
from __future__ import annotations

import argparse, time
import numpy as np
import torch

from polaris.data import load_dataset
from polaris.models import POLARIS, GCN, GAT, GCNII, FAGCN
from polaris.training import train_polaris, TrainConfig

HETERO = ["Texas", "Wisconsin", "Cornell", "Actor"]

# (make_model, beta) per variant; H,L bound at call time
VARIANTS = {
    "POLARIS-U":  (lambda nf, nc, H, L: POLARIS(nf, H, nc, L, agg="sum", signed=False), 0.0),
    "POLARIS-E":  (lambda nf, nc, H, L: POLARIS(nf, H, nc, L, agg="sum", signed=False), 1.0),
    "POLARIS":    (lambda nf, nc, H, L: POLARIS(nf, H, nc, L, agg="sum", signed=True),  0.0),
    "POLARIS-SE": (lambda nf, nc, H, L: POLARIS(nf, H, nc, L, agg="sum", signed=True),  1.0),
}


def acc_per_split(make, beta, dname, splits, epochs, H, L):
    accs = []
    for sp in range(splits):
        data, nf, ncls = load_dataset(dname, split=sp)
        torch.manual_seed(sp); np.random.seed(sp)
        m = make(nf, ncls, H, L)
        r = train_polaris(m, data, ncls, TrainConfig(epochs=epochs, beta=beta))
        accs.append(r["acc"])
    return np.array(accs)


def cmd_polarisse(args):
    H, L = args.hidden, args.depth
    names = list(VARIANTS)
    out = args.out or "results_polarisse.txt"
    per = {v: [] for v in names}
    with open(out, "w") as f:
        f.write(f"POLARIS-SE ablation | depth {L}, hidden {H}, "
                f"{args.splits} Geom-GCN splits\n\n")
        f.write(f"{'dataset':>10}" + "".join(f"{v:>10}" for v in names) + "\n")
        f.flush()
        for dn in HETERO:
            row = f"{dn:>10}"
            for v in names:
                make, beta = VARIANTS[v]
                a = acc_per_split(make, beta, dn, args.splits, args.epochs, H, L)
                per[v].append(a.mean())
                row += f"{a.mean()*100:>10.1f}"
            f.write(row + "\n"); f.flush()
        f.write(f"{'mean':>10}" + "".join(f"{np.mean(per[v])*100:>10.1f}"
                                          for v in names) + "\n")
    print("polarisse done", flush=True)


def cmd_timing(args):
    H, L = args.hidden, args.depth
    data, nf, ncls = load_dataset(args.dataset, split=0)
    n_nodes = data.x.size(0); n_edges = data.edge_index.size(1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(dev)
    import torch.nn.functional as F
    builders = {
        "GCN":    lambda: GCN(nf, H, ncls, L),
        "GAT":    lambda: GAT(nf, H, ncls, max(L, 2), heads=4),
        "FAGCN":  lambda: FAGCN(nf, H, ncls, layers=L),
        "POLARIS-U": lambda: POLARIS(nf, H, ncls, L, agg="sum", signed=False),
        "POLARIS":   lambda: POLARIS(nf, H, ncls, L, agg="sum", signed=True),
    }
    warm, reps = 5, 30
    out = args.out or "results_timing.txt"
    with open(out, "w") as f:
        f.write(f"Computational overhead | {args.dataset}: {n_nodes} nodes, "
                f"{n_edges} edges, depth {L}, hidden {H}, device={dev.type}\n\n")
        f.write(f"{'model':>10}{'params':>12}{'ms/epoch':>12}{'ms/fwd':>10}\n")
        f.flush()
        for name, build in builders.items():
            torch.manual_seed(0); np.random.seed(0)
            model = build().to(dev)
            nparam = sum(p.numel() for p in model.parameters())
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            for _ in range(warm):
                model.train(); opt.zero_grad()
                out_ = model(data.x, data.edge_index)
                loss = F.cross_entropy(out_[data.train_mask], data.y[data.train_mask])
                loss.backward(); opt.step()
            if dev.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                model.train(); opt.zero_grad()
                out_ = model(data.x, data.edge_index)
                loss = F.cross_entropy(out_[data.train_mask], data.y[data.train_mask])
                loss.backward(); opt.step()
            if dev.type == "cuda": torch.cuda.synchronize()
            ms_ep = 1000.0 * (time.perf_counter() - t0) / reps
            model.eval()
            if dev.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(reps):
                    model(data.x, data.edge_index)
            if dev.type == "cuda": torch.cuda.synchronize()
            ms_fwd = 1000.0 * (time.perf_counter() - t0) / reps
            f.write(f"{name:>10}{nparam:>12d}{ms_ep:>12.2f}{ms_fwd:>10.2f}\n")
            f.flush()
    print("timing done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("polarisse")
    s.add_argument("--hidden", type=int, default=64); s.add_argument("--depth", type=int, default=4)
    s.add_argument("--splits", type=int, default=10); s.add_argument("--epochs", type=int, default=120)
    s.add_argument("--out")
    t = sub.add_parser("timing")
    t.add_argument("--dataset", default="Actor"); t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--depth", type=int, default=4); t.add_argument("--out")
    args = ap.parse_args()
    torch.set_num_threads(4)
    {"polarisse": cmd_polarisse, "timing": cmd_timing}[args.cmd](args)


if __name__ == "__main__":
    main()
