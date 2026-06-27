#!/usr/bin/env python
"""Q7: quantify the label-safe discriminative bias.

(a) How many edges does the bias actually touch under standard label rates?
    The bias fires only on edges whose BOTH endpoints are labelled (training)
    and i != j: nu = (train_mask[src] & train_mask[dst]) & (src != dst). We
    report the fraction of (self-looped) edges with nu=1 per dataset.

(b) Does the learned attention concentrate same-class mass above chance at
    INFERENCE (where the bias is off)? For a trained POLARIS, we recompute the
    softmax weights alpha on the full graph in eval mode (bias disabled), and
    for each node measure the share of its attention mass landing on same-label
    neighbours, p_i = sum_{j: y_j=y_i} alpha_ij. We compare mean p to the edge
    homophily h (the chance level under uniform attention, Prop. 3).

Writes results/label_bias_coverage.txt. No fabrication: every number is measured from a
trained model on the real splits.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops

from polaris.data import load_dataset, edge_homophily
from polaris.models import POLARIS
from polaris.training import train_polaris, TrainConfig

HETERO = ["Texas", "Wisconsin", "Cornell", "Actor"]


@torch.no_grad()
def first_layer_alpha(model, data):
    """Recompute layer-0 softmax weights in eval mode (bias OFF), returning
    (alpha, src, dst) over the augmented neighbourhood."""
    model.eval()
    layer = model.layers[0]
    x = data.x
    h = layer.W(x)
    ei, _ = add_self_loops(data.edge_index, num_nodes=x.size(0))
    src, dst = ei[0], ei[1]
    z = (h * layer.att_dst).sum(-1)[dst] + (h * layer.att_src).sum(-1)[src]
    z = F.leaky_relu(z, layer.negative_slope)
    from torch_geometric.utils import softmax as scatter_softmax
    alpha = scatter_softmax(z / layer.tau, dst, num_nodes=x.size(0))
    return alpha, src, dst


def bias_edge_fraction(data):
    ei, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
    src, dst = ei[0], ei[1]
    tm = data.train_mask
    nu = (tm[src] & tm[dst]) & (src != dst)
    return nu.float().mean().item(), tm.float().mean().item()


def same_class_mass(alpha, src, dst, y):
    same = (y[src] == y[dst]).float()
    n = y.size(0)
    from torch_geometric.utils import scatter
    p = scatter(alpha * same, dst, dim=0, dim_size=n, reduce="sum")
    deg = scatter(alpha, dst, dim=0, dim_size=n, reduce="sum").clamp(min=1e-9)
    return (p / deg).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--out", default="results/label_bias_coverage.txt")
    args = ap.parse_args()
    torch.set_num_threads(4)

    with open(args.out, "w") as f:
        f.write("Q7: label-safe bias coverage and inference-time attention "
                f"concentration (mean over {args.splits} Geom-GCN splits; POLARIS signed).\n")
        f.write(f"{'dataset':>10}{'h':>7}{'label_rate':>11}{'bias_edge_frac':>15}"
                f"{'mean p (infer)':>15}{'p - h':>8}\n")
        for dn in HETERO:
            fr, lb, ps, hs = [], [], [], []
            for sp in range(args.splits):
                data, nf, ncls = load_dataset(dn, split=sp)
                hs.append(edge_homophily(data))
                frac, lab = bias_edge_fraction(data); fr.append(frac); lb.append(lab)
                torch.manual_seed(sp); np.random.seed(sp)
                m = POLARIS(nf, args.hidden, ncls, args.depth, agg="sum", signed=True)
                train_polaris(m, data, ncls, TrainConfig(epochs=args.epochs))
                alpha, src, dst = first_layer_alpha(m, data)
                ps.append(same_class_mass(alpha, src, dst, data.y))
            h, frac, lab, p = np.mean(hs), np.mean(fr), np.mean(lb), np.mean(ps)
            f.write(f"{dn:>10}{h:>7.3f}{lab:>11.3f}{frac:>15.3f}{p:>15.3f}{p-h:>+8.3f}\n")
            f.flush()
            print(f"{dn}: h={h:.3f} bias_edges={frac:.3f} p={p:.3f} (p-h={p-h:+.3f})", flush=True)
    print("BIAS_ALPHA_DONE", flush=True)


if __name__ == "__main__":
    main()
