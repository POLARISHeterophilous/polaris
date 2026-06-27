#!/usr/bin/env python
"""Diagnostic: decompose POLARIS's same-class attention mass into self-loop vs
real-neighbour parts, per layer and per class, to see what carries the
published p-h gap. Shared protocol; split 0 (fast), signed POLARIS.
"""
from __future__ import annotations
import numpy as np, torch, torch.nn.functional as F
from torch_geometric.utils import add_self_loops, softmax as ssm, scatter
from polaris.data import load_dataset, edge_homophily
from polaris.models import POLARIS
from polaris.training import train_polaris, TrainConfig
torch.set_num_threads(4)


def alpha_at(layer, x, edge_index, n):
    h = layer.W(x)
    ei, _ = add_self_loops(edge_index, num_nodes=n)
    s, t = ei
    z = F.leaky_relu((h * layer.att_dst).sum(-1)[t] + (h * layer.att_src).sum(-1)[s],
                     layer.negative_slope)
    return ssm(z / layer.tau, t, num_nodes=n), s, t


@torch.no_grad()
def per_layer(model, data):
    model.eval(); x = data.x; n = x.size(0); rows = []
    for i, L in enumerate(model.layers):
        a, s, t = alpha_at(L, x, data.edge_index, n)
        same = (data.y[s] == data.y[t]).float()
        pin = (scatter(a * same, t, 0, dim_size=n, reduce='sum') /
               scatter(a, t, 0, dim_size=n, reduce='sum').clamp(min=1e-9)).mean().item()
        selfm = scatter(a * (s == t).float(), t, 0, dim_size=n, reduce='sum').mean().item()
        ns = (s != t).float(); aa = a * ns
        pex = (scatter(aa * same * ns, t, 0, dim_size=n, reduce='sum') /
               scatter(aa, t, 0, dim_size=n, reduce='sum').clamp(min=1e-9)).mean().item()
        rows.append((i, pin, selfm, pex))
        x = L(x, data.edge_index)
        if i < len(model.layers) - 1:
            x = F.elu(x)
    return rows


@torch.no_grad()
def per_class_layer0(model, data, ncls):
    model.eval(); n = data.x.size(0)
    a, s, t = alpha_at(model.layers[0], data.x, data.edge_index, n)
    same = (data.y[s] == data.y[t]).float(); ns = (s != t).float()
    aa = a * ns
    pex_i = (scatter(aa * same * ns, t, 0, dim_size=n, reduce='sum') /
             scatter(aa, t, 0, dim_size=n, reduce='sum').clamp(min=1e-9))
    h_i = (scatter(same * ns, t, 0, dim_size=n, reduce='sum') /
           scatter(ns, t, 0, dim_size=n, reduce='sum').clamp(min=1e-9))
    out = []
    for c in range(ncls):
        m = (data.y == c)
        if m.any():
            out.append((c, h_i[m].mean().item(), pex_i[m].mean().item()))
    return out


for d in ["Texas", "Wisconsin", "Cornell", "Actor"]:
    data, nf, ncls = load_dataset(d, split=0)
    torch.manual_seed(0); np.random.seed(0)
    m = POLARIS(nf, 64, ncls, 4, agg="sum", signed=True)
    train_polaris(m, data, ncls, TrainConfig(epochs=120))
    h = edge_homophily(data)
    print(f"\n=== {d} (h={h:.3f}, {ncls} classes) ===")
    print("  layer  p_incl  selfmass  p_excl  (p_excl - h)")
    for i, pin, sm, pex in per_layer(m, data):
        print(f"   {i:>3}   {pin:.3f}   {sm:.3f}   {pex:.3f}   {pex-h:+.3f}")
    if d in ("Cornell", "Actor"):
        print("  per-class (layer 0, neighbours only):  class  h_c   p_excl_c  diff")
        for c, hc, pc in per_class_layer0(m, data, ncls):
            print(f"     {c:>3}   {hc:.3f}   {pc:.3f}   {pc-hc:+.3f}")
