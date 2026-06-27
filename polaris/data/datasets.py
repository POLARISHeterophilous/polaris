"""Dataset loading utilities (homophilic + heterophilic)."""
from __future__ import annotations

import torch
from torch_geometric.datasets import (
    Planetoid, WebKB, Actor, HeterophilousGraphDataset,
)
from torch_geometric.utils import add_self_loops, degree

PLANETOID = {"Cora", "Citeseer", "Pubmed"}
WEBKB = {"Texas", "Wisconsin", "Cornell"}
# Platonov et al. (2023) large heterophilous benchmarks (10 splits each).
PLATONOV = {"Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"}


def _select_split(data, split: int):
    """WebKB/Actor ship 10 split columns (mask shape [N, 10]); pick one."""
    if data.train_mask.dim() == 2:
        data = data.clone()
        data.train_mask = data.train_mask[:, split]
        data.val_mask = data.val_mask[:, split]
        data.test_mask = data.test_mask[:, split]
    return data


def load_dataset(name: str, root: str = "/tmp/pyg_data", split: int = 0):
    """Return (data, num_features, num_classes) for a supported dataset.

    Homophilic: Cora, Citeseer, Pubmed (public split).
    Heterophilic: Texas, Wisconsin, Cornell (WebKB), Actor -- `split` in 0..9
    selects one of the 10 standard Geom-GCN random splits.
    """
    if name in PLANETOID:
        ds = Planetoid(root=f"{root}/{name}", name=name)
        return ds[0], ds.num_features, ds.num_classes
    if name in WEBKB:
        ds = WebKB(root=f"{root}/{name}", name=name)
        return _select_split(ds[0], split), ds.num_features, ds.num_classes
    if name == "Actor":
        ds = Actor(root=f"{root}/Actor")
        return _select_split(ds[0], split), ds.num_features, ds.num_classes
    if name in PLATONOV:
        ds = HeterophilousGraphDataset(root=f"{root}/{name}", name=name)
        return _select_split(ds[0], split), ds.num_features, ds.num_classes
    raise ValueError(f"unsupported dataset {name}")


@torch.no_grad()
def entropy_ceiling(data) -> float:
    """Mean log|N~(i)| -- the maximum attainable mean aggregation entropy."""
    ei, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    deg = degree(ei[1], num_nodes=data.num_nodes).clamp(min=1)
    return deg.log().mean().item()


@torch.no_grad()
def edge_homophily(data) -> float:
    """Fraction of edges connecting same-label nodes (edge homophily ratio).
    ~0.8 for citation graphs; <0.3 for WebKB/Actor heterophilic graphs."""
    src, dst = data.edge_index
    return (data.y[src] == data.y[dst]).float().mean().item()
