"""Baseline GNNs (GCN, GAT) sharing the POLARIS forward signature."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid, ncls, layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        dims = [in_dim] + [hid] * (layers - 1) + [ncls]
        for i in range(layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))

    def forward(self, x, edge_index, **kw):
        for i, c in enumerate(self.convs):
            x = c(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_dim, hid, ncls, layers=2, heads=8, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_dim, hid, heads=heads, dropout=dropout))
        for _ in range(layers - 2):
            self.convs.append(GATConv(hid * heads, hid, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hid * heads, ncls, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index, **kw):
        for i, c in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = c(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return x
