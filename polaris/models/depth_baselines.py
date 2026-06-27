"""Depth-robust baselines: GCNII and APPNP.

These are the methods actually designed to prevent oversmoothing, and are the
fair comparison for any depth-robustness claim (GCN/GAT collapse and are a
strawman at large depth). Both share the POLARIS forward signature
(x, edge_index, **kw) so the trainer can use them interchangeably.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv, APPNP as APPNPProp
from torch_geometric.utils import add_self_loops, degree


class GCNII(torch.nn.Module):
    """GCNII (Chen et al., ICML 2020): initial residual + identity mapping.

    h^(l+1) = sigma( ((1-a)P h^(l) + a h^(0)) ((1-b_l)I + b_l W_l) ),
    with b_l = log(theta/l + 1). The canonical deep-GNN baseline.
    """

    def __init__(self, in_dim, hid, ncls, layers=2, alpha=0.1, theta=0.5,
                 dropout=0.5, shared_weights=True):
        super().__init__()
        self.dropout = dropout
        self.lin_in = torch.nn.Linear(in_dim, hid)
        self.lin_out = torch.nn.Linear(hid, ncls)
        self.convs = torch.nn.ModuleList()
        for l in range(layers):
            self.convs.append(
                GCN2Conv(hid, alpha=alpha, theta=theta, layer=l + 1,
                         shared_weights=shared_weights)
            )

    def forward(self, x, edge_index, **kw):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x0 = F.relu(self.lin_in(x))
        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(conv(x, x0, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.lin_out(x)


class APPNP(torch.nn.Module):
    """APPNP (Gasteiger et al., ICLR 2019): decoupled MLP + personalized
    PageRank propagation. 'Depth' = number of propagation steps K (parameter
    count is fixed), so it is structurally immune to oversmoothing collapse."""

    def __init__(self, in_dim, hid, ncls, layers=10, alpha=0.1, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(in_dim, hid)
        self.lin2 = torch.nn.Linear(hid, ncls)
        self.prop = APPNPProp(K=layers, alpha=alpha)

    def forward(self, x, edge_index, **kw):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)
        return self.prop(x, edge_index)


class GPRGNN(torch.nn.Module):
    """GPR-GNN (Chien et al., ICLR 2021): Generalized PageRank.

    Decoupled MLP encoder, then H = sum_{k=0}^{K} gamma_k P^k Z with LEARNABLE
    per-hop weights gamma_k. Unlike APPNP's fixed PPR weights, the gamma_k can
    go negative, which lets GPR-GNN learn high-pass (heterophily) filters. This
    is the canonical heterophily-specialist baseline closest to POLARIS's
    'learn the propagation' framing. gamma init: PPR-style alpha(1-alpha)^k.
    """

    def __init__(self, in_dim, hid, ncls, layers=10, alpha=0.1, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.K = layers
        self.lin1 = torch.nn.Linear(in_dim, hid)
        self.lin2 = torch.nn.Linear(hid, ncls)
        # PPR initialisation of the per-hop coefficients
        gamma = alpha * (1 - alpha) ** torch.arange(layers + 1)
        gamma[-1] = (1 - alpha) ** layers
        self.gamma = torch.nn.Parameter(gamma)

    def _norm_adj(self, edge_index, n, device):
        ei, _ = add_self_loops(edge_index, num_nodes=n)
        row, col = ei
        deg = degree(col, n, dtype=torch.float).clamp(min=1)
        dinv = deg.pow(-0.5)
        w = dinv[row] * dinv[col]                      # symmetric normalisation
        return ei, w

    def forward(self, x, edge_index, **kw):
        n = x.size(0)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        z = self.lin2(x)
        ei, w = self._norm_adj(edge_index, n, x.device)
        row, col = ei
        out = self.gamma[0] * z
        h = z
        for k in range(1, self.K + 1):
            # h <- P h  (sparse propagation)
            msg = h[row] * w.unsqueeze(-1)
            h = torch.zeros_like(h).index_add_(0, col, msg)
            out = out + self.gamma[k] * h
        return out


class FAGCN(torch.nn.Module):
    """FAGCN (Bo et al., AAAI 2021): Frequency-Adaptive GCN.

    Each propagation step mixes low- and high-frequency signals with a learned,
    signed edge coefficient eps_ij = tanh(g^T [h_i || h_j]) in (-1,1):
        h_i' = eps_self * h_i^0 + sum_j (eps_ij / sqrt(d_i d_j)) h_j .
    The signed coefficient is what handles heterophily (it can subtract a
    dissimilar neighbour). A heterophily-specialist baseline.
    """

    def __init__(self, in_dim, hid, ncls, layers=4, eps=0.3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.K = layers
        self.eps = eps
        self.lin_in = torch.nn.Linear(in_dim, hid)
        self.lin_out = torch.nn.Linear(hid, ncls)
        self.gate = torch.nn.Linear(2 * hid, 1)

    def forward(self, x, edge_index, **kw):
        n = x.size(0)
        x = F.dropout(x, self.dropout, training=self.training)
        h0 = torch.relu(self.lin_in(x))
        ei, _ = add_self_loops(edge_index, num_nodes=n)
        row, col = ei
        deg = degree(col, n, dtype=torch.float).clamp(min=1)
        dnorm = deg.pow(-0.5)
        h = h0
        for _ in range(self.K):
            g = torch.tanh(self.gate(torch.cat([h[row], h[col]], dim=-1))).squeeze(-1)
            w = g * dnorm[row] * dnorm[col]
            msg = h[row] * w.unsqueeze(-1)
            agg = torch.zeros_like(h).index_add_(0, col, msg)
            h = self.eps * h0 + agg
            h = F.dropout(h, self.dropout, training=self.training)
        return self.lin_out(h)
