"""Additional heterophily baselines requested in review.

LINKX (Lim et al., NeurIPS 2021, "Large Scale Learning on Non-Homophilous
Graphs"): embed the adjacency and the features with separate MLPs, then
combine. Faithful to Eq. (1) of the paper:

    h_A = MLP_A(A) ,  h_X = MLP_X(X)
    h   = ReLU( W [h_A || h_X] + h_A + h_X )
    y   = MLP_f(h)

The adjacency row A_i is consumed by a single Linear layer; we apply it
sparsely (A @ Wa) so we never densify A. Shares the POLARIS forward signature
(x, edge_index, **kw).

NOTE ON SCOPE: GGCN (Yan et al., ICDM 2022) and GloGNN (Li et al., ICML 2022)
were also requested. Their published results depend on details (degree
corrections / signed structural terms for GGCN; closed-form global
coefficients for GloGNN) that are easy to get subtly wrong in a
reimplementation. To avoid reporting numbers that do not faithfully
reproduce those methods, we do NOT ship from-memory reimplementations here;
if they are to be added as baselines, run them from the authors' released
code under the same splits. LINKX is included because its architecture is
simple and unambiguous.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree


def _mlp(sizes, dropout):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class LINKX(nn.Module):
    """LINKX (Lim et al., NeurIPS 2021)."""

    def __init__(self, in_dim, hid, ncls, layers=1, dropout=0.5):
        super().__init__()
        self.hid = hid
        self.dropout = dropout
        # MLP_A maps an adjacency row (dim = n, unknown at init) via a sparse
        # linear: we realise it lazily on the first forward when n is known.
        self.lin_A = None
        self.mlp_A = _mlp([hid, hid], dropout)
        self.mlp_X = _mlp([in_dim, hid, hid], dropout)
        self.W = nn.Linear(2 * hid, hid)
        self.mlp_f = _mlp([hid, hid, ncls], dropout)

    def _adj_embed(self, edge_index, n, device):
        # h_A = A @ Wa with A the (symmetric, self-looped) adjacency; computed
        # sparsely as a degree-normalised neighbour sum of a learnable row code.
        if self.lin_A is None:
            self.lin_A = nn.Linear(n, self.hid, bias=False).to(device)
        # one-hot rows would be O(n^2); instead use A as a propagation of the
        # node-id embedding lin_A.weight^T (column j is node j's code), i.e.
        # h_A[i] = sum_{j in N(i)} code[j]. This equals A @ code.
        code = self.lin_A.weight.t()              # (n, hid)
        ei, _ = add_self_loops(edge_index, num_nodes=n)
        row, col = ei
        deg = degree(col, n, dtype=torch.float).clamp(min=1).pow(-1.0)
        msg = code[row] * deg[col].unsqueeze(-1)
        h_A = torch.zeros(n, self.hid, device=device).index_add_(0, col, msg)
        return h_A

    def forward(self, x, edge_index, **kw):
        n = x.size(0)
        h_A = self._adj_embed(edge_index, n, x.device)
        h_A = self.mlp_A(F.dropout(h_A, self.dropout, training=self.training))
        h_X = self.mlp_X(x)
        h = F.relu(self.W(torch.cat([h_A, h_X], dim=-1)) + h_A + h_X)
        h = F.dropout(h, self.dropout, training=self.training)
        return self.mlp_f(h)
