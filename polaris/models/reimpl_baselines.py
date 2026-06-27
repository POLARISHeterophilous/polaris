"""Reimplementations of SADE-GCN and SIMGA for comparison.

PROVENANCE
----------
Reimplemented from the papers (no public code for either):
  * SADE-GCN (Lai et al., arXiv:2305.18385): dual node/topology embeddings with
    signed asymmetric self-attention.
  * SIMGA (Liu et al., arXiv:2305.09958): decoupled MLP encoder + global
    aggregation by a SimRank similarity matrix, computed here with the standard
    iterative fixed point (the authors' release uses an external C++ tool).
Both follow the papers' described architectures, expose the POLARIS harness
signature forward(x, edge_index), and return raw logits; they are run under the
same shared protocol as every other model.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, add_self_loops, degree


# ---------------------------------------------------------------------------
# SADE-GCN: dual (node + topology) embeddings, each with SIGNED ASYMMETRIC
# self-attention; combined only at the final layer. (Lai et al., 2305.18385.)
# ---------------------------------------------------------------------------
class _SignedAsymAttn(nn.Module):
    """One signed, asymmetric self-attention layer restricted to graph edges.
    Asymmetric: separate query/key maps so score(i,j) != score(j,i). Signed:
    a tanh nonlinearity allows negative attention (high-pass) weights."""
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, h, adj_mask):
        Q, K, V = self.q(h), self.k(h), self.v(h)
        s = torch.tanh((Q @ K.t()) * self.scale)     # signed, asymmetric
        s = s * adj_mask                              # restrict to edges (+self)
        # normalise by per-row |score| mass: a signed weighted average that
        # preserves magnitude across layers (an unnormalised s @ V attenuates
        # the signal each layer and underfits at fixed epochs).
        denom = s.abs().sum(-1, keepdim=True).clamp(min=1e-6)
        return (s / denom) @ V


class SADEGCN(nn.Module):
    """Our reimplementation of SADE-GCN."""
    def __init__(self, in_dim, hid, ncls, layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.n_layers = max(layers, 1)
        # node-feature stream
        self.fc_x = nn.Linear(in_dim, hid)
        self.attn_x = nn.ModuleList([_SignedAsymAttn(hid) for _ in range(self.n_layers)])
        # topology stream: rows of the adjacency are the topology "features"
        self.fc_a = None          # lazily sized to n on first forward
        self.attn_a = nn.ModuleList([_SignedAsymAttn(hid) for _ in range(self.n_layers)])
        self.hid = hid
        self.out = nn.Linear(2 * hid, ncls)          # combine streams at the end
        self._mask = None; self._n = None

    def _adj_mask(self, edge_index, n, device):
        ei, _ = add_self_loops(edge_index, num_nodes=n)
        A = to_dense_adj(ei, max_num_nodes=n)[0].to(device)
        return (A > 0).float()

    def forward(self, x, edge_index, **kw):
        n = x.size(0)
        if self._mask is None or self._n != n or self._mask.device != x.device:
            self._mask = self._adj_mask(edge_index, n, x.device); self._n = n
            if self.fc_a is None:
                self.fc_a = nn.Linear(n, self.hid).to(x.device)
        # node-feature stream
        hx = F.relu(self.fc_x(F.dropout(x, self.dropout, training=self.training)))
        for att in self.attn_x:
            hx = F.dropout(F.relu(att(hx, self._mask)) + hx,    # residual
                           self.dropout, training=self.training)
        # topology stream (adjacency rows as input)
        ha = F.relu(self.fc_a(self._mask))
        for att in self.attn_a:
            ha = F.dropout(F.relu(att(ha, self._mask)) + ha,    # residual
                           self.dropout, training=self.training)
        return self.out(torch.cat([hx, ha], dim=-1))


# ---------------------------------------------------------------------------
# SIMGA: decoupled MLP encoder + GLOBAL aggregation by a SimRank similarity
# matrix S: out = (1-d) * MLP(X) + d * S @ MLP(X). (Liu et al., 2305.09958.)
# SimRank computed by its standard iterative fixed point (small graphs).
# ---------------------------------------------------------------------------
@torch.no_grad()
def _simrank(edge_index, n, c=0.8, iters=10, device="cpu"):
    """Iterative all-pairs SimRank S = c * P^T S P with diag reset to 1,
    P column-normalised adjacency. Faithful to the SimRank definition; this is
    the matrix SIMGA's C++ tool approximates."""
    A = to_dense_adj(edge_index, max_num_nodes=n)[0].to(device)
    A = A + A.t()
    A = (A > 0).float()
    col = A.sum(0, keepdim=True).clamp(min=1)
    P = A / col                                   # column-stochastic
    S = torch.eye(n, device=device)
    I = torch.eye(n, device=device)
    for _ in range(iters):
        S = c * (P.t() @ S @ P)
        S = S - torch.diag(torch.diag(S)) + I     # reset self-similarity to 1
    return S


class SIMGA(nn.Module):
    """Our reimplementation of SIMGA (small-graph SimRank variant)."""
    def __init__(self, in_dim, hid, ncls, layers=2, dropout=0.5, delta=0.7):
        super().__init__()
        self.dropout = dropout
        self.delta = delta            # mix between local MLP and global aggregation
        dims = [in_dim] + [hid] * (layers - 1) + [ncls]
        self.mlp = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))
        self._S = None; self._n = None

    def _encode(self, x):
        h = x
        for i, lin in enumerate(self.mlp):
            h = lin(F.dropout(h, self.dropout, training=self.training))
            if i < len(self.mlp) - 1:
                h = F.relu(h)
        return h

    def forward(self, x, edge_index, **kw):
        n = x.size(0)
        if self._S is None or self._n != n or self._S.device != x.device:
            self._S = _simrank(edge_index, n, device=x.device); self._n = n
        z = self._encode(x)                          # decoupled encoder (logits)
        return (1 - self.delta) * z + self.delta * (self._S @ z)   # global agg
