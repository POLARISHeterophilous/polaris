"""Additional heterophily baselines requested in review (Q5): H2GCN, MixHop,
and LSGNN.

All three share the POLARIS forward signature ``forward(x, edge_index, **kw)`` and
are trained by the SAME harness, protocol, seeds, and splits as every other
model (hidden 64, depth 4, 120 epochs, Adam lr 0.01 / wd 5e-4, label smoothing
0.1, dropout 0.5, 10 Geom-GCN splits). They are faithful but compact
reimplementations of the published operators; like our other reimplementations
they may differ from the authors' tuned code.

References
----------
H2GCN  -- Zhu et al., NeurIPS 2020, "Beyond Homophily in Graph Neural Networks".
          Ego/neighbour separation, higher-order (2-hop) neighbourhoods, and a
          jumping-knowledge concat of every intermediate representation.
MixHop -- Abu-El-Haija et al., ICML 2019, "MixHop: Higher-Order Graph
          Convolutional Architectures via Sparsified Neighborhood Mixing".
          Each layer concatenates features propagated by several adjacency
          powers A^0, A^1, A^2.
LSGNN  -- Chen et al., IJCAI 2023, "LSGNN: Towards General GNN in Node
          Classification by Local Similarity". Per-node weighted fusion of
          multi-hop representations, with weights from a local-similarity score,
          on top of an initial-residual difference connection (IRDC).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree


def _sym_norm(edge_index, n, device, add_loops=True):
    """Return (row, col, w) of the symmetrically normalised adjacency
    D^-1/2 (A[+I]) D^-1/2 as edge weights."""
    ei = edge_index
    if add_loops:
        ei, _ = add_self_loops(ei, num_nodes=n)
    row, col = ei
    deg = degree(col, n, dtype=torch.float).clamp(min=1.0)
    dinv = deg.pow(-0.5)
    w = dinv[row] * dinv[col]
    return row, col, w


def _propagate(h, row, col, w, n):
    """One sparse hop: out[i] = sum_{j} w_ij h[j] (j=row -> i=col)."""
    msg = h[row] * w.unsqueeze(-1)
    return torch.zeros(n, h.size(-1), device=h.device).index_add_(0, col, msg)


# --------------------------------------------------------------------------- #
# MixHop                                                                       #
# --------------------------------------------------------------------------- #
class MixHop(nn.Module):
    """MixHop (Abu-El-Haija et al., ICML 2019).

    Each layer: for powers p in {0,1,2}, propagate the current representation p
    times through the normalised adjacency, pass each through its own linear map,
    and concatenate. A final linear maps to classes.
    """

    def __init__(self, in_dim, hid, ncls, layers=2, dropout=0.5, powers=(0, 1, 2)):
        super().__init__()
        self.powers = powers
        self.dropout = dropout
        self.lins = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.lins.append(nn.ModuleList(nn.Linear(d, hid) for _ in powers))
            d = hid * len(powers)
        self.out = nn.Linear(d, ncls)

    def forward(self, x, edge_index, **kw):
        n = x.size(0)
        row, col, w = _sym_norm(edge_index, n, x.device)
        h = x
        for layer in self.lins:
            outs = []
            hp = h
            for p, lin in zip(self.powers, layer):
                hpow = h
                for _ in range(p):
                    hpow = _propagate(hpow, row, col, w, n)
                outs.append(lin(hpow))
            h = F.relu(torch.cat(outs, dim=-1))
            h = F.dropout(h, self.dropout, training=self.training)
        return self.out(h)


# --------------------------------------------------------------------------- #
# H2GCN                                                                        #
# --------------------------------------------------------------------------- #
class H2GCN(nn.Module):
    """H2GCN (Zhu et al., NeurIPS 2020).

    Design 1: separate ego (self) and neighbour embeddings (no self-loops in
    propagation). Design 2: 1-hop and 2-hop neighbourhoods, concatenated each
    round. Design 3: jumping-knowledge -- concatenate the initial embedding and
    every round's output, then classify. Non-parametric propagation (no weights
    in the rounds), as in the paper.
    """

    def __init__(self, in_dim, hid, ncls, layers=2, dropout=0.5):
        super().__init__()
        self.rounds = layers
        self.dropout = dropout
        self.embed = nn.Linear(in_dim, hid)
        # Final dim: initial (hid) + each round doubles via {1-hop, 2-hop} concat.
        final = hid * (2 ** (layers + 1) - 1)
        self.out = nn.Linear(final, ncls)

    def forward(self, x, edge_index, **kw):
        n = x.size(0)
        # Neighbour adjacency WITHOUT self-loops (ego/neighbour separation).
        row, col, w = _sym_norm(edge_index, n, x.device, add_loops=False)
        h = F.relu(self.embed(x))
        h = F.dropout(h, self.dropout, training=self.training)
        reps = [h]
        for _ in range(self.rounds):
            h1 = _propagate(h, row, col, w, n)                 # 1-hop
            h2 = _propagate(h1, row, col, w, n)                # 2-hop
            h = torch.cat([h1, h2], dim=-1)
            reps.append(h)
        h = torch.cat(reps, dim=-1)
        h = F.dropout(h, self.dropout, training=self.training)
        return self.out(h)


# --------------------------------------------------------------------------- #
# LSGNN                                                                        #
# --------------------------------------------------------------------------- #
class LSGNN(nn.Module):
    """LSGNN (Chen et al., IJCAI 2023) -- compact faithful version.

    (1) IRDC-style multi-hop features: H_0 = MLP(X); H_k = prop(H_{k-1}) plus an
        initial-residual difference term that subtracts already-used signal,
        H_k = prop(H_{k-1}) + (X_emb - H_{k-1}) * beta.
    (2) Local similarity: for each node a scalar s_i = mean cosine similarity to
        its neighbours, mapped to per-node, per-hop fusion weights so that
        low-similarity (heterophilous) nodes can up-weight farther / ego hops.
    (3) Weighted fusion of the hop representations -> classifier.
    """

    def __init__(self, in_dim, hid, ncls, layers=3, dropout=0.5, irdc_beta=0.5):
        super().__init__()
        self.K = layers
        self.dropout = dropout
        self.beta = irdc_beta
        self.embed = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                   nn.Dropout(dropout))
        # Per-node fusion weights over (K+1) hops from the local-similarity scalar.
        self.fuse = nn.Linear(1, self.K + 1)
        self.out = nn.Linear(hid, ncls)

    @torch.no_grad()
    def _local_sim(self, x, row, col, n):
        # Mean cosine similarity of each node to its (raw-feature) neighbours.
        xn = F.normalize(x, dim=-1)
        sim = (xn[row] * xn[col]).sum(-1)                      # per-edge cosine
        s = torch.zeros(n, device=x.device).index_add_(0, col, sim)
        cnt = torch.zeros(n, device=x.device).index_add_(
            0, col, torch.ones_like(sim))
        return (s / cnt.clamp(min=1.0)).unsqueeze(-1)         # (n,1)

    def forward(self, x, edge_index, **kw):
        n = x.size(0)
        row, col, w = _sym_norm(edge_index, n, x.device)
        s = self._local_sim(x, row, col, n)
        emb = self.embed(x)
        reps = [emb]
        h = emb
        for _ in range(self.K):
            h = _propagate(h, row, col, w, n) + self.beta * (emb - h)   # IRDC
            reps.append(h)
        # Per-node softmax fusion weights over hops, from local similarity.
        alpha = torch.softmax(self.fuse(s), dim=-1)            # (n, K+1)
        stacked = torch.stack(reps, dim=1)                    # (n, K+1, hid)
        fused = (alpha.unsqueeze(-1) * stacked).sum(1)        # (n, hid)
        fused = F.dropout(fused, self.dropout, training=self.training)
        return self.out(fused)
