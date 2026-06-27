"""Harness adapters for external baselines (GGCN, GloGNN).

These import the AUTHORS' OFFICIAL model classes from their cloned repos and
run them through the POLARIS harness (our load_dataset / train_polaris / splits /
val-selection / cross-entropy), so the comparison is apples-to-apples: every
model sees the same data pipeline and differs only in the operator.

We do NOT reimplement the methods -- we import their nn.Module and only:
  (i)  build the adjacency the way the original repo's process.py does, from
       our edge_index;
  (ii) expose the (x, edge_index) forward signature our trainer calls;
  (iii) return raw logits (the originals end in log_softmax; we strip that so
        our CrossEntropyLoss is applied to logits exactly as for every other
        model).

Repo locations are resolved relative to the project root; set
POLARIS_EXTERNAL_DIR to override.

Provenance for the paper:
  GGCN   -- Yan et al., "Two Sides of the Same Coin", ICDM 2022
            (github.com/Yujun-Yan/Heterophily_and_oversmoothing)
  GloGNN -- Li et al., "Finding Global Homophily ...", ICML 2022
            (github.com/RecklessRonan/GloGNN)
"""
from __future__ import annotations
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch_geometric.utils import to_scipy_sparse_matrix

_DEFAULT_EXT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "external_baselines")
EXT = os.environ.get("POLARIS_EXTERNAL_DIR", os.path.abspath(_DEFAULT_EXT))

GGCN_REPO = os.path.join(EXT, "Heterophily_and_oversmoothing")
GLOGNN_REPO = os.path.join(EXT, "GloGNN", "small-scale")


# ----------------------------------------------------------------------------
def _sys_norm_adj_dense(edge_index, n, device):
    """Symmetric-normalized A+I as a DENSE tensor -- identical to the GGCN
    repo's utils.sys_normalized_adjacency (A+I, D^-1/2 (A+I) D^-1/2)."""
    A = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes=n).tocoo()
    A = A + sp.eye(n)
    rowsum = np.array(A.sum(1)).flatten()
    rowsum = (rowsum == 0) * 1.0 + rowsum
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = sp.diags(d_inv_sqrt)
    norm = D.dot(A).dot(D).tocoo()
    return torch.tensor(norm.todense(), dtype=torch.float32, device=device)


def _import_ggcn():
    import sys
    if GGCN_REPO not in sys.path:
        sys.path.insert(0, GGCN_REPO)
    from model import GGCN as _GGCN          # official class
    return _GGCN


class GGCNAdapter(nn.Module):
    """Official GGCN (dense), wrapped for the POLARIS harness. Defaults match the
    repo's GGCN(...) construction in full-supervised.py (degree correction +
    signed weights on, no decay/bn/ln)."""

    def __init__(self, in_dim, hid, ncls, layers=4, dropout=0.5,
                 decay_rate=1.0, exponent=3.0):
        super().__init__()
        _GGCN = _import_ggcn()
        # use_decay False -> coeff path is plain residual (no per-dataset decay tuning)
        self.net = _GGCN(nfeat=in_dim, nlayers=layers, nhidden=hid, nclass=ncls,
                         dropout=dropout, decay_rate=decay_rate, exponent=exponent,
                         use_degree=True, use_sign=True, use_decay=False,
                         use_sparse=False)
        self._adj = None
        self._n = None

    def forward(self, x, edge_index, **kw):
        import torch.nn.functional as F
        n = x.size(0)
        if self._adj is None or self._n != n or self._adj.device != x.device:
            self._adj = _sys_norm_adj_dense(edge_index, n, x.device)
            self._n = n
            self.net.degree_precompute = None     # recompute for this adj
        # The official net ends in F.log_softmax; our trainer applies
        # cross_entropy (which log-softmaxes internally) and argmax. Returning
        # the net's log-probabilities and reversing the log_softmax would be
        # ill-posed, so we recompute the GGCN forward verbatim but stop before
        # the final log_softmax, yielding raw logits. (Mirrors model.py forward.)
        net = self.net
        if net.use_degree and net.degree_precompute is None:
            net.precompute_degree_d(self._adj)
        h = F.dropout(x, net.dropout, training=net.training)
        layer_previous = net.act_fn(net.fcn(h))
        layer_inner = net.convs[0](h, self._adj, net.degree_precompute)
        for i, con in enumerate(net.convs[1:]):
            if net.use_norm:
                layer_inner = net.norms[i](layer_inner)
            layer_inner = net.act_fn(layer_inner)
            layer_inner = F.dropout(layer_inner, net.dropout, training=net.training)
            if i == 0:
                layer_previous = layer_inner + layer_previous
            else:
                import math
                coeff = (math.log(net.decay / (i + 2) ** net.exponent + 1)
                         if net.use_decay else 1)
                layer_previous = coeff * layer_inner + layer_previous
            layer_inner = con(layer_previous, self._adj, net.degree_precompute)
        return layer_inner                         # raw logits (no log_softmax)


# ----------------------------------------------------------------------------
def _row_norm_adj_dense(edge_index, n, device, dtype=torch.float64):
    """Row-normalized A+I as a DENSE tensor -- matches the GloGNN repo's
    normalize(adj + I) used for its `adj` input (an n x n feature matrix)."""
    A = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes=n).tocoo()
    A = (A + sp.eye(n)).tocoo()
    rowsum = np.array(A.sum(1), dtype=np.float64).flatten()
    r_inv = np.power(rowsum, -1.0)
    r_inv[np.isinf(r_inv)] = 0.0
    norm = sp.diags(r_inv).dot(A).tocoo()
    return torch.tensor(norm.todense(), dtype=dtype, device=device)


def _import_glognn():
    # main.py runs argparse + a training loop at module level, so we cannot
    # import it wholesale. We exec ONLY the file header + the MLP_NORM class
    # source (through the line just before the next top-level def), which
    # defines the official class with its own imports and nothing else.
    path = os.path.join(GLOGNN_REPO, "main.py")
    lines = open(path).read().splitlines()
    end = next(i for i, l in enumerate(lines) if l.startswith("def encode_onehot"))
    src = "\n".join(lines[:end])
    ns = {}
    # main.py header calls torch.set_default_dtype(float64) at module level;
    # exec it, then RESTORE the default so other models (POLARIS) stay float32.
    prev = torch.get_default_dtype()
    try:
        exec(compile(src, path, "exec"), ns)
    finally:
        torch.set_default_dtype(prev)
    return ns["MLP_NORM"]


class GloGNNAdapter(nn.Module):
    """Official GloGNN (MLP_NORM), wrapped for the POLARIS harness. Structural
    coefficients are left at the repo's library defaults (alpha 0.1, beta 0.1,
    gamma 0.2, delta 1.0, orders 2) -- the method, not per-dataset tuning;
    norm_layers is set to our shared depth. The model consumes a dense,
    row-normalized A+I and is float64 internally (as in the repo)."""

    def __init__(self, in_dim, hid, ncls, n_nodes, layers=4, dropout=0.5):
        super().__init__()
        MLP_NORM = _import_glognn()
        self.net = MLP_NORM(
            nnodes=n_nodes, nfeat=in_dim, nhid=hid, nclass=ncls, dropout=dropout,
            alpha=0.1, beta=0.1, gamma=0.2, delta=1.0,
            norm_func_id=2, norm_layers=layers, orders=2, orders_func_id=3,
            cuda=False).double()
        self._adj = None
        self._n = None

    def forward(self, x, edge_index, **kw):
        import torch.nn.functional as F
        n = x.size(0)
        # MLP_NORM stores eye/coeff tensors as plain attributes (not buffers),
        # so .to(device) does not move them; place them on the input device.
        for attr in ("class_eye", "nodes_eye", "alpha", "beta", "gamma", "delta"):
            t = getattr(self.net, attr, None)
            if torch.is_tensor(t) and t.device != x.device:
                setattr(self.net, attr, t.to(x.device))
        if self._adj is None or self._n != n or self._adj.device != x.device:
            self._adj = _row_norm_adj_dense(edge_index, n, x.device)
            self._n = n
        # The official forward ends in F.log_softmax; our trainer applies
        # cross_entropy to logits. Temporarily make log_softmax an identity so
        # the net returns its raw logits unchanged (everything else verbatim).
        orig = F.log_softmax
        F.log_softmax = lambda inp, *a, **k: inp
        try:
            out = self.net(x.double(), self._adj)
        finally:
            F.log_softmax = orig
        return out.float()                         # raw logits
