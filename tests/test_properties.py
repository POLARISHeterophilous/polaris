"""Sanity tests for POLARIS theoretical properties (run with pytest, or directly)."""
from __future__ import annotations

import torch

from polaris.data import load_dataset
from polaris.models import POLARIS, POLARISLayer


def _cora():
    data, nf, ncls = load_dataset("Cora")
    return data, nf, ncls


def test_label_safety():
    """Property 2: inference output identical with/without labels passed."""
    data, nf, ncls = _cora()
    torch.manual_seed(0)
    model = POLARIS(nf, 32, ncls, num_layers=2, agg="sum").eval()
    with torch.no_grad():
        o1 = model(data.x, data.edge_index)
        o2 = model(data.x, data.edge_index, y=data.y, train_mask=data.train_mask)
    assert (o1 - o2).abs().max().item() < 1e-6


def test_disc_bias_active_in_training():
    data, nf, ncls = _cora()
    torch.manual_seed(0)
    model = POLARIS(nf, 32, ncls, num_layers=2, agg="sum").train()
    with torch.no_grad():
        t1 = model(data.x, data.edge_index, y=data.y, train_mask=data.train_mask)
        t2 = model(data.x, data.edge_index)
    assert (t1 - t2).abs().max().item() > 1e-6


def test_permutation_equivariance():
    """Property 1: permuting nodes permutes the output (eval mode)."""
    data, nf, _ = _cora()
    torch.manual_seed(0)
    layer = POLARISLayer(nf, 16, agg="sum").eval()
    n = data.x.size(0)
    perm = torch.randperm(n)
    inv = torch.argsort(perm)
    with torch.no_grad():
        out = layer(data.x, data.edge_index)
        out_perm = layer(data.x[perm], inv[data.edge_index])
    assert (out_perm[inv] - out).abs().max().item() < 1e-5


def test_entropy_regulariser_has_gradient():
    data, nf, ncls = _cora()
    torch.manual_seed(0)
    model = POLARIS(nf, 32, ncls, num_layers=2, agg="sum").train()
    _, ent_loss = model(data.x, data.edge_index, y=data.y,
                        train_mask=data.train_mask, return_entropy_loss=True)
    ent_loss.backward()
    assert model.layers[0].theta_rho.grad is not None
    assert model.layers[0].theta_rho.grad.abs().item() > 0


if __name__ == "__main__":
    for fn in [test_label_safety, test_disc_bias_active_in_training,
               test_permutation_equivariance, test_entropy_regulariser_has_gradient]:
        fn()
        print(f"PASS  {fn.__name__}")
