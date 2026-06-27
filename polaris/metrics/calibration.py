"""Calibration metrics.

The reframing of this work: a confident model is not automatically a good one.
We therefore evaluate *calibration* -- the agreement between predicted confidence
and empirical accuracy -- rather than treating low predictive entropy as the goal.

Provided: Expected Calibration Error (ECE), Negative Log-Likelihood (NLL),
multi-class Brier score, and the reliability curve used for reliability diagrams.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def expected_calibration_error(logits: torch.Tensor, y: torch.Tensor,
                               n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins."""
    p = F.softmax(logits, dim=-1)
    conf, pred = p.max(-1)
    acc = pred.eq(y).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros((), device=logits.device)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()
    return ece.item()


@torch.no_grad()
def negative_log_likelihood(logits: torch.Tensor, y: torch.Tensor) -> float:
    return F.cross_entropy(logits, y).item()


@torch.no_grad()
def brier_score(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Multi-class Brier score: mean squared error between softmax and one-hot."""
    p = F.softmax(logits, dim=-1)
    onehot = F.one_hot(y, num_classes=p.size(-1)).float()
    return ((p - onehot) ** 2).sum(-1).mean().item()


@torch.no_grad()
def reliability_curve(logits: torch.Tensor, y: torch.Tensor, n_bins: int = 15):
    """Return (bin_confidence, bin_accuracy, bin_weight) for reliability diagrams."""
    p = F.softmax(logits, dim=-1)
    conf, pred = p.max(-1)
    acc = pred.eq(y).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    confs, accs, weights = [], [], []
    n = y.numel()
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            confs.append(conf[m].mean().item())
            accs.append(acc[m].mean().item())
            weights.append(m.float().sum().item() / n)
        else:
            confs.append((bins[i] + bins[i + 1]).item() / 2)
            accs.append(float("nan"))
            weights.append(0.0)
    return confs, accs, weights
