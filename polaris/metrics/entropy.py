"""Entropy metrics.

We carefully distinguish the two quantities the reviews said were conflated:

  - predictive entropy  : entropy of the OUTPUT softmax distribution p(y|x).
                          A property of the classifier head, NOT what message
                          passing controls. Low value == confident (possibly
                          over-confident) predictions.

  - aggregation entropy : H_i = -sum_j alpha_ij log alpha_ij, the entropy of the
                          per-node ATTENTION distribution over neighbours. This is
                          the quantity POLARIS is designed to control.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def predictive_entropy(logits: torch.Tensor) -> float:
    p = F.softmax(logits, dim=-1)
    return (-(p * (p + 1e-12).log()).sum(-1)).mean().item()


@torch.no_grad()
def mean_aggregation_entropy(model, x, edge_index) -> list[float]:
    """Per-layer mean aggregation entropy H_i, evaluated in eval mode."""
    was_training = model.training
    model.eval()
    _, ents = model(x, edge_index, return_entropy=True)
    if was_training:
        model.train()
    return ents.tolist()
