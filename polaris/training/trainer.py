"""Training loop with validation-based model selection and the controlled-entropy
regulariser.

Protocol (clean, no test leakage):
    train_mask -> loss / gradients
    val_mask   -> checkpoint selection (strict >, ties keep the earlier epoch)
    test_mask  -> final reported metrics only
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from polaris.metrics.calibration import (
    expected_calibration_error, negative_log_likelihood, brier_score,
)
from polaris.metrics.entropy import predictive_entropy


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    beta: float = 0.0           # strength of the aggregation-entropy regulariser
    use_labels: bool = True     # pass labels for the training-only discriminative bias


def _auc(logits, y, ncls):
    import warnings
    p = F.softmax(logits, dim=-1).cpu().numpy()
    yt = y.cpu().numpy()
    try:
        with warnings.catch_warnings():     # silence "only one class in y_true"
            warnings.simplefilter("ignore")
            if ncls == 2:                   # binary: AUC on positive-class prob
                return roc_auc_score(yt, p[:, 1])
            return roc_auc_score(yt, p, multi_class="ovr",
                                 average="macro", labels=list(range(ncls)))
    except ValueError:
        return float("nan")


@torch.no_grad()
def evaluate(model, data, ncls, mask_name: str = "test_mask") -> dict:
    model.eval()
    logits = model(data.x, data.edge_index)
    m = getattr(data, mask_name)
    lt, yt = logits[m], data.y[m]
    return {
        "acc": lt.argmax(-1).eq(yt).float().mean().item(),
        "f1": f1_score(yt.cpu(), lt.argmax(-1).cpu(), average="macro"),
        "auc": _auc(lt, yt, ncls),
        "nll": negative_log_likelihood(lt, yt),
        "brier": brier_score(lt, yt),
        "ece": expected_calibration_error(lt, yt),
        "pred_entropy": predictive_entropy(lt),
    }


def train_polaris(model, data, ncls, cfg: TrainConfig = TrainConfig()) -> dict:
    """Train and return test metrics for the val-selected checkpoint, plus the
    measured per-layer aggregation entropy."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val, best_state = -1.0, None
    is_dcm = hasattr(model, "layers")

    for _ in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        if is_dcm and cfg.beta > 0:
            out, ent_loss = model(
                data.x, data.edge_index,
                y=data.y if cfg.use_labels else None,
                train_mask=data.train_mask if cfg.use_labels else None,
                return_entropy_loss=True,
            )
        else:
            out = model(
                data.x, data.edge_index,
                y=data.y if cfg.use_labels else None,
                train_mask=data.train_mask if cfg.use_labels else None,
            )
            ent_loss = torch.zeros((), device=out.device)
        ce = F.cross_entropy(out[data.train_mask], data.y[data.train_mask],
                             label_smoothing=cfg.label_smoothing)
        (ce + cfg.beta * ent_loss).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_acc = model(data.x, data.edge_index).argmax(-1)[data.val_mask] \
                .eq(data.y[data.val_mask]).float().mean().item()
        if val_acc > best_val:                       # strict: ties keep earlier epoch
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    res = evaluate(model, data, ncls, "test_mask")
    res["val_acc"] = best_val
    if is_dcm:
        model.eval()
        with torch.no_grad():
            _, ents = model(data.x, data.edge_index, return_entropy=True)
        res["agg_entropy"] = ents.mean().item()
        res["agg_entropy_layers"] = ents.tolist()
    return res
