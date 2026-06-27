"""Polarity- and Selectivity-controlled Graph Attention (POLARIS) — entropy-controlled message passing for GNNs.

Public API:
    POLARISLayer, POLARIS            -- the model (models/)
    train_polaris               -- training loop with optional entropy regulariser (training/)
    expected_calibration_error, predictive_entropy, aggregation_entropy, ...  (metrics/)
"""
from polaris.models.polaris import POLARIS, POLARISLayer
from polaris.training.trainer import train_polaris, TrainConfig
from polaris.metrics.calibration import (
    expected_calibration_error,
    negative_log_likelihood,
    brier_score,
    reliability_curve,
)
from polaris.metrics.entropy import predictive_entropy, mean_aggregation_entropy

__all__ = [
    "POLARIS",
    "POLARISLayer",
    "train_polaris",
    "TrainConfig",
    "expected_calibration_error",
    "negative_log_likelihood",
    "brier_score",
    "reliability_curve",
    "predictive_entropy",
    "mean_aggregation_entropy",
]

__version__ = "0.1.0"
