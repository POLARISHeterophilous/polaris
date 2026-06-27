"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def enable_cublas_determinism() -> None:
    """Set the env var deterministic cuBLAS matmul needs. Must run BEFORE the
    first CUDA matmul (the cuBLAS handle reads it once at creation), so call it
    at process start, before any model touches the GPU."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_determinism(seed: int, warn_only: bool = True) -> None:
    """Seed every RNG and switch on deterministic kernels.

    Same seed + same device => bit-identical results. With deterministic
    algorithms on, the CPU and GPU paths still differ at the kernel level, but
    each backend is reproducible run to run. ``warn_only=True`` lets the few ops
    without a deterministic CUDA kernel fall back (with a warning) instead of
    raising, so a sweep never crashes mid-run; set it False to hard-fail on any
    nondeterministic op.
    """
    enable_cublas_determinism()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
