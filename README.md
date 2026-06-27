# POLARIS — Polarity- and Selectivity-controlled Graph Attention

An attention operator for **heterophilous** graphs whose selectivity (the
entropy of its attention distribution) is **explicit and controllable**, and
whose convex self-mixing keeps it **non-expansive** with depth. A per-edge
signed coefficient lets it *repel* disagreeing neighbours; a temperature plus an
optional entropy regulariser set a graph-appropriate aggregation entropy.

Variants: **POLARIS** (signed, main), **POLARIS-U** (unsigned ablation),
**POLARIS-E** (unsigned + entropy regulariser).

## Install
```bash
cd src
pip install -e .            # or: pip install -r requirements.txt
```

## Layout
```
src/
├── polaris/
│   ├── models/        POLARIS layer + model (polaris.py); baselines GCN/GAT,
│   │                  depth baselines (GCNII/APPNP/GPRGNN/FAGCN),
│   │                  LINKX (hetero_baselines.py), SADE-GCN/SIMGA
│   │                  reimplementations (reimpl_baselines.py), and
│   │                  harness adapters for official GGCN/GloGNN
│   │                  (external_adapters.py)
│   ├── data/          dataset loading (WebKB/Actor/Planetoid/Platonov),
│   │                  edge-homophily, entropy-ceiling helpers
│   ├── metrics/       calibration (ECE/NLL/Brier/reliability) + entropy
│   ├── training/      val-selected trainer with entropy regulariser
│   └── utils.py       set_determinism() for reproducible runs
├── scripts/           runnable studies (see below)
├── results/           verified run outputs (see map below)
└── tests/             property tests (label-safety, perm-equivariance, ...)
```

## Results

All numbers in the paper come from the verified logs in `results/`:

| file | paper artifact |
|------|----------------|
| `heterophily.txt`            | Table 2 — main heterophily benchmark (WebKB/Actor/Cora) |
| `fair_harness.txt`           | Table 2 — recent baselines in one shared harness |
| `ablation_lambda_gate.txt`   | Table 4 — self-mixing / gate ablation |
| `depth.txt`                  | Table 5 — depth robustness on Wisconsin |
| `platonov.txt`               | Table 7 — large heterophilous benchmarks |
| `mechanism.txt`              | Fig. 2 — aggregation entropy vs. homophily |
| `linkx.txt`                  | LINKX vs. POLARIS (incl. Actor) |
| `significance_bias_fagcn.txt`| label-safe-bias ablation + paired Wilcoxon vs. FAGCN |
| `significance_glognn.txt`    | paired Wilcoxon vs. GloGNN (official) |
| `label_bias_coverage.txt`    | label-safe bias coverage + inference attention mass |
| `entropy_calibration.txt`    | entropy target (POLARIS-E) vs. calibration |

## Reproduce
```bash
cd src
python -m scripts.run_heterophily             # -> results/heterophily.txt
python -m scripts.run_fair_baselines          # -> results/fair_harness.txt
python -m scripts.run_lambda_gate_ablation    # -> results/ablation_lambda_gate.txt
python -m scripts.run_depth_robustness        # -> results/depth.txt
python -m scripts.run_platonov                # -> results/platonov.txt
python -m scripts.run_linkx_baseline          # -> results/linkx.txt
python -m scripts.run_mechanism_plot          # -> results/mechanism.txt (+ figure)
python -m pytest tests/ -q
```
Every script fixes the same shared protocol (hidden 64, depth 4, 120 epochs,
Adam lr 0.01 / wd 5e-4, label smoothing 0.1, dropout 0.5, validation-selected
checkpoint, 10 Geom-GCN splits; Cora single public split); only the operator
changes. `polaris.utils.set_determinism(seed)` makes a run bit-identical on a fixed
device.

GGCN and GloGNN run from the authors' official repositories through the same
harness; their dense O(n^2) variants are out of memory on the large Platonov
graphs and on Actor, marked `oom` in the tables.

## Key idea
- `H_i = -sum_j a_ij log a_ij` — **aggregation entropy** (per-node attention),
  the controllable quantity. Target `H*_i = rho * log|N(i)|`; regulariser
  `(H_i - H*_i)^2`.
- A signed per-edge coefficient `psi_ij = tanh(c.[h_i||h_j]) in (-1,1)` lets the
  operator subtract dissimilar neighbours while staying non-expansive (`|psi|<=1`).
- Calibration (ECE/NLL/Brier), not raw confidence, is a reported objective.
