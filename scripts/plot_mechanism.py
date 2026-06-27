#!/usr/bin/env python
"""Render Figure 2 (mechanism: aggregation entropy vs homophily) from the
verified run values in results/mechanism.txt. Standalone so the figure can be
regenerated without re-training; numbers are the measured means.

Usage:  python -m scripts.plot_mechanism
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (dataset, edge-homophily h, normalised aggregation entropy H_i/log|N~(i)|)
# verified means from results/mechanism.txt
DATA = [
    ("Texas",     0.108, 0.904),
    ("Cornell",   0.131, 0.843),
    ("Wisconsin", 0.196, 0.933),
    ("Actor",     0.219, 0.993),
    ("Citeseer",  0.736, 0.996),
    ("Pubmed",    0.802, 1.000),
    ("Cora",      0.810, 0.996),
]
# per-point label placement (dx, dy in points; ha) to avoid collisions
LABELS = {
    "Texas":     (7, -3, "left"),
    "Cornell":   (7, 2, "left"),
    "Wisconsin": (7, 3, "left"),
    "Actor":     (7, 3, "left"),
    "Citeseer":  (-10, 16, "right"),
    "Pubmed":    (-8, -14, "right"),
    "Cora":      (2, 12, "center"),
}
# datasets whose label needs a thin leader line to its point
LEADER = {"Citeseer", "Pubmed", "Cora"}
HET = "#c0392b"     # heterophilous points
HOM = "#2471a3"     # homophilous points


def main():
    hs = np.array([d[1] for d in DATA])
    nH = np.array([d[2] for d in DATA])
    r = np.corrcoef(hs, nH)[0, 1]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))

    # regime shading
    ax.axvspan(0.0, 0.5, color=HET, alpha=0.05, zorder=0)
    ax.axvspan(0.5, 1.0, color=HOM, alpha=0.05, zorder=0)
    ax.axvline(0.5, color="grey", ls=":", lw=1, zorder=1)
    ax.text(0.25, 0.818, "heterophilous", color=HET, ha="center",
            fontsize=9, style="italic")
    ax.text(0.70, 0.818, "homophilous", color=HOM, ha="center",
            fontsize=9, style="italic")

    # trend line
    a, b = np.polyfit(hs, nH, 1)
    xx = np.array([0.05, 0.87])
    ax.plot(xx, a * xx + b, color="grey", lw=1.3, ls="--", zorder=2,
            label=f"linear fit ($r={r:.2f}$)")

    for (dn, h, y) in DATA:
        c = HET if h < 0.5 else HOM
        ax.scatter(h, y, s=70, color=c, edgecolor="black", linewidth=0.6,
                   zorder=3)
        dx, dy, ha = LABELS[dn]
        arrow = dict(arrowstyle="-", color="grey", lw=0.6,
                     shrinkA=1, shrinkB=3) if dn in LEADER else None
        ax.annotate(dn, (h, y), fontsize=9, ha=ha, zorder=4,
                    xytext=(dx, dy), textcoords="offset points",
                    arrowprops=arrow)

    ax.set_xlabel("edge homophily $h$", fontsize=11)
    ax.set_ylabel(r"normalised aggregation entropy"
                  "\n" r"$H_i/\log|\tilde{\mathcal{N}}(i)|$", fontsize=11)
    ax.set_xlim(0.0, 0.88)
    ax.set_ylim(0.805, 1.04)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(loc="center right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    os.makedirs("../paper/figures", exist_ok=True)
    out = "../paper/figures/entropy_vs_homophily.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Pearson r = {r:.3f};  saved -> {out}")


if __name__ == "__main__":
    main()
