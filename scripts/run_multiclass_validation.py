#!/usr/bin/env python
"""Validate the formal multi-class theory (App. A): the symmetric K-class
simplex-ETF model predicts the averaged neighbour signal carries
gamma_K(h) = (Kh-1)/(K-1) of mu_c, so the sign flips at the chance level
h = 1/K (not 1/2), and self-mixing with lambda >= 1/K restores the correct
class. We check this three ways.

  synthA1 : Monte-Carlo the model; measured aggregate coefficient vs gamma_K(h),
            and the empirical sign-flip homophily vs 1/K, for K in {2,3,5,10}.
  synthA2 : train POLARIS-U (averaging) vs POLARIS on synthetic K=5 graphs across h;
            POLARIS-U should collapse to chance below h=1/K, POLARIS should hold.
  realK   : tabulate (K, h, 1/K, regime) for the real datasets, to relate the
            threshold to where POLARIS actually wins.

Same POLARIS protocol/seeds as elsewhere. Appends to
results/appendix/multiclass_validation.txt.
"""
from __future__ import annotations
import os, numpy as np, torch
from polaris.data import load_dataset, edge_homophily
from polaris.models import POLARIS
from polaris.training import train_polaris, TrainConfig

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "results", "appendix", "multiclass_validation.txt")


def log(m=""):
    print(m, flush=True)
    with open(OUT, "a") as f:
        f.write(m + "\n")


def etf_means(K, d, s=1.0, seed=0):
    """K simplex-ETF means in R^d: ||mu_k||^2=s, <mu_k,mu_l>=-s/(K-1), sum=0."""
    base = np.eye(K) - 1.0 / K                       # rows e_k - 1/K, sum to 0
    base = base / np.linalg.norm(base, axis=1, keepdims=True)   # unit simplex ETF
    M = np.zeros((K, d))
    M[:, :K] = base
    return M * np.sqrt(s)


def gen_graph(n, K, h, d=16, deg=10, sigma=1.0, seed=0):
    """Symmetric K-class model (Def. A2): features mu_{c}+noise; each neighbour
    same-class w.p. h, else uniform over the other K-1 classes."""
    rng = np.random.default_rng(seed)
    mu = etf_means(K, d, s=1.0)
    y = rng.integers(0, K, size=n)
    x = mu[y] + sigma * rng.standard_normal((n, d))
    by_class = [np.where(y == c)[0] for c in range(K)]
    src, dst = [], []
    for i in range(n):
        c = y[i]
        for _ in range(deg):
            if rng.random() < h:
                cls = c
            else:
                others = [k for k in range(K) if k != c]
                cls = others[rng.integers(0, K - 1)]
            pool = by_class[cls]
            if len(pool) == 0:
                continue
            j = pool[rng.integers(0, len(pool))]
            src.append(j); dst.append(i)          # message j -> i
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return x, y, edge_index, mu


def aggregate_coeff(x, y, edge_index, mu, K):
    """Mean over class-c nodes of <m_i, mu_c>/s, m_i = uniform neighbour mean."""
    n = x.shape[0]
    src = edge_index[0].numpy(); dst = edge_index[1].numpy()
    s = (np.linalg.norm(mu[0]) ** 2)
    agg = np.zeros_like(x); cnt = np.zeros(n)
    np.add.at(agg, dst, x[src]); np.add.at(cnt, dst, 1.0)
    cnt = np.maximum(cnt, 1.0)
    agg = agg / cnt[:, None]
    coeff = (agg * mu[y]).sum(1) / s
    return float(coeff.mean())


def run_synthA1():
    log("\n" + "=" * 68)
    log("synthA1  measured aggregate coeff vs gamma_K(h)=(Kh-1)/(K-1)")
    log("=" * 68)
    for K in [2, 3, 5, 10]:
        log(f"\nK={K}  (theory sign-flip at 1/K={1.0/K:.3f})")
        log(f"  {'h':>6}{'measured':>10}{'gamma_K':>10}")
        hs = sorted(set(np.round(np.linspace(0.0, 0.6, 13), 3)) | {round(1.0/K, 3)})
        prev_h, prev_c, flip = None, None, None
        for h in hs:
            x, y, ei, mu = gen_graph(2000, K, h, seed=0)
            c = aggregate_coeff(x, y, ei, mu, K)
            g = (K * h - 1) / (K - 1)
            if prev_c is not None and prev_c < 0 <= c:
                flip = prev_h + (0 - prev_c) * (h - prev_h) / (c - prev_c)
            prev_h, prev_c = h, c
            if h in (0.0, round(1.0/K, 3)) or abs(h - 0.3) < 1e-9 or abs(h-0.6) < 1e-9:
                log(f"  {h:>6.3f}{c:>10.3f}{g:>10.3f}")
        if flip is not None:
            log(f"  -> empirical sign-flip at h={flip:.3f} (theory 1/K={1.0/K:.3f})")


def run_synthA2():
    log("\n" + "=" * 68)
    log("synthA2  POLARIS-U (averaging) vs POLARIS on synthetic K=5 graphs (1/K=0.20)")
    log("         test accuracy %, mean over 3 seeds")
    log("=" * 68)
    K = 5
    log(f"  {'h':>6}{'POLARIS-U':>9}{'POLARIS':>9}{'chance':>8}")
    for h in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        au, ag = [], []
        for seed in range(3):
            x, y, ei, _ = gen_graph(1500, K, h, seed=seed)
            n = x.shape[0]
            idx = np.random.default_rng(seed).permutation(n)
            tr = torch.zeros(n, dtype=torch.bool); va = tr.clone(); te = tr.clone()
            tr[idx[:n//2]] = True; va[idx[n//2:int(.7*n)]] = True; te[idx[int(.7*n):]] = True
            import types
            data = types.SimpleNamespace(
                x=torch.tensor(x, dtype=torch.float), y=torch.tensor(y),
                edge_index=ei, train_mask=tr, val_mask=va, test_mask=te)
            for store, signed in [(au, False), (ag, True)]:
                torch.manual_seed(seed); np.random.seed(seed)
                m = POLARIS(x.shape[1], 64, K, 4, agg="sum", signed=signed)
                r = train_polaris(m, data, K, TrainConfig(epochs=120))
                store.append(r["acc"])
        log(f"  {h:>6.2f}{np.mean(au)*100:>9.1f}{np.mean(ag)*100:>9.1f}"
            f"{100.0/K:>8.1f}")


def run_realK():
    log("\n" + "=" * 68)
    log("realK  threshold vs regime on real datasets (h, K, 1/K)")
    log("=" * 68)
    log(f"  {'dataset':<14}{'K':>4}{'h':>8}{'1/K':>8}{'h<1/K?':>8}")
    sets = ["Texas", "Cornell", "Wisconsin", "Actor", "Cora",
            "Roman-empire", "Amazon-ratings", "Minesweeper"]
    for d in sets:
        try:
            data, nf, ncls = load_dataset(d, split=0)
            h = edge_homophily(data)
            log(f"  {d:<14}{ncls:>4}{h:>8.3f}{1.0/ncls:>8.3f}"
                f"{'yes' if h < 1.0/ncls else 'no':>8}")
        except Exception as e:
            log(f"  {d:<14} [load error: {e}]")


GROUPS = {"synthA1": run_synthA1, "synthA2": run_synthA2, "realK": run_realK}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(GROUPS), default=list(GROUPS))
    args = ap.parse_args()
    torch.set_num_threads(4)
    log("\n##### MULTI-CLASS THEORY VALIDATION #####")
    for g in args.only:
        try:
            GROUPS[g]()
        except Exception as e:
            import traceback; log(f"\n[ERROR {g}] {e}\n{traceback.format_exc()}")
    log("\n##### DONE #####")
