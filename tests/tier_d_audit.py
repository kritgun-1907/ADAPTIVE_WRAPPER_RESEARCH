"""
Audit of everything Tier D added this session.

Two bugs reached reported numbers before anyone checked them:

  1. build_sparse_orthogonal's offset stride collapsed to 0 at density 0.05, so
     every feature landed in ONE particle. Mean density still read 0.05, which
     is exactly why it survived inspection -- the summary statistic was right
     while the structure was wrong.
  2. build_importance_guided_positions re-seeded at ~n_features/3 (~33% density)
     regardless of the swarm it was rejoining, undoing sparsity mid-run.

Both were invisible to "does it run" and to spot-checking means. So this file
asserts STRUCTURAL invariants, not just plausible-looking outputs. Every check
prints PASS/FAIL and the script exits non-zero if any fail.

Run:  python tests/tier_d_audit.py
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main():
    from orpsoc.utils import (build_orthogonal_positions,
                              build_importance_guided_positions,
                              walk_forward_folds, sigmoid)
    from experiments.dimensionality.tier_d_curve import build_sparse_orthogonal
    from experiments.dimensionality.tier_d_comparators import (select_mrmr, select_lasso,
                                                select_rfe, signal_recall,
                                                score_selection)
    rng = np.random.RandomState(0)

    print("\nA. sparse-orthogonal initialisation")
    for dens in (0.05, 0.10, 0.5):        # 1/n_particles floor = 0.05 at 20
        g = build_sparse_orthogonal(20, 2000, dens, rng)
        per = g.sum(1)
        check(f"density {dens}: every particle non-empty", per.min() > 0,
              f"min={int(per.min())} max={int(per.max())}")
        check(f"density {dens}: realised ≈ requested",
              abs(g.mean() - dens) < max(0.02, 0.25 * dens),
              f"realised={g.mean():.3f}")
        check(f"density {dens}: load balanced across particles",
              per.max() <= 2 * max(per.mean(), 1),
              f"spread {int(per.min())}..{int(per.max())}")

    print("\nB. importance-guided re-seeding respects the swarm density")
    prob = rng.rand(2000); prob /= prob.sum()
    for dens in (0.05, 0.10):
        g = build_importance_guided_positions(10, 2000, prob, min_f=3,
                                              seed=1, density=dens)
        check(f"density {dens}: re-seed density ≈ requested",
              abs(g.mean() - dens) < max(0.03, 0.5 * dens),
              f"realised={g.mean():.3f}")
    g_old = build_importance_guided_positions(10, 2000, prob, min_f=3, seed=1)
    check("density=None preserves the published ~1/3 target",
          0.2 < g_old.mean() < 0.45, f"realised={g_old.mean():.3f}")

    print("\nC. transfer bias moves the zero-velocity fixed point")
    for b, want in ((0.0, 0.5), (-3.0, 0.047), (-4.0, 0.018)):
        check(f"bias {b}: sigmoid(0+bias) ≈ {want}",
              abs(float(sigmoid(np.array([0.0]))[0] if b == 0 else
                        sigmoid(np.array([b]))[0]) - want) < 0.01,
              f"{float(sigmoid(np.array([float(b)]))[0]):.3f}")

    print("\nD. comparators honour the matched budget")
    Xy = None
    for p in ("data/hd_ar1_p200.pkl", "data/hd_regime_switch_p200.pkl"):
        if os.path.exists(p):
            with open(p, "rb") as fh:
                d = pickle.load(fh)
            Xy = (d["X"], d["y"]); break
    if Xy is None:
        check("dataset available for comparator checks", False, "none found")
    else:
        X, y = Xy
        feat = list(X.columns)
        X_tr, y_tr = X.iloc[:600], y.iloc[:600]
        for k in (5, 25, 60):
            for nm, fn in (("mrmr", select_mrmr), ("lasso", select_lasso),
                           ("rfe", select_rfe)):
                sel = fn(X_tr, y_tr, feat, k_target=k, seed=0)["selected"]
                ok = (len(sel) == k) if nm != "lasso" else (abs(len(sel) - k) <= max(2, 0.25 * k))
                check(f"{nm} k_target={k}", ok, f"got {len(sel)}")
                check(f"{nm} k={k}: no duplicate features", len(set(sel)) == len(sel))
                check(f"{nm} k={k}: all selected exist in X", set(sel) <= set(feat))

    print("\nE. ground-truth recall is only claimed where truth exists")
    check("signal_recall counts the 5 synthetic signals",
          signal_recall([f"signal_{i}" for i in range(5)]) == 1.0)
    check("signal_recall returns 0.0 (not None) on a real-feature list",
          signal_recall(["mkt_ret_1d", "Agric_lvl"]) == 0.0,
          "callers MUST guard on dataset type")

    print("\nF. walk-forward folds are causal")
    if Xy is not None:
        X, y = Xy
        folds = walk_forward_folds(X, y, n_splits=8)
        check("8 folds returned", len(folds) == 8)
        ok_gap, ok_grow = True, True
        prev_end = -1
        for X_tr, y_tr, X_te, y_te, tr_end in folds:
            if not (X_te.index.min() > X_tr.index.max()):
                ok_gap = False
            if tr_end <= prev_end:
                ok_grow = False
            prev_end = tr_end
        check("test window always strictly after train window", ok_gap)
        check("training window grows monotonically", ok_grow)

    print("\nG. real high-dim datasets")
    for name in ("ff100", "ff49ind"):
        f = f"data/{name}.pkl"
        if not os.path.exists(f):
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        X, y = d["X"], d["y"]
        check(f"{name}: col 0 is a return series (HMM driver)",
              X.columns[0] == "mkt_ret_1d", X.columns[0])
        check(f"{name}: all finite", bool(np.isfinite(X.to_numpy()).all()))
        check(f"{name}: target not degenerate", 0.2 < float(y.mean()) < 0.8,
              f"y-mean={float(y.mean()):.3f}")
        check(f"{name}: rows align", len(X) == len(y))

    print("\nH. industry ladder holds the target fixed across widths")
    inds = sorted([f for f in os.listdir("data")
                   if f.startswith("ind") and f.endswith(".pkl")
                   and not f.startswith("raw_")])
    ymeans, lens = {}, {}
    for f in inds:
        with open(f"data/{f}", "rb") as fh:
            d = pickle.load(fh)
        ymeans[f] = round(float(d["y"].mean()), 6)
        lens[f] = len(d["y"])
    if len(ymeans) >= 2:
        check("target identical across all widths",
              len(set(ymeans.values())) == 1, f"{sorted(set(ymeans.values()))}")
        check("row count identical across all widths",
              len(set(lens.values())) == 1, f"{sorted(set(lens.values()))}")
    else:
        print("  (skip) fewer than 2 industry datasets present locally")

    print()
    if FAILS:
        print(f"  {len(FAILS)} CHECK(S) FAILED:")
        for f in FAILS:
            print(f"    - {f}")
        raise SystemExit(1)
    print("  ALL AUDIT CHECKS PASSED")


if __name__ == "__main__":
    main()
