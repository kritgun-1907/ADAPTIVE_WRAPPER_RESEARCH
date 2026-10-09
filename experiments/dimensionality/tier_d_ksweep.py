"""
Tier D follow-up: does a genuinely COMPACT selection beat baseline at high p?

The main Tier D run cannot answer this. It gives every comparator the budget
OrPSOC chose (`k_target`), and at p>=1000 OrPSOC degenerates to ~48% density --
roughly where random initialisation puts it. So mRMR/LASSO/RFE were forced to
return ~960-feature subsets: they ranked the 5 true signals at the top (measured
recall 0.77-1.00, see tier_d_diagnose.py) and then padded with ~955 noise
features. No method was ever evaluated at a small k.

That leaves the interesting question open. The dimensionality hypothesis says a
boosted classifier stops absorbing the selection problem as p grows. Testing it
needs a selector that ACTUALLY COMPACTS. A filter at fixed small k does; PSO
under this configuration does not.

This sweeps k directly, with no PSO in the loop, so it is cheap: mRMR is a
ranking, and scoring is one LightGBM fit per (k, fold).

Reports, for each k: mean test AUC across folds against the all-features
baseline on the identical folds, plus ground-truth recall.

Run:  python experiments/dimensionality/tier_d_ksweep.py
      TIER_D_KSWEEP_DATASETS=hd_regime_switch_p2000,hd_ar1_p2000 python ...
"""
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orpsoc.utils import walk_forward_folds
from experiments.dimensionality.tier_d_comparators import (select_mrmr, select_lasso,
                                            score_selection, signal_recall)

K_GRID = [5, 10, 25, 50, 100, 250, 500]
_env = os.environ.get("TIER_D_KSWEEP_DATASETS")
DATASETS = ([d.strip() for d in _env.split(",")] if _env else
            ["hd_regime_switch_p2000", "hd_ar1_p2000",
             "hd_regime_switch_p1000", "hd_ar1_p1000"])
METHODS = {"mrmr": select_mrmr, "lasso": select_lasso}


def main():
    out = {}
    for ds in DATASETS:
        path = f"data/{ds}.pkl"
        if not os.path.exists(path):
            print(f"  SKIP {ds}: missing")
            continue
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        X, y = d["X"], d["y"]
        feat = list(X.columns)
        p = len(feat)
        folds = walk_forward_folds(X, y, n_splits=8)

        base = [score_selection(feat, a, b, c, e, seed=42)
                for a, b, c, e, _ in folds]
        base_mean = float(np.mean(base))
        print("=" * 84)
        print(f"  {ds}   p={p}   baseline (all features) = {base_mean:.4f}")
        print(f"  {'method':<8}{'k':>6}{'AUC':>9}{'vs base':>10}{'recall':>9}{'sec':>8}")
        print("-" * 84)

        out[ds] = {"p": p, "baseline": base_mean, "runs": []}
        for mname, fn in METHODS.items():
            for k in K_GRID:
                if k >= p:
                    continue
                t0 = time.time()
                aucs, recs, ks = [], [], []
                for X_tr, y_tr, X_te, y_te, _ in folds:
                    sel = fn(X_tr, y_tr, feat, k_target=k)["selected"]
                    aucs.append(score_selection(sel, X_tr, y_tr, X_te, y_te,
                                                seed=42))
                    recs.append(signal_recall(sel))
                    ks.append(len(sel))
                m = float(np.mean(aucs))
                delta = m - base_mean
                flag = "  <-- BEATS BASELINE" if delta > 0 else ""
                print(f"  {mname:<8}{np.mean(ks):>6.0f}{m:>9.4f}{delta:>+10.4f}"
                      f"{np.mean(recs):>9.2f}{time.time()-t0:>8.1f}{flag}")
                out[ds]["runs"].append(
                    {"method": mname, "k_target": k, "k_actual": float(np.mean(ks)),
                     "auc": m, "delta": delta, "recall": float(np.mean(recs)),
                     "fold_aucs": aucs})
        print()
        with open("results/tier_d_ksweep.json", "w") as fh:
            json.dump(out, fh, indent=2, default=float)

    print("  -> results/tier_d_ksweep.json")


if __name__ == "__main__":
    main()
