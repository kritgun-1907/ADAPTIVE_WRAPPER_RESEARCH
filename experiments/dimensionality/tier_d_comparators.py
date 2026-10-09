"""
Tier D / D1 — EXTERNAL COMPARATORS under the identical walk-forward protocol.

Work order §3 Tier D D1.a-d. The current ablation compares OrPSOC only against
the all-features baseline and against its own internal variants, so it cannot
answer the first question a reviewer asks: *is the orthogonal-init + crossover
machinery worth anything against a standard selector?*

Four comparators, each selecting from the TRAINING WINDOW ONLY and then scored
through the EXACT final pipeline used by orpsoc_utils.run_standard_orpsoc:

  D1.a  binary_pso   Standard binary PSO -- same particle count, iterations,
                     fitness and min_f as OrPSOC, but RANDOM initialisation and
                     NO crossover. This is the ablation that isolates the two
                     mechanisms OrPSOC actually claims as novel. Any OrPSOC
                     advantage that survives here is attributable to them.
  D1.b  mrmr        Minimum-redundancy maximum-relevance, FCQ variant
                     (F-statistic relevance / mean |corr| redundancy).
                     A pure filter: no model in the loop.
  D1.c  lasso       L1-penalised logistic regression; support = selection.
        rfe          Recursive feature elimination wrapped around LightGBM.
                     Embedded and wrapper respectively.

MATCHED BUDGET
──────────────
Filters and embedded methods need a target subset size k, while PSO discovers
its own. Comparing a 5-feature filter against a 13-feature PSO confounds
selection quality with compactness. Every comparator here is therefore given
`k_target` = the number of features OrPSOC selected ON THAT SAME FOLD, so the
comparison is at equal compactness and the only free variable is *which* k
features get chosen. Subset sizes are still reported per D1.d, because LASSO
cannot always hit an exact support size.

WHAT IS AND IS NOT CAUSAL
─────────────────────────
Every selector sees `X_tr` only. The test fold is never touched during
selection, scoring, or budget setting. `k_target` comes from OrPSOC's choice on
the same fold's TRAINING data, not from test performance.

Run:  python experiments/dimensionality/tier_d_comparators.py            (self-test, synthetic)
Imported by: experiments/dimensionality/tier_d_run.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from orpsoc.utils import sigmoid, evaluate


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED SCORING — byte-identical to run_standard_orpsoc's final block
# ══════════════════════════════════════════════════════════════════════════════

def score_selection(sel, X_tr, y_tr, X_te, y_te, seed=42, model_factory=None):
    """
    Fit the selected subset on the full training window and score the test fold.

    This mirrors orpsoc_utils.run_standard_orpsoc lines 1011-1025 exactly --
    same estimator, same hyper-parameters, same Pipeline, same
    `except -> 0.5` fallback. If this drifts from that block the comparator
    numbers stop being comparable to the ablation numbers, which is the whole
    point of the exercise.
    """
    if not sel:
        return 0.5
    try:
        final_model = (model_factory() if model_factory is not None else
                       LGBMClassifier(n_estimators=100, num_leaves=31,
                                      learning_rate=0.1, verbosity=-1,
                                      random_state=seed, n_jobs=1))
        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model",  final_model),
        ])
        pipe.fit(X_tr[sel], y_tr)
        return float(roc_auc_score(y_te, pipe.predict_proba(X_te[sel])[:, 1]))
    except Exception:
        return 0.5


def _matrix(X_tr):
    """Imputed + standardised training matrix, reused by the filter methods."""
    imp = SimpleImputer(strategy="mean")
    sc  = StandardScaler()
    return sc.fit_transform(imp.fit_transform(X_tr))


# ══════════════════════════════════════════════════════════════════════════════
#  D1.a — STANDARD BINARY PSO
# ══════════════════════════════════════════════════════════════════════════════

def select_binary_pso(X_tr, y_tr, feat_names, seed=42, n_particles=20,
                      max_iter=60, w_max=0.9, w_min=0.4, min_f=3, theta=0.5,
                      **kwargs):
    """
    Textbook binary PSO (Kennedy & Eberhart 1997) on the same fitness function.

    Differences from run_standard_orpsoc, and ONLY these:
      - initialisation is uniform random, not build_orthogonal_positions()
      - no two-point crossover stage

    Everything else -- velocity update, inertia schedule, sigmoid transfer,
    min_f repair, internal 75/25 fitness split, position cache -- is copied so
    the contrast isolates the two OrPSOC mechanisms.
    """
    rng = np.random.RandomState(seed)
    n   = len(feat_names)
    t0  = time.time()

    cut = int(len(X_tr) * 0.75)
    X_p, y_p = X_tr.iloc[:cut], y_tr.iloc[:cut]
    X_v, y_v = X_tr.iloc[cut:], y_tr.iloc[cut:]

    _cache = {}
    def _eval(pos):
        key = pos.astype(np.uint8).tobytes()
        val = _cache.get(key)
        if val is None:
            val = _cache[key] = evaluate(pos, feat_names, X_p, y_p, X_v, y_v,
                                         min_f, theta)
        return val

    # Uniform random init -- the thing OrPSOC replaces with orthogonal arrays.
    particles = []
    for _ in range(n_particles):
        pos = (rng.rand(n) < 0.5).astype(float)
        if pos.sum() < min_f:
            pos[rng.choice(n, size=min_f, replace=False)] = 1
        particles.append({"pos": pos.copy(), "vel": rng.randn(n) * 0.1,
                          "best_pos": pos.copy(), "best_fit": _eval(pos)})

    gbest_pos = max(particles, key=lambda p: p["best_fit"])["best_pos"].copy()
    gbest_fit = max(p["best_fit"] for p in particles)

    for it in range(max_iter):
        w = w_max - (w_max - w_min) * (it / max(max_iter - 1, 1))
        for p in particles:
            r1, r2 = rng.rand(n), rng.rand(n)
            p["vel"] = (w * p["vel"]
                        + 2.0 * r1 * (p["best_pos"] - p["pos"])
                        + 2.0 * r2 * (gbest_pos     - p["pos"]))
            new_pos = (rng.rand(n) < sigmoid(p["vel"])).astype(float)
            if new_pos.sum() < min_f:
                zeros = np.where(new_pos == 0)[0]
                need  = min_f - int(new_pos.sum())
                if len(zeros) >= need:
                    new_pos[rng.choice(zeros, size=need, replace=False)] = 1
            p["pos"] = new_pos

        # NOTE: no crossover stage here -- that is the point of this comparator.

        for p in particles:
            fit = _eval(p["pos"])
            if fit > p["best_fit"]:
                p["best_fit"] = fit
                p["best_pos"] = p["pos"].copy()
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_pos = p["pos"].copy()

    sel = [feat_names[i] for i in np.where(gbest_pos == 1)[0]]
    return {"selected": sel, "runtime": time.time() - t0}


# ══════════════════════════════════════════════════════════════════════════════
#  D1.b — mRMR  (filter)
# ══════════════════════════════════════════════════════════════════════════════

def select_mrmr(X_tr, y_tr, feat_names, k_target, **kwargs):
    """
    mRMR, FCQ variant (Ding & Peng 2005; the quotient form used by most fast
    implementations):

        first     j* = argmax  F(j)
        then      j* = argmax  F(j) / ( mean_{s in S} |corr(j, s)| + eps )

    F is the ANOVA F-statistic against y (relevance); the denominator is mean
    absolute Pearson correlation with the already-selected set (redundancy).
    No model is fitted, which is the point -- it is the cheap filter baseline.
    """
    t0 = time.time()
    n  = len(feat_names)
    k  = int(max(1, min(k_target, n)))

    M = _matrix(X_tr)
    y = np.asarray(y_tr)

    rel = f_classif(M, y)[0]
    rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)

    # Correlations are computed against the running selection only (O(k*p)),
    # never as a full p x p matrix -- at p=2000 that matrix is the bottleneck.
    Mz = M / (M.std(axis=0, ddof=0) + 1e-12)
    nrow = Mz.shape[0]

    chosen  = [int(np.argmax(rel))]
    red_sum = np.abs(Mz.T @ Mz[:, chosen[0]]) / nrow

    while len(chosen) < k:
        denom = red_sum / len(chosen) + 1e-8
        scores = rel / denom
        scores[chosen] = -np.inf
        j = int(np.argmax(scores))
        if not np.isfinite(scores[j]):
            break
        chosen.append(j)
        red_sum += np.abs(Mz.T @ Mz[:, j]) / nrow

    return {"selected": [feat_names[i] for i in chosen],
            "runtime": time.time() - t0}


# ══════════════════════════════════════════════════════════════════════════════
#  D1.c — LASSO and RFE  (embedded / wrapper)
# ══════════════════════════════════════════════════════════════════════════════

def select_lasso(X_tr, y_tr, feat_names, k_target, **kwargs):
    """
    L1-penalised logistic regression; the non-zero support is the selection.

    The penalty strength is not free: it is chosen so the support size lands as
    close as possible to `k_target`, so the comparison against the other methods
    happens at matched compactness. Ties break toward the SPARSER model -- the
    standard lasso/CV convention, and the same rule experiments/adaptation/pareto_knee.py
    already uses.

    Support size is monotone non-decreasing in C, so C is found by BISECTION on
    log C rather than by sweeping a fixed grid. At p~950 with several thousand
    training rows a 25-point liblinear sweep dominated the entire comparator
    budget; bisection reaches the same answer in ~10 fits and lands closer to
    k_target, since the grid rarely contains a C that hits it exactly.
    """
    t0 = time.time()
    M  = _matrix(X_tr)
    y  = np.asarray(y_tr)
    k  = int(max(1, min(k_target, len(feat_names))))

    def support_at(C):
        try:
            lr = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                    max_iter=2000, random_state=42)
            lr.fit(M, y)
            return np.where(np.abs(lr.coef_[0]) > 1e-10)[0]
        except Exception:
            return None

    best, best_key = None, None
    def consider(support):
        nonlocal best, best_key
        if support is None or len(support) == 0:
            return
        key = (abs(len(support) - k), len(support))   # closest, then sparser
        if best_key is None or key < best_key:
            best_key, best = key, support

    lo, hi = -4.0, 2.0                    # log10 C
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        sup = support_at(10.0 ** mid)
        consider(sup)
        if sup is None:
            break
        if len(sup) < k:                  # too sparse -> weaken the penalty
            lo = mid
        elif len(sup) > k:
            hi = mid
        else:
            break

    if best is None:                       # degenerate fit: fall back to relevance
        rel = np.nan_to_num(f_classif(M, y)[0], nan=0.0)
        best = np.argsort(rel)[::-1][:k]

    return {"selected": [feat_names[i] for i in best],
            "runtime": time.time() - t0}


def select_rfe(X_tr, y_tr, feat_names, k_target, seed=42, **kwargs):
    """
    Recursive feature elimination around LightGBM.

    `step=0.1` drops 10% of the remaining features per round rather than one at
    a time. At p=1000 a step of 1 would mean ~1000 sequential LightGBM fits per
    fold, which is not affordable inside a 30-seed study; 0.1 gives ~40 rounds.
    This is a stated configuration of RFE, not a tuned one.
    """
    t0 = time.time()
    M  = _matrix(X_tr)
    y  = np.asarray(y_tr)
    k  = int(max(1, min(k_target, len(feat_names))))

    est = LGBMClassifier(n_estimators=40, num_leaves=15, learning_rate=0.1,
                         verbosity=-1, random_state=seed, n_jobs=1)
    try:
        rfe = RFE(estimator=est, n_features_to_select=k, step=0.1)
        rfe.fit(M, y)
        idx = np.where(rfe.support_)[0]
    except Exception:
        rel = np.nan_to_num(f_classif(M, y)[0], nan=0.0)
        idx = np.argsort(rel)[::-1][:k]

    return {"selected": [feat_names[i] for i in idx],
            "runtime": time.time() - t0}


# ══════════════════════════════════════════════════════════════════════════════
#  REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

# needs_k: comparator requires the matched budget from OrPSOC's fold selection.
COMPARATORS = {
    "binary_pso": (select_binary_pso, False),
    "mrmr":       (select_mrmr,       True),
    "lasso":      (select_lasso,      True),
    "rfe":        (select_rfe,        True),
}


def signal_recall(sel, n_signal=5, group=None):
    """
    Ground-truth recall on the synthetic benchmark.

    group=None -> all signal_0..4;  'r1' -> signal_0..2 (pre-switch regime);
    'r2' -> signal_3..4 (post-switch regime).

    CALLER MUST GUARD. This function cannot tell "real dataset, no ground truth"
    apart from "synthetic run that selected none of the signals" -- both look
    like an empty intersection, and it returns 0.0 for each. A real dataset
    scored here would silently contribute 0.0 to a recall mean. tier_d_run.py
    stores None for any non-`hd_` dataset for that reason.
    """
    if group == "r1":
        truth = {f"signal_{i}" for i in range(3)}
    elif group == "r2":
        truth = {f"signal_{i}" for i in range(3, n_signal)}
    else:
        truth = {f"signal_{i}" for i in range(n_signal)}
    if not truth:
        return None
    return len(truth & set(sel)) / len(truth)


if __name__ == "__main__":
    # Self-test: one fold of a small synthetic problem, all four comparators.
    import importlib.util
    spec = importlib.util.spec_from_file_location("mb", "data_prep/make_benchmark_v2.py")
    mb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mb)

    X, y, _ = mb.generate_v2("regime_switch", n_features=100)
    feat = list(X.columns)
    X_tr, y_tr = X.iloc[:600], y.iloc[:600]
    X_te, y_te = X.iloc[605:800], y.iloc[605:800]

    print(f"self-test  p={len(feat)}  train={len(X_tr)}  test={len(X_te)}")
    print(f"{'method':<12} {'k':>4} {'AUC':>7} {'recall':>7} {'sec':>7}")

    allf = score_selection(feat, X_tr, y_tr, X_te, y_te)
    print(f"{'all':<12} {len(feat):>4} {allf:>7.4f} {1.0:>7.2f} {0.0:>7.1f}")

    r = select_binary_pso(X_tr, y_tr, feat, n_particles=10, max_iter=15)
    k = len(r["selected"])
    print(f"{'binary_pso':<12} {k:>4} "
          f"{score_selection(r['selected'], X_tr, y_tr, X_te, y_te):>7.4f} "
          f"{signal_recall(r['selected']):>7.2f} {r['runtime']:>7.1f}")

    for name in ("mrmr", "lasso", "rfe"):
        fn = COMPARATORS[name][0]
        r = fn(X_tr, y_tr, feat, k_target=k)
        print(f"{name:<12} {len(r['selected']):>4} "
              f"{score_selection(r['selected'], X_tr, y_tr, X_te, y_te):>7.4f} "
              f"{signal_recall(r['selected']):>7.2f} {r['runtime']:>7.1f}")
