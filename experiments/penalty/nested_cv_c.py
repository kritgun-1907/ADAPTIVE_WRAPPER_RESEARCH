"""
Item 1 — nested cross-validation tuning C for the baseline AND every arm.

THE QUESTION
────────────
Every logistic cell in the paper is scored at sklearn's default C = 1.0. The
ridge sweep (experiments/penalty/ridge_sweep.py, Table 12) then found that the
ALL-FEATURES baseline is not flat in C: it rises from 0.6986 at C = 100 to
0.7742 at C = 0.001 on ind17, and from 0.6352 to 0.7727 on ind30. At C = 0.001
the unselected model matches or beats every selection arm scored at C = 1.

That makes the paper's headline logistic gain ambiguous. It may be measuring
how much external selection helps, or it may be measuring how under-regularised
C = 1 is on these data. The two are not distinguishable from a fixed-C grid,
because a grid tunes nothing -- it reports a curve and leaves the reader to
pick a column.

This file removes the ambiguity the way the comparison requires: C is chosen by
an INNER cross-validation on the training window only, INDEPENDENTLY for the
baseline and for each selection arm, at every fold of every seed. Each arm
therefore gets the penalty that is best for the feature set it actually chose,
and the baseline gets the penalty that is best for all p features. Nothing is
tuned on a test fold.

WHAT IS TUNED AND WHAT IS NOT
─────────────────────────────
The SELECTION procedures are left exactly as published: the swarm's internal
fitness still evaluates LogisticRegression(max_iter=200) at C = 1, so the
subsets this file scores are the same subsets the paper reports. Only the
final fit-and-score step is tuned.

That is deliberate and it is the only design that answers the question asked.
Tuning C inside the fitness loop as well would change WHICH FEATURES each arm
selects, and the resulting delta could no longer be attributed to the penalty
rather than to a different search. It is also ~25x the compute: 20 particles x
60 iterations x 8 folds x 20 seeds, each fitness evaluation becoming a 6-value
x 4-block inner CV.

The consequence is stated rather than hidden: this run answers "does the
reported gain survive a fairly regularised baseline", not "what would a
fully nested pipeline select".

BOTH SCORES ARE RECORDED ON THE SAME SUBSET
───────────────────────────────────────────
Every selected subset is scored twice at each fold -- once at the fixed C = 1
the paper uses, once at the inner-CV choice. Scoring is negligible next to
selection, so this is nearly free, and it buys the thing that matters: the
fixed-C column is a REPLICATION CHECK against Table 6, computed inside this
file on these exact subsets. If it does not reproduce Table 6, this file
differs from the main pipeline somewhere and the tuned column cannot be
trusted either.

INNER CV IS TIME-ORDERED, NOT STRATIFIED K-FOLD
───────────────────────────────────────────────
These are ordered, autocorrelated series. sklearn's default cv=5 in
LogisticRegressionCV is StratifiedKFold, which shuffles future into past and
would leak. The inner splitter is TimeSeriesSplit, matching the 75/25
time-ordered validation split the swarm itself uses (Algorithm 1, line 1).
Blocks with a single class present cannot be scored by AUC and are skipped;
if no block in a fold is scorable the arm falls back to C = 1 and the fold is
flagged, so a silent fallback cannot be mistaken for a tuning result.

TIE-BREAK
─────────
Ties on mean inner AUC resolve to the SMALLER C, i.e. the stronger penalty.
This is the conservative direction: it cannot manufacture a win for the tuned
baseline by accident, because on an exact tie it picks the more regularised
model rather than the one closest to the paper's default.

Run:  python experiments/penalty/nested_cv_c.py
Env:  NCV_DATASETS=ind17,ind30  NCV_SEEDS=20
      NCV_CGRID=0.001,0.01,0.1,1,10,100  NCV_ARMS=all,sparse_pso,...
      NCV_WORKERS=8
Writes results/nested_cv_c.json
"""
import os
import sys
import json
import time
import pickle

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from orpsoc.runner import (pin_threads, default_workers, provenance,
                           CheckpointStore)
from orpsoc.utils import walk_forward_folds, run_hybrid_orpsoc
from experiments.dimensionality.tier_d_comparators import COMPARATORS
from experiments.dimensionality.tier_d_sparse_pso import sparse_pso
from experiments.dimensionality.tier_d_curve import sparse_orpsoc

pin_threads(1)

N_PARTICLES, MAX_ITER, N_SPLITS = 20, 60, 8
MIN_F, THETA = 3, 0.5
INIT_DENSITY, BIAS = 0.05, -3.0
INNER_SPLITS = 4

N_SEEDS = int(os.environ.get("NCV_SEEDS", "20"))
_ds = os.environ.get("NCV_DATASETS")
DATASETS = ([d.strip() for d in _ds.split(",")] if _ds else ["ind17", "ind30"])
_cg = os.environ.get("NCV_CGRID")
C_GRID = ([float(c) for c in _cg.split(",")] if _cg else
          [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])

ARMS = ("all", "sparse_pso", "sparse_orpsoc", "full_hybrid",
        "mrmr", "lasso", "rfe")
_a = os.environ.get("NCV_ARMS")
ACTIVE_ARMS = ([s.strip() for s in _a.split(",")] if _a else list(ARMS))
for _bad in set(ACTIVE_ARMS) - set(ARMS):
    raise SystemExit(f"unknown arm: {_bad}; valid = {ARMS}")

# The selector sees the published estimator, unchanged. Only scoring is tuned.
SEL_MODEL = lambda: LogisticRegression(max_iter=200)

CONFIG = {"n_particles": N_PARTICLES, "max_iter": MAX_ITER,
          "n_splits": N_SPLITS, "theta": THETA, "min_f": MIN_F,
          "init_density": INIT_DENSITY, "bias": BIAS,
          "c_grid": C_GRID, "inner_splits": INNER_SPLITS,
          "inner_cv": "TimeSeriesSplit", "tuned_step": "score_only"}
PROV = provenance(CONFIG, ["orpsoc/utils.py", "experiments/penalty/nested_cv_c.py"])
STORE = CheckpointStore("results/checkpoints", "nested_cv_c", PROV)


def _pipe(C):
    return Pipeline([
        ("imp",    SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(max_iter=200, C=C)),
    ])


def tune_C(X_tr, y_tr, sel):
    """
    Choose C by inner TimeSeriesSplit CV on the training window alone.

    Returns (best_C, mean inner AUC at best_C, per-C inner means, n scorable
    blocks). A fold with no scorable block returns C = 1.0 with n_blocks = 0,
    which the caller records so that fallbacks are countable rather than
    invisible.
    """
    Xs = X_tr[sel]
    yv = np.asarray(y_tr)
    n = len(Xs)
    if n < (INNER_SPLITS + 1) * 8:
        return 1.0, float("nan"), {}, 0

    splits = [(tr, va) for tr, va in TimeSeriesSplit(INNER_SPLITS).split(Xs)
              if len(np.unique(yv[tr])) > 1 and len(np.unique(yv[va])) > 1]
    if not splits:
        return 1.0, float("nan"), {}, 0

    means = {}
    for C in C_GRID:
        s = []
        for tr, va in splits:
            try:
                p = _pipe(C).fit(Xs.iloc[tr], yv[tr])
                s.append(roc_auc_score(yv[va],
                                       p.predict_proba(Xs.iloc[va])[:, 1]))
            except Exception:
                pass
        if s:
            means[C] = float(np.mean(s))
    if not means:
        return 1.0, float("nan"), {}, 0

    best = max(means.values())
    # ties -> stronger penalty (smaller C); see module docstring
    C_best = min(C for C, v in means.items() if v >= best - 1e-12)
    return float(C_best), float(means[C_best]), means, len(splits)


def score_at(X_tr, y_tr, X_te, y_te, sel, C):
    """Mirror of tier_d_comparators.score_selection with C exposed."""
    if not sel:
        return 0.5
    try:
        p = _pipe(C).fit(X_tr[sel], y_tr)
        return float(roc_auc_score(y_te, p.predict_proba(X_te[sel])[:, 1]))
    except Exception:
        return 0.5


def run_unit(ds, seed):
    unit = f"{ds}|s{seed}|" + ",".join(sorted(ACTIVE_ARMS))
    hit = STORE.load(unit)
    if hit is not None:
        return hit

    with open(f"data/{ds}.pkl", "rb") as fh:
        d = pickle.load(fh)
    X, y = d["X"], d["y"]
    feat = list(X.columns)
    arms = {a: {"auc_fixed": [], "auc_tuned": [], "k": [],
                "C": [], "inner_auc": [], "inner_blocks": []}
            for a in ACTIVE_ARMS}

    for X_tr, y_tr, X_te, y_te, _ in walk_forward_folds(X, y, n_splits=N_SPLITS):
        def record(name, sel):
            C, ia, _means, nb = tune_C(X_tr, y_tr, sel)
            a = arms[name]
            a["auc_fixed"].append(score_at(X_tr, y_tr, X_te, y_te, sel, 1.0))
            a["auc_tuned"].append(score_at(X_tr, y_tr, X_te, y_te, sel, C))
            a["k"].append(len(sel))
            a["C"].append(C)
            a["inner_auc"].append(ia)
            a["inner_blocks"].append(nb)

        if "all" in arms:
            record("all", feat)

        need_k = bool({"mrmr", "lasso", "rfe"} & set(arms))
        sel_p = None
        if "sparse_pso" in arms or need_k:
            sel_p = sparse_pso(X_tr, y_tr, feat, INIT_DENSITY, BIAS, seed,
                               model_factory=SEL_MODEL)
            if "sparse_pso" in arms:
                record("sparse_pso", sel_p)
        if "sparse_orpsoc" in arms:
            record("sparse_orpsoc",
                   sparse_orpsoc(X_tr, y_tr, feat, seed,
                                 model_factory=SEL_MODEL))
        if "full_hybrid" in arms:
            h = run_hybrid_orpsoc(X_tr, y_tr, X_te, y_te, feat,
                                  hmm_trigger=False, seed=seed,
                                  n_particles=N_PARTICLES, max_iter=MAX_ITER,
                                  min_f=MIN_F, theta=THETA,
                                  use_importance_reinit=True,
                                  model_factory=SEL_MODEL,
                                  init_density=INIT_DENSITY,
                                  transfer_bias=BIAS)
            record("full_hybrid", h["selected"])

        if need_k:
            k_t = max(MIN_F, len(sel_p))
            for cname in ("mrmr", "lasso", "rfe"):
                if cname in arms:
                    fn = COMPARATORS[cname][0]
                    record(cname, fn(X_tr, y_tr, feat, k_target=k_t,
                                     seed=seed)["selected"])

    rec = {"ds": ds, "model": "LogRegNestedCV", "seed": seed,
           "n_features": int(X.shape[1]), "arms": arms}
    STORE.save(unit, rec)
    return rec


def main():
    present = [d for d in DATASETS if os.path.exists(f"data/{d}.pkl")]
    missing = set(DATASETS) - set(present)
    if missing:
        print(f"  SKIPPING (no data): {', '.join(sorted(missing))}", flush=True)
    grid = [(ds, s) for ds in present for s in range(N_SEEDS)]
    workers = int(os.environ.get("NCV_WORKERS", default_workers()))

    print("=" * 96)
    print("  NESTED CV ON C — baseline and every arm tuned independently")
    print(f"  arms: {', '.join(ACTIVE_ARMS)}")
    print(f"  C grid: {C_GRID}   inner: TimeSeriesSplit({INNER_SPLITS})")
    print(f"  seeds={N_SEEDS} workers={workers} provenance={PROV['hash']}")
    print(f"  datasets: {', '.join(present)}   units: {len(grid)}")
    print("=" * 96, flush=True)

    t0 = time.time()
    res = Parallel(n_jobs=workers, backend="loky", verbose=5)(
        delayed(run_unit)(*g) for g in grid)
    print(f"\n  done in {(time.time()-t0)/60:.1f} min", flush=True)
    with open("results/nested_cv_c.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print("  -> results/nested_cv_c.json")


if __name__ == "__main__":
    main()
