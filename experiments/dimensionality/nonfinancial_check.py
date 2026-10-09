"""
The stationary check: does the classifier-sets-the-sign result hold off
financial data?

WHY THIS IS A SEPARATE SCRIPT AND NOT A FLAG ON tier_d_fullnovelty.py
──────────────────────────────────────────────────────────────────────
Two reasons, one methodological and one practical.

METHODOLOGICAL. Every dataset in the paper is financial, and the real ladder is
one market partitioned more finely. The review's point is that the Conclusion
makes a general machine-learning claim on that evidence. These datasets are IID,
so they are scored under ordinary stratified k-fold, NOT walk-forward. Mixing
two cross-validation schemes inside one script and one results file is how a
reader ends up comparing numbers that were never comparable. The output lands in
its own file and is reported as its own subsection.

The mechanism being tested is not about non-stationarity at all: if the sign of
the selection effect is set by the estimator, that should hold on stationary
data. Testing it here ISOLATES the claim rather than complicating it.

PRACTICAL. provenance() hashes the source of every file it is given, so any edit
to tier_d_fullnovelty.py changes its hash and silently invalidates every
existing checkpoint under it -- including the 8-seed ladder units that make
finishing the remaining widths cheap. Keeping this separate leaves that hash at
29a57cced36f, which is the value the 2026-08-20 20-seed run recorded.

A TRAP THIS AVOIDS
──────────────────
walk_forward_folds has min_train=150. golub has 72 rows, so it returns an EMPTY
fold list -- no exception, just a unit with zero folds that the checkpoint store
would happily persist as valid. The guard below refuses to store an empty unit.

COST
────
golub is p=7129 with n=72; madelon is p=500 with n=2600. Neither is close to the
anchor the timing model was fitted on, so run --probe FIRST and read the measured
estimate rather than trusting an extrapolation 40x beyond its range.

Run:  python experiments/dimensionality/nonfinancial_check.py --probe     (time one seed, ETA)
      python experiments/dimensionality/nonfinancial_check.py             (the real thing)

Env:  NF_DATASETS=madelon,golub   NF_MODELS=LogReg,LightGBM   NF_SEEDS=20
      NF_ARMS=all,sparse_pso,...  ORPSOC_N_JOBS=6
"""
import json
import os
import pickle
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orpsoc.runner import (pin_threads, default_workers, provenance,
                           CheckpointStore)
from orpsoc.utils import run_hybrid_orpsoc
from experiments.dimensionality.tier_d_comparators import COMPARATORS, score_selection
from experiments.dimensionality.tier_d_sparse_pso import MODELS, sparse_pso
from experiments.dimensionality.tier_d_curve import sparse_orpsoc
from experiments.dimensionality.pso_competitors import COMPETITORS

pin_threads(1)

N_PARTICLES, MAX_ITER = 20, 60
# 5, not the ladder's 8: golub has 72 rows, and 8 stratified folds would leave
# ~3 positives per test fold, which makes a fold-level AUC almost meaningless.
N_SPLITS = int(os.environ.get("NF_SPLITS", "5"))
MIN_F, THETA = 3, 0.5
INIT_DENSITY, BIAS = 0.05, -3.0

N_SEEDS = int(os.environ.get("NF_SEEDS", "20"))
_ds = os.environ.get("NF_DATASETS")
DATASETS = ([d.strip() for d in _ds.split(",")] if _ds else ["madelon", "golub"])
_m = os.environ.get("NF_MODELS")
MODEL_NAMES = ([s.strip() for s in _m.split(",")] if _m else ["LogReg", "LightGBM"])

# Same MODELS extension the ladder uses, so a model name means the same thing in
# both files.
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
MODELS = dict(MODELS)
MODELS["LinearSVM"] = lambda: CalibratedClassifierCV(
    LinearSVC(max_iter=5000, dual="auto"), cv=3)
MODELS["RandomForest"] = lambda: RandomForestClassifier(
    n_estimators=200, n_jobs=1, random_state=42)
MODELS["kNN"] = lambda: KNeighborsClassifier(n_neighbors=15)

ARMS = ("all", "sparse_pso", "sparse_orpsoc", "full_hybrid",
        "mrmr", "lasso", "rfe",
        "cgpso", "taguchi_cbpso", "hpso_de", "bwaoa_c")
_a = os.environ.get("NF_ARMS")
ACTIVE_ARMS = ([s.strip() for s in _a.split(",")] if _a else list(ARMS))
for _bad in set(ACTIVE_ARMS) - set(ARMS):
    raise SystemExit(f"unknown arm: {_bad}; valid = {ARMS}")

# "cv" is in CONFIG so this can never collide with a walk-forward checkpoint,
# even if every other parameter happens to agree.
CONFIG = {"n_particles": N_PARTICLES, "max_iter": MAX_ITER,
          "n_splits": N_SPLITS, "theta": THETA, "min_f": MIN_F,
          "init_density": INIT_DENSITY, "bias": BIAS,
          "cv": "stratified_kfold"}
PROV = provenance(CONFIG, ["orpsoc/utils.py",
                           "experiments/dimensionality/nonfinancial_check.py"])
STORE = CheckpointStore("results/checkpoints", "nonfinancial", PROV)


def stratified_folds(X, y, seed):
    """(X_tr, y_tr, X_te, y_te, n_train) tuples, shaped like walk_forward_folds."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    yv = np.asarray(y)
    return [(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], len(tr))
            for tr, te in skf.split(np.zeros(len(yv)), yv)]


def run_unit(ds, model_name, seed):
    unit = f"{ds}|{model_name}|s{seed}|" + ",".join(sorted(ACTIVE_ARMS))
    hit = STORE.load(unit)
    if hit is not None:
        return hit

    with open(f"data/{ds}.pkl", "rb") as fh:
        d = pickle.load(fh)
    X, y = d["X"], d["y"]
    feat = list(X.columns)
    mf = MODELS[model_name]
    arms = {a: {"auc": [], "k": []} for a in ACTIVE_ARMS}

    folds = stratified_folds(X, y, seed)
    if not folds:
        raise SystemExit(f"{ds}: no folds built (n={len(X)}) -- refusing to "
                         f"store an empty unit")

    for X_tr, y_tr, X_te, y_te, _ in folds:
        def record(name, sel, auc=None):
            arms[name]["auc"].append(
                auc if auc is not None else
                score_selection(sel, X_tr, y_tr, X_te, y_te, seed=seed,
                                model_factory=mf))
            arms[name]["k"].append(len(sel))

        if "all" in arms:
            record("all", feat)

        need_k = bool({"mrmr", "lasso", "rfe"} & set(arms))
        sel_p = None
        if "sparse_pso" in arms or need_k:
            sel_p = sparse_pso(X_tr, y_tr, feat, INIT_DENSITY, BIAS, seed,
                               model_factory=mf)
            if "sparse_pso" in arms:
                record("sparse_pso", sel_p)
        if "sparse_orpsoc" in arms:
            record("sparse_orpsoc",
                   sparse_orpsoc(X_tr, y_tr, feat, seed, model_factory=mf))
        if "full_hybrid" in arms:
            h = run_hybrid_orpsoc(X_tr, y_tr, X_te, y_te, feat,
                                  hmm_trigger=False, seed=seed,
                                  n_particles=N_PARTICLES, max_iter=MAX_ITER,
                                  min_f=MIN_F, theta=THETA,
                                  use_importance_reinit=True, model_factory=mf,
                                  init_density=INIT_DENSITY, transfer_bias=BIAS)
            record("full_hybrid", h["selected"], auc=h["auc"])

        for cname, cfn in COMPETITORS.items():
            if cname in arms:
                record(cname, cfn(X_tr, y_tr, feat, seed=seed,
                                  model_factory=mf)["selected"])

        if need_k:
            k_t = max(MIN_F, len(sel_p))
            for cname in ("mrmr", "lasso", "rfe"):
                if cname in arms:
                    fn = COMPARATORS[cname][0]
                    record(cname, fn(X_tr, y_tr, feat, k_target=k_t,
                                     seed=seed)["selected"])

    rec = {"ds": ds, "model": model_name, "seed": seed,
           "n_features": int(X.shape[1]), "n_rows": int(X.shape[0]),
           "cv": "stratified_kfold", "n_splits": N_SPLITS, "arms": arms}
    STORE.save(unit, rec)
    return rec


def main():
    probe = "--probe" in sys.argv
    present = [d for d in DATASETS if os.path.exists(f"data/{d}.pkl")]
    missing = [d for d in DATASETS if d not in present]
    workers = default_workers()

    print("=" * 88)
    print("  NON-FINANCIAL STATIONARY CHECK  (stratified k-fold, NOT walk-forward)")
    print(f"  datasets: {', '.join(present) or '(none found)'}"
          + (f"   MISSING: {', '.join(missing)}" if missing else ""))
    print(f"  models: {', '.join(MODEL_NAMES)}   arms: {len(ACTIVE_ARMS)}")
    print(f"  seeds={N_SEEDS} splits={N_SPLITS} workers={workers}  prov={PROV['hash']}")
    print("=" * 88, flush=True)
    if not present:
        raise SystemExit("no datasets found -- run data_prep/build_nonfinancial.py first")

    if probe:
        print("\n  PROBE: one seed per (dataset, model), timed serially.\n")
        total = 0.0
        for ds in present:
            for m in MODEL_NAMES:
                t0 = time.time()
                run_unit(ds, m, 9999)          # seed outside the real range
                dt = time.time() - t0
                total += dt
                print(f"    {ds:<10}{m:<12}{dt/60:>7.1f} min/unit", flush=True)
        est = total * N_SEEDS / max(1, workers) / 3600
        print(f"\n  measured {total/60:.1f} min for {len(present)*len(MODEL_NAMES)} units")
        print(f"  ESTIMATE for {N_SEEDS} seeds on {workers} workers: {est:.1f} h wall")
        print("  (probe units are stored under seed 9999 and excluded from results)")
        return

    grid = [(ds, m, s) for ds in present for m in MODEL_NAMES
            for s in range(N_SEEDS)]
    print(f"  units: {len(grid)}\n", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=workers, backend="loky", verbose=5)(
        delayed(run_unit)(*g) for g in grid)
    print(f"\n  done in {(time.time()-t0)/60:.1f} min", flush=True)
    with open("results/nonfinancial_check.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print("  -> results/nonfinancial_check.json")


if __name__ == "__main__":
    main()
