"""
Item 3 — the ridge sweep: does regularisation strength explain the sign?

THE CLAIM BEING TESTED
──────────────────────
Section 3.9 attributes the positive selection effect under logistic regression
to COLLINEARITY SENSITIVITY rather than to absent embedded-selection capacity.
Nothing in the paper tests that directly. This does.

If variance inflation under correlated inputs is the mechanism, then an
L2-regularised logistic regression should need external selection LESS as the
penalty strengthens, because ridge already shrinks the inflated coefficients.
The prediction is specific and falsifiable:

    the selection gain shrinks MONOTONICALLY as C decreases.

If the gain is flat in C, collinearity sensitivity is not the mechanism and the
manuscript's attribution has to change. A null here is a real result, not a
failed experiment, and is reported either way.

WHY THIS IS A SEPARATE FILE
───────────────────────────
tier_d_fullnovelty.py cannot be edited to add ridge models. Its provenance hash
covers its own source, so ANY edit invalidates all 1,226 stored checkpoints --
including the ones a concurrent EC2 run is actively writing. This file therefore
carries its own MODELS, its own CheckpointStore namespace ("ridge_sweep") and
its own provenance, and imports the arm implementations unchanged so the
selection procedures are bit-identical to the rest of the paper.

C GRID
──────
Six values spanning four orders of magnitude, centred on sklearn's default
C = 1.0 which is what every other logistic cell in the paper uses. The C = 1.0
column is therefore a REPLICATION CHECK: it must reproduce the corresponding
Table 10 cell, and if it does not, something in this file differs from the main
pipeline and the sweep is not interpretable.

Widths p = 171 and p = 301: the two the supervisor specified, both already at
20 seeds in the main ladder so the comparison is like-for-like.

Run:  python experiments/penalty/ridge_sweep.py
Env:  RS_DATASETS=ind17,ind30  RS_SEEDS=20  RS_CGRID=0.001,0.01,0.1,1,10,100
Writes results/ridge_sweep.json
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

from orpsoc.runner import (pin_threads, default_workers, provenance,
                           CheckpointStore)
from orpsoc.utils import walk_forward_folds, run_hybrid_orpsoc
from experiments.dimensionality.tier_d_comparators import COMPARATORS, score_selection
from experiments.dimensionality.tier_d_sparse_pso import sparse_pso
from experiments.dimensionality.tier_d_curve import sparse_orpsoc
from experiments.dimensionality.pso_competitors import COMPETITORS

pin_threads(1)

N_PARTICLES, MAX_ITER, N_SPLITS = 20, 60, 8
MIN_F, THETA = 3, 0.5
INIT_DENSITY, BIAS = 0.05, -3.0

N_SEEDS = int(os.environ.get("RS_SEEDS", "20"))
_ds = os.environ.get("RS_DATASETS")
DATASETS = ([d.strip() for d in _ds.split(",")] if _ds else ["ind17", "ind30"])
_cg = os.environ.get("RS_CGRID")
C_GRID = ([float(c) for c in _cg.split(",")] if _cg else
          [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])

# max_iter=200 matches the main pipeline's LogReg EXACTLY. Only C varies.
MODELS = {f"RidgeC{c:g}": (lambda c=c: LogisticRegression(max_iter=200, C=c))
          for c in C_GRID}
MODEL_NAMES = list(MODELS)

CONFIG = {"n_particles": N_PARTICLES, "max_iter": MAX_ITER,
          "n_splits": N_SPLITS, "theta": THETA, "min_f": MIN_F,
          "init_density": INIT_DENSITY, "bias": BIAS, "c_grid": C_GRID}
PROV = provenance(CONFIG, ["orpsoc/utils.py", "experiments/penalty/ridge_sweep.py"])
STORE = CheckpointStore("results/checkpoints", "ridge_sweep", PROV)

ARMS = ("all", "sparse_pso", "sparse_orpsoc", "full_hybrid",
        "mrmr", "lasso", "rfe",
        "cgpso", "taguchi_cbpso", "hpso_de", "bwaoa_c")
_a = os.environ.get("RS_ARMS")
ACTIVE_ARMS = ([s.strip() for s in _a.split(",")] if _a else list(ARMS))
for _bad in set(ACTIVE_ARMS) - set(ARMS):
    raise SystemExit(f"unknown arm: {_bad}; valid = {ARMS}")


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

    for X_tr, y_tr, X_te, y_te, _ in walk_forward_folds(X, y, n_splits=N_SPLITS):
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

    rec = {"ds": ds, "model": model_name,
           "C": float(model_name.replace("RidgeC", "")),
           "seed": seed, "n_features": int(X.shape[1]), "arms": arms}
    STORE.save(unit, rec)
    return rec


def main():
    present = [d for d in DATASETS if os.path.exists(f"data/{d}.pkl")]
    missing = set(DATASETS) - set(present)
    if missing:
        print(f"  SKIPPING (no data): {', '.join(sorted(missing))}", flush=True)
    # Cheapest width first, and C ascending, so an interrupted run still leaves
    # a usable prefix of the curve rather than a scatter of unrelated cells.
    grid = [(ds, m, s) for ds in present for m in MODEL_NAMES
            for s in range(N_SEEDS)]

    print("=" * 96)
    print("  RIDGE SWEEP — does regularisation strength explain the sign?")
    print(f"  arms: {', '.join(ACTIVE_ARMS)}")
    print(f"  C grid: {C_GRID}")
    print(f"  seeds={N_SEEDS} workers={default_workers()} provenance={PROV['hash']}")
    print(f"  datasets: {', '.join(present)}   units: {len(grid)}")
    print("=" * 96, flush=True)

    t0 = time.time()
    res = Parallel(n_jobs=default_workers(), backend="loky", verbose=5)(
        delayed(run_unit)(*g) for g in grid)
    print(f"\n  done in {(time.time()-t0)/60:.1f} min", flush=True)
    with open("results/ridge_sweep.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print("  -> results/ridge_sweep.json")


if __name__ == "__main__":
    main()
