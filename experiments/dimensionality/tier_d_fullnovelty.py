"""
Full-novelty OrPSOC at high p: does the complete published mechanism set help
once the density collapse is fixed?

THE LADDER. Each arm adds exactly one layer, so any gain is attributable:

  all              every feature (per-classifier baseline)
  sparse_pso       random sparse init + transfer bias. NO OrPSOC novelty.
                   This is what produced the significant LogReg wins, which is
                   why the ladder is needed: those wins say nothing about
                   OrPSOC.
  sparse_orpsoc    + orthogonal (density-general) init + two-point crossover
                   = run_standard_orpsoc's mechanism set, made sparse
  full_hybrid      + APSOLL adaptive-c, three-leader velocity, three-phase
                   cr/w schedule, importance-guided reinit
                   = run_hybrid_orpsoc, i.e. EVERY published novelty
  mrmr/lasso/rfe   external comparators at the budget the PSO chose

full_hybrid runs the real engine (orpsoc_utils.run_hybrid_orpsoc) through its
new init_density / transfer_bias parameters rather than a re-implementation, so
there is no risk of the arm drifting from the published method. Those two
parameters default to the old behaviour and test_equivalence.py passes
unchanged, so nothing already reported moves.

hmm_trigger is left FALSE: the detector does not transfer to real markets
(DETECTOR_FINDINGS.md §4d), so firing it here would add a known-broken signal
and confound the mechanism comparison.

Run:  python experiments/dimensionality/tier_d_fullnovelty.py
"""
import json
import os
import pickle
import sys
import time

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orpsoc.runner import (pin_threads, default_workers, provenance,
                           CheckpointStore)
from orpsoc.utils import walk_forward_folds, run_hybrid_orpsoc
from experiments.dimensionality.tier_d_comparators import (COMPARATORS, score_selection,
                                            signal_recall)
from experiments.dimensionality.tier_d_sparse_pso import MODELS, sparse_pso
from experiments.dimensionality.tier_d_curve import sparse_orpsoc
from experiments.dimensionality.pso_competitors import COMPETITORS

pin_threads(1)

N_PARTICLES, MAX_ITER, N_SPLITS = 20, 60, 8
MIN_F, THETA = 3, 0.5
INIT_DENSITY, BIAS = 0.05, -3.0

N_SEEDS = int(os.environ.get("FN_SEEDS", "8"))
_ds = os.environ.get("FN_DATASETS")
DATASETS = ([d.strip() for d in _ds.split(",")] if _ds else
            ["ff100", "hd_ar1_p2000", "hd_regime_switch_p2000"])
# Review: two classifiers give nothing to interpolate between. These three span
# the embedded-selection spectrum -- RandomForest has it, LinearSVM and kNN do
# not -- and are the same three the earlier collinearity round used at D=50.
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
MODELS = dict(MODELS)
MODELS["LinearSVM"] = lambda: CalibratedClassifierCV(
    LinearSVC(max_iter=5000, dual="auto"), cv=3)   # needs predict_proba for AUC
MODELS["RandomForest"] = lambda: RandomForestClassifier(
    n_estimators=200, n_jobs=1, random_state=42)
MODELS["kNN"] = lambda: KNeighborsClassifier(n_neighbors=15)

_m = os.environ.get("FN_MODELS")
MODEL_NAMES = ([s.strip() for s in _m.split(",")] if _m else list(MODELS))

CONFIG = {"n_particles": N_PARTICLES, "max_iter": MAX_ITER,
          "n_splits": N_SPLITS, "theta": THETA, "min_f": MIN_F,
          "init_density": INIT_DENSITY, "bias": BIAS}
PROV = provenance(CONFIG, ["orpsoc/utils.py",
                           "experiments/dimensionality/tier_d_fullnovelty.py"])
STORE = CheckpointStore("results/checkpoints", "tier_d_fullnovelty", PROV)

ARMS = ("all", "sparse_pso", "sparse_orpsoc", "full_hybrid",
        "mrmr", "lasso", "rfe",
        "cgpso", "taguchi_cbpso", "hpso_de", "bwaoa_c")

# Arm selection, so a bug in ONE arm does not cost a full recompute of the
# others. Two bugs (the sparse-orthogonal stride collapse and the ~33%-dense
# importance re-seed) invalidated only `sparse_orpsoc` and `full_hybrid`;
# `all`, `sparse_pso` and the three filters never touch either code path and
# their existing numbers stand. Re-run just the broken pair with
#     FN_ARMS=sparse_orpsoc,full_hybrid
# and merge at analysis time. Arms are recorded per unit, so a merged result is
# explicit about which arms came from which run rather than silently blended.
_a = os.environ.get("FN_ARMS")
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
    is_synth = ds.startswith("hd_")
    mf = MODELS[model_name]
    arms = {a: {"auc": [], "k": [], "recall": []} for a in ACTIVE_ARMS}

    for X_tr, y_tr, X_te, y_te, _ in walk_forward_folds(X, y, n_splits=N_SPLITS):
        def record(name, sel, auc=None):
            arms[name]["auc"].append(
                auc if auc is not None else
                score_selection(sel, X_tr, y_tr, X_te, y_te, seed=seed,
                                model_factory=mf))
            arms[name]["k"].append(len(sel))
            arms[name]["recall"].append(signal_recall(sel) if is_synth else None)

        if "all" in arms:
            record("all", feat)

        # sparse_pso is needed either as an arm or as the matched-budget source
        # for the filters, so compute it when any of those are active.
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

        # The real engine, every novelty on, made sparse via the two new params.
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
                if cname not in arms:
                    continue
                fn = COMPARATORS[cname][0]
                record(cname, fn(X_tr, y_tr, feat, k_target=k_t,
                                 seed=seed)["selected"])

    rec = {"ds": ds, "model": model_name, "seed": seed,
           "n_features": int(X.shape[1]), "arms": arms}
    STORE.save(unit, rec)
    return rec


def main():
    present = [d for d in DATASETS if os.path.exists(f"data/{d}.pkl")]
    grid = [(ds, m, s) for ds in present for m in MODEL_NAMES
            for s in range(N_SEEDS)]
    print("=" * 96)
    print("  FULL-NOVELTY OrPSOC LADDER at high p")
    print(f"  arms: {', '.join(ACTIVE_ARMS)}")
    print(f"  init={INIT_DENSITY} bias={BIAS} seeds={N_SEEDS} "
          f"workers={default_workers()}  provenance={PROV['hash']}")
    print(f"  datasets: {', '.join(present)}   units: {len(grid)}")
    print("=" * 96, flush=True)

    t0 = time.time()
    res = Parallel(n_jobs=default_workers(), backend="loky", verbose=5)(
        delayed(run_unit)(*g) for g in grid)
    print(f"\n  done in {(time.time()-t0)/60:.1f} min", flush=True)
    tag = "" if set(ACTIVE_ARMS) == set(ARMS) else "_" + "-".join(sorted(ACTIVE_ARMS))
    with open(f"results/tier_d_fullnovelty{tag}.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"  -> results/tier_d_fullnovelty{tag}.json")


if __name__ == "__main__":
    main()
