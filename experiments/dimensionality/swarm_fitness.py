"""
The fitness column for Table 13 — what objective value did each swarm actually
reach, and does Correlation-Guided PSO's AUC margin come from a better search
or from a larger subset?

WHY THIS EXISTS
───────────────
The review asked for a fitness column alongside AUC in Table 13. It could not
be produced from the stored results: every arm in the checkpoint store persists
only {auc, k, recall}, and the fitness the swarm optimised was computed inside
each selector, used to drive the search, and then discarded. `sparse_pso`,
`sparse_orpsoc` and all four competitors compute a gbest fitness `gf` and
return the subset without it.

WHAT IS RECOMPUTED, AND WHY IT IS EXACT
───────────────────────────────────────
Fitness is a pure function of the selected subset and the fold:

    Fitness = θ·AUC(validation) + (1-θ)·(1 - k/p),      θ = 0.5, min_f = 3

evaluated by orpsoc_utils.evaluate_ctx against a FoldEvalContext built on the
trailing 25% of the training window. All three selector families -- sparse_pso
(tier_d_sparse_pso), sparse_orpsoc (tier_d_curve), the four competitors
(pso_competitors) and run_hybrid_orpsoc itself -- use the IDENTICAL split
(cut = int(len(X_tr) * 0.75)), the identical MIN_F = 3 and THETA = 0.5, and the
identical model_factory. Verified by reading all four call sites.

So evaluating the returned gbest subset against a context built the same way
reproduces the gbest fitness the swarm actually achieved, to the bit. Nothing
is approximated and no selector source file is touched -- which matters,
because editing a selector would risk perturbing its RNG stream while
provenance for the main ladder (which hashes only orpsoc_utils.py and
tier_d_fullnovelty.py) would not invalidate the affected checkpoints.

WHAT THE COLUMN DECIDES
───────────────────────
The compactness term is the point. A swarm that buys AUC with a larger subset
pays for it in fitness, so if CGPSO leads on AUC but not on fitness, its margin
is budget rather than search. If it leads on both, the search really is better.
The table cannot be read off the AUC column alone, which is exactly the
review's objection.

THE SUBSETS ARE PERSISTED THIS TIME
───────────────────────────────────
`selected` is written per arm per fold. Cross-seed Jaccard on real data was
previously blocked for want of exactly this, and any future question about
which features an arm chose is answerable from this file without a re-run.

Run:  python experiments/dimensionality/swarm_fitness.py
      SF_SEEDS=2 SF_DATASETS=ind17 python experiments/dimensionality/swarm_fitness.py   # smoke
Writes results/swarm_fitness.json
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
from orpsoc.utils import (walk_forward_folds, run_hybrid_orpsoc,
                          FoldEvalContext, evaluate_ctx)
from experiments.dimensionality.tier_d_comparators import score_selection
from experiments.dimensionality.tier_d_sparse_pso import MODELS, sparse_pso
from experiments.dimensionality.tier_d_curve import sparse_orpsoc
from experiments.dimensionality.pso_competitors import COMPETITORS

pin_threads(1)

# Identical to tier_d_fullnovelty.py. These are not free parameters here: the
# numbers must be comparable cell-for-cell with Table 13, so any divergence
# would make the fitness column describe a different run than the AUC column.
N_PARTICLES, MAX_ITER, N_SPLITS = 20, 60, 8
MIN_F, THETA = 3, 0.5
INIT_DENSITY, BIAS = 0.05, -3.0

N_SEEDS = int(os.environ.get("SF_SEEDS", "20"))
_ds = os.environ.get("SF_DATASETS")
DATASETS = ([d.strip() for d in _ds.split(",")] if _ds else ["ind17", "ind30"])
_m = os.environ.get("SF_MODELS")
MODEL_NAMES = ([s.strip() for s in _m.split(",")] if _m else ["LogReg"])

# The seven swarm arms of Table 13, in table order. The three non-swarm methods
# (mRMR, LASSO, RFE) are deliberately absent: they do not optimise this
# objective -- they are handed a budget k and return a ranking -- so a
# "fitness" for them would be a number we computed rather than one they
# achieved, and putting it in the same column would invite exactly the
# comparison it is not entitled to.
SWARM_ARMS = ("sparse_pso", "sparse_orpsoc", "full_hybrid",
              "cgpso", "taguchi_cbpso", "hpso_de", "bwaoa_c")
_a = os.environ.get("SF_ARMS")
ACTIVE_ARMS = ([s.strip() for s in _a.split(",")] if _a else list(SWARM_ARMS))
for _bad in set(ACTIVE_ARMS) - set(SWARM_ARMS):
    raise SystemExit(f"unknown arm: {_bad}; valid = {SWARM_ARMS}")

CONFIG = {"n_particles": N_PARTICLES, "max_iter": MAX_ITER,
          "n_splits": N_SPLITS, "theta": THETA, "min_f": MIN_F,
          "init_density": INIT_DENSITY, "bias": BIAS}
# Hashes the selector sources as well as the engine, because this file's whole
# purpose is to report a quantity those selectors produce.
PROV = provenance(CONFIG, ["orpsoc/utils.py",
                           "experiments/dimensionality/swarm_fitness.py",
                           "experiments/dimensionality/tier_d_sparse_pso.py",
                           "experiments/dimensionality/tier_d_curve.py",
                           "experiments/dimensionality/pso_competitors.py"])
STORE = CheckpointStore("results/checkpoints", "swarm_fitness", PROV)


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
    arms = {a: {"auc": [], "k": [], "fitness": [], "selected": []}
            for a in ACTIVE_ARMS}

    for X_tr, y_tr, X_te, y_te, _ in walk_forward_folds(X, y, n_splits=N_SPLITS):
        # One context per fold, reused across arms: building it is the same
        # imputer/scaler fit for every arm, and hoisting it keeps the fitness
        # recomputation cheap relative to the searches themselves.
        cut = int(len(X_tr) * 0.75)
        ctx = FoldEvalContext(X_tr.iloc[:cut], y_tr.iloc[:cut],
                              X_tr.iloc[cut:], y_tr.iloc[cut:], feat,
                              model_factory=mf)
        index = {f: i for i, f in enumerate(feat)}

        def record(name, sel, auc=None):
            pos = np.zeros(len(feat), dtype=float)
            for f in sel:
                pos[index[f]] = 1.0
            arms[name]["fitness"].append(float(evaluate_ctx(pos, ctx, MIN_F, THETA)))
            arms[name]["auc"].append(
                auc if auc is not None else
                score_selection(sel, X_tr, y_tr, X_te, y_te, seed=seed,
                                model_factory=mf))
            arms[name]["k"].append(len(sel))
            arms[name]["selected"].append(list(sel))

        if "sparse_pso" in arms:
            record("sparse_pso",
                   sparse_pso(X_tr, y_tr, feat, INIT_DENSITY, BIAS, seed,
                              model_factory=mf))
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

    rec = {"ds": ds, "model": model_name, "seed": seed,
           "n_features": int(X.shape[1]), "arms": arms}
    STORE.save(unit, rec)
    return rec


def main():
    present = [d for d in DATASETS if os.path.exists(f"data/{d}.pkl")]
    missing = set(DATASETS) - set(present)
    if missing:
        raise SystemExit(f"missing datasets: {sorted(missing)}")
    grid = [(ds, m, s) for ds in present for m in MODEL_NAMES
            for s in range(N_SEEDS)]
    print("=" * 96)
    print("  SWARM FITNESS — the objective value each swarm actually achieved")
    print(f"  arms: {', '.join(ACTIVE_ARMS)}")
    print(f"  fitness = {THETA}*AUC(val) + {1-THETA}*(1 - k/p), min_f={MIN_F}")
    print(f"  seeds={N_SEEDS} workers={default_workers()}  "
          f"provenance={PROV['hash']}")
    print(f"  datasets: {', '.join(present)}   units: {len(grid)}")
    print("=" * 96, flush=True)

    t0 = time.time()
    res = Parallel(n_jobs=default_workers(), backend="loky", verbose=5)(
        delayed(run_unit)(*g) for g in grid)
    print(f"\n  done in {(time.time()-t0)/60:.1f} min", flush=True)
    with open("results/swarm_fitness.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print("  -> results/swarm_fitness.json")


if __name__ == "__main__":
    main()
