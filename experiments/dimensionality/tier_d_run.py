"""
Tier D driver — D1 comparators x D2 dimensionality, one walk-forward protocol.

Answers two questions in one design:

  D1  Does OrPSOC's orthogonal-init + crossover machinery beat a standard
      binary PSO, a filter (mRMR), and embedded/wrapper methods (LASSO, RFE)?
  D2  Does the all-features baseline's advantage shrink as p grows, and does it
      shrink FASTER under regime shift than on a stationary series?

DATASETS
────────
  synthetic  hd_regime_switch_p{50,200,500,1000,2000}   primary arm
             hd_ar1_p{...}                              stationary control
  real       fama_french (50)  ff49ind (491)  ff100 (951)

`fama_french` is included as the real anchor: it is the published 50-feature
result, so if it does not reproduce, the wide datasets are not comparable to it.

CONDITIONS
──────────
  baseline    all features. Computed ONCE per (dataset, fold): with
              random_state fixed it is deterministic, which is exactly why
              TIERC_FINDINGS reports its per-seed std as 0.0000. Running it per
              seed would burn 20x the compute to reproduce one number.
  orpsoc      run_standard_orpsoc -- the method under test
  binary_pso  D1.a
  mrmr        D1.b            } all three at MATCHED BUDGET: k_target is the
  lasso       D1.c            } number of features OrPSOC chose on this same
  rfe         D1.c            } (seed, fold), so compactness is held equal

CHECKPOINTING
─────────────
Per (dataset, seed), provenance-scoped, via the same CheckpointStore step7/step9
use. Changing this file or orpsoc_utils.py changes the hash and correctly
invalidates old units rather than silently reloading them (guardrail G3).

PARALLELISM — WHY THIS DIFFERS FROM step7/step9
───────────────────────────────────────────────
Those stages parallelise over SEEDS within one dataset, which caps useful
workers at n_seeds (deploy/README.md: "no 31st seed to give it", hence the
32-vCPU box). Tier D has 13 datasets x 20 seeds = 260 independent units, so it
parallelises over the flat (dataset, seed) grid instead and can therefore use a
box far larger than 32 vCPU. Workers load their own dataset from disk rather
than receiving it through the joblib pickle: ff100 is ~50 MB, and shipping that
to every worker would cost more than the fit it enables.

Ordering: units are dispatched LARGEST DATASET FIRST. The long pole is ff100;
starting it last would leave a tail where most workers sit idle waiting for it.

Run:  python experiments/dimensionality/tier_d_run.py
      ORPSOC_FAST=1 python experiments/dimensionality/tier_d_run.py      # cheap smoke test
      TIER_D_DATASETS=hd_regime_switch_p50,ff100 python experiments/dimensionality/tier_d_run.py
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
from orpsoc.utils import run_standard_orpsoc, walk_forward_folds
from experiments.dimensionality.tier_d_comparators import (COMPARATORS, score_selection,
                                            signal_recall)

pin_threads(1)

FAST_MODE   = os.environ.get("ORPSOC_FAST", "0") == "1"
N_SEEDS     = 3  if FAST_MODE else 20
MAX_ITER    = 15 if FAST_MODE else 60
N_PARTICLES = 10 if FAST_MODE else 20
N_SPLITS    = 6  if FAST_MODE else 8
THETA       = 0.5
MIN_F       = 3
N_JOBS      = default_workers()

CONFIG = {"fast_mode": FAST_MODE, "n_seeds": N_SEEDS, "max_iter": MAX_ITER,
          "n_particles": N_PARTICLES, "n_splits": N_SPLITS, "theta": THETA}
PROV = provenance(CONFIG, ["orpsoc/utils.py", "experiments/dimensionality/tier_d_run.py",
                           "experiments/dimensionality/tier_d_comparators.py"])

SYNTH = [f"hd_{lvl}_p{p}"
         for lvl in ("regime_switch", "ar1")
         for p in (50, 200, 500, 1000, 2000)]
REAL  = ["fama_french", "ff49ind", "ff100"]
ALL_DATASETS = SYNTH + REAL

_env = os.environ.get("TIER_D_DATASETS")
DATASETS = [d.strip() for d in _env.split(",")] if _env else ALL_DATASETS


_CACHE = {}


def load_dataset(key):
    """Load from disk, memoised per PROCESS. Workers each pay this once."""
    if key not in _CACHE:
        with open(f"data/{key}.pkl", "rb") as fh:
            d = pickle.load(fh)
        _CACHE[key] = (d["X"], d["y"])
    return _CACHE[key]


def run_seed(X, y, ds_key, seed):
    """One complete walk-forward sequence for one seed. The checkpoint unit."""
    folds = walk_forward_folds(X, y, n_splits=N_SPLITS)
    feat = list(X.columns)
    is_synth = ds_key.startswith("hd_")

    rec = {c: {"fold_aucs": [], "fold_n_sel": [], "fold_selected": [],
               "fold_recall": [], "runtime": 0.0}
           for c in ("orpsoc",) + tuple(COMPARATORS)}

    for X_tr, y_tr, X_te, y_te, _ in folds:
        # OrPSOC first -- its subset size sets the matched budget.
        r = run_standard_orpsoc(X_tr, y_tr, X_te, y_te, feat, seed=seed,
                                n_particles=N_PARTICLES, max_iter=MAX_ITER,
                                min_f=MIN_F, theta=THETA)
        k_target = max(MIN_F, r["n_sel"])
        results = {"orpsoc": {"selected": r["selected"], "auc": r["auc"],
                              "runtime": r["runtime"]}}

        for cname, (fn, needs_k) in COMPARATORS.items():
            t0 = time.time()
            out = fn(X_tr, y_tr, feat, k_target=k_target, seed=seed,
                     n_particles=N_PARTICLES, max_iter=MAX_ITER,
                     min_f=MIN_F, theta=THETA)
            sel = out["selected"]
            results[cname] = {
                "selected": sel,
                "auc": score_selection(sel, X_tr, y_tr, X_te, y_te, seed=seed),
                "runtime": out.get("runtime", time.time() - t0)}

        for cname, res in results.items():
            rec[cname]["fold_aucs"].append(res["auc"])
            rec[cname]["fold_n_sel"].append(len(res["selected"]))
            rec[cname]["fold_selected"].append(res["selected"])
            rec[cname]["fold_recall"].append(
                signal_recall(res["selected"]) if is_synth else None)
            rec[cname]["runtime"] += res["runtime"]

    return rec


def run_baseline(X, y):
    """All-features reference, once per fold. Deterministic -> not per-seed."""
    feat = list(X.columns)
    aucs = []
    for X_tr, y_tr, X_te, y_te, _ in walk_forward_folds(X, y, n_splits=N_SPLITS):
        aucs.append(score_selection(feat, X_tr, y_tr, X_te, y_te, seed=42))
    return aucs


STORE = CheckpointStore("results/checkpoints", "tier_d", PROV)


def unit_job(ds, seed):
    """
    One checkpoint unit, executed in a worker. `seed=None` means the
    all-features baseline for that dataset.

    The worker loads its own dataset rather than receiving it via pickle --
    see PARALLELISM in the module docstring.
    """
    u = f"{ds}_s{seed}" if seed is not None else f"{ds}_baseline"
    hit = STORE.load(u)
    if hit is not None:
        return ds, seed, hit

    X, y = load_dataset(ds)
    r = run_baseline(X, y) if seed is None else run_seed(X, y, ds, seed)
    STORE.save(u, r)
    return ds, seed, r


def main():
    os.makedirs("results", exist_ok=True)

    present = [d for d in DATASETS if os.path.exists(f"data/{d}.pkl")]
    for d in DATASETS:
        if d not in present:
            print(f"  SKIP {d}: data/{d}.pkl missing", flush=True)
    # Largest first: ff100 is the long pole and must not start last.
    present.sort(key=lambda d: os.path.getsize(f"data/{d}.pkl"), reverse=True)

    grid = [(d, s) for d in present for s in [None] + list(range(N_SEEDS))]

    print("=" * 78)
    print("  TIER D — external comparators x dimensionality")
    print(f"  Mode: {'FAST (smoke)' if FAST_MODE else 'FULL (paper)'}   "
          f"seeds={N_SEEDS} iter={MAX_ITER} particles={N_PARTICLES} "
          f"splits={N_SPLITS}")
    print(f"  Workers={N_JOBS}   provenance={PROV['hash']}")
    print(f"  Datasets ({len(present)}): {', '.join(present)}")
    print(f"  Units: {len(grid)}  (datasets x (baseline + {N_SEEDS} seeds))")
    print("=" * 78, flush=True)

    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(unit_job)(d, s) for d, s in grid)
    print(f"\n  all units done in {(time.time()-t0)/60:.1f} min", flush=True)

    by_ds = {}
    for ds, seed, r in results:
        by_ds.setdefault(ds, {"baseline": None, "seeds": []})
        if seed is None:
            by_ds[ds]["baseline"] = r
        else:
            by_ds[ds]["seeds"].append(r)

    out = {"config": dict(CONFIG), "provenance": PROV, "datasets": {}}

    for ds in present:
        X, y = load_dataset(ds)
        base_aucs = by_ds[ds]["baseline"]
        recs = by_ds[ds]["seeds"]
        print(f"\n  [{ds}] X={X.shape}  baseline mean AUC = "
              f"{np.mean(base_aucs):.4f}", flush=True)

        ds_out = {"n_features": int(X.shape[1]), "n_rows": int(X.shape[0]),
                  "baseline": {"fold_aucs": base_aucs,
                               "mean_auc": float(np.mean(base_aucs))},
                  "conditions": {}}

        for cname in ("orpsoc",) + tuple(COMPARATORS):
            seed_means = [float(np.mean(r[cname]["fold_aucs"])) for r in recs]
            ds_out["conditions"][cname] = {
                "seed_aucs": seed_means,
                "mean_auc": float(np.mean(seed_means)),
                "std_auc": float(np.std(seed_means)),
                "mean_n_sel": float(np.mean(
                    [np.mean(r[cname]["fold_n_sel"]) for r in recs])),
                "seed_fold_aucs": [r[cname]["fold_aucs"] for r in recs],
                "seed_fold_n_sel": [r[cname]["fold_n_sel"] for r in recs],
                "seed_fold_selected": [r[cname]["fold_selected"] for r in recs],
                "seed_fold_recall": [r[cname]["fold_recall"] for r in recs],
                "total_runtime": float(np.sum([r[cname]["runtime"] for r in recs])),
            }
            d = ds_out["conditions"][cname]["mean_auc"] - ds_out["baseline"]["mean_auc"]
            print(f"      {cname:<12} AUC={ds_out['conditions'][cname]['mean_auc']:.4f}"
                  f"  vs base {d:+.4f}"
                  f"  k={ds_out['conditions'][cname]['mean_n_sel']:.1f}", flush=True)

        out["datasets"][ds] = ds_out
        with open("results/tier_d.json", "w") as fh:
            json.dump(out, fh, indent=2, default=float)

    print("\n  -> results/tier_d.json", flush=True)


if __name__ == "__main__":
    main()
