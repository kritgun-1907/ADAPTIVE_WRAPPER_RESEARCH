"""
Re-run the RFE arm under the pinned scikit-learn, cheaply, for LightGBM.

THE PROBLEM
───────────
RFE for LightGBM at ind10/ind12/ind17/ind30 was computed 2026-08-19/20, before
the scikit-learn 1.8.0 pin was verified, and may have run under 1.9.0. RFE is
the only arm the version change is known to move: across the 64 cells the two
runs share, every other arm reproduces bit-identically, while RFE differs in all
64, because sklearn's elimination order changed between releases.

WHY NOT JUST RUN `FN_ARMS=rfe`
──────────────────────────────
Because `run_unit` computes the matched budget by re-running sparse PSO:

    need_k = {"mrmr","lasso","rfe"} & arms   ->   sel_p = sparse_pso(...)
    k_t = max(MIN_F, len(sel_p))

So asking for RFE alone still pays for a full 20x60 particle swarm per fold.
Measured on this project (results/tier_d_timing.json), that is the difference
between 1.4-3.5 s and 185-469 s per fold -- roughly 130x. Re-running RFE that
way for four widths of LightGBM was estimated at ~160 core-hours.

THE SHORTCUT, AND WHY IT IS SOUND
─────────────────────────────────
The matched budget k does not depend on the scikit-learn version. That is not an
assumption: merge_ladder.py verified it directly over all 64 shared cells --
"Matched budgets (k) are identical in both runs, so the change is in which
features RFE keeps, not in how many."

The stored checkpoints already contain sparse PSO's per-fold k. This script
reads those, and re-runs ONLY the RFE selection and its final scoring against
the same folds. Everything that is version-sensitive is recomputed; nothing that
is version-invariant is paid for twice.

It writes units under exactly the key `FN_ARMS=rfe` would produce, so
build_ladder.py picks them up by its normal "later-written unit wins per arm"
rule with no special-casing.

Run:  python experiments/dimensionality/rfe_repin.py
Env:  RP_DATASETS=ind10,ind12,ind17,ind30   RP_MODELS=LightGBM   RP_SEEDS=20
"""
import os
import sys
import time
import pickle

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sklearn
if sklearn.__version__ != "1.8.0":
    raise SystemExit(f"REFUSING TO RUN: scikit-learn {sklearn.__version__}, "
                     f"pinned at 1.8.0. This script exists to remove a version "
                     f"inconsistency; running it off-pin would create another.")

from orpsoc.runner import pin_threads, default_workers
from orpsoc.utils import walk_forward_folds
from experiments.dimensionality.tier_d_comparators import COMPARATORS, score_selection
import experiments.dimensionality.tier_d_fullnovelty as fn
pin_threads(1)

MIN_F = 3
N_SPLITS = 8
STORE = fn.STORE
MODELS = fn.MODELS

_ds = os.environ.get("RP_DATASETS")
DATASETS = ([d.strip() for d in _ds.split(",")] if _ds else
            ["ind10", "ind12", "ind17", "ind30"])
_m = os.environ.get("RP_MODELS")
MODEL_NAMES = ([s.strip() for s in _m.split(",")] if _m else ["LightGBM"])
N_SEEDS = int(os.environ.get("RP_SEEDS", "20"))


def stored_k(ds, model, seed):
    """Per-fold sparse-PSO subset sizes from whatever unit already holds them."""
    root = os.path.join("results/checkpoints/tier_d_fullnovelty", STORE.prov["hash"]) \
        if hasattr(STORE, "prov") else None
    root = root or os.path.join("results/checkpoints/tier_d_fullnovelty",
                                fn.PROV["hash"])
    prefix = f"{ds}|{model}|s{seed}|"
    for f in os.listdir(root):
        if not f.startswith(prefix) or not f.endswith(".pkl"):
            continue
        try:
            with open(os.path.join(root, f), "rb") as fh:
                obj = pickle.load(fh)
        except Exception:
            continue
        arms = obj.get("payload", obj).get("arms", {})
        if "sparse_pso" in arms and arms["sparse_pso"].get("k"):
            return list(arms["sparse_pso"]["k"])
    return None


def run_unit(ds, model_name, seed):
    unit = f"{ds}|{model_name}|s{seed}|rfe"
    hit = STORE.load(unit)
    if hit is not None:
        return ("cached", ds, model_name, seed)

    ks = stored_k(ds, model_name, seed)
    if ks is None:
        return ("no_k", ds, model_name, seed)

    with open(f"data/{ds}.pkl", "rb") as fh:
        d = pickle.load(fh)
    X, y = d["X"], d["y"]
    feat = list(X.columns)
    mf = MODELS[model_name]
    aucs, kk = [], []

    folds = walk_forward_folds(X, y, n_splits=N_SPLITS)
    if len(folds) != len(ks):
        return ("fold_mismatch", ds, model_name, seed)

    fnr = COMPARATORS["rfe"][0]
    for (X_tr, y_tr, X_te, y_te, _), k_fold in zip(folds, ks):
        k_t = max(MIN_F, int(k_fold))
        sel = fnr(X_tr, y_tr, feat, k_target=k_t, seed=seed)["selected"]
        aucs.append(score_selection(sel, X_tr, y_tr, X_te, y_te, seed=seed,
                                    model_factory=mf))
        kk.append(len(sel))

    rec = {"ds": ds, "model": model_name, "seed": seed,
           "n_features": int(X.shape[1]),
           "arms": {"rfe": {"auc": aucs, "k": kk,
                            "recall": [None] * len(aucs)}}}
    STORE.save(unit, rec)
    return ("ok", ds, model_name, seed)


def main():
    grid = [(ds, m, s) for ds in DATASETS if os.path.exists(f"data/{ds}.pkl")
            for m in MODEL_NAMES for s in range(N_SEEDS)]
    print("=" * 78)
    print("  RFE RE-PIN — recompute only the version-sensitive arm")
    print(f"  scikit-learn {sklearn.__version__} (pinned)")
    print(f"  datasets: {', '.join(DATASETS)}   models: {', '.join(MODEL_NAMES)}")
    print(f"  units: {len(grid)}   workers: {default_workers()}")
    print(f"  provenance: {fn.PROV['hash']}")
    print("=" * 78, flush=True)

    t0 = time.time()
    res = Parallel(n_jobs=default_workers(), backend="loky", verbose=5)(
        delayed(run_unit)(*g) for g in grid)
    import collections
    c = collections.Counter(r[0] for r in res)
    print(f"\n  done in {(time.time()-t0)/60:.1f} min   {dict(c)}")
    for status, ds, m, s in res:
        if status not in ("ok", "cached"):
            print(f"    {status}: {ds} {m} s{s}")


if __name__ == "__main__":
    main()
