"""
Tier D curve + the LogReg comparator table, in one pass.

THREE THINGS THIS CLOSES
────────────────────────
1. THE CURVE. The main Tier D run was stopped at p=2000 (synthetic) and
   p=951/491 (real), so "no dimensionality crossover" rests on endpoints rather
   than a curve. This fills p = 50, 200, 500, 1000 on both levels. It is now
   affordable because sparse PSO settles at k~50-130 instead of ~48% of p, so a
   fitness evaluation fits ~100 columns rather than ~1000.

2. THE COMPARATOR TABLE UNDER LogReg. D1 was run entirely under LightGBM -- the
   classifier that pareto_knee and the sparse-PSO grid both show cannot benefit
   from selection. A reviewer asking "is this better than mRMR?" would be
   answered from the one regime where nothing helps. Comparators are re-run here
   at the budget the PSO actually chose (k~50-130), under both classifiers.

3. D1.a, WHICH IS CURRENTLY UNANSWERED. `sparse_pso` in tier_d_sparse_pso.py is
   a PLAIN binary PSO: random sparse init, NO crossover. So the significant
   LogReg wins it produced say nothing about OrPSOC's two claimed novelties.
   This adds a `sparse_orpsoc` arm -- orthogonal sparse init + two-point
   crossover, the same machinery run_standard_orpsoc uses -- so the contrast
   isolates exactly those mechanisms at a density where the search actually
   works. If sparse_orpsoc does not beat sparse_pso, that is a finding about
   OrPSOC, and it is the finding D1.a was written to obtain.

ORTHOGONAL INITIALISATION AT ARBITRARY DENSITY
──────────────────────────────────────────────
build_orthogonal_positions puts each feature in exactly n_particles//2
particles, i.e. 50% density by construction -- the defect Tier D uncovered.
build_sparse_orthogonal generalises it: each feature appears in exactly
round(density * n_particles) particles, spread by a rotating offset so pairwise
coverage stays balanced. At density 0.05 with 20 particles that is 1 particle
per feature, and each particle carries ~density*p features. Same balance
property, arbitrary density.

Run:  python experiments/dimensionality/tier_d_curve.py
      CURVE_DATASETS=hd_ar1_p500 CURVE_SEEDS=4 python experiments/dimensionality/tier_d_curve.py
"""
import json
import os
import pickle
import re
import sys
import time

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orpsoc.runner import (pin_threads, default_workers, provenance,
                           CheckpointStore)
from orpsoc.utils import (sigmoid, crossover, walk_forward_folds,
                          FoldEvalContext, evaluate_ctx,
                          build_sparse_positions)
from experiments.dimensionality.tier_d_comparators import (COMPARATORS, score_selection,
                                            signal_recall)
from experiments.dimensionality.tier_d_sparse_pso import MODELS, sparse_pso

pin_threads(1)

N_PARTICLES, MAX_ITER, N_SPLITS = 20, 60, 8
MIN_F, THETA = 3, 0.5
W_MAX, W_MIN = 0.9, 0.4
INIT_DENSITY, BIAS = 0.05, -3.0        # the configuration that worked

N_SEEDS = int(os.environ.get("CURVE_SEEDS", "8"))
_ds = os.environ.get("CURVE_DATASETS")
DATASETS = ([d.strip() for d in _ds.split(",")] if _ds else
            [f"hd_{lvl}_p{p}" for p in (50, 200, 500, 1000)
             for lvl in ("regime_switch", "ar1")])
_m = os.environ.get("CURVE_MODELS")
MODEL_NAMES = ([s.strip() for s in _m.split(",")] if _m else list(MODELS))

CONFIG = {"n_particles": N_PARTICLES, "max_iter": MAX_ITER,
          "n_splits": N_SPLITS, "theta": THETA, "min_f": MIN_F,
          "init_density": INIT_DENSITY, "bias": BIAS}
PROV = provenance(CONFIG, ["orpsoc/utils.py", "experiments/dimensionality/tier_d_curve.py",
                           "experiments/dimensionality/tier_d_sparse_pso.py"])
STORE = CheckpointStore("results/checkpoints", "tier_d_curve", PROV)


# Single shared implementation lives in orpsoc_utils.build_sparse_positions.
# This file previously carried its own copy; the stride bug was fixed in one and
# not the other, so full_hybrid kept running on a broken swarm for a whole EC2
# job. Never duplicate an initialiser.
def build_sparse_orthogonal(n_particles, n_features, density, rng):
    seed = int(rng.randint(1_000_000))
    return build_sparse_positions(n_particles, n_features, density, seed=seed)


def sparse_orpsoc(X_tr, y_tr, feat, seed, model_factory=None):
    """Sparse PSO WITH OrPSOC's two mechanisms: orthogonal sparse init and
    two-point crossover. Everything else identical to sparse_pso, so the
    difference between them is exactly those mechanisms."""
    rng = np.random.RandomState(seed)
    n = len(feat)
    cut = int(len(X_tr) * 0.75)
    ctx = FoldEvalContext(X_tr.iloc[:cut], y_tr.iloc[:cut],
                          X_tr.iloc[cut:], y_tr.iloc[cut:], feat,
                          model_factory=model_factory)
    cache = {}
    def ev(pos):
        key = pos.astype(np.uint8).tobytes()
        if key not in cache:
            cache[key] = evaluate_ctx(pos, ctx, MIN_F, THETA)
        return cache[key]

    init = build_sparse_orthogonal(N_PARTICLES, n, INIT_DENSITY, rng)
    parts = []
    for i in range(N_PARTICLES):
        pos = init[i].copy()
        if pos.sum() < MIN_F:
            pos[rng.choice(n, MIN_F, replace=False)] = 1
        parts.append({"pos": pos.copy(), "vel": rng.randn(n) * 0.1,
                      "bp": pos.copy(), "bf": ev(pos)})

    gb = max(parts, key=lambda p: p["bf"])["bp"].copy()
    gf = max(p["bf"] for p in parts)

    for it in range(MAX_ITER):
        w = W_MAX - (W_MAX - W_MIN) * (it / max(MAX_ITER - 1, 1))
        for p in parts:
            r1, r2 = rng.rand(n), rng.rand(n)
            p["vel"] = (w * p["vel"] + 2.0 * r1 * (p["bp"] - p["pos"])
                        + 2.0 * r2 * (gb - p["pos"]))
            newp = (rng.rand(n) < sigmoid(p["vel"] + BIAS)).astype(float)
            if newp.sum() < MIN_F:
                z = np.where(newp == 0)[0]
                newp[rng.choice(z, MIN_F - int(newp.sum()), replace=False)] = 1
            p["pos"] = newp

        # OrPSOC's two-point crossover -- the second mechanism under test.
        idx = list(range(N_PARTICLES))
        rng.shuffle(idx)
        for k in range(0, N_PARTICLES - 1, 2):
            pa, pb = parts[idx[k]], parts[idx[k + 1]]
            ca, cb = crossover(pa["pos"], pb["pos"], 0.6, MIN_F, rng)
            for child, parent in [(ca, pa), (cb, pb)]:
                f = ev(child)
                if f > parent["bf"]:
                    parent["pos"], parent["bp"], parent["bf"] = child, child.copy(), f

        for p in parts:
            f = ev(p["pos"])
            if f > p["bf"]:
                p["bf"], p["bp"] = f, p["pos"].copy()
            if f > gf:
                gf, gb = f, p["pos"].copy()

    return [feat[i] for i in np.where(gb == 1)[0]]


def run_unit(ds, model_name, seed):
    unit = f"{ds}|{model_name}|s{seed}"
    hit = STORE.load(unit)
    if hit is not None:
        return hit

    with open(f"data/{ds}.pkl", "rb") as fh:
        d = pickle.load(fh)
    X, y = d["X"], d["y"]
    feat = list(X.columns)
    is_synth = ds.startswith("hd_")
    mf = MODELS[model_name]

    arms = {a: {"auc": [], "k": [], "recall": []}
            for a in ("all", "sparse_pso", "sparse_orpsoc",
                      "mrmr", "lasso", "rfe")}

    for X_tr, y_tr, X_te, y_te, _ in walk_forward_folds(X, y, n_splits=N_SPLITS):
        def record(name, sel):
            arms[name]["auc"].append(
                score_selection(sel, X_tr, y_tr, X_te, y_te, seed=seed,
                                model_factory=mf))
            arms[name]["k"].append(len(sel))
            arms[name]["recall"].append(signal_recall(sel) if is_synth else None)

        record("all", feat)
        sel_p = sparse_pso(X_tr, y_tr, feat, INIT_DENSITY, BIAS, seed,
                           model_factory=mf)
        record("sparse_pso", sel_p)
        record("sparse_orpsoc",
               sparse_orpsoc(X_tr, y_tr, feat, seed, model_factory=mf))

        # Comparators at the budget the PSO actually chose -- k~50-130 here,
        # not the degenerate ~48% of p that forced them in the main Tier D run.
        k_t = max(MIN_F, len(sel_p))
        for cname in ("mrmr", "lasso", "rfe"):
            fn = COMPARATORS[cname][0]
            record(cname, fn(X_tr, y_tr, feat, k_target=k_t, seed=seed)["selected"])

    rec = {"ds": ds, "model": model_name, "seed": seed,
           "n_features": int(X.shape[1]), "arms": arms}
    STORE.save(unit, rec)
    return rec


def main():
    present = [d for d in DATASETS if os.path.exists(f"data/{d}.pkl")]
    # Cheapest datasets first: the low-p points cost minutes and make the trend
    # visible long before the widest one finishes. Width is read from the name
    # where possible (hd_*_p2000, ind30); anything unparseable sorts LAST rather
    # than raising, so a new dataset cannot crash the run at startup -- ff100
    # has no width in its name and is genuinely the most expensive.
    def _width(name):
        m = re.findall(r"(\d+)", name)
        return int(m[-1]) if m else 10 ** 6
    present.sort(key=_width)
    grid = [(ds, m, s) for ds in present for m in MODEL_NAMES
            for s in range(N_SEEDS)]

    print("=" * 96)
    print("  TIER D CURVE + LogReg comparator table")
    print(f"  arms: all, sparse_pso, sparse_orpsoc, mrmr, lasso, rfe")
    print(f"  init={INIT_DENSITY} bias={BIAS} seeds={N_SEEDS} "
          f"workers={default_workers()}   provenance={PROV['hash']}")
    print(f"  datasets ({len(present)}): {', '.join(present)}")
    print(f"  units: {len(grid)}")
    print("=" * 96, flush=True)

    t0 = time.time()
    res = Parallel(n_jobs=default_workers(), backend="loky", verbose=5)(
        delayed(run_unit)(*g) for g in grid)
    print(f"\n  done in {(time.time()-t0)/60:.1f} min", flush=True)

    with open("results/tier_d_curve.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print("  -> results/tier_d_curve.json")


if __name__ == "__main__":
    main()
