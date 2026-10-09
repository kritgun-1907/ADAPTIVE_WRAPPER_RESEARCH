"""
Sparse binary PSO: does fixing the density collapse make selection work at high p?

BACKGROUND — the two defects this is built to correct
─────────────────────────────────────────────────────
Tier D found OrPSOC selecting ~48% of features at p>=1000, i.e. essentially its
own initialisation. Two separate causes were then measured
(experiments/dimensionality/tier_d_sparse_init.py):

  1. INITIALISATION. build_orthogonal_positions sets each feature on in exactly
     n_particles//2 particles -- 50% density by construction, at every p.

  2. THE TRANSFER FUNCTION, which is the deeper one. With
         new_pos = rand < sigmoid(v)
     a decaying velocity drives sigmoid(v) -> 0.5, so every bit becomes a coin
     flip again. Measured: starting at 1% density, the swarm climbed to 30.6%
     by iteration 30 and was still rising. Sparse init ALONE is undone.
     Adding a negative bias, sigmoid(v + bias), moves the zero-velocity fixed
     point to sigmoid(bias) and held density flat at ~2% for 30 iterations.

  3. THE FOOTHOLD, discovered by the same test. Every 1%-init run found ZERO of
     the 5 true signals. PSO can only amplify a signal it has already sampled,
     and a random 20-of-2000 subset contains one only 4.9% of the time -- 63%
     for a swarm of 20, which came up empty. At k=100 (5%) the swarm-level
     probability is 99%, and that run did find a signal.

     P(swarm of 20 finds >=1 of 5 signals in p=2000):
         k=10 -> 40%    k=20 -> 63%    k=50 -> 92%    k=100 -> 99%

So init density must be set from the foothold calculation, not chosen by feel.
Primary configuration: init 5%, bias -3.

WHAT IS BEING TESTED, PRECISELY
───────────────────────────────
Not "is sparse PSO better than OrPSOC" -- that is a foregone conclusion given
OrPSOC returns half the feature set. The question is whether a selector that
ACTUALLY COMPACTS can beat the all-features LightGBM baseline at high p, which
is the dimensionality hypothesis the whole of Tier D was built to test and which
the main run could not answer because no method ever ran at small k.

Everything else matches run_standard_orpsoc: same fitness (evaluate), same
inner 75/25 causal split, same min_f repair, same inertia schedule, same final
scoring pipeline. Only initialisation and the transfer bias differ, so any
difference is attributable to them.

Run:  python experiments/dimensionality/tier_d_sparse_pso.py
      SPARSE_DATASETS=hd_regime_switch_p2000 SPARSE_SEEDS=5 python ...
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

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from orpsoc.runner import (pin_threads, default_workers, provenance,
                           CheckpointStore)
from orpsoc.utils import (sigmoid, walk_forward_folds, FoldEvalContext,
                          evaluate_ctx)
from experiments.dimensionality.tier_d_comparators import score_selection, signal_recall

pin_threads(1)

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

N_PARTICLES = 20
MAX_ITER    = 60
N_SPLITS    = 8
MIN_F       = 3
THETA       = 0.5
W_MAX, W_MIN = 0.9, 0.4

N_SEEDS  = int(os.environ.get("SPARSE_SEEDS", "10"))
_ds      = os.environ.get("SPARSE_DATASETS")
DATASETS = ([d.strip() for d in _ds.split(",")] if _ds else
            ["hd_regime_switch_p2000", "hd_ar1_p2000", "ff100"])

# (init_density, bias). The first is the primary configuration; the others
# bracket it so the result is not a single lucky point. The (0.05, 0.0) arm is
# the control: sparse init with the UNMODIFIED transfer, which is the fix as
# originally proposed and is expected to drift back toward 50% density.
CONFIGS = [(0.05, -3.0), (0.05, -2.0), (0.10, -3.0), (0.05, 0.0)]
_cfg = os.environ.get("SPARSE_CONFIGS")     # e.g. "0.05:-3"
if _cfg:
    CONFIGS = [(float(s.split(":")[0]), float(s.split(":")[1]))
               for s in _cfg.split(",") if s.strip()]

# THE MODEL ARM. pareto_knee.py at p=951/2000 showed the classifier, not the
# dimensionality, decides whether selection helps: under LogReg every selection
# arm beat all-features (ff100 +0.0871, p=0.0078), under LightGBM every arm lost
# or tied. Both are run here so the sparse selector is tested against the
# classifier it might actually rescue, not only against the one that already
# absorbs the problem. LogReg matches pareto_knee's MODELS entry EXACTLY --
# max_iter=200, defaults otherwise -- so the cells are directly comparable.
MODELS = {
    "LogReg":   lambda: LogisticRegression(max_iter=200),
    "LightGBM": lambda: LGBMClassifier(n_estimators=100, num_leaves=31,
                                       learning_rate=0.1, verbosity=-1,
                                       random_state=42, n_jobs=1),
}
_m = os.environ.get("SPARSE_MODELS")
MODEL_NAMES = ([s.strip() for s in _m.split(",")] if _m else list(MODELS))

# ── CHECKPOINTING ────────────────────────────────────────────────────────────
# A first version of this file wrote results only at the very end. It ran 17 h
# on the laptop, consumed 39 core-hours, gave no readable progress, and would
# have lost everything on a crash. Same CheckpointStore that step7/step9/tier_d
# use: one file per (dataset, model, init, bias, seed), provenance-scoped so
# editing this file or orpsoc_utils.py correctly invalidates stale units rather
# than silently reloading them (guardrail G3).
CONFIG = {"n_particles": N_PARTICLES, "max_iter": MAX_ITER,
          "n_splits": N_SPLITS, "theta": THETA, "min_f": MIN_F,
          "w_max": W_MAX, "w_min": W_MIN}
PROV = provenance(CONFIG, ["orpsoc/utils.py",
                           "experiments/dimensionality/tier_d_sparse_pso.py"])
STORE = CheckpointStore("results/checkpoints", "tier_d_sparse", PROV)

# Cheapest-and-most-decisive first. The primary config and its control run
# across every dataset/model BEFORE the bracket configs, so the question
# "does sparse PSO rescue LogReg?" is answered by the partial results and the
# run can be stopped early without losing the finding.
CONFIG_RANK  = {(0.05, -3.0): 0, (0.05, 0.0): 1, (0.05, -2.0): 2, (0.10, -3.0): 3}
DATASET_RANK = {"ff100": 0, "hd_ar1_p2000": 1, "hd_regime_switch_p2000": 2}
MODEL_RANK   = {"LogReg": 0, "LightGBM": 1}


def sparse_pso(X_tr, y_tr, feat, init_density, bias, seed, model_factory=None):
    rng = np.random.RandomState(seed)
    n = len(feat)
    cut = int(len(X_tr) * 0.75)
    X_p, y_p = X_tr.iloc[:cut], y_tr.iloc[:cut]
    X_v, y_v = X_tr.iloc[cut:], y_tr.iloc[cut:]

    # FoldEvalContext hoists the imputer/scaler out of the particle loop and
    # carries the model_factory, so the PSO's fitness uses the SAME classifier
    # the selection will finally be scored with. Fitting with LightGBM and
    # scoring with LogReg would make the search optimise the wrong objective.
    ctx = FoldEvalContext(X_p, y_p, X_v, y_v, feat, model_factory=model_factory)

    cache = {}
    def ev(pos):
        key = pos.astype(np.uint8).tobytes()
        if key not in cache:
            cache[key] = evaluate_ctx(pos, ctx, MIN_F, THETA)
        return cache[key]

    parts = []
    for _ in range(N_PARTICLES):
        pos = (rng.rand(n) < init_density).astype(float)
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
            newp = (rng.rand(n) < sigmoid(p["vel"] + bias)).astype(float)
            if newp.sum() < MIN_F:
                z = np.where(newp == 0)[0]
                newp[rng.choice(z, MIN_F - int(newp.sum()), replace=False)] = 1
            p["pos"] = newp
        for p in parts:
            f = ev(p["pos"])
            if f > p["bf"]:
                p["bf"], p["bp"] = f, p["pos"].copy()
            if f > gf:
                gf, gb = f, p["pos"].copy()

    return [feat[i] for i in np.where(gb == 1)[0]]


def run_unit(ds, model_name, init_density, bias, seed):
    """One checkpoint unit: a complete walk-forward sequence for one cell."""
    unit = f"{ds}|{model_name}|i{init_density}|b{bias}|s{seed}".replace("/", "_")
    hit = STORE.load(unit)
    if hit is not None:
        return hit
    r = _run_unit_uncached(ds, model_name, init_density, bias, seed)
    STORE.save(unit, r)
    return r


def _run_unit_uncached(ds, model_name, init_density, bias, seed):
    with open(f"data/{ds}.pkl", "rb") as fh:
        d = pickle.load(fh)
    X, y = d["X"], d["y"]
    feat = list(X.columns)
    is_synth = ds.startswith("hd_")
    mf = MODELS[model_name]

    aucs, ks, recs = [], [], []
    for X_tr, y_tr, X_te, y_te, _ in walk_forward_folds(X, y, n_splits=N_SPLITS):
        sel = sparse_pso(X_tr, y_tr, feat, init_density, bias, seed,
                         model_factory=mf)
        aucs.append(score_selection(sel, X_tr, y_tr, X_te, y_te, seed=seed,
                                    model_factory=mf))
        ks.append(len(sel))
        recs.append(signal_recall(sel) if is_synth else None)
    return {"ds": ds, "model": model_name, "init": init_density, "bias": bias,
            "seed": seed, "fold_aucs": aucs, "fold_k": ks,
            "recall": ([r for r in recs if r is not None] or None)}


def baseline_of(ds, model_name):
    """All-features reference for THIS classifier. There is no single baseline:
    at p=951 LogReg scores 0.7672 on all features where LightGBM scores 0.8306,
    so comparing a LogReg selection against a LightGBM baseline would be
    measuring the classifier swap, not the selection."""
    with open(f"data/{ds}.pkl", "rb") as fh:
        d = pickle.load(fh)
    X, y = d["X"], d["y"]
    feat = list(X.columns)
    mf = MODELS[model_name]
    return [score_selection(feat, a, b, c, e, seed=42, model_factory=mf)
            for a, b, c, e, _ in walk_forward_folds(X, y, n_splits=N_SPLITS)]


def main():
    present = [d for d in DATASETS if os.path.exists(f"data/{d}.pkl")]
    grid = [(ds, m, i, b, s) for ds in present for m in MODEL_NAMES
            for (i, b) in CONFIGS for s in range(N_SEEDS)]
    grid.sort(key=lambda g: (CONFIG_RANK.get((g[2], g[3]), 99),
                             DATASET_RANK.get(g[0], 99),
                             MODEL_RANK.get(g[1], 99), g[4]))

    print("=" * 92)
    print("  SPARSE PSO — does a selector that actually compacts beat baseline at high p?")
    print(f"  particles={N_PARTICLES} iter={MAX_ITER} splits={N_SPLITS} "
          f"seeds={N_SEEDS}   workers={default_workers()}")
    print(f"  datasets: {', '.join(present)}")
    print(f"  models:   {', '.join(MODEL_NAMES)}")
    print(f"  configs (init,bias): {CONFIGS}")
    print(f"  units: {len(grid)}   provenance={PROV['hash']}")
    print(f"  order: primary config across all cells first, brackets last")
    print("=" * 92, flush=True)

    base = {(ds, m): baseline_of(ds, m) for ds in present for m in MODEL_NAMES}
    for (ds, m), b in base.items():
        print(f"  baseline {ds:<26} {m:<9} = {np.mean(b):.4f}", flush=True)

    t0 = time.time()
    res = Parallel(n_jobs=default_workers(), backend="loky", verbose=5)(
        delayed(run_unit)(*g) for g in grid)
    print(f"\n  done in {(time.time()-t0)/60:.1f} min\n", flush=True)

    out = {"baselines": {f"{ds}|{m}": list(map(float, v))
                         for (ds, m), v in base.items()},
           "runs": res}
    with open("results/tier_d_sparse_pso.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    for ds in present:
      for mname in MODEL_NAMES:
        bm = float(np.mean(base[(ds, mname)]))
        print("=" * 92)
        print(f"  {ds}  [{mname}]   all-features baseline={bm:.4f}")
        print(f"  {'init':>6}{'bias':>7}{'meanAUC':>10}{'vs base':>10}"
              f"{'p':>8}{'k':>8}{'recall':>9}")
        print("-" * 92)
        for (i, b) in CONFIGS:
            rs = [r for r in res if r["ds"] == ds and r["model"] == mname
                  and r["init"] == i and r["bias"] == b]
            if not rs:
                continue
            sm = np.array([np.mean(r["fold_aucs"]) for r in rs])
            kk = np.mean([np.mean(r["fold_k"]) for r in rs])
            rc = [np.mean(r["recall"]) for r in rs if r["recall"]]
            p = float("nan")
            if wilcoxon is not None and len(sm) >= 6 and np.any(sm - bm != 0):
                try:
                    p = wilcoxon(sm - bm).pvalue
                except Exception:
                    pass
            flag = "  <-- BEATS BASELINE" if (sm.mean() - bm) > 0 and p < 0.05 else ""
            print(f"  {i:>6.2f}{b:>7.1f}{sm.mean():>10.4f}{sm.mean()-bm:>+10.4f}"
                  f"{p:>8.4f}{kk:>8.1f}"
                  f"{(np.mean(rc) if rc else float('nan')):>9.2f}{flag}")
        print()
    print("  -> results/tier_d_sparse_pso.json")


if __name__ == "__main__":
    main()
