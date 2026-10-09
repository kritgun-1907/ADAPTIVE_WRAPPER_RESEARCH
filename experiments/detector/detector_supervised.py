"""
Supervised drift detection: watch the model's ERROR, not the market's volatility.

WHY THIS IS THE ONLY THING THAT CAN WORK HERE
─────────────────────────────────────────────
The v2 benchmark's "regime switch" is PURE CONCEPT DRIFT. Verified directly:

    X(v2_ar1) == X(v2_drift) == X(v2_regime_switch)     byte-identical
    y(v2_ar1) != y(v2_regime_switch)
    mean |corr| pre-break  = 0.0600
    mean |corr| post-break = 0.0603

P(y|X) changes at row 500; P(X) does not. `make_benchmark_v2.py:146` says so
explicitly ("identical marginals ... the ONLY thing that differs is the drifting
concept"). Every detector tested so far -- HMM level/deviation/occupancy/tail,
twin-window KS, BOCPD, PCA reconstruction error -- reads ONLY X. On this
benchmark they are uninformative by construction, which is why all of them fire
on pure noise and why the PCA statistic returns identical values on the switching
and stationary datasets.

A supervised detector asks a different question: not "is the market strange?" but
"has my model started being wrong?". Concept drift is exactly what that detects.

DESIGN (causal -- the test fold is never touched)
─────────────────────────────────────────────────
Within each fold's TRAINING window only:
    1. fit on the first FIT_FRAC of the window
    2. predict on the remainder, producing a per-sample log-loss SEQUENCE
    3. run ADWIN over that sequence

ADWIN (Bifet & Gavalda 2007): maintain a window of observations; for every split
into W0|W1, declare drift when |mean(W0) - mean(W1)| exceeds a Hoeffding bound

    eps_cut = sqrt( (1 / 2m) * ln(4 / delta) ),   m = harmonic mean(|W0|, |W1|)

It is parameter-light (one confidence delta) and makes no distributional
assumption beyond bounded values, which is why it is the standard choice.

NULL CONTROL, identical to the PCA experiment
─────────────────────────────────────────────
  signal set : v2_regime_switch (concept changes at row 500)
  null set   : v2_ar1           (stationary concept, no break exists)
Any fire on the null is a false alarm. Both evaluated over N_DRAWS independent
draws, because single-draw rankings already proved unreliable twice here.

  hit@4  fires at the first POST fold -- the earliest CAUSALLY possible, since
         the break must enter the TRAINING window before the model can be wrong
         about it
  FP     fires before that fold on v2_regime_switch
  NULL   fires at all on v2_ar1

Run:  python experiments/detector/detector_supervised.py [n_draws]
Writes results/detector_supervised.json
"""
import os
import sys
import json
import importlib

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from orpsoc.runner import pin_threads, default_workers
pin_threads(1)

from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from orpsoc.utils import walk_forward_folds, classify_folds

N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BASE_SEED = 9000            # the same draws every other detector was scored on
SWITCH = 500
FIT_FRAC = 0.60             # share of the TRAINING window used to fit
ADWIN_DELTA = 0.002         # ADWIN confidence
MIN_SUB = 20                # minimum sub-window for a split to be considered


def adwin_drift(losses, delta=ADWIN_DELTA, min_sub=MIN_SUB):
    """
    Returns (drift_detected, max_ratio).

    max_ratio = max over splits of |mean(W0) - mean(W1)| / eps_cut.
    Drift is declared when that exceeds 1, i.e. some split violates the
    Hoeffding bound. Reported as a continuous score so the margin is visible
    rather than just the binary decision.
    """
    x = np.asarray(losses, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * min_sub:
        return False, 0.0
    # Bound the values so the Hoeffding bound is valid (log-loss is unbounded
    # above; clipping at 10 nats is far beyond any real per-sample loss here).
    x = np.clip(x, 0.0, 10.0)
    csum = np.concatenate(([0.0], np.cumsum(x)))
    best = 0.0
    for c in range(min_sub, n - min_sub + 1):
        n0, n1 = c, n - c
        m0 = csum[c] / n0
        m1 = (csum[n] - csum[c]) / n1
        m = 1.0 / (1.0 / n0 + 1.0 / n1)          # harmonic mean / 2
        eps = np.sqrt((1.0 / (2.0 * m)) * np.log(4.0 / delta))
        if eps <= 0:
            continue
        best = max(best, abs(m0 - m1) / eps)
    return bool(best > 1.0), float(best)


def fold_losses(X_tr, y_tr):
    """Per-sample log-loss on the held-out tail of the TRAINING window."""
    n = len(X_tr)
    cut = int(n * FIT_FRAC)
    if cut < 40 or n - cut < 2 * MIN_SUB:
        return np.array([])
    Xa, ya = X_tr.iloc[:cut], y_tr.iloc[:cut]
    Xb, yb = X_tr.iloc[cut:], y_tr.iloc[cut:]
    if ya.nunique() < 2:
        return np.array([])
    pipe = Pipeline([("i", SimpleImputer(strategy="mean")),
                     ("s", StandardScaler()),
                     ("m", LogisticRegression(max_iter=200))])
    try:
        pipe.fit(Xa, ya)
        p = pipe.predict_proba(Xb)[:, 1]
    except Exception:
        return np.array([])
    p = np.clip(p, 1e-6, 1 - 1e-6)
    yv = yb.values.astype(float)
    return -(yv * np.log(p) + (1 - yv) * np.log(1 - p))


def fires_on(level, seed):
    m = importlib.import_module("data_prep.make_benchmark_v2")
    X, y, _b = m.generate_v2(level, seed=seed)
    folds = walk_forward_folds(X, y, n_splits=8, gap=5, min_train=150)
    switch = SWITCH if level == "regime_switch" else None
    ph = [p["phase"] for p in classify_folds(folds, switch)]
    fires, scores = [], []
    for fi, (X_tr, y_tr, *_r) in enumerate(folds):
        L = fold_losses(X_tr, y_tr)
        d, s = adwin_drift(L) if len(L) else (False, 0.0)
        scores.append(s)
        if d:
            fires.append(fi)
    return ph, scores, fires


def one_draw(seed):
    ph, s_rs, f_rs = fires_on("regime_switch", seed)
    post0 = ph.index("post")
    _p, s_ar, f_ar = fires_on("ar1", seed)
    return {"seed": seed, "post0": post0,
            "rs_scores": s_rs, "rs_fires": f_rs,
            "ar_scores": s_ar, "ar_fires": f_ar,
            "hit4": int(post0 in f_rs),
            "hitge": int(any(i >= post0 for i in f_rs)),
            "fp": sum(1 for i in f_rs if i < post0),
            "null": len(f_ar)}


def main():
    seeds = [BASE_SEED + i for i in range(N_DRAWS)]
    print("=" * 90)
    print(f"  SUPERVISED ERROR-DRIFT DETECTOR (log-loss + ADWIN) — {N_DRAWS} draws")
    print(f"  signal: v2_regime_switch     null: v2_ar1 (stationary concept)")
    print(f"  fit on first {FIT_FRAC:.0%} of the TRAINING window, ADWIN over the rest")
    print("=" * 90, flush=True)

    rows = Parallel(n_jobs=default_workers(), verbose=5)(
        delayed(one_draw)(s) for s in seeds)

    h4 = np.array([r["hit4"] for r in rows], float)
    hg = np.array([r["hitge"] for r in rows], float)
    fp = np.array([r["fp"] for r in rows], float)
    nl = np.array([r["null"] for r in rows], float)
    with open("results/detector_supervised.json", "w") as f:
        json.dump({"n_draws": N_DRAWS, "seeds": seeds, "fit_frac": FIT_FRAC,
                   "adwin_delta": ADWIN_DELTA, "rows": rows}, f, indent=2)

    post0 = rows[0]["post0"]
    print()
    print("=" * 90)
    print(f"  RESULTS over {N_DRAWS} draws")
    print("=" * 90)
    print(f"  fires at fold {post0} (earliest causal)  : {h4.mean():.2f}"
          f"  [{h4.sum():.0f}/{N_DRAWS}]")
    print(f"  fires at ANY post-break fold          : {hg.mean():.2f}"
          f"  [{hg.sum():.0f}/{N_DRAWS}]")
    print(f"  false positives per draw (pre-break)  : {fp.mean():.2f} +- {fp.std():.2f}"
          f"  [{fp.min():.0f},{fp.max():.0f}]")
    print(f"  NULL fires per draw (stationary)      : {nl.mean():.2f} +- {nl.std():.2f}"
          f"  [{nl.min():.0f},{nl.max():.0f}]")
    print(f"  draws with ZERO false positives       : {np.mean(fp == 0):.2f}")
    print(f"  draws clean on BOTH (FP=0, NULL=0)    : {np.mean((fp == 0) & (nl == 0)):.2f}")
    print()
    print("  BENCHMARK — every UNSUPERVISED detector on these same 20 draws:")
    print("    hmm_level (current) : FP 13.2+-1.4, NULL 11.3+-1.5, frac FP=0 = 0.00")
    print("    hmm_occupancy       : FP  1.3+-1.1, NULL  2.3+-1.4, frac FP=0 = 0.25")
    print("    pca_reconstruction  : FP  0.00+-0.00, NULL 0.35+-0.48, frac FP=0 = 1.00")
    print("    (PCA is clean because X does not change -- it correctly finds nothing)")
    print()
    S = np.array([r["rs_scores"] for r in rows], float)
    A = np.array([r["ar_scores"] for r in rows], float)
    print(f"  mean ADWIN score by fold (drift declared above 1.0):")
    print("    " + "".join(f"f{i}".rjust(9) for i in range(S.shape[1])))
    print("    switch " + "".join(f"{v:>9.2f}" for v in S.mean(0)))
    print("    null   " + "".join(f"{v:>9.2f}" for v in A.mean(0)))
    print(f"    {'':<7}" + "".join(("  <-break".rjust(9) if i == post0 else "".rjust(9))
                                   for i in range(S.shape[1])))


if __name__ == "__main__":
    main()
