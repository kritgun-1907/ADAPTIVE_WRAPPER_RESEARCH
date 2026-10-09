"""
PCA reconstruction-error detector: watch the covariance structure, not one feature.

THE ARGUMENT
────────────
Every statistic tested so far reduces the training window to ONE feature's
rolling volatility and asks whether that level is high. Measured across 20 draws,
the incumbent fires 13.2 +- 1.4 times before the break and 11.3 +- 1.5 times on
pure noise features -- it cannot tell a regime change from ordinary volatility
clustering, because at one-feature-level resolution those are the same thing.

A regime shift is not "things got loud". It is "things stopped moving together
the way they did". Temporary volatility leaves the covariance structure intact
and merely scales it; a genuine regime change rotates it. So:

    fit PCA on an EARLIER reference window
    project a RECENT window onto those components
    a structural break shows up as reconstruction error the old basis cannot absorb

Scale-invariance matters here and is handled explicitly: the recent window is
standardised with the REFERENCE window's own mean and scale, then the statistic
is normalised by the reference's in-sample error. A pure volatility spike
inflates numerator and denominator together and therefore does NOT fire, which is
the entire point.

THE NULL CONTROL
────────────────
A multivariate detector cannot be scored per-feature, so the null is a whole
dataset with NO break: `v2_ar1` is stationary by construction (same AR(1)
latents, no concept change). Any fire there is a false alarm. This is a stronger
null than the noise-column control used for the univariate statistics, because
the features are genuinely correlated -- it just never changes.

    hit@4    fires at fold 4 on v2_regime_switch (earliest CAUSALLY possible:
             the break must enter the TRAINING window first)
    FP       fires before fold 4 on v2_regime_switch
    NULL     fires at all on v2_ar1 (stationary, no break exists)

Evaluated over N_DRAWS independent draws of BOTH datasets, because a single-draw
ranking already proved unreliable once in this investigation.

Run:  python experiments/detector/detector_pca.py [n_draws]
Writes results/detector_pca.json
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
from orpsoc.utils import (walk_forward_folds, classify_folds,
                          AdaptiveRegimeThreshold)

N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BASE_SEED = 9000            # same draws the univariate comparison used
SWITCH = 500
VAR_KEEP = 0.90             # components retained from the reference window
RECENT_FRAC = 0.25          # trailing share of the window treated as "recent"
PCT_K = 85                  # same gate as every other detector tested


def pca_recon_stat(X_train):
    """
    Relative rise in PCA reconstruction error, recent window vs reference window.

    Causal: only X_train is touched. Returns a value in [0, 1].
    """
    A = np.asarray(X_train, float)
    n = len(A)
    if n < 80:
        return 0.0
    cut = int(n * (1.0 - RECENT_FRAC))
    ref_all, rec = A[:cut], A[cut:]
    if len(ref_all) < 60 or len(rec) < 20:
        return 0.0
    # The reference window is itself SPLIT: the basis is fitted on ref_fit and
    # the baseline error is measured on the held-out ref_val.
    #
    # An earlier version compared IN-sample error on the whole reference against
    # OUT-of-sample error on the recent window. Retaining 90% of the reference's
    # variance makes the in-sample residual tiny by construction, so that ratio
    # was enormous on every fold of every dataset -- the statistic sat pinned at
    # the clip ceiling of 1.000 and carried no information at all. Both errors
    # must be out-of-sample for the comparison to mean anything.
    split = int(len(ref_all) * 0.75)
    ref, ref_val = ref_all[:split], ref_all[split:]
    if len(ref) < 40 or len(ref_val) < 15:
        return 0.0

    # Standardise BOTH with the reference window's statistics. Using the recent
    # window's own scale would divide out the very shift we want to detect.
    mu = ref.mean(0)
    sd = ref.std(0)
    sd[sd < 1e-12] = 1e-12
    R = (ref - mu) / sd
    V = (ref_val - mu) / sd
    Q = (rec - mu) / sd

    # Basis from the reference window only.
    try:
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    ev = S ** 2
    tot = ev.sum()
    if tot <= 0:
        return 0.0
    k = int(np.searchsorted(np.cumsum(ev) / tot, VAR_KEEP) + 1)
    k = max(1, min(k, Vt.shape[0] - 1)) if Vt.shape[0] > 1 else 1
    P = Vt[:k]                                     # (k, n_features)

    def recon_err(M):
        proj = (M @ P.T) @ P
        return float(np.mean((M - proj) ** 2))

    e_base = recon_err(V)      # held-out, same distribution as the basis
    e_rec = recon_err(Q)       # held-out, possibly post-break
    if e_base <= 1e-12:
        return 0.0
    # Both terms are out-of-sample, so the ratio reflects a CHANGE in covariance
    # structure rather than the in-sample optimism of the fit. Normalising by
    # e_base is what makes a pure volatility spike inert: it inflates numerator
    # and denominator together, leaving the ratio near zero.
    return float(np.clip((e_rec - e_base) / e_base, 0.0, 1.0))


def fires_on(level, seed):
    m = importlib.import_module("data_prep.make_benchmark_v2")
    X, y, _b = m.generate_v2(level, seed=seed)
    folds = walk_forward_folds(X, y, n_splits=8, gap=5, min_train=150)
    switch = SWITCH if level == "regime_switch" else None
    ph = [p["phase"] for p in classify_folds(folds, switch)]
    thr = AdaptiveRegimeThreshold(method="percentile", lookback=50,
                                  percentile_k=PCT_K, cooldown=1)
    fired, vals = [], []
    for (X_tr, *_r) in folds:
        p = pca_recon_stat(X_tr.values)
        vals.append(p)
        if thr.update(p):
            fired.append(len(vals) - 1)
    return ph, vals, fired


def one_draw(seed):
    ph_rs, v_rs, f_rs = fires_on("regime_switch", seed)
    post0 = ph_rs.index("post")
    _ph_ar, v_ar, f_ar = fires_on("ar1", seed)
    return {"seed": seed, "post0": post0,
            "rs_vals": v_rs, "rs_fires": f_rs,
            "ar_vals": v_ar, "ar_fires": f_ar,
            "hit4": int(post0 in f_rs),
            "hitge": int(any(i >= post0 for i in f_rs)),
            "fp": sum(1 for i in f_rs if i < post0),
            "null": len(f_ar)}


def main():
    seeds = [BASE_SEED + i for i in range(N_DRAWS)]
    print("=" * 88)
    print(f"  PCA RECONSTRUCTION-ERROR DETECTOR — {N_DRAWS} draws")
    print(f"  signal set: v2_regime_switch    null set: v2_ar1 (stationary, no break)")
    print(f"  keep {VAR_KEEP:.0%} of reference variance, recent window = last "
          f"{RECENT_FRAC:.0%}, gate k={PCT_K}")
    print("=" * 88, flush=True)

    rows = Parallel(n_jobs=default_workers(), verbose=5)(
        delayed(one_draw)(s) for s in seeds)

    h4 = np.array([r["hit4"] for r in rows], float)
    hg = np.array([r["hitge"] for r in rows], float)
    fp = np.array([r["fp"] for r in rows], float)
    nl = np.array([r["null"] for r in rows], float)

    with open("results/detector_pca.json", "w") as f:
        json.dump({"n_draws": N_DRAWS, "seeds": seeds, "var_keep": VAR_KEEP,
                   "recent_frac": RECENT_FRAC, "pct_k": PCT_K, "rows": rows}, f,
                  indent=2)

    print()
    print("=" * 88)
    print(f"  RESULTS over {N_DRAWS} draws")
    print("=" * 88)
    print(f"  fires at fold {rows[0]['post0']} (earliest causal) : "
          f"{h4.mean():.2f} of 1.0 draws  [{h4.sum():.0f}/{N_DRAWS}]")
    print(f"  fires at ANY post-break fold             : "
          f"{hg.mean():.2f}  [{hg.sum():.0f}/{N_DRAWS}]")
    print(f"  false positives per draw (pre-break)     : {fp.mean():.2f} +- {fp.std():.2f}"
          f"  [{fp.min():.0f},{fp.max():.0f}]")
    print(f"  NULL fires per draw (stationary v2_ar1)  : {nl.mean():.2f} +- {nl.std():.2f}"
          f"  [{nl.min():.0f},{nl.max():.0f}]")
    print(f"  draws with ZERO false positives          : {np.mean(fp == 0):.2f}")
    print(f"  draws clean on BOTH (FP=0 and NULL=0)    : "
          f"{np.mean((fp == 0) & (nl == 0)):.2f}")
    print()
    print("  BENCHMARK — univariate statistics on the same 20 draws:")
    print("    hmm_level (current) : FP 13.2+-1.4, NULL 11.3+-1.5, frac FP=0 = 0.00")
    print("    hmm_occupancy       : FP  1.3+-1.1, NULL  2.3+-1.4, frac FP=0 = 0.25")
    print("    (their NULL is noise-column fires, a WEAKER null than v2_ar1 here)")
    print()
    print("  mean statistic value by fold, v2_regime_switch "
          f"(break enters TRAINING at fold {rows[0]['post0']}):")
    V = np.array([r["rs_vals"] for r in rows], float)
    print("    " + "".join(f"f{i}".rjust(9) for i in range(V.shape[1])))
    print("    " + "".join(f"{v:>9.3f}" for v in V.mean(0)))
    A = np.array([r["ar_vals"] for r in rows], float)
    print("  same, on the stationary null (should be flat and low):")
    print("    " + "".join(f"{v:>9.3f}" for v in A.mean(0)))


if __name__ == "__main__":
    main()
