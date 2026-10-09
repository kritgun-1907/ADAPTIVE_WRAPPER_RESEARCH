"""
Detector-gated warm start: suppress population memory AFTER a detected break.

WHY THE OBVIOUS FIX IS THE WRONG ONE
────────────────────────────────────
lag_factorial.py established that the Full Hybrid's three-fold adaptation lag is
caused by the warm start, not by the detector (which fires at the earliest
causally possible fold in 30/30 seeds) and not by the diversity burst (removing
its throttle changes nothing, p=0.70). Removing the warm start entirely gives
+0.0199 AUC (p=0.0004) and pulls adaptation from fold 7 to fold 6.

The natural repair is "when the detector fires, wipe the memory". That does
almost nothing, and the reason is visible in the code path:

    if not hmm_trigger:
        particles[0]["pos"] = ws      # stale gbest, EVERY non-trigger fold

The trigger fires ONCE, at fold 4. Folds 5-7 then re-seed the pre-switch
solution through the *other* branch. Wiping at the trigger fold therefore
addresses the one fold that was already handled and leaves the three that do the
damage untouched.

WHAT THIS TESTS INSTEAD
───────────────────────
A gate with memory: once the detector fires, suppress the warm start for the
next `gate_len` folds, then let it resume. This keeps population memory where it
is useful (stationary stretches, where lag_factorial shows warm=on is harmless)
and removes it exactly where it is harmful (the folds immediately after a break,
where the swarm should be abandoning the old subset).

  gate_len = 0   control -- current Full Hybrid, must reproduce fold 7
  gate_len = 1   the naive "wipe at trigger" fix
  gate_len = 3   suppress through the folds lag_factorial implicates
  gate_len = 99  never resume -- equivalent to lag_factorial's warm=off arm

PRE-REGISTERED OUTCOMES, fixed before running, matching lag_factorial so the two
are directly comparable:
  primary   adaptation_fold = first POST fold whose mean r2 recall >= RECALL_THRESH
  secondary mean test AUC, paired per-seed Wilcoxon against the gate_len=0 control
An earlier adaptation that costs accuracy is not a win, so both are reported.

Run:  python experiments/adaptation/warm_gate.py [n_seeds]
Writes results/warm_gate.json
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

from orpsoc.runner import pin_threads, default_workers
from orpsoc.utils import (run_hybrid_orpsoc, walk_forward_folds, classify_folds,
                          AdaptiveRegimeThreshold)

pin_threads(1)

LEVEL = "v2_regime_switch"
SWITCH = 500
N_SPLITS, GAP, MIN_TRAIN = 8, 5, 150
N_PARTICLES, MAX_ITER, THETA = 20, 60, 0.5
RECALL_THRESH = 1.0
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 30

GATES = [0, 1, 3, 99]
NAME = {0: "gate=0   [CONTROL, current Full Hybrid]",
        1: "gate=1   wipe at trigger only (the naive fix)",
        3: "gate=3   suppress for 3 folds after detection",
        99: "gate=99  never resume (= lag_factorial warm=off)"}


# Reuse step7's OWN SimpleHMM + get_hmm_trigger rather than a re-implementation,
# via the exec-head idiom the repo already uses (see experiments/adaptation/apsoll_sweep.py).
#
# BUG FIXED: the marker must be "#  MASTER ABLATION LOOP". An earlier version
# split on "#  ABLATION LOOP", which is NOT a substring of it, so split()
# returned the WHOLE file and exec ran the entire 4-hour ablation -- inside every
# joblib worker. step7's import guard does not catch this either: it trips on
# __name__ != "__main__", and an empty exec namespace reports "__main__" by
# default. Done once at module level, not per call.
_s7_src = open("pipeline/step7_ablation.py").read().split("#  MASTER ABLATION LOOP")[0]
assert "def get_hmm_trigger" in _s7_src and "MASTER ABLATION LOOP" not in _s7_src, \
    "step7 head split failed -- refusing to exec the full ablation"
_s7_ns = {}
exec(compile(_s7_src, "step7_ablation<head>", "exec"), _s7_ns)
_GET_HMM_TRIGGER = _s7_ns["get_hmm_trigger"]


def _hmm_trigger(X_tr, feat, thr):
    return _GET_HMM_TRIGGER(X_tr, feat_name=feat[0], threshold_obj=thr)


def run_one_seed(seed, gate_len):
    with open(f"data/{LEVEL}.pkl", "rb") as f:
        d = pickle.load(f)
    X, y = d["X"], d["y"]
    feat = list(X.columns)
    folds = walk_forward_folds(X, y, n_splits=N_SPLITS, gap=GAP,
                              min_train=MIN_TRAIN)
    phases = [p["phase"] for p in classify_folds(folds, SWITCH)]
    r1 = [c for c in feat if c in ("signal_0", "signal_1", "signal_2")]
    r2 = [c for c in feat if c in ("signal_3", "signal_4")]

    hmm = AdaptiveRegimeThreshold(method="percentile", lookback=50,
                                  percentile_k=85.0)
    warm = None
    folds_since_fire = None          # None => no break detected yet
    out = {"fold_aucs": [], "fold_r2_hits": [], "fold_r1_hits": [],
           "n_sel": [], "fold_triggered": [], "warm_suppressed": [],
           "_phases": phases}

    for fi, (X_tr, y_tr, X_te, y_te, _) in enumerate(folds):
        if len(y_te.unique()) < 2:
            continue
        trig, p_trans, _ = _hmm_trigger(X_tr, feat, hmm)

        if trig:
            folds_since_fire = 0
        elif folds_since_fire is not None:
            folds_since_fire += 1

        # The gate: memory is withheld from the trigger fold through the next
        # gate_len folds. gate_len=0 reproduces current behaviour exactly.
        gated = (folds_since_fire is not None and folds_since_fire <= gate_len)
        use_warm = None if gated else warm

        r = run_hybrid_orpsoc(
            X_tr, y_tr, X_te, y_te,
            hmm_trigger=trig, warm_start_pos=use_warm, p_trans=p_trans,
            feat_names=feat, seed=seed + fi * 1000,
            n_particles=N_PARTICLES, max_iter=MAX_ITER, min_f=3, theta=THETA,
            cr_low=0.3, cr_high=0.8, w_max=0.9, w_min=0.4,
            N_explore=max(5, MAX_ITER // 4), lam=0.1)

        warm = r["gbest_pos"]
        sel = set(r["selected"])
        out["fold_aucs"].append(r["auc"])
        out["n_sel"].append(r["n_sel"])
        out["fold_r1_hits"].append(len(sel & set(r1)))
        out["fold_r2_hits"].append(len(sel & set(r2)))
        out["fold_triggered"].append(bool(trig))
        out["warm_suppressed"].append(bool(gated))
    return out


def adaptation_fold(seeds):
    ph = seeds[0]["_phases"]
    for i in [j for j, p in enumerate(ph) if p == "post"]:
        v = [s["fold_r2_hits"][i] for s in seeds if len(s["fold_r2_hits"]) > i]
        if v and np.mean(v) >= RECALL_THRESH:
            return i
    return None


def main():
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    print("=" * 84)
    print("  DETECTOR-GATED WARM START")
    print(f"  {LEVEL}  seeds={N_SEEDS}  particles={N_PARTICLES} iter={MAX_ITER}"
          f"  workers={default_workers()}")
    print("=" * 84, flush=True)

    res, t0 = {}, time.time()
    for g in GATES:
        seeds = Parallel(n_jobs=default_workers(), backend="loky")(
            delayed(run_one_seed)(s, g) for s in range(N_SEEDS))
        res[g] = seeds
        print(f"  {NAME[g]:<48} done", flush=True)
    print(f"\n  all cells in {(time.time()-t0)/60:.1f} min\n", flush=True)

    ctrl = np.array([np.mean(s["fold_aucs"]) for s in res[0]])
    print(f"  {'cell':<48}{'AUC':>8}{'vs ctrl':>10}{'p':>9}{'k':>7}{'adapt':>7}")
    print("-" * 92)
    summary = {}
    for g in GATES:
        auc = np.array([np.mean(s["fold_aucs"]) for s in res[g]])
        k = np.mean([np.mean(s["n_sel"]) for s in res[g]])
        af = adaptation_fold(res[g])
        p = float("nan")
        if wilcoxon is not None and g != 0 and np.any(auc - ctrl != 0):
            try:
                p = wilcoxon(auc - ctrl).pvalue
            except Exception:
                pass
        star = "  <--" if (g != 0 and auc.mean() > ctrl.mean() and p < 0.05) else ""
        print(f"  {NAME[g]:<48}{auc.mean():>8.4f}"
              f"{(auc.mean()-ctrl.mean()) if g else 0:>+10.4f}"
              f"{p:>9.4f}{k:>7.1f}"
              f"{(af+1) if af is not None else 0:>7}{star}")
        summary[g] = {"auc_mean": float(auc.mean()), "auc_seeds": auc.tolist(),
                      "delta_vs_control": float(auc.mean()-ctrl.mean()),
                      "p_vs_control": None if np.isnan(p) else float(p),
                      "mean_k": float(k),
                      "adaptation_fold": None if af is None else int(af+1)}

    print("\n  per-fold r2 recall (post-switch signals):")
    for g in GATES:
        m = np.mean([s["fold_r2_hits"] for s in res[g]], axis=0)
        print(f"    gate={g:<3} " + "  ".join(f"{v:.2f}" for v in m))

    os.makedirs("results", exist_ok=True)
    with open("results/warm_gate.json", "w") as fh:
        json.dump({"level": LEVEL, "n_seeds": N_SEEDS, "gates": GATES,
                   "recall_thresh": RECALL_THRESH, "summary": summary}, fh,
                  indent=2, default=float)
    print("\n  -> results/warm_gate.json")


if __name__ == "__main__":
    main()
