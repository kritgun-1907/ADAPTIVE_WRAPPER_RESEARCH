"""
Supervised error-drift detector on REAL markets.

On synthetic v2_regime_switch the log-loss + ADWIN detector reached 90% detection
at the earliest causally possible fold with ZERO false positives and ZERO fires
on a stationary null, across 20 draws. That benchmark's break is pure concept
drift (X byte-identical pre/post), which is the regime the detector is built for.

Real markets are the untested case, and they differ in three ways that could
break it:
  * breaks alter P(X) as well as P(y|X), so the loss can move for reasons the
    detector is not meant to fire on
  * there are SEVERAL documented breaks (2001-09-17, 2008-09-15, 2020-02-20),
    not one
  * there is no ground truth per row -- only fold-level break windows, and no
    stationary counterpart to serve as a null

EVALUATION, adapted honestly to that
────────────────────────────────────
  break-adjacent = break_folds ∪ (break_folds + 1)
      the +1 is causal: a model cannot be wrong about a break until that break
      has entered its TRAINING window, one fold after the fold whose TEST window
      contains it. Both folds are counted as legitimate.
  hit  = fires at any break-adjacent fold
  FP   = fires at any fold that is NOT break-adjacent

There is no null dataset here, so "zero false positives" cannot be established
the way it was on synthetic. The strongest available claim is the ratio of
break-adjacent fires to total fires, plus the score SEPARATION between
break-adjacent and other folds.

The incumbent HMM's actual fire folds (from results/step9_real_data.json) are
printed alongside, so the comparison is like-for-like on the same folds.

Run:  python experiments/detector/detector_supervised_real.py
Writes results/detector_supervised_real.json
"""
import os
import sys
import json
import pickle

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from orpsoc.runner import pin_threads
pin_threads(1)

from orpsoc.utils import walk_forward_folds
from experiments.detector.detector_supervised import fold_losses, adwin_drift

DATASETS = ["sector_etf", "fama_french"]
MIN_TRAIN = 500          # matches step9 / ff_adaptation fold construction
CUTOFFS = [1.0, 2.0, 3.0, 4.0, 6.0]


def main():
    s9 = json.load(open("results/step9_real_data.json"))
    out = {}

    print("=" * 96)
    print("  SUPERVISED ERROR-DRIFT DETECTOR (log-loss + ADWIN) — REAL MARKETS")
    print("  documented breaks: 2001-09-17, 2008-09-15, 2020-02-20")
    print("=" * 96)

    for ds in DATASETS:
        d = pickle.load(open(f"data/{ds}.pkl", "rb"))
        X, y = d["X"], d["y"]
        folds = walk_forward_folds(X, y, n_splits=8, gap=5, min_train=MIN_TRAIN)
        blk = s9["datasets"].get(ds, {})
        brk = blk.get("break_folds") or []
        adjacent = sorted({b for b in brk} | {b + 1 for b in brk})
        hmm_fires = [i for i, v in enumerate(blk.get("trigger_fire_rate") or [])
                     if v > 0]

        scores = []
        for (X_tr, y_tr, *_r) in folds:
            L = fold_losses(X_tr, y_tr)
            _d, s = adwin_drift(L) if len(L) else (False, 0.0)
            scores.append(float(s))

        n = len(scores)
        print(f"\n── {ds}   X={X.shape}   folds={n}")
        print(f"   break_folds={brk}   break-adjacent={adjacent}")
        print(f"   incumbent HMM fired at {hmm_fires}")
        print(f"   {'fold':<10}" + "".join(f"f{i}".rjust(9) for i in range(n)))
        print(f"   {'ADWIN':<10}" + "".join(f"{v:>9.2f}" for v in scores))
        print(f"   {'':<10}" + "".join(("  <-brk".rjust(9) if i in adjacent else "".rjust(9))
                                       for i in range(n)))

        adj_s = [scores[i] for i in adjacent if i < n]
        oth_s = [scores[i] for i in range(n) if i not in adjacent]
        print(f"   mean score  break-adjacent={np.mean(adj_s):.2f}   "
              f"other folds={np.mean(oth_s):.2f}   "
              f"ratio={np.mean(adj_s)/max(np.mean(oth_s), 1e-9):.2f}x")

        print(f"\n   {'cutoff':<9}{'fires':<22}{'hit':>6}{'FP':>5}{'precision':>11}")
        rows = []
        for c in CUTOFFS:
            fires = [i for i, v in enumerate(scores) if v > c]
            hit = int(any(i in adjacent for i in fires))
            fp = sum(1 for i in fires if i not in adjacent)
            prec = (len(fires) - fp) / len(fires) if fires else float("nan")
            rows.append({"cutoff": c, "fires": fires, "hit": hit, "fp": fp,
                         "precision": None if fires == [] else prec})
            print(f"   {c:<9.1f}{str(fires):<22}{hit:>6}{fp:>5}"
                  f"{(f'{prec:.2f}' if fires else '-'):>11}")

        out[ds] = {"scores": scores, "break_folds": brk, "adjacent": adjacent,
                   "hmm_fires": hmm_fires, "cutoffs": rows,
                   "mean_adjacent": float(np.mean(adj_s)),
                   "mean_other": float(np.mean(oth_s))}

    with open("results/detector_supervised_real.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/detector_supervised_real.json")
    print()
    print("  READING THIS")
    print("    The synthetic result (90% hit, 0 FP) came with a stationary NULL")
    print("    dataset. No such null exists here, so precision -- the share of")
    print("    fires that are break-adjacent -- is the honest substitute, and the")
    print("    separation ratio is the effect size. A detector that fires on every")
    print("    fold scores hit=1 and is worthless; check FP and precision first.")


if __name__ == "__main__":
    main()
