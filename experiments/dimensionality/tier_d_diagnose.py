"""
Tier D diagnostic: is the high-p negative result a FINDING or a SEARCH FAILURE?

Those are different claims and the headline numbers cannot separate them:

  FINDING        selection genuinely cannot beat an all-features LightGBM at
                 high p -- the selector finds the right features and still
                 loses, because the classifier already handles the noise.
  SEARCH FAILURE the selector never located the signal at all, so the
                 comparison says nothing about selection as an idea.

The synthetic arms have ground truth (signal_0..4), which settles it. The test:

    chance recall = (mean selected density k/p)
        because a subset of density d drawn without regard to which features
        matter includes each of the 5 signals with probability d.

    observed recall = |selected ∩ {signal_0..4}| / 5

If observed ≈ chance, the selector is choosing at random with respect to the
signal and the high-p rows are a verdict on the SEARCH, not on selection.
If observed >> chance, the selector works and the loss is a real result.

Run:  python experiments/dimensionality/tier_d_diagnose.py
"""
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CONDITIONS = ["orpsoc", "binary_pso", "mrmr", "lasso", "rfe"]
N_SIGNAL = 5


def main():
    dirs = sorted(glob.glob("results/checkpoints/tier_d/*/"))
    for d in dirs:
        files = glob.glob(d + "*.pkl")
        if not files:
            continue

        units = {}
        for f in files:
            with open(f, "rb") as fh:
                rec = pickle.load(fh)
            name = os.path.basename(f)[:-4]
            ds, _, tail = name.rpartition("_")
            units.setdefault(ds, {})[tail] = rec["payload"]

        for ds in sorted(units):
            if not ds.startswith("hd_"):
                continue                      # ground truth only on synthetic
            p = int(ds.split("_p")[-1])
            seeds = [k for k in units[ds] if k.startswith("s")]
            if not seeds:
                continue

            print("=" * 88)
            print(f"  {ds}   p={p}   seeds={len(seeds)}")
            print(f"  {'condition':<12}{'density':>9}{'chance':>9}"
                  f"{'observed':>10}{'ratio':>8}   verdict")
            print("-" * 88)

            for c in CONDITIONS:
                dens, rec_obs = [], []
                for s in seeds:
                    r = units[ds][s].get(c)
                    if not r:
                        continue
                    ks = np.array(r["fold_n_sel"], dtype=float)
                    dens.append(np.mean(ks / p))
                    rr = [x for x in r["fold_recall"] if x is not None]
                    if rr:
                        rec_obs.append(np.mean(rr))
                if not dens or not rec_obs:
                    continue
                density = float(np.mean(dens))
                observed = float(np.mean(rec_obs))
                ratio = observed / density if density > 0 else float("nan")
                if ratio < 1.15:
                    verdict = "SEARCH FAILURE (at chance)"
                elif ratio < 1.6:
                    verdict = "weak signal"
                else:
                    verdict = "searching (above chance)"
                print(f"  {c:<12}{density:>9.3f}{density:>9.3f}"
                      f"{observed:>10.3f}{ratio:>8.2f}   {verdict}")
            print()


if __name__ == "__main__":
    main()
