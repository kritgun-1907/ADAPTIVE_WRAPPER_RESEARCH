"""
Tier D / D2.b — HIGH-DIMENSIONAL SYNTHETIC SWEEP.

The paper's central negative result -- that external selection does not beat the
all-features baseline -- is explicitly attributed to dimensionality: a
gradient-boosted classifier handed 50 features reweights importance internally
across a regime change, absorbing the selection problem. The stated, testable
consequence is that this stops being true as p grows.

A single wide dataset cannot test that. It yields one point, and one point
cannot locate a crossover. This sweep holds EVERYTHING fixed except p:

    n_steps  = 1000   (unchanged)
    n_signal = 5      (unchanged -- signal_0..2 drive regime 1, signal_3..4
                       drive regime 2, exactly as in the 50-feature benchmark)
    seed     = 42     (unchanged)
    p        in {50, 200, 500, 1000, 2000}

so the only thing varying across the curve is the number of pure-noise
distractors. p=50 reproduces the existing v2 benchmark exactly and acts as the
anchor: if it does not match the published numbers, something else changed.

TWO LEVELS, AND WHY THE CONTROL IS NOT OPTIONAL
───────────────────────────────────────────────
  regime_switch  the primary arm -- concept drift at row 500
  ar1            a STATIONARY control at the same p

Without the control, an OrPSOC win at p=1000 is uninterpretable. "Feature
selection helps in high dimensions" is a well-known result and would not be a
contribution. The claim this project can actually make is about selection under
REGIME SHIFT, and that requires showing the high-p advantage is larger on
regime_switch than on a stationary series with identical dimensionality. If the
two curves rise together, the finding is dimensionality, not regime shift, and
must be reported as such.

NOTE ON p > n
─────────────
At p=2000 there are more features than rows (1000). That is a legitimate and
interesting regime for feature selection, but it is a genuinely different
problem from p=50, not merely a wider one -- LightGBM's own column subsampling
starts to bind. Flagged here so it is discussed rather than silently averaged
into a trend line.

Run:  python data_prep/make_highdim_sweep.py
Writes data/hd_<level>_p<p>.pkl  as {"X":…, "y":…, "base":…} dicts, matching
the pickle contract used by make_benchmark_v2.py.
"""
import os as _os, sys as _sys
_ROOT = (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
         if "__file__" in globals() else _os.getcwd())   # exec'd by a sibling: cwd is the root
_sys.path.insert(0, _ROOT)   # repo root, so `orpsoc` and siblings import
_os.chdir(_ROOT)             # data/, results/ and tables/ are root-relative

import importlib.util
import os
import pickle

import numpy as np

# Import the v2 generator rather than re-implementing it: any divergence
# between this sweep and the published benchmark would invalidate the p=50
# anchor, which is the only thing tying the curve to existing results.
_spec = importlib.util.spec_from_file_location("mb", "data_prep/make_benchmark_v2.py")
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)

P_GRID = [50, 200, 500, 1000, 2000]
LEVELS = ["regime_switch", "ar1"]
SEED   = mb.SEED
N_STEPS = mb.N_STEPS
N_SIGNAL = mb.N_SIGNAL
SWITCH_INDEX = mb.SWITCH_INDEX


def build(level: str, p: int):
    """Generate one (X, y, base) triple at dimensionality p."""
    return mb.generate_v2(level, n_steps=N_STEPS, n_features=p,
                          n_signal=N_SIGNAL, seed=SEED)


def main():
    os.makedirs("data", exist_ok=True)
    print("=" * 68)
    print("  TIER D — HIGH-DIMENSIONAL SWEEP")
    print(f"  n_steps={N_STEPS}  n_signal={N_SIGNAL}  seed={SEED}  "
          f"switch={SWITCH_INDEX}")
    print("=" * 68, flush=True)

    for level in LEVELS:
        for p in P_GRID:
            X, y, base = build(level, p)
            path = f"data/hd_{level}_p{p}.pkl"
            with open(path, "wb") as fh:
                pickle.dump({"X": X, "y": y, "base": base}, fh)

            cors = np.array([abs(np.corrcoef(X[c], y)[0, 1]) for c in X.columns])
            sig, noi = cors[:N_SIGNAL], cors[N_SIGNAL:]
            flag = "  <- p > n" if p > N_STEPS else ""
            print(f"  {level:<14} p={p:<5} shape={X.shape}  y={y.mean():.3f}  "
                  f"|corr| sig={sig.min():.3f}-{sig.max():.3f}  "
                  f"noise_max={noi.max():.3f}{flag}", flush=True)

    print("\n  Anchor check: hd_regime_switch_p50 must reproduce the published")
    print("  v2_regime_switch benchmark exactly.", flush=True)
    Xa, ya, _ = build("regime_switch", 50)
    ref = "data/v2_regime_switch.pkl"
    if os.path.exists(ref):
        with open(ref, "rb") as fh:
            d = pickle.load(fh)
        Xr, yr = d["X"], d["y"]
        same_X = Xa.shape == Xr.shape and np.allclose(
            Xa.to_numpy(), Xr[Xa.columns].to_numpy(), atol=0, rtol=0)
        same_y = bool((ya.to_numpy() == np.asarray(yr)).all())
        print(f"  X identical: {same_X}    y identical: {same_y}", flush=True)
        if not (same_X and same_y):
            raise SystemExit("ANCHOR FAILED — p=50 does not reproduce the "
                             "published benchmark; the sweep is not comparable.")
    else:
        print(f"  [skip] {ref} not present locally", flush=True)


if __name__ == "__main__":
    main()
