"""
Does SPARSE INITIALISATION actually fix the high-p density collapse?

Tier D showed OrPSOC selecting ~48% of features at p>=1000 -- essentially where
build_orthogonal_positions puts it, which is 50% BY CONSTRUCTION (each feature
on in exactly n_particles//2 particles). The natural fix is to start sparse.

But there is a reason to doubt that initialisation alone is enough. The binary
transfer is

    v         = w*v + 2*r1*(pbest - pos) + 2*r2*(gbest - pos)
    new_pos   = rand < sigmoid(v)

With a sparse swarm, pos/pbest/gbest are nearly all zero, so both attraction
terms vanish and v decays under w (0.9 -> 0.4). As v -> 0, sigmoid(v) -> 0.5 and
every bit becomes a coin flip again. If that is what happens, the 50% attractor
lives in the TRANSFER FUNCTION and sparse init is undone within a few iterations.

This measures the density trajectory directly, for several init densities and
two transfer variants:

    "sigmoid"       the current rule
    "biased"        sigmoid(v + bias), bias<0, which moves the zero-velocity
                    fixed point from 0.5 down to sigmoid(bias)

Reports mean particle density per iteration plus gbest fitness/size/recall, so
the question "does it stay sparse, and does staying sparse help?" is answered
with numbers rather than argument.

Run:  python experiments/dimensionality/tier_d_sparse_init.py
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orpsoc.utils import sigmoid, evaluate
from experiments.dimensionality.tier_d_comparators import score_selection, signal_recall

DATASET   = os.environ.get("SPARSE_DS", "hd_regime_switch_p2000")
N_PART    = 20
MAX_ITER  = 30
MIN_F     = 3
THETA     = 0.5
W_MAX, W_MIN = 0.9, 0.4


def run(X_tr, y_tr, X_te, y_te, feat, init_density, transfer, bias=0.0,
        seed=42, max_iter=MAX_ITER):
    rng = np.random.RandomState(seed)
    n = len(feat)
    cut = int(len(X_tr) * 0.75)
    X_p, y_p = X_tr.iloc[:cut], y_tr.iloc[:cut]
    X_v, y_v = X_tr.iloc[cut:], y_tr.iloc[cut:]

    cache = {}
    def ev(pos):
        k = pos.astype(np.uint8).tobytes()
        if k not in cache:
            cache[k] = evaluate(pos, feat, X_p, y_p, X_v, y_v, MIN_F, THETA)
        return cache[k]

    parts = []
    for _ in range(N_PART):
        pos = (rng.rand(n) < init_density).astype(float)
        if pos.sum() < MIN_F:
            pos[rng.choice(n, MIN_F, replace=False)] = 1
        parts.append({"pos": pos.copy(), "vel": rng.randn(n) * 0.1,
                      "bp": pos.copy(), "bf": ev(pos)})

    gb = max(parts, key=lambda p: p["bf"])["bp"].copy()
    gf = max(p["bf"] for p in parts)
    traj = [float(np.mean([p["pos"].mean() for p in parts]))]

    for it in range(max_iter):
        w = W_MAX - (W_MAX - W_MIN) * (it / max(max_iter - 1, 1))
        for p in parts:
            r1, r2 = rng.rand(n), rng.rand(n)
            p["vel"] = (w * p["vel"] + 2.0 * r1 * (p["bp"] - p["pos"])
                        + 2.0 * r2 * (gb - p["pos"]))
            prob = sigmoid(p["vel"] + bias) if transfer == "biased" else sigmoid(p["vel"])
            np_pos = (rng.rand(n) < prob).astype(float)
            if np_pos.sum() < MIN_F:
                z = np.where(np_pos == 0)[0]
                np_pos[rng.choice(z, MIN_F - int(np_pos.sum()), replace=False)] = 1
            p["pos"] = np_pos
        for p in parts:
            f = ev(p["pos"])
            if f > p["bf"]:
                p["bf"], p["bp"] = f, p["pos"].copy()
            if f > gf:
                gf, gb = f, p["pos"].copy()
        traj.append(float(np.mean([p["pos"].mean() for p in parts])))

    sel = [feat[i] for i in np.where(gb == 1)[0]]
    return {"traj": traj, "gbest_fit": gf, "k": len(sel),
            "auc": score_selection(sel, X_tr, y_tr, X_te, y_te),
            "recall": signal_recall(sel)}


def main():
    with open(f"data/{DATASET}.pkl", "rb") as fh:
        d = pickle.load(fh)
    X, y = d["X"], d["y"]
    feat = list(X.columns)
    n = len(X)
    X_tr, y_tr = X.iloc[:600], y.iloc[:600]
    X_te, y_te = X.iloc[605:800], y.iloc[605:800]

    base = score_selection(feat, X_tr, y_tr, X_te, y_te)
    print("=" * 92)
    print(f"  {DATASET}   p={len(feat)}   baseline(all)={base:.4f}   "
          f"{N_PART} particles x {MAX_ITER} iter")
    print("=" * 92)
    print(f"  {'init':>6}{'transfer':>10}{'bias':>6}  density trajectory (it 0,5,10,20,30)"
          f"{'k':>7}{'AUC':>9}{'recall':>8}")
    print("-" * 92)

    for init_d, transfer, bias in [
            (0.50, "sigmoid", 0.0),      # current behaviour
            (0.01, "sigmoid", 0.0),      # sparse init only  <- the proposed fix
            (0.05, "sigmoid", 0.0),
            (0.01, "biased", -4.0),      # sparse init + shifted fixed point
            (0.01, "biased", -3.0)]:
        r = run(X_tr, y_tr, X_te, y_te, feat, init_d, transfer, bias)
        t = r["traj"]
        pts = [t[0], t[min(5, len(t)-1)], t[min(10, len(t)-1)],
               t[min(20, len(t)-1)], t[-1]]
        traj_s = " ".join(f"{v:.3f}" for v in pts)
        print(f"  {init_d:>6.2f}{transfer:>10}{bias:>6.1f}  {traj_s}"
              f"{r['k']:>7}{r['auc']:>9.4f}{r['recall']:>8.2f}")


if __name__ == "__main__":
    main()
