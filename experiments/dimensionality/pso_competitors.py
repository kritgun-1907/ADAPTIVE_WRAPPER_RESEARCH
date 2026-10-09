"""
Four enhanced-swarm competitors, requested at review (2026-08-18).

The comparator table currently holds OrPSOC against a plain binary PSO and three
non-swarm methods (mRMR, LASSO, RFE). The reviewer's point is fair: OrPSOC is the
only *enhanced* swarm in the table, so it is not being asked to beat anything of
its own class.

  cgpso           Correlation-Guided PSO
  taguchi_cbpso   Taguchi chaotic binary PSO
  hpso_de         Hybrid PSO / Differential Evolution
  bwaoa_c         Binary whale-inspired optimiser with crossover

HONEST PROVENANCE, AND THE LIMIT OF THIS COMPARISON
───────────────────────────────────────────────────
These are implementations of the mechanisms as described in the literature, NOT
verbatim reproductions of any one paper's code, and each below states exactly
which mechanism it implements and which parameter values were chosen here. Where
a paper's setting is not reproducible from its text, the value is set to match
our own configuration (20 particles, 60 iterations, min_f=3, theta=0.5) so the
budget is identical across arms. A competitor that loses under a parameter set
its authors did not choose is weak evidence against that method, and the
manuscript must say so rather than claiming a clean win.

FAIRNESS: SPARSITY IS GIVEN TO EVERY ARM
────────────────────────────────────────
Section 3.9 established that binary PSO with a sigmoid transfer converges to
selecting ~50% of features at any p, because sigma(0)=0.5 is a fixed point --
a defect of the encoding, not of any particular algorithm. Every arm here is
therefore given the same corrected treatment our own selector receives: sparse
initialisation via build_sparse_positions and a negative transfer bias. Running
the competitors at 50% density while our method runs sparse would manufacture a
win, and would be the first thing a referee checked.

Run:  python experiments/dimensionality/pso_competitors.py     (self-test on one fold)
Imported by: experiments/dimensionality/tier_d_fullnovelty.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orpsoc.utils import (sigmoid, crossover, FoldEvalContext, evaluate_ctx,
                          build_sparse_positions)

N_PARTICLES, MAX_ITER = 20, 60
MIN_F, THETA = 3, 0.5
INIT_DENSITY, BIAS = 0.05, -3.0
W_MAX, W_MIN = 0.9, 0.4


def _ctx_and_eval(X_tr, y_tr, feat, model_factory):
    cut = int(len(X_tr) * 0.75)
    ctx = FoldEvalContext(X_tr.iloc[:cut], y_tr.iloc[:cut],
                          X_tr.iloc[cut:], y_tr.iloc[cut:], feat,
                          model_factory=model_factory)
    cache = {}

    def ev(pos):
        k = pos.astype(np.uint8).tobytes()
        if k not in cache:
            cache[k] = evaluate_ctx(pos, ctx, MIN_F, THETA)
        return cache[k]
    return ev


def _repair(pos, rng, n):
    if pos.sum() < MIN_F:
        z = np.where(pos == 0)[0]
        need = MIN_F - int(pos.sum())
        if len(z) >= need:
            pos[rng.choice(z, need, replace=False)] = 1
    return pos


def _init(n, seed, rng):
    return build_sparse_positions(N_PARTICLES, n, INIT_DENSITY, seed=seed)


# ── 1. Correlation-Guided PSO ────────────────────────────────────────────────
def select_cgpso(X_tr, y_tr, feat, seed=42, model_factory=None, **kw):
    """
    MECHANISM: a filter signal guides the swarm. Feature-target correlation is
    computed on the training window only and enters the VELOCITY UPDATE as a
    standing attraction term, so highly-correlated features are easier to
    switch on and harder to switch off (after Wang et al. 2018, IJDSN 14(4)).
    Initialisation is the shared sparse draw used by every arm (_init); the
    correlation does NOT bias it. (An earlier version of this docstring said it
    did; the manuscript describes the code as run.)

    PARAMETERS CHOSEN HERE: correlation weight c3 = 1.0, matched to the social
    and cognitive weights (2.0 each) at half strength so the guidance informs
    without dominating. Correlations are |Pearson| rescaled to [0,1].
    """
    rng = np.random.RandomState(seed)
    n = len(feat)
    ev = _ctx_and_eval(X_tr, y_tr, feat, model_factory)

    Xv = X_tr.to_numpy(dtype=float)
    yv = np.asarray(y_tr, dtype=float)
    sd = Xv.std(axis=0); sd[sd == 0] = 1.0
    corr = np.abs(((Xv - Xv.mean(0)) / sd * (yv - yv.mean())[:, None]).mean(0))
    corr = np.nan_to_num(corr); corr = corr / (corr.max() + 1e-12)
    c3 = 1.0

    parts = []
    init = _init(n, seed, rng)
    for i in range(N_PARTICLES):
        pos = _repair(init[i].copy(), rng, n)
        parts.append({"pos": pos, "vel": rng.randn(n) * 0.1,
                      "bp": pos.copy(), "bf": ev(pos)})
    gb = max(parts, key=lambda p: p["bf"])["bp"].copy()
    gf = max(p["bf"] for p in parts)

    for it in range(MAX_ITER):
        w = W_MAX - (W_MAX - W_MIN) * (it / max(MAX_ITER - 1, 1))
        for p in parts:
            r1, r2, r3 = rng.rand(n), rng.rand(n), rng.rand(n)
            p["vel"] = (w * p["vel"]
                        + 2.0 * r1 * (p["bp"] - p["pos"])
                        + 2.0 * r2 * (gb - p["pos"])
                        + c3 * r3 * corr)          # correlation guidance
            p["pos"] = _repair((rng.rand(n) < sigmoid(p["vel"] + BIAS)).astype(float), rng, n)
        for p in parts:
            f = ev(p["pos"])
            if f > p["bf"]:
                p["bf"], p["bp"] = f, p["pos"].copy()
            if f > gf:
                gf, gb = f, p["pos"].copy()
    return {"selected": [feat[i] for i in np.where(gb == 1)[0]]}


# ── 2. Taguchi chaotic binary PSO ────────────────────────────────────────────
def select_taguchi_cbpso(X_tr, y_tr, feat, seed=42, model_factory=None, **kw):
    """
    MECHANISM: two additions to binary PSO. (a) CHAOTIC inertia -- the inertia
    weight is driven by a logistic map z <- 4z(1-z) rather than a linear decay,
    which keeps the schedule non-monotonic and is the usual chaotic-PSO device.
    (b) TAGUCHI crossover -- rather than blending two parents at random cut
    points, a two-level orthogonal array assigns each feature to one parent, the
    combination is evaluated, and the better level per factor is retained.

    PARAMETERS CHOSEN HERE: logistic map with z0 = 0.7 (avoiding the fixed
    points 0, 0.25, 0.5, 0.75); inertia mapped to [0.4, 0.9] to match the range
    the other arms use; the orthogonal array is a Hadamard-style sign matrix of
    the next power of two above n_particles, truncated -- the standard
    construction when an exact L-array of the required size is unavailable.
    """
    rng = np.random.RandomState(seed)
    n = len(feat)
    ev = _ctx_and_eval(X_tr, y_tr, feat, model_factory)

    def hadamard(m):
        h = np.array([[1.0]])
        while h.shape[0] < m:
            h = np.block([[h, h], [h, -h]])
        return h[:m, :m]

    H = hadamard(int(2 ** np.ceil(np.log2(max(N_PARTICLES, 2)))))
    z = 0.7

    parts = []
    init = _init(n, seed, rng)
    for i in range(N_PARTICLES):
        pos = _repair(init[i].copy(), rng, n)
        parts.append({"pos": pos, "vel": rng.randn(n) * 0.1,
                      "bp": pos.copy(), "bf": ev(pos)})
    gb = max(parts, key=lambda p: p["bf"])["bp"].copy()
    gf = max(p["bf"] for p in parts)

    for it in range(MAX_ITER):
        z = 4.0 * z * (1.0 - z)
        z = min(max(z, 1e-6), 1 - 1e-6)
        w = W_MIN + (W_MAX - W_MIN) * z          # chaotic inertia
        for p in parts:
            r1, r2 = rng.rand(n), rng.rand(n)
            p["vel"] = (w * p["vel"] + 2.0 * r1 * (p["bp"] - p["pos"])
                        + 2.0 * r2 * (gb - p["pos"]))
            p["pos"] = _repair((rng.rand(n) < sigmoid(p["vel"] + BIAS)).astype(float), rng, n)

        # Taguchi combination between each particle's best and the global best
        for p in parts:
            row = H[rng.randint(H.shape[0])]
            lvl = np.resize((row > 0).astype(float), n)
            child = np.where(lvl > 0, p["bp"], gb)
            child = _repair(child.astype(float), rng, n)
            f = ev(child)
            if f > p["bf"]:
                p["bf"], p["bp"], p["pos"] = f, child.copy(), child.copy()
                if f > gf:
                    gf, gb = f, child.copy()

        for p in parts:
            f = ev(p["pos"])
            if f > p["bf"]:
                p["bf"], p["bp"] = f, p["pos"].copy()
            if f > gf:
                gf, gb = f, p["pos"].copy()
    return {"selected": [feat[i] for i in np.where(gb == 1)[0]]}


# ── 3. Hybrid PSO / Differential Evolution ───────────────────────────────────
def select_hpso_de(X_tr, y_tr, feat, seed=42, model_factory=None, **kw):
    """
    MECHANISM: PSO drives the velocity, DE supplies the variation. Each
    iteration performs a standard binary-PSO update, then a DE/rand/1 mutation
    in velocity space (v_a + F(v_b - v_c)) with binomial crossover against the
    incumbent, and keeps the trial vector only if it improves -- DE's greedy
    selection rather than PSO's unconditional move.

    PARAMETERS CHOSEN HERE: F = 0.5 and CR = 0.9, the values most commonly
    reported as robust defaults for DE/rand/1/bin; donors are drawn without
    replacement and excluding the target.
    """
    rng = np.random.RandomState(seed)
    n = len(feat)
    ev = _ctx_and_eval(X_tr, y_tr, feat, model_factory)
    F, CR = 0.5, 0.9

    parts = []
    init = _init(n, seed, rng)
    for i in range(N_PARTICLES):
        pos = _repair(init[i].copy(), rng, n)
        parts.append({"pos": pos, "vel": rng.randn(n) * 0.1,
                      "bp": pos.copy(), "bf": ev(pos)})
    gb = max(parts, key=lambda p: p["bf"])["bp"].copy()
    gf = max(p["bf"] for p in parts)

    for it in range(MAX_ITER):
        w = W_MAX - (W_MAX - W_MIN) * (it / max(MAX_ITER - 1, 1))
        for i, p in enumerate(parts):
            r1, r2 = rng.rand(n), rng.rand(n)
            p["vel"] = (w * p["vel"] + 2.0 * r1 * (p["bp"] - p["pos"])
                        + 2.0 * r2 * (gb - p["pos"]))
            p["pos"] = _repair((rng.rand(n) < sigmoid(p["vel"] + BIAS)).astype(float), rng, n)

            idx = [j for j in range(N_PARTICLES) if j != i]
            a, b, c = rng.choice(idx, 3, replace=False)
            donor = parts[a]["vel"] + F * (parts[b]["vel"] - parts[c]["vel"])
            mask = rng.rand(n) < CR
            mask[rng.randint(n)] = True          # DE guarantees >=1 donor gene
            trial_v = np.where(mask, donor, p["vel"])
            trial = _repair((rng.rand(n) < sigmoid(trial_v + BIAS)).astype(float), rng, n)
            ft = ev(trial)
            if ft > ev(p["pos"]):                # DE greedy selection
                p["pos"], p["vel"] = trial, trial_v

        for p in parts:
            f = ev(p["pos"])
            if f > p["bf"]:
                p["bf"], p["bp"] = f, p["pos"].copy()
            if f > gf:
                gf, gb = f, p["pos"].copy()
    return {"selected": [feat[i] for i in np.where(gb == 1)[0]]}


# ── 4. Binary whale-inspired optimiser with crossover ────────────────────────
def select_bwaoa_c(X_tr, y_tr, feat, seed=42, model_factory=None, **kw):
    """
    MECHANISM: whale-optimisation dynamics binarised, plus crossover. Each
    iteration an agent either (a) encircles the incumbent best, (b) spirals
    toward it via the log-spiral bubble-net term, or (c) explores around a
    random agent, selected by the standard |A|<1 / |A|>=1 rule with a
    coefficient a decaying linearly from 2 to 0. The continuous position is
    mapped to bits through the same biased sigmoid the other arms use, and a
    two-point crossover stage is added, which is the "-C" in the name.

    PARAMETERS CHOSEN HERE: spiral constant b = 1.0 and l ~ U(-1,1), the values
    in the original whale formulation; crossover rate 0.6, matched to OrPSOC's
    so the two differ in swarm dynamics rather than in crossover intensity.
    """
    rng = np.random.RandomState(seed)
    n = len(feat)
    ev = _ctx_and_eval(X_tr, y_tr, feat, model_factory)
    b_spiral = 1.0

    init = _init(n, seed, rng)
    agents = [{"pos": _repair(init[i].copy(), rng, n)} for i in range(N_PARTICLES)]
    for ag in agents:
        ag["fit"] = ev(ag["pos"])
    best = max(agents, key=lambda a: a["fit"])
    gb, gf = best["pos"].copy(), best["fit"]

    for it in range(MAX_ITER):
        a_coef = 2.0 - 2.0 * it / max(MAX_ITER - 1, 1)
        for ag in agents:
            r = rng.rand(n)
            A = 2 * a_coef * r - a_coef
            C = 2 * rng.rand(n)
            if rng.rand() < 0.5:
                if np.mean(np.abs(A)) < 1:
                    cont = gb - A * np.abs(C * gb - ag["pos"])          # encircle
                else:
                    rnd = agents[rng.randint(N_PARTICLES)]["pos"]
                    cont = rnd - A * np.abs(C * rnd - ag["pos"])        # explore
            else:
                l = rng.uniform(-1, 1, n)
                cont = (np.abs(gb - ag["pos"]) * np.exp(b_spiral * l)
                        * np.cos(2 * np.pi * l) + gb)                   # bubble-net
            ag["pos"] = _repair((rng.rand(n) < sigmoid(cont + BIAS)).astype(float), rng, n)

        idx = list(range(N_PARTICLES)); rng.shuffle(idx)
        for k in range(0, N_PARTICLES - 1, 2):
            pa, pb = agents[idx[k]], agents[idx[k + 1]]
            ca, cb = crossover(pa["pos"], pb["pos"], 0.6, MIN_F, rng)
            for child, parent in ((ca, pa), (cb, pb)):
                f = ev(child)
                if f > parent["fit"]:
                    parent["pos"], parent["fit"] = child, f

        for ag in agents:
            ag["fit"] = ev(ag["pos"])
            if ag["fit"] > gf:
                gf, gb = ag["fit"], ag["pos"].copy()
    return {"selected": [feat[i] for i in np.where(gb == 1)[0]]}


COMPETITORS = {"cgpso": select_cgpso, "taguchi_cbpso": select_taguchi_cbpso,
               "hpso_de": select_hpso_de, "bwaoa_c": select_bwaoa_c}


if __name__ == "__main__":
    import pickle
    from experiments.dimensionality.tier_d_comparators import score_selection
    with open("data/hd_ar1_p200.pkl", "rb") as fh:
        d = pickle.load(fh)
    X, y = d["X"], d["y"]
    feat = list(X.columns)
    X_tr, y_tr = X.iloc[:600], y.iloc[:600]
    X_te, y_te = X.iloc[605:800], y.iloc[605:800]
    print(f"self-test  p={len(feat)}  (1 fold, LightGBM)")
    print(f"  {'arm':<16}{'k':>5}{'AUC':>9}{'sec':>8}")
    print(f"  {'all':<16}{len(feat):>5}"
          f"{score_selection(feat, X_tr, y_tr, X_te, y_te):>9.4f}{0.0:>8.1f}")
    for nm, fn in COMPETITORS.items():
        t0 = time.time()
        sel = fn(X_tr, y_tr, feat, seed=0)["selected"]
        print(f"  {nm:<16}{len(sel):>5}"
              f"{score_selection(sel, X_tr, y_tr, X_te, y_te):>9.4f}"
              f"{time.time()-t0:>8.1f}")
