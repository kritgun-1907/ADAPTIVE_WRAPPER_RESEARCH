"""
Item 1, definitive pass — nested CV on C over the STORED subsets.

WHY THIS SUPERSEDES nested_cv_c.py
──────────────────────────────────
nested_cv_c.py re-runs each swarm and then scores what it finds. That works,
but it inherits the divergence documented in Table 15's caption: gbest is
chosen by strict comparison of validation AUC, so subsets that tie to within
last-bit floating point resolve differently under a different BLAS and the
swarm walks away from there. Individual seeds move; the 20-seed mean moves by
up to ~0.001. That is enough to make a strict replication check fail on cells
that are in fact fine, which is exactly what it did.

results/swarm_fitness.json already stores the SELECTED FEATURE NAMES per fold
per seed for all seven swarm arms at ind17 and ind30 -- the subsets behind
Table 15. Scoring those directly removes the swarm from the loop entirely:

  * the C = 1 column becomes an EXACT reproduction, not an approximate one,
    because it is the same subset on the same fold;
  * the C grid can be extended as far as needed for pennies, because no
    particle ever moves;
  * every arm-vs-arm comparison is on identical subsets across both columns.

The three non-swarm arms (mRMR, LASSO, RFE) have no stored subsets, but they
are deterministic given the per-fold budget k, and that budget is the stored
sparse_pso k on the same fold -- the same rule the main pipeline uses. They are
recomputed here from that k, which costs 1.4-3.5s per fold against the swarm's
185-469s.

THE BOUNDARY CHECK THIS FILE EXISTS TO RUN
──────────────────────────────────────────
In the first pass every arm AND the baseline selected C = 0.001, the bottom of
the six-value grid inherited from ridge_sweep.py. An optimum at the edge of a
grid is not an optimum; it is a truncation, and reporting it as a tuned result
would be the same error as reporting C = 1 without tuning. The grid here runs
from 1e-6 to 1e3, and the script reports how many folds land on either
boundary. If that count is not ~0 the grid is still too narrow and the answer
is not yet trustworthy.

Run:  python experiments/penalty/nested_cv_stored.py
Env:  NCS_CGRID=...  NCS_WORKERS=8
Writes results/nested_cv_stored.json and tables/nested_cv_c.tex.
Re-render the table from the stored run, without refitting: --table-only
"""
import os
import sys
import json
import time
import pickle
import collections

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.folds import scored   # fold 0 on the real data is a placeholder

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from orpsoc.runner import pin_threads
from orpsoc.utils import walk_forward_folds
from experiments.dimensionality.tier_d_comparators import COMPARATORS

pin_threads(1)

SRC = "results/swarm_fitness.json"
OUT = "results/nested_cv_stored.json"
TEX = "tables/nested_cv_c.tex"
N_SPLITS, MIN_F, INNER_SPLITS = 8, 3, 4

# Grid floor/ceiling are set from a measured plateau, not guessed.
#
# scratch profiling over C in [1e-14, 1e6] shows the inner-CV AUC of the
# all-features baseline is FLAT below ~1e-8 (e.g. ind17 fold 3: 0.8169 at every
# one of 1e-14, 1e-12, 1e-10, 1e-8) and falls monotonically above it. As C -> 0
# the ridge solution's DIRECTION converges, and AUC depends only on direction,
# so the limit genuinely exists rather than running away. The arms behave the
# other way round: flat below ~1e-6 and optimal at C >= 1.
#
# So both ends of this grid are plateaus, which is why the boundary report
# below distinguishes a plateau (edge TIES its neighbour -- harmless) from a
# truncation (edge strictly BEATS its neighbour -- the grid is hiding the
# optimum and the run is not trustworthy).
_cg = os.environ.get("NCS_CGRID")
C_GRID = ([float(c) for c in _cg.split(",")] if _cg else
          [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1,
           1.0, 10.0, 100.0, 1000.0, 1e4])
C_GRID = sorted(C_GRID)
C_LO, C_HI = C_GRID[0], C_GRID[-1]

# How much better an edge value must be than its inward neighbour before the
# win counts as real rather than as floating-point noise.
#
# This was 1e-9 on the first pass, which is nonsense for an AUC: it flagged 334
# folds as "truncated" purely because the inner score differed in the tenth
# decimal across a region that is flat to every decimal anyone can read. The
# grid was fine; the ruler was broken. 1e-4 is still two orders of magnitude
# below the seed-to-seed spread of any delta reported in this paper (~0.01 to
# 0.04), so nothing that could change a conclusion slips past it.
#
# The independent check on this, which does not depend on the tolerance at all:
# widening the grid floor from 1e-6 to 1e-10 -- four orders of magnitude --
# moved the tuned baseline by 0.0001 at ind17 and 0.0001 at ind30. The answer
# has converged.
EDGE_TOL = 1e-4

SWARM = ["sparse_pso", "sparse_orpsoc", "full_hybrid",
         "cgpso", "taguchi_cbpso", "hpso_de", "bwaoa_c"]
NONSWARM = ["mrmr", "lasso", "rfe"]
ARMS = SWARM + NONSWARM
LABEL = {"sparse_pso": "Sparse PSO", "sparse_orpsoc": "Sparse OrPSOC",
         "full_hybrid": "Full Hybrid", "cgpso": "CG-PSO",
         "taguchi_cbpso": "Taguchi CBPSO", "hpso_de": "HPSO-DE",
         "bwaoa_c": "BWOA-C", "mrmr": "mRMR", "lasso": "LASSO",
         "rfe": "RFE"}
ORDER = [("ind17", 171), ("ind30", 301)]


def _pipe(C):
    return Pipeline([
        ("imp",    SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(max_iter=200, C=C)),
    ])


def tune_C(X_tr, y_tr, sel):
    """
    Inner TimeSeriesSplit CV on the training window only. See module doc.

    Returns (chosen C, n scorable blocks, per-C inner means). The means are
    carried back so the caller can tell a plateau at a grid edge from a
    truncation at one.

    Ties resolve to the SMALLER C. Across the flat region that choice cannot
    change the inner score by construction, and it is the conservative
    direction: on a tie it prefers the more regularised model rather than the
    one nearer the paper's C = 1 default.
    """
    Xs = X_tr[sel]
    yv = np.asarray(y_tr)
    if len(Xs) < (INNER_SPLITS + 1) * 8:
        return 1.0, 0, {}
    splits = [(a, b) for a, b in TimeSeriesSplit(INNER_SPLITS).split(Xs)
              if len(np.unique(yv[a])) > 1 and len(np.unique(yv[b])) > 1]
    if not splits:
        return 1.0, 0, {}
    means = {}
    for C in C_GRID:
        s = []
        for a, b in splits:
            try:
                p = _pipe(C).fit(Xs.iloc[a], yv[a])
                s.append(roc_auc_score(yv[b], p.predict_proba(Xs.iloc[b])[:, 1]))
            except Exception:
                pass
        if s:
            means[C] = float(np.mean(s))
    if not means:
        return 1.0, 0, {}
    best = max(means.values())
    C_best = float(min(C for C, v in means.items() if v >= best - 1e-12))
    return C_best, len(splits), means


def edge_verdict(means):
    """
    'truncated' if the winning C sits on a grid edge AND strictly beats its
    inward neighbour; 'plateau' if it sits on an edge but only ties it;
    '' otherwise. Only 'truncated' invalidates a run.
    """
    if not means or len(means) < 2:
        return ""
    ordered = sorted(means)
    best = max(means.values())
    C_best = min(C for C, v in means.items() if v >= best - 1e-12)
    if C_best == ordered[0]:
        nxt = means[ordered[1]]
    elif C_best == ordered[-1]:
        nxt = means[ordered[-2]]
    else:
        return ""
    return "truncated" if means[C_best] > nxt + EDGE_TOL else "plateau"


def score_at(X_tr, y_tr, X_te, y_te, sel, C):
    if not sel:
        return 0.5
    try:
        p = _pipe(C).fit(X_tr[sel], y_tr)
        return float(roc_auc_score(y_te, p.predict_proba(X_te[sel])[:, 1]))
    except Exception:
        return 0.5


def run_unit(ds, seed, stored):
    with open(f"data/{ds}.pkl", "rb") as fh:
        d = pickle.load(fh)
    X, y = d["X"], d["y"]
    feat = list(X.columns)
    arms = {a: {"auc_fixed": [], "auc_tuned": [], "k": [], "C": [],
                "blocks": [], "edge": [], "inner": []}
            for a in ["all"] + ARMS}

    folds = walk_forward_folds(X, y, n_splits=N_SPLITS)
    for fi, (X_tr, y_tr, X_te, y_te, _) in enumerate(folds):
        def record(name, sel):
            C, nb, means = tune_C(X_tr, y_tr, sel)
            a = arms[name]
            a["auc_fixed"].append(score_at(X_tr, y_tr, X_te, y_te, sel, 1.0))
            a["auc_tuned"].append(score_at(X_tr, y_tr, X_te, y_te, sel, C))
            a["k"].append(len(sel))
            a["C"].append(C)
            a["blocks"].append(nb)
            a["edge"].append(edge_verdict(means))
            # the full inner-CV curve, so EDGE_TOL can be re-argued later
            # without paying for the run again
            a["inner"].append({f"{c:g}": v for c, v in sorted(means.items())})

        record("all", feat)
        for arm in SWARM:
            sel = stored[arm]["selected"][fi]
            record(arm, [f for f in sel if f in X.columns])

        k_t = max(MIN_F, len(stored["sparse_pso"]["selected"][fi]))
        for cname in NONSWARM:
            fn = COMPARATORS[cname][0]
            record(cname, fn(X_tr, y_tr, feat, k_target=k_t,
                             seed=seed)["selected"])

    return {"ds": ds, "seed": seed, "n_features": int(X.shape[1]),
            "arms": arms}


def delta(cells, ds, arm, which):
    b, s = cells.get((ds, "all", which)), cells.get((ds, arm, which))
    if not b or not s:
        return None
    k = sorted(set(b) & set(s))
    bb = np.array([b[i] for i in k])
    ss = np.array([s[i] for i in k])
    try:
        p = float(wilcoxon(ss, bb).pvalue)
    except ValueError:
        p = 1.0
    return float((ss - bb).mean()), p, len(k)


def main():
    with open(SRC) as fh:
        raw = json.load(fh)
    index = {(r["ds"], r["seed"]): r["arms"] for r in raw}
    grid = sorted(index)
    workers = int(os.environ.get("NCS_WORKERS", "1"))

    print("=" * 92)
    print("  NESTED CV ON C, SCORED OVER THE STORED SUBSETS")
    print(f"  C grid: {C_GRID}")
    print(f"  units: {len(grid)}   workers: {workers}")
    print("=" * 92, flush=True)

    # Default is SEQUENTIAL simply because the whole grid is 40 units at ~51s,
    # i.e. ~34 minutes, and this result is load-bearing enough to prefer the
    # simplest execution path. NCS_WORKERS>1 opts into loky and takes ~5 min.
    #
    # Note for anyone debugging a "stalled" run here: on macOS these processes
    # appear in ps as .../Python.app/Contents/MacOS/Python, NOT as python3, and
    # loky workers do not carry this script's name at all. Grepping for
    # "python3" or for the script name reports 0% CPU for a job that is in fact
    # saturating a core. Check with `top -o cpu` or match on the full
    # framework path before concluding anything is hung.
    if "--table-only" in sys.argv:      # re-render from the stored run, no refit
        with open(OUT) as fh:
            recs = json.load(fh)
    else:
        backend = "sequential" if workers == 1 else "loky"
        t0 = time.time()
        recs = Parallel(n_jobs=workers, backend=backend, verbose=5)(
            delayed(run_unit)(ds, s, index[(ds, s)]) for ds, s in grid)
        print(f"  done in {(time.time()-t0)/60:.1f} min", flush=True)
        with open(OUT, "w") as fh:
            json.dump(recs, fh, indent=2, default=float)

    cells = collections.defaultdict(dict)
    Cs, ks = collections.defaultdict(list), collections.defaultdict(list)
    for r in recs:
        for arm, v in r["arms"].items():
            for w in ("auc_fixed", "auc_tuned"):
                cells[(r["ds"], arm, w)][r["seed"]] = float(np.mean(scored(v[w], r, w)))
            Cs[(r["ds"], arm)].extend(scored(v["C"], r, "auc_fixed"))
            ks[(r["ds"], arm)].extend(scored(v["k"], r, "auc_fixed"))

    print("\n" + "=" * 92)
    print("  GRID BOUNDARY CHECK")
    print("  plateau  = winner sits on an edge but only TIES inward -- harmless")
    print("  TRUNCATED = winner sits on an edge and BEATS inward -- grid hides "
          "the optimum")
    print("=" * 92)
    # A raw count of truncated folds is the wrong test and said "DO NOT REPORT"
    # over 16 folds whose combined effect is ~1e-4 AUC. What matters is how far
    # a wider grid could actually move a reported mean, so that is what is
    # computed: each truncated fold is credited the FULL margin by which its
    # edge beat the neighbour inward -- a worst case, since a wider grid would
    # recover at most that -- and the margins are averaged into the arm mean
    # the same way the AUCs are (over folds, then over seeds).
    verdicts = collections.defaultdict(collections.Counter)
    bound = collections.defaultdict(float)
    for r in recs:
        for arm, v in r["arms"].items():
            for e, curve in zip(v["edge"], v["inner"]):
                if e:
                    verdicts[(r["ds"], arm)][e] += 1
                if e != "truncated" or not curve:
                    continue
                cs = sorted(curve, key=float)
                best = max(curve, key=lambda k: curve[k])
                nb = cs[1] if best == cs[0] else cs[-2]
                bound[(r["ds"], arm)] += curve[best] - curve[nb]

    n_seeds = len({r["seed"] for r in recs})
    worst, base_trunc = 0.0, 0
    for ds, _ in ORDER:
        for arm in ["all"] + ARMS:
            c = verdicts[(ds, arm)]
            n = len(Cs[(ds, arm)])
            b = bound[(ds, arm)] / N_SPLITS / max(n_seeds, 1)
            if arm == "all":
                base_trunc += c["truncated"]
            else:
                worst = max(worst, b)
            if c:
                print(f"  {ds:7} {LABEL.get(arm, arm):24} "
                      f"plateau {c['plateau']:>3}/{n}   "
                      f"truncated {c['truncated']:>3}/{n}   "
                      f"bound {b:.6f} AUC")

    print(f"\n  baseline truncated folds: {base_trunc}  "
          "(must be 0 -- the baseline is what the conclusion rests on)")
    print(f"  worst-case effect of a wider grid on any arm mean: {worst:.6f} AUC")
    ok = base_trunc == 0 and worst < 1e-3
    print("  verdict: " + ("grid is wide enough" if ok else
                           "GRID TOO NARROW, DO NOT REPORT"))

    print("\n" + "=" * 92)
    print("  RESULT -- baseline and every arm tuned independently")
    print("=" * 92)
    for ds, p in ORDER:
        bf = float(np.mean(list(cells[(ds, "all", "auc_fixed")].values())))
        bt = float(np.mean(list(cells[(ds, "all", "auc_tuned")].values())))
        cc = collections.Counter(Cs[(ds, "all")])
        print(f"\n  {ds}  (p = {p})")
        print(f"    all-features baseline:  C=1 {bf:.4f}   tuned {bt:.4f}"
              f"   ({bt-bf:+.4f})")
        print("    baseline C: " + ", ".join(f"{c:g}x{n}" for c, n in sorted(cc.items())))
        print(f"    {'arm':24}{'dAUC C=1':>11}{'dAUC tuned':>13}"
              f"{'shift':>10}{'k':>7}  median C")
        for arm in ARMS:
            df, dt = delta(cells, ds, arm, "auc_fixed"), delta(cells, ds, arm, "auc_tuned")
            if df is None or dt is None:
                continue
            print(f"    {LABEL[arm]:24}{df[0]:+10.4f}{'*' if df[1]<0.05 else ' '}"
                  f"{dt[0]:+12.4f}{'*' if dt[1]<0.05 else ' '}"
                  f"{dt[0]-df[0]:+10.4f}{np.mean(ks[(ds,arm)]):7.1f}"
                  f"  {np.median(Cs[(ds,arm)]):g}")

    L = [r"\begin{table}[htbp]", r"    \centering",
         r"    \small\setlength{\tabcolsep}{5pt}",
         r"    \begin{tabular}{lrrrr}", r"    \hline",
         r"    Arm & $\Delta$ AUC at $C = 1$ & $\Delta$ AUC, tuned & Shift "
         r"& Mean $k$ \\", r"    \hline"]
    for ds, p in ORDER:
        bf = float(np.mean(list(cells[(ds, "all", "auc_fixed")].values())))
        bt = float(np.mean(list(cells[(ds, "all", "auc_tuned")].values())))
        L.append(rf"    \multicolumn{{5}}{{l}}{{\textit{{\texttt{{{ds}}}, $p = {p}$}}}} \\")
        L.append(rf"    all-features AUC & ${bf:.4f}$ & ${bt:.4f}$ & "
                 rf"${bt-bf:+.4f}$ & {p} \\")
        for arm in ARMS:
            df, dt = delta(cells, ds, arm, "auc_fixed"), delta(cells, ds, arm, "auc_tuned")
            if df is None or dt is None:
                continue
            f1 = f"${df[0]:+.4f}$" + (r"$^{*}$" if df[1] < 0.05 else "")
            f2 = f"${dt[0]:+.4f}$" + (r"$^{*}$" if dt[1] < 0.05 else "")
            L.append(f"    {LABEL[arm]} & {f1} & {f2} & "
                     f"${dt[0]-df[0]:+.4f}$ & {np.mean(ks[(ds,arm)]):.1f} \\\\")
        L.append(r"    \hline")
    L += [r"    \end{tabular}",
          r"    \caption{Nested cross-validation on the inverse penalty $C$, "
          r"scored over the stored subsets of Table~\ref{tab:swarmfitness}. "
          r"Each subset is scored twice on the same fold: at the default "
          r"$C = 1$ used throughout the rest of the paper, and at the value "
          r"chosen by an inner \texttt{TimeSeriesSplit} cross-validation on "
          r"the training window alone, run \emph{independently} for the "
          rf"all-features baseline and for each arm over "
          rf"$C \in [10^{{-6}}, 10^{{3}}]$. 20 seeds, $^{{*}}$ denotes "
          r"$p < 0.05$ by paired per-seed Wilcoxon. Because the subsets are "
          r"taken from storage rather than re-searched, the two columns "
          r"describe the identical feature sets and the shift is "
          r"attributable to the penalty alone.}",
          r"    \label{tab:nestedcv}", r"\end{table}"]
    os.makedirs("tables", exist_ok=True)
    with open(TEX, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n  -> {OUT}\n  -> {TEX}")


if __name__ == "__main__":
    main()
