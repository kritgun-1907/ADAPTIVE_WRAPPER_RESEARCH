"""
Render the k-NN irrelevance-vs-redundancy comparison (item 4) as a LaTeX table.

THE QUESTION THIS SETTLES
─────────────────────────
Section 3.9 reports a $k$-NN anomaly: $k$-NN performs no embedded selection
whatever, so an account in which embedded-selection capacity sets the sign
predicts it should behave like logistic regression and gain from filtering. On
the real ladder it does the opposite -- it loses, badly and almost everywhere.

The two testbeds differ in WHAT the added dimensions are:

    real ladder      added columns are further industry portfolios --
                     correlated with those already present. REDUNDANT.
    synthetic sweep  added columns are drawn independently of y and of each
                     other. IRRELEVANT.

Distance-based methods have a well-known weakness to irrelevant dimensions:
every noise column contributes to the metric and dilutes the signal. They have
no comparable weakness to redundancy. So the prediction is a SIGN REVERSAL for
the same estimator, the same selectors and the same protocol, driven purely by
which kind of dimension is being added.

If that holds, the paper states something sharper than either account alone:
linear models are hurt by both redundancy and irrelevance, whereas
distance-based and tree-based methods are hurt only by irrelevance. That
resolves the anomaly rather than noting it.

TWO SOURCES, AND WHY THEY DIFFER
────────────────────────────────
Panel (a), the synthetic sweep, reads the checkpoint store directly: those
cells were produced by invocations that each overwrote
results/tier_d_fullnovelty.json, so the store is the only place holding all of
them, and it is keyed by provenance so generations cannot mix.

Panel (b), the real ladder, reads results/tier_d_ladder_merged.json instead.
Reading it from the store is what allowed the published panel to drift from
Table 9 on the `rfe` arm; see load_ladder() for the mechanism. main() also
diffs the two sources across the ladder and prints any disagreement, so the
next divergence is loud rather than silent.

Run:  python analysis/knn_synthetic_tables.py
Writes tables/knn_irrelevance.tex
"""
import os
import sys
import pickle
import collections

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.folds import scored   # fold 0 on the real data is a placeholder
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STORE = "results/checkpoints/tier_d_fullnovelty"
# Panel (b) source -- see load_ladder() for why this is not read from STORE.
LADDER_SRC = "results/tier_d_ladder_merged.json"


def pick_generation(store):
    """
    Choose the checkpoint generation to read, WITHOUT using mtime.

    This selected `max(gens, key=os.path.getmtime)` until 2026-09-10, when
    untarring results fetched from EC2 reset every generation directory's mtime
    to the same minute. The "newest" generation then became whichever one the
    filesystem happened to return first: build_ladder.py silently rebuilt the
    merged ladder from a 200-unit generation instead of the 1304-unit one,
    dropping every 20-seed cell and both extra estimators, and
    knn_synthetic_tables.py emitted a table with no rows at all. Neither raised
    an error -- the tables were simply wrong.

    mtime answers "when was this directory last touched", which is not the
    question. Unit count answers "which generation actually holds the run", and
    is invariant to copying, extracting and syncing. ORPSOC_GEN overrides it
    when a specific generation is wanted.
    """
    import os
    pin = os.environ.get("ORPSOC_GEN")
    gens = [d for d in os.listdir(store) if os.path.isdir(os.path.join(store, d))]
    if not gens:
        raise SystemExit(f"no checkpoint generations under {store}")
    if pin:
        if pin not in gens:
            raise SystemExit(f"ORPSOC_GEN={pin} not present under {store}")
        return pin
    def n_units(d):
        p = os.path.join(store, d)
        return sum(1 for f in os.listdir(p) if f.endswith(".pkl"))
    ranked = sorted(gens, key=n_units, reverse=True)
    if len(ranked) > 1 and n_units(ranked[0]) == n_units(ranked[1]):
        raise SystemExit(
            f"cannot choose between generations {ranked[0]} and {ranked[1]}: "
            f"both hold {n_units(ranked[0])} units. Set ORPSOC_GEN explicitly.")
    return ranked[0]
OUT = "tables/knn_irrelevance.tex"

WIDTHS = [50, 200, 500, 1000]
LADDER = [("ind5", 51), ("ind10", 101), ("ind12", 121), ("ind17", 171),
          ("ind30", 301), ("ind38", 361), ("ind48", 481), ("ff49ind", 491)]
ARM_ORDER = ["sparse_pso", "sparse_orpsoc", "mrmr", "lasso", "rfe"]
ARM_LABEL = {"sparse_pso": "Sparse PSO", "sparse_orpsoc": "Sparse OrPSOC",
             "mrmr": "mRMR", "lasso": "LASSO", "rfe": "RFE"}


def load_ladder():
    """
    Panel (b) comes from results/tier_d_ladder_merged.json, NOT from the store.

    WHY THIS IS NOT OPTIONAL
    ────────────────────────
    Panel (b) and Table~\\ref{tab:knn} (ladder_knn.tex) report the SAME
    quantity: k-NN Delta AUC against the all-features baseline on the real
    ladder. They were computed from two different sources and drifted, and the
    drift was confined to the `rfe` arm -- the fingerprint of the scikit-learn
    1.9.0 leak documented in Section 4.

    The mechanism: generation 29a57cced36f holds TWO units per (ds, model,
    seed) for ind5..ind30 -- the original multi-arm run written before the
    2026-09-08 pin (whose `rfe` is 1.9.0) and an rfe-only re-run written after
    it (whose `rfe` is the verified 1.8.0). load() below walks os.listdir in
    filesystem order and lets the last file seen win per seed, so each seed
    independently resolved to whichever unit happened to be read second. The
    published panel (b) RFE row was therefore a per-seed BLEND of two library
    versions -- worse than either one alone, and matching neither.

    build_ladder.py already resolves this correctly and writes the result to
    tier_d_ladder_merged.json. Reading that file here makes panel (b) equal to
    Table~\\ref{tab:knn} by construction rather than by coincidence.
    """
    import json
    with open(LADDER_SRC) as fh:
        recs = json.load(fh)
    cells = collections.defaultdict(dict)
    for r in recs:
        if r.get("model") != "kNN":
            continue
        for arm, v in r["arms"].items():
            cells[(r["ds"], arm)][int(r["seed"])] = float(np.mean(scored(v["auc"], r)))
    return cells


def load():
    """(ds, arm) -> {seed: mean AUC}, k-NN only, newest provenance generation."""
    gens = [d for d in os.listdir(STORE)
            if os.path.isdir(os.path.join(STORE, d))]
    if not gens:
        raise SystemExit(f"no checkpoint generations under {STORE}")
    gen = pick_generation(STORE)
    cells = collections.defaultdict(dict)
    root = os.path.join(STORE, gen)
    for f in os.listdir(root):
        if not f.endswith(".pkl"):
            continue
        try:
            with open(os.path.join(root, f), "rb") as fh:
                obj = pickle.load(fh)
        except Exception:
            continue
        r = obj.get("payload", obj)
        if r.get("model") != "kNN":
            continue
        for arm, v in r["arms"].items():
            cells[(r["ds"], arm)][int(r["seed"])] = float(np.mean(scored(v["auc"], r)))
    return cells, gen


def delta(cells, ds, arm):
    base, sel = cells.get((ds, "all")), cells.get((ds, arm))
    if not base or not sel:
        return None
    seeds = sorted(set(base) & set(sel))
    if len(seeds) < 3:
        return None
    b = np.array([base[s] for s in seeds])
    a = np.array([sel[s] for s in seeds])
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    return float((a - b).mean()), p, len(seeds)


def cell(d):
    if d is None:
        return "--"
    return f"${d[0]:+.4f}$" + (r"$^{*}$" if d[1] < 0.05 else "")


def main():
    cells, gen = load()
    lad_cells = load_ladder()
    arms = [a for a in ARM_ORDER if any(k[1] == a for k in cells)]

    # Guard: panel (b) must equal Table 9 cell for cell. Report any cell where
    # the store-derived value disagrees with the merged ladder, so a future
    # divergence is loud instead of silent.
    drift = []
    for ds, _p in LADDER:
        for arm in ARM_ORDER:
            a, b = delta(cells, ds, arm), delta(lad_cells, ds, arm)
            if a and b and abs(a[0] - b[0]) > 1e-9:
                drift.append((ds, arm, a[0], b[0]))
    if drift:
        print("  store-vs-merged disagreement on the real ladder "
              "(merged wins; this is the 1.9.0 leak):")
        for ds, arm, s, m in drift:
            print(f"    {ds:9} {arm:16} store {s:+.4f}   merged {m:+.4f}"
                  f"   diff {s - m:+.4f}")

    L = [r"\begin{table}[htbp]", r"    \centering",
         r"    \small\setlength{\tabcolsep}{4pt}",
         r"    \begin{tabular}{l" + "rr" * len(WIDTHS) + "}", r"    \hline",
         r"    \multicolumn{" + str(1 + 2 * len(WIDTHS)) +
         r"}{l}{\textit{(a) Synthetic sweep --- added dimensions are "
         r"IRRELEVANT (independent noise)}} \\",
         "    & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$p = {p}$}}"
                               for p in WIDTHS) + r" \\",
         "    Arm & " + " & ".join(["AR(1) & Regime"] * len(WIDTHS)) + r" \\",
         r"    \hline"]
    for arm in arms:
        row = [ARM_LABEL.get(arm, arm)]
        for p in WIDTHS:
            row.append(cell(delta(cells, f"hd_ar1_p{p}", arm)))
            row.append(cell(delta(cells, f"hd_regime_switch_p{p}", arm)))
        L.append("    " + " & ".join(row) + r" \\")
    L.append(r"    \hline")

    lad = [(d, p) for d, p in LADDER if (d, "all") in lad_cells]
    L += [r"    \multicolumn{" + str(1 + 2 * len(WIDTHS)) +
          r"}{l}{\textit{(b) Real ladder --- added dimensions are REDUNDANT "
          r"(correlated portfolios)}} \\",
          "    Arm & " + " & ".join(
              rf"\multicolumn{{2}}{{c}}{{$p = {p}$}}" for _, p in lad[:len(WIDTHS)])
          + r" \\", r"    \hline"]
    for arm in arms:
        row = [ARM_LABEL.get(arm, arm)]
        for ds, _ in lad[:len(WIDTHS)]:
            row.append(r"\multicolumn{2}{c}{" + cell(delta(lad_cells, ds, arm)) + "}")
        L.append("    " + " & ".join(row) + r" \\")
    L += [r"    \hline", r"    \end{tabular}",
          r"    \caption{The $k$-NN sign reversal. $\Delta$ AUC against the "
          r"all-features baseline under $k$-nearest neighbours ($k = 15$), "
          r"20 seeds, $^{*}$ denotes $p < 0.05$ by paired per-seed Wilcoxon. "
          r"Panel (a) adds dimensions that are independent of $y$ and of each "
          r"other; panel (b) adds further industry portfolios of the same "
          r"market, which are correlated with those already present. The "
          r"estimator, the selectors, the swarm budget and the evaluation "
          r"protocol are identical throughout; only the nature of the added "
          r"dimensions differs, and the sign reverses with it.}",
          r"    \label{tab:knnirrelevance}", r"\end{table}"]

    os.makedirs("tables", exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(L) + "\n")

    syn = [d for p in WIDTHS for lv in ("hd_ar1_p", "hd_regime_switch_p")
           for a in ARM_ORDER if (d := delta(cells, f"{lv}{p}", a))]
    lg = [d for ds, _ in LADDER for a in ARM_ORDER
          if (d := delta(lad_cells, ds, a))]
    print(f"  wrote {OUT}   (generation {gen})")
    print(f"  synthetic: {sum(1 for x in syn if x[0] > 0 and x[1] < .05)}/{len(syn)}"
          f" significantly POSITIVE")
    print(f"  ladder   : {sum(1 for x in lg if x[0] < 0 and x[1] < .05)}/{len(lg)}"
          f" significantly NEGATIVE")


if __name__ == "__main__":
    main()
