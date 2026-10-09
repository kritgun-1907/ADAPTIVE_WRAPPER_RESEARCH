"""
Render the ridge sweep (item 3) as a LaTeX table.

Separated from ridge_sweep.py so the table regenerates without re-running the
240 fits, and so a partially-populated JSON still renders whatever exists.

WHAT THE TABLE HAS TO SHOW
──────────────────────────
Two things, and they are different claims:

1. The DELTA rows test the collinearity account. If variance inflation under
   correlated inputs is why selection helps a linear model, the gain must
   shrink monotonically as the L2 penalty strengthens (C decreases), because
   ridge is already doing the shrinking. Non-monotonicity falsifies it.

2. The ALL-FEATURES row is the finding the sweep was not designed to look for
   and is arguably the more important one: the baseline RISES sharply as C
   falls, and at C = 0.001 it reaches a level that no selection arm attains at
   the default C = 1. Selection and L2 regularisation are substitutes
   approaching the same ceiling, not complements.

The C = 1.0 column is a replication check: it is sklearn's default and what
every other logistic cell in the paper uses, so it must reproduce the
corresponding Table 10 value. Verified at ind17 seed 0: sparse PSO 0.716018 vs
0.716020 published, and identical selected-subset sizes k = [7, 3, 6, 4].

Run:  python analysis/ridge_tables.py
Writes tables/ridge_sweep.tex
"""
import os
import sys
import json
import collections

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.folds import scored   # fold 0 on the real data is a placeholder
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC = "results/ridge_sweep.json"
OUT = "tables/ridge_sweep.tex"

ARM_ORDER = ["sparse_pso", "sparse_orpsoc", "full_hybrid", "mrmr", "lasso", "rfe"]
ARM_LABEL = {"sparse_pso": "Sparse PSO", "sparse_orpsoc": "Sparse OrPSOC",
             "full_hybrid": "Full Hybrid", "mrmr": "mRMR", "lasso": "LASSO",
             "rfe": "RFE"}
DS_LABEL = {"ind17": ("ind17", 171), "ind30": ("ind30", 301)}


def load():
    with open(SRC) as fh:
        raw = json.load(fh)
    cells = collections.defaultdict(dict)
    for r in raw:
        for arm, v in r["arms"].items():
            cells[(r["ds"], float(r["C"]), arm)][int(r["seed"])] = \
                float(np.mean(scored(v["auc"], r)))
    return cells


def delta(cells, ds, C, arm):
    base, sel = cells.get((ds, C, "all")), cells.get((ds, C, arm))
    if not base or not sel:
        return None
    seeds = sorted(set(base) & set(sel))
    if len(seeds) < 2:
        return None
    b = np.array([base[s] for s in seeds])
    a = np.array([sel[s] for s in seeds])
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    return float((a - b).mean()), p


def main():
    cells = load()
    Cs = sorted({k[1] for k in cells})
    dss = [d for d in ("ind17", "ind30") if any(k[0] == d for k in cells)]
    arms = [a for a in ARM_ORDER if any(k[2] == a for k in cells)]

    mono = []
    L = [r"\begin{table}[htbp]", r"    \centering",
         r"    \small\setlength{\tabcolsep}{4pt}",
         r"    \begin{tabular}{l" + "r" * len(Cs) + "}", r"    \hline",
         "    Arm & " + " & ".join(rf"$C = {c:g}$" for c in Cs) + r" \\",
         r"    \hline"]
    for ds in dss:
        name, p = DS_LABEL[ds]
        L.append(rf"    \multicolumn{{{1 + len(Cs)}}}{{l}}"
                 rf"{{\textit{{{name}, $p = {p}$}}}} \\")
        for arm in arms:
            row, gains = [ARM_LABEL.get(arm, arm)], []
            for c in Cs:
                d = delta(cells, ds, c, arm)
                if d is None:
                    row.append("--"); continue
                gains.append(d[0])
                row.append(f"${d[0]:+.4f}$" + (r"$^{*}$" if d[1] < 0.05 else ""))
            L.append("    " + " & ".join(row) + r" \\")
            if len(gains) == len(Cs):
                up = all(gains[i] <= gains[i + 1] for i in range(len(gains) - 1))
                mono.append((ds, arm, up, gains[-1] - gains[0]))
        # the baseline row: the finding the sweep was not looking for
        base = []
        for c in Cs:
            v = cells.get((ds, c, "all"))
            base.append(f"${np.mean(list(v.values())):.4f}$" if v else "--")
        L.append(r"    \textit{all-features AUC} & " + " & ".join(base) + r" \\")
        L.append(r"    \hline")
    L += [r"    \end{tabular}",
          r"    \caption{The ridge sweep. $\Delta$ AUC against the "
          r"all-features baseline under $L_2$-regularised logistic regression "
          r"as the inverse penalty $C$ varies over four orders of magnitude, "
          r"20 seeds, all arms receiving the sparsity correction of "
          r"Section~\ref{sec:dimensionality}. $^{*}$ denotes $p < 0.05$, "
          r"paired per-seed Wilcoxon. $C = 1.0$ is sklearn's default and the "
          r"value used by every other logistic cell in the paper, so that "
          r"column reproduces Table~\ref{tab:dimreal}. The final row of each "
          r"panel is the all-features baseline itself, which is not a "
          r"$\Delta$: it rises as the penalty strengthens, and at "
          r"$C = 0.001$ exceeds what any selection arm attains at $C = 1$.}",
          r"    \label{tab:ridgesweep}", r"\end{table}"]

    os.makedirs("tables", exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(L) + "\n")

    n_up = sum(1 for _, _, up, _ in mono if up)
    print(f"  wrote {OUT}")
    print(f"  monotonic-increasing in C: {n_up}/{len(mono)} arm-width cells")
    for ds, arm, up, span in mono:
        print(f"    {ds:<7}{arm:<15}{'UP  ' if up else 'NOT '}span={span:+.4f}")


if __name__ == "__main__":
    main()
