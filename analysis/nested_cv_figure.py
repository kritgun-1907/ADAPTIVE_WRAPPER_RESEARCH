"""
Item 1 as a figure: what tuning C does to every selection arm, and why.

THREE PANELS, AND WHY EACH IS THERE
───────────────────────────────────
(a), (b)  The result. For each arm at p = 171 and p = 301, the Delta AUC the
          paper currently reports (scored at the default C = 1) and the Delta
          AUC the same subset earns once C is chosen by inner cross-validation
          on the training folds. Every arm crosses zero. A table of twenty
          sign flips is hard to read; twenty arrows crossing a vertical line
          is not.

(c)       The mechanism, which is the part worth publishing. Mean inner-CV AUC
          against C, for the all-features baseline and for the selected
          subsets. The two curves peak at opposite ends of the range: the full
          matrix wants heavy shrinkage, the k ~ 5-12 subsets want almost none.
          C = 1 sits between them, so the single fixed default handicaps the
          baseline and leaves the arms near their optimum. That is the whole
          result in one picture -- the comparison was tilted by a default
          nobody chose.

Reads results/nested_cv_stored.json (written by nested_cv_stored.py), which
stores the per-fold inner-CV curve for every arm, so panel (c) costs nothing
extra to draw.

Run:  python analysis/nested_cv_figure.py
Writes plots/fig_nested_cv.png
"""
import os
import sys
import json
import collections

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analysis.fig_style as FS
FS.apply()

SRC = "results/nested_cv_stored.json"
OUT = "plots"
ORDER = [("ind17", 171), ("ind30", 301)]
ARMS = ["sparse_pso", "sparse_orpsoc", "full_hybrid", "cgpso",
        "taguchi_cbpso", "hpso_de", "bwaoa_c", "mrmr", "lasso", "rfe"]
LABEL = FS.ARM                    # names fixed by manuscript Figs 1 and 3
FIXED, TUNED = "#1f77b4", "#d62728"


def load():
    with open(SRC) as fh:
        recs = json.load(fh)
    cell = collections.defaultdict(dict)          # (ds,arm,which) -> seed -> AUC
    curve = collections.defaultdict(list)         # (ds,arm) -> [ {C: inner} ]
    for r in recs:
        for arm, v in r["arms"].items():
            for w in ("auc_fixed", "auc_tuned"):
                # Fold 0 is excluded: its 150-row training window holds one
                # class, so it is an AUC=0.5 placeholder for every arm and the
                # baseline (Methods §2.4.2). Matches Table 9 of the manuscript.
                cell[(r["ds"], arm, w)][r["seed"]] = float(np.mean(v[w][1:]))
            curve[(r["ds"], arm)].extend([c for c in v.get("inner", []) if c])
    return cell, curve


def delta(cell, ds, arm, which):
    b, s = cell[(ds, "all", which)], cell[(ds, arm, which)]
    k = sorted(set(b) & set(s))
    return float(np.mean([s[i] - b[i] for i in k]))


def mean_curve(curve, ds, arms):
    """Mean inner-CV AUC per C, pooled over folds, seeds and the given arms."""
    acc = collections.defaultdict(list)
    for arm in arms:
        for d in curve[(ds, arm)]:
            for c, v in d.items():
                acc[float(c)].append(v)
    xs = sorted(acc)
    return xs, [float(np.mean(acc[c])) for c in xs]


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"missing {SRC} -- run experiments/penalty/nested_cv_stored.py")
    cell, curve = load()
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(FS.FULL_W * 1.55, 4.1),
                             gridspec_kw={"width_ratios": [1, 1, 1.35]})

    # ── (a), (b): the sign flip ──────────────────────────────────────────
    for ax, (ds, p) in zip(axes[:2], ORDER):
        ys = np.arange(len(ARMS))[::-1]
        for y, arm in zip(ys, ARMS):
            f = delta(cell, ds, arm, "auc_fixed")
            t = delta(cell, ds, arm, "auc_tuned")
            ax.annotate("", xy=(t, y), xytext=(f, y),
                        arrowprops=dict(arrowstyle="-|>", lw=1.3,
                                        color="#999999",
                                        shrinkA=3.5, shrinkB=1.5))
            ax.scatter([f], [y], s=42, color=FIXED, zorder=3)
            ax.scatter([t], [y], s=42, color=TUNED, zorder=3, marker="D")
        ax.axvline(0, color="k", lw=1.4)
        ax.set_yticks(ys)
        ax.set_yticklabels([LABEL[a] for a in ARMS])
        ax.set_xlabel(r"$\Delta$AUC vs all-features baseline")
        ax.set_title(rf"$p = {p}$")
        ax.margins(x=0.16)
    axes[0].scatter([], [], s=42, color=FIXED, label=r"scored at $C = 1$")
    axes[0].scatter([], [], s=42, color=TUNED, marker="D",
                    label=r"$C$ tuned by inner CV")
    for ax in axes[:2]:
        ax.set_ylim(-0.7, len(ARMS) - 1 + 0.7)
    # one legend above the two dumbbell panels (supervisor's layout, Fig. 3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=2, bbox_to_anchor=(0.2, 1.06))

    # ── (c): the mechanism ───────────────────────────────────────────────
    ax = axes[2]
    for ds, p, ls in [("ind17", 171, "-"), ("ind30", 301, "--")]:
        xb, yb = mean_curve(curve, ds, ["all"])
        xs, ys_ = mean_curve(curve, ds, ARMS)
        ax.plot(xb, yb, ls, color="#444444", lw=1.8,
                label=f"all features, $p={p}$")
        ax.plot(xs, ys_, ls, color=TUNED, lw=1.8,
                label=f"selected subsets, $p={p}$")
    ax.set_xscale("log")
    ax.axvline(1.0, color="#2ca02c", ls=":", lw=1.8)
    lo, hi = ax.get_ylim()
    ax.text(1.6, hi - 0.002, "default $C=1$", fontsize=9, color="#2ca02c",
            va="top", ha="left")            # top-right corner is empty
    ax.set_xlabel(r"inverse penalty $C$ (log scale)")
    ax.set_ylabel("mean inner-CV AUC")
    ax.set_title("Why: the two sides prefer opposite penalties")
    ax.legend(fontsize=8, loc="lower left", frameon=True)

    fig.tight_layout()
    FS.save(fig, "fig_nested_cv", "fig_nested_cv")
    for ds, p in ORDER:
        n_flip = sum(1 for a in ARMS
                     if delta(cell, ds, a, "auc_fixed") > 0
                     and delta(cell, ds, a, "auc_tuned") < 0)
        print(f"     p={p}: {n_flip}/{len(ARMS)} arms flip from + to -")


if __name__ == "__main__":
    main()
