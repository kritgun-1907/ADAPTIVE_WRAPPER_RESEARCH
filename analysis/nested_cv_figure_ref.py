"""
Manuscript Figure 3 (the nested-CV penalty figure) in its published layout, drawn
from the stored results and exported as vector PDF.

This layout is the reference for names, so it reproduces that look
(matplotlib defaults, full frame, legend above the panels, horizontal "default C = 1"
label) rather than the shared fig_style. Data and helpers are the same as
nested_cv_figure.py.

Run:  python analysis/nested_cv_figure_ref.py
Writes figures/fig_nested_cv-updated.pdf (and a PNG preview in plots/)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analysis.nested_cv_figure as N            # load(), delta(), mean_curve(), ARMS, ORDER

plt.rcdefaults()                          # undo fig_style: this layout uses the defaults

LABEL = {"sparse_pso": "Sparse PSO", "sparse_orpsoc": "Sparse OrPSOC",
         "full_hybrid": "Full Hybrid", "cgpso": "CG-PSO",
         "taguchi_cbpso": "Taguchi CBPSO", "hpso_de": "HPSO-DE",
         "bwaoa_c": "BWOA-C", "mrmr": "mRMR", "lasso": "LASSO", "rfe": "RFE"}
FIXED, TUNED = "#1f77b4", "#d62728"


def main():
    cell, curve = N.load()
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.0),
                             gridspec_kw={"width_ratios": [1, 1, 1],
                                          "wspace": 0.42})

    for ax, (ds, p) in zip(axes[:2], N.ORDER):
        ys = np.arange(len(N.ARMS))[::-1]
        for y, arm in zip(ys, N.ARMS):
            f = N.delta(cell, ds, arm, "auc_fixed")
            t = N.delta(cell, ds, arm, "auc_tuned")
            ax.annotate("", xy=(t, y), xytext=(f, y),
                        arrowprops=dict(arrowstyle="->", lw=1.0, color="#888888",
                                        shrinkA=3, shrinkB=2))
            ax.scatter([f], [y], s=40, color=FIXED, zorder=3)
            ax.scatter([t], [y], s=45, color=TUNED, marker="D", zorder=3)
        ax.axvline(0, color="k", lw=1.0)
        ax.set_yticks(ys)
        ax.set_yticklabels([LABEL[a] for a in N.ARMS])
        ax.set_ylim(-0.5, len(N.ARMS) - 0.5)
        ax.grid(axis="x", alpha=0.35)
        ax.set_xlabel(r"$\Delta$AUC vs all-features baseline")
        ax.set_title(rf"$p = {p}$")
        ax.xaxis.set_major_locator(MultipleLocator(0.025))

    ax = axes[2]
    for ds, p, ls in [("ind17", 171, "-"), ("ind30", 301, "--")]:
        xb, yb = N.mean_curve(curve, ds, ["all"])
        xs, ys_ = N.mean_curve(curve, ds, N.ARMS)
        ax.plot(xb, yb, ls, color="#333333", lw=1.5, label=f"all features, $p={p}$")
        ax.plot(xs, ys_, ls, color=TUNED, lw=1.5, label=f"selected subsets, $p={p}$")
    ax.set_xscale("log")
    ax.set_xticks([10.0 ** e for e in range(-9, 4, 2)])
    ax.axvline(1.0, color="green", ls=":", lw=1.5)
    ax.grid(alpha=0.35)
    ax.text(0.7, 0.843, "default $C=1$", color="green", ha="right", va="bottom")
    ax.set_xlabel(r"inverse penalty $C$ (log scale)")
    ax.set_ylabel("mean inner-CV AUC")
    ax.set_title("Why: the two sides prefer opposite penalties")
    ax.legend(loc="lower left", fontsize=9)

    h = [axes[0].scatter([], [], s=40, color=FIXED, label=r"scored at $C=1$"),
         axes[0].scatter([], [], s=45, color=TUNED, marker="D",
                         label=r"$C$ tuned by inner CV")]
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.33, 1.06), columnspacing=3)

    os.makedirs("figures", exist_ok=True); os.makedirs("plots", exist_ok=True)
    out = "figures/fig_nested_cv-updated.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig("plots/fig_nested_cv-updated_preview.png", dpi=150, bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
