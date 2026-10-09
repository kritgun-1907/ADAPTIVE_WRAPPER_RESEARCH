"""
One source of truth for how every manuscript and supplementary figure reads
and looks: arm names, colours, font sizes and export format.

Names follow manuscript Figures 1 and 3, which are the reference. Every figure script imports from here, so a
name changed here changes everywhere and cannot drift again.

Conventions
-----------
* Single-panel figures carry no title (the caption does that job);
  multi-panel figures carry one short title per panel. No figure-level titles.
* Figures are exported as vector PDF under their manuscript file names into
  figures/, plus a PNG preview into plots/.
"""
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(REPO, "plots")
PAPER_FIGS = os.path.join(REPO, "figures")


def apply():
    plt.rcParams.update({
        "font.size": 10.5, "axes.titlesize": 10.5, "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
        "axes.titleweight": "normal",
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300,
        # embed TrueType so text stays editable and searchable in the PDF
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# ── arm names (high-dimensional comparison, p > 50) ─────────────────────────
ARM = {
    "sparse_pso": "Sparse PSO", "sparse_orpsoc": "Sparse OrPSOC",
    "full_hybrid": "Full Hybrid", "cgpso": "CG-PSO",
    "taguchi_cbpso": "Taguchi CBPSO", "hpso_de": "HPSO-DE",
    "bwaoa_c": "BWOA-C", "mrmr": "mRMR", "lasso": "LASSO", "rfe": "RFE",
}
ARM_COLOR = {
    "sparse_pso": "#1f77b4", "sparse_orpsoc": "#d62728",
    "full_hybrid": "#9467bd", "mrmr": "#2ca02c", "lasso": "#8c564b",
    "rfe": "#e377c2", "cgpso": "#ff7f0e", "taguchi_cbpso": "#17becf",
    "hpso_de": "#bcbd22", "bwaoa_c": "#7f7f7f",
}

# ── condition names (core ablation, D = 50) ─────────────────────────────────
COND = {
    "baseline": "Baseline", "standard_orpsoc": "Standard OrPSOC",
    "apsoll": "+APSOLL", "full_hybrid": "Full Hybrid",
    "full_hybrid_noimp": "FH no-imp",
}
COND_COLOR = {
    "baseline": "#444444", "standard_orpsoc": "#1f77b4",
    "apsoll": "#ff7f0e", "full_hybrid": "#d62728",
    "full_hybrid_noimp": "#9467bd",
}
COND_ORDER = ["baseline", "standard_orpsoc", "apsoll", "full_hybrid",
              "full_hybrid_noimp"]

# ── synthetic level names, as in the manuscript tables ──────────────────────
LEVEL = {
    "null": "L0 null", "white_noise": "L1 white noise", "ar1": "L2 AR(1)",
    "drift": "L3 drift", "regime_switch": "L4 regime switch",
}

# ── method families (Figure S10) ────────────────────────────────────────────
FAMILY_LABEL = {
    "ours": "OrPSOC family (chooses its own $k$)",
    "rival": "Enhanced-swarm comparators (choose their own $k$)",
    "nonswarm": "Non-swarm comparators (budget-matched to Sparse PSO)",
}

# full width of the elsarticle preprint text block, in inches
FULL_W = 7.0


def save(fig, stem, paper_name=None):
    """Write plots/<stem>.png (preview) and figures/<paper_name>.pdf."""
    os.makedirs(PLOTS, exist_ok=True)
    os.makedirs(PAPER_FIGS, exist_ok=True)
    fig.savefig(os.path.join(PLOTS, stem + ".png"), bbox_inches="tight")
    pdf = os.path.join(PLOTS, stem + ".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    if paper_name:
        shutil.copyfile(pdf, os.path.join(PAPER_FIGS, paper_name + ".pdf"))
    plt.close(fig)
    print(f"  wrote {stem}.pdf" + (f"  -> figures/{paper_name}.pdf" if paper_name else ""))
