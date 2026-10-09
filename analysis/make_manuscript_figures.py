"""
Generate the figures the manuscript still lacks.

The existing plots/ directory covers the four figures already in the draft
(mean AUC by condition, sector-ETF per-fold, L4 recovery, Jaccard stability).
This adds the exhibits required by the outstanding "additional analysis"
blocks and by the Tier D results, which have no figure at all yet:

  fig_recall_per_fold        the companion recall analysis the draft calls
                             "the missing link that turns this from suggestive
                             to conclusive" (Figure 4, item 1)
  fig_compactness_tradeoff   AUC retained vs features used, per level
                             (Section 4.6, which currently has no exhibit)
  fig_dimensionality_synth   delta AUC vs p on the synthetic sweep, both
                             classifiers -- locates the crossover
  fig_dimensionality_real    the same on the Ken French industry ladder
  fig_detector_separation    separation ratio, synthetic vs real markets
  fig_density_collapse       why binary PSO stalls at high p, and the fix

Run:  python analysis/make_manuscript_figures.py
Writes plots/fig_*.png at 200 dpi.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis.fig_style as FS
FS.apply()
OUT = "plots"
# The 20-seed synthetic sweep (deploy/queue_synthetic_sweep.sh). Kept in
# sync with the same constant in analysis/ladder_tables.py.
SWEEP_20 = ("results/tier_d_fullnovelty_"
            "all-lasso-mrmr-rfe-sparse_orpsoc-sparse_pso.json")
# Two separate palettes: the ablation CONDITIONS and the Tier D ARMS share
# some names ("full_hybrid") but are different experiments, and one shared dict
# silently gave two conditions the same colour.
C, ARMC = FS.COND_COLOR, FS.ARM_COLOR       # names and colours: analysis/fig_style.py
ARMLBL, LBL = FS.ARM, FS.COND


# ── 1. per-fold recall of the post-switch signals ────────────────────────────
def fig_recall_per_fold():
    """Figure 4's companion. Tier A established r2 recall rises for BOTH
    OrPSOC and Full Hybrid, and 3x more for OrPSOC -- refuting the draft's
    claim that it rises for Full Hybrid but not standard OrPSOC. The per-fold
    trace is what makes that visible."""
    src = "results/step7_ablation_v2.json"
    if not os.path.exists(src):
        print("  skip recall: no step7 ablation json"); return
    d = json.load(open(src))
    lvl = next((k for k in d.get("full_results", {})
                if "regime" in k and "@" not in k), None)
    if lvl is None:
        print("  skip recall: no regime level"); return
    conds = ["standard_orpsoc", "apsoll", "full_hybrid", "full_hybrid_noimp"]

    fig, axes = plt.subplots(1, 2, figsize=(FS.FULL_W, 3.3))
    for ax, key, ttl in [
            (axes[0], "fold_r2_hits", "Post-switch signals $s_3$–$s_4$ (should rise)"),
            (axes[1], "fold_r1_hits", "Pre-switch signals $s_0$–$s_2$ (should fall)")]:
        for c in conds:
            seeds = d["full_results"][lvl]["conditions"].get(c, []) or []
            traces = [s[key] for s in seeds if key in s]
            if not traces:
                continue
            n = min(len(t) for t in traces)
            m = np.mean([t[:n] for t in traces], axis=0)
            ax.plot(range(1, n + 1), m, marker="o", ms=4, lw=1.8,
                    color=C[c], label=LBL[c])
        ax.axvline(4.5, color="red", ls="--", lw=1.2, alpha=0.8)
        ax.text(4.6, ax.get_ylim()[1] * 0.92, "switch enters\ntraining",
                color="red", fontsize=7.5, va="top")
        ax.set_xlabel("Walk-forward fold"); ax.set_ylabel("Mean features recovered")
        ax.set_title(ttl)
    # one legend below both panels: inside the left panel it ran into the
    # "switch enters training" label
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(h), fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    FS.save(fig, "fig_recall_per_fold", "RecallPerFold")


# ── 2. compactness trade-off ─────────────────────────────────────────────────
def fig_compactness():
    """Section 4.6 has no exhibit. Tier A A4 is the strongest supportable O4
    statement: 90.9-97.3% of baseline AUC at 14-19% of the features."""
    a = json.load(open("results/tier_a.json")).get("A4", {})
    rows = a.get("rows") or a
    lv, pct_auc, pct_k = [], [], []
    for k, v in rows.items():
        if not isinstance(v, dict):
            continue
        pa = v.get("pct_of_baseline_auc") or v.get("pct_baseline_auc")
        pk = v.get("pct_of_N") or v.get("pct_n")
        if pa is None or pk is None:
            continue
        lv.append(k.replace("v2_", "")); pct_auc.append(float(pa)); pct_k.append(float(pk))
    if not lv:   # fall back to the published Tier A table
        lv = ["null", "white_noise", "ar1", "drift", "regime_switch"]
        pct_auc = [95.3, 97.3, 93.1, 95.8, 90.9]
        pct_k = [15, 14, 19, 17, 19]

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    # per-level label offsets (points), chosen so no two labels overlap
    OFF = {"null": (-8, -13, "right"), "white_noise": (-8, 4, "right"),
           "ar1": (8, 2, "left"), "drift": (8, 3, "left"),
           "regime_switch": (8, -3, "left")}
    for x, y, t in zip(pct_k, pct_auc, lv):
        floor = (t == "null")       # L0: both AUCs at chance, ratio not interpretable
        ax.scatter([x], [y], s=70, zorder=3, color="white" if floor else "#d62728",
                   edgecolor="#888888" if floor else "#d62728", linewidth=1.4)
        dx, dy, ha = OFF.get(t, (8, 3, "left"))
        ax.annotate(FS.LEVEL.get(t, t) + (" (chance floor)" if floor else ""), (x, y),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=9,
                    color="#666666" if floor else "black")
    ax.axhline(100, color="#444", ls="--", lw=1.2)
    ax.text(19.5, 100.35, "all-features baseline (100% AUC, 100% of features)",
            fontsize=8.5, color="#444", ha="center")
    ax.set_xlim(5, 34); ax.set_ylim(min(pct_auc) - 3, 101.8)
    ax.set_xlabel("Features used (% of $D=50$)")
    ax.set_ylabel("Baseline AUC retained (%)")
    fig.tight_layout()
    FS.save(fig, "fig_compactness_tradeoff", "Compactness")


def _curve(path, order, arms, drop_first=False):
    # drop_first: on the real ladder fold 0's 150-row training window holds one
    # class, so every arm and the baseline score an AUC=0.5 placeholder there
    # (Methods §2.4.2). The manuscript tables exclude it; so must the figure.
    # Synthetic data has both classes in fold 0 and keeps all eight folds.
    d = json.load(open(path))
    c = defaultdict(lambda: defaultdict(list))
    for r in d:
        for a in list(arms) + ["all"]:
            if a in r["arms"]:
                auc = r["arms"][a]["auc"]
                c[(r["ds"], r["model"])][a].append(np.mean(auc[1:] if drop_first else auc))
        c[(r["ds"], r["model"])]["_p"] = r["n_features"]
    return c


def _plot_curve(c, order, arms, title, fname, xlabel, paper_name=None):
    fig, axes = plt.subplots(1, 2, figsize=(FS.FULL_W, 3.6), sharey=True)
    for ax, model in zip(axes, ["LogReg", "LightGBM"]):
        ps, series, sigs = [], defaultdict(list), defaultdict(list)
        for ds in order:
            k = (ds, model)
            if k not in c or "all" not in c[k]:
                continue
            ps.append(c[k]["_p"])
            base = np.mean(c[k]["all"])
            for a in arms:
                series[a].append(np.mean(c[k][a]) - base if a in c[k] else np.nan)
                # Paired per-seed Wilcoxon, the same test the tables report.
                # c[k][a] and c[k]["all"] are appended inside one pass over the
                # records, so index i is the same seed in both -- but only when
                # the arm is present in every record (rfe is not, past seed 7).
                v = c[k].get(a)
                ok = False
                if v is not None and len(v) == len(c[k]["all"]):
                    d = np.asarray(v) - np.asarray(c[k]["all"])
                    if np.any(d != 0):
                        ok = wilcoxon(d).pvalue < 0.05
                sigs[a].append(ok)
        if not ps:
            continue
        for a in arms:
            ax.plot(ps, series[a], marker="o", ms=4.5, lw=1.9,
                    color=ARMC.get(a), label=ARMLBL.get(a, a))
        ax.axhline(0, color="k", lw=1.2)

        # Shade only where selection actually wins, and mark the crossover:
        # the first width at which the PSO arm clears the baseline AND the
        # margin is significant.
        #
        # Significance is required deliberately. On mean sign alone the 20-seed
        # ladder marks p=121, where sparse PSO is +0.0048 at p=0.216 -- a
        # crossover the table beside this figure does not support. Marking it
        # would repeat exactly the figure-contradicts-table defect corrected in
        # 2c12d8d. With the test applied the marker lands on p=171, which the
        # table does support.
        pso = np.array(series.get("sparse_pso", [np.nan] * len(ps)), dtype=float)
        sig = np.array(sigs.get("sparse_pso", [False] * len(ps)), dtype=bool)
        if np.nanmax(pso) > 0:
            idx = next((i for i, v in enumerate(pso) if v > 0 and sig[i]
                        and all(x > 0 for x in pso[i:] if not np.isnan(x))), None)
            if idx is not None and idx > 0:
                ax.axvline(ps[idx], color="#2ca02c", ls=":", lw=1.6)
                ax.annotate(f"crossover $p\\approx{ps[idx]}$", (ps[idx], 0),
                            textcoords="offset points", xytext=(8, -34),
                            fontsize=8.5, color="#2ca02c", fontweight="bold")
            top = ax.get_ylim()[1]
            ax.axhspan(0, top, color="#2ca02c", alpha=0.06)

        ax.set_xscale("log")
        # Every width gets a tick. Neighbouring widths are close on a log axis
        # (101/121, 481/491), so alternate labels drop to a second row and no
        # two labels on the same row can touch.
        ax.set_xticks(ps)
        low = []                                  # labels dropped to the second row
        for i in range(len(ps)):
            low.append(i > 0 and ps[i] / ps[i - 1] < 1.35 and not low[i - 1])
        ax.set_xticklabels([("\n" if lo else "") + str(v) for v, lo in zip(ps, low)],
                           fontsize=8.5)
        ax.minorticks_off()
        ax.set_xlabel(xlabel); ax.set_title(ARMLBL.get(model, {"LogReg": "Logistic regression"}.get(model, model)))
    axes[0].set_ylabel("$\\Delta$ AUC vs all-features")
    # one legend below both panels, so it never sits on a curve
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(arms), fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    FS.save(fig, fname.replace(".png", ""), paper_name)


def fig_dim_synth():
    # Prefer the 20-seed re-run when it exists, exactly as
    # analysis/ladder_tables.py does for the table beside this figure. The
    # figure and Table 12b must not end up drawn from different runs -- that is
    # the class of drift reports/manuscript_audit_20260909.md §A2 records.
    src = next((p for p in (SWEEP_20, "results/tier_d_curve.json")
                if os.path.exists(p)), None)
    if src is None:
        print("  skip synth curve"); return
    print(f"  synth curve source: {src}")
    arms = ["sparse_pso", "sparse_orpsoc", "mrmr", "lasso", "rfe"]   # = Tables 7 and S4
    order = [f"hd_regime_switch_p{p}" for p in (50, 200, 500, 1000)]
    c = _curve(src, order, arms)
    _plot_curve(c, order, arms,
                "Synthetic dimensionality sweep (regime switch): the benefit of "
                "selection is gated by $p$, and by the classifier",
                "fig_dimensionality_synth.png", "Number of candidate features $p$",
                "DimSynthetic")


def fig_dim_real():
    # Merged source, not tier_d_fullnovelty.json directly: the 20-seed run covers
    # only p=101..301, so the ladder is 20 seeds there and 8 seeds at the other
    # four widths. experiments/merge_ladder.py builds it and documents the rule
    # (including why rfe is held at the pinned scikit-learn 1.8.0 values).
    src = "results/tier_d_ladder_merged.json"
    if not os.path.exists(src):
        print("  skip real curve -- run experiments/merge_ladder.py first"); return
    arms = ["sparse_pso", "sparse_orpsoc", "full_hybrid", "mrmr", "lasso", "rfe"]
    order = ["ind5", "ind10", "ind12", "ind17", "ind30", "ind38", "ind48", "ff49ind"]
    c = _curve(src, order, arms, drop_first=True)
    _plot_curve(c, order, arms,
                "Real-data dimensionality ladder (Ken French industry portfolios, "
                "identical target at every width)",
                "fig_dimensionality_real.png", "Number of candidate features $p$",
                "fig_ladder")


# ── 5. detector separation ───────────────────────────────────────────────────
def fig_detector_separation():
    labels = ["Synthetic\n(regime switch)", "Sector ETFs", "Fama–French"]
    vals = [6.98, 1.01, 0.92]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    bars = ax.bar(labels, vals, color=["#2ca02c", "#bbbbbb", "#bbbbbb"], width=0.55)
    ax.axhline(1.0, color="red", ls="--", lw=1.4, label="ratio = 1 (no separation)")
    ax.legend(loc="upper right", fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, max(v, 1.0) + 0.3, f"{v:.2f}$\\times$",   # clear of the ratio-1 line
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Separation ratio\n(break-adjacent score $\\div$ elsewhere)")
    ax.set_ylim(0, 8.2)
    fig.tight_layout()
    FS.save(fig, "fig_detector_separation", "DetectorSeparation")


# ── 6. density collapse and its correction ───────────────────────────────────
def fig_density_collapse():
    """Measured in experiments/dimensionality/tier_d_sparse_init.py at p=2000."""
    it = [0, 5, 10, 20, 30]
    traces = {
        "orthogonal init, no bias": ([0.498, 0.498, 0.489, 0.486, 0.485], "#444444", "-"),
        "sparse init 1%, no bias": ([0.009, 0.181, 0.205, 0.269, 0.306], "#ff7f0e", "-"),
        "sparse init 5%, no bias": ([0.048, 0.207, 0.230, 0.288, 0.322], "#1f77b4", "-"),
        "sparse init 1%, bias $-3$": ([0.009, 0.047, 0.050, 0.059, 0.058], "#2ca02c", "--"),
        "sparse init 1%, bias $-4$": ([0.009, 0.021, 0.023, 0.021, 0.020], "#d62728", "--"),
    }
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for lab, (y, col, ls) in traces.items():
        ax.plot(it, y, marker="o", ms=4.5, lw=1.9, color=col, ls=ls, label=lab)
    ax.axhline(0.5, color="#444", ls=":", lw=1.2)
    ax.text(30, 0.515, "$\\sigma(0)=0.5$: the transfer function's fixed point",
            fontsize=8, ha="right", color="#444")
    ax.set_xlabel("PSO iteration"); ax.set_ylabel("Mean particle density\n(fraction of $p$ selected)")
    ax.set_ylim(0, 0.58)
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.02, 0.84))   # empty band under the 0.5 line
    fig.tight_layout()
    FS.save(fig, "fig_density_collapse", "fig_density")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("generating manuscript figures:")
    for fn in (fig_recall_per_fold, fig_compactness, fig_dim_synth,
               fig_dim_real, fig_detector_separation, fig_density_collapse):
        try:
            fn()
        except Exception as e:
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")
