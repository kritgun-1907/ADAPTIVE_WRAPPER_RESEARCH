"""
Supplementary Figures S1, S3, S4, S8 and S9, redrawn from stored results in the
shared style (analysis/fig_style.py): manuscript names, readable fonts, no
internal labels, panel titles only on multi-panel figures, vector PDF output.

These figures were previously produced as side effects of step8_results.py,
step9_real_data.py and experiments/adaptation/tier_c.py, which also recompute statistics
and rewrite result files. This script only reads results, so redrawing a figure
can never change a number.

  S1  MeanAUC           synthetic ablation, mean AUC by condition, 4 levels
  S3  Step9_realdata    sector-ETF per-fold AUC, 5 conditions
  S4  TierC_FamaFrench  Fama-French per-fold AUC and inter-fold Jaccard
  S8  level4Synthetic   synthetic L4 per-fold AUC across the switch
  S9  JaccardSim        inter-fold Jaccard, L2 and L4

Run:  python analysis/make_supplementary_figures.py
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(os.path.dirname(HERE))

import analysis.fig_style as FS
import matplotlib.pyplot as plt
FS.apply()

ABL = json.load(open("results/step7_ablation_v2.json"))
S9 = json.load(open("results/step9_real_data.json"))["datasets"]
LEVELS = ["white_noise", "ar1", "drift", "regime_switch"]


def _cond_handles():
    return [plt.Line2D([], [], color=FS.COND_COLOR[c], marker="o", lw=2,
                       label=FS.COND[c]) for c in FS.COND_ORDER]


def _seeds(level, cond):
    return ABL["full_results"]["v2_" + level]["conditions"][cond]


def _fold_matrix(level, cond, key="fold_aucs"):
    rows = [s[key] for s in _seeds(level, cond)]
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows], float)


# ── S1: mean AUC by condition ────────────────────────────────────────────────
def s1_mean_auc():
    fig, axes = plt.subplots(1, 4, figsize=(FS.FULL_W * 1.25, 3.4), sharey=True)
    summ = ABL["summary"]
    for ax, lv in zip(axes, LEVELS):
        m = [np.mean(summ["v2_" + lv][c]["seed_aucs"]) for c in FS.COND_ORDER]
        sd = [np.std(summ["v2_" + lv][c]["seed_aucs"]) for c in FS.COND_ORDER]
        ax.bar(range(5), m, yerr=sd, capsize=3, width=0.7,
               color=[FS.COND_COLOR[c] for c in FS.COND_ORDER])
        ax.set_xticks(range(5))
        ax.set_xticklabels([FS.COND[c] for c in FS.COND_ORDER], rotation=40,
                           ha="right", fontsize=8.5)
        ax.set_title(FS.LEVEL[lv])
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean test AUC")
    fig.tight_layout()
    FS.save(fig, "supp_mean_auc", "MeanAUC")


# ── S3: sector-ETF per-fold AUC ──────────────────────────────────────────────
def s3_sector_etf():
    d = S9["sector_etf"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for c in FS.COND_ORDER:
        m = np.array(d["conditions"][c]["fold_auc_mean"])
        s = np.array(d["conditions"][c]["fold_auc_std"])
        x = np.arange(1, len(m) + 1)
        ax.plot(x, m, "-o", ms=4, lw=1.8, color=FS.COND_COLOR[c], label=FS.COND[c])
        ax.fill_between(x, m - s, m + s, color=FS.COND_COLOR[c], alpha=0.10)
    for i, b in enumerate(d["break_folds"]):
        ax.axvline(b + 1, color="red", ls="--", lw=1.3, alpha=0.8,
                   label="documented break" if i == 0 else None)
    ax.set_xlabel("Walk-forward fold")
    ax.set_ylabel("Test AUC (mean $\\pm$ s.d. over seeds)")
    ax.set_xticks(range(1, 9))
    ax.legend(fontsize=8.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    FS.save(fig, "supp_sector_etf_per_fold", "Step9_realdata")


# ── S4: Fama-French per-fold AUC and inter-fold Jaccard ─────────────────────
def s4_fama_french():
    import experiments.adaptation.tier_c as tier_c  # step9 checkpoint loader (read-only)
    seeds = tier_c.load_checkpoints()["fama_french"]
    d = S9["fama_french"]
    fig, axes = plt.subplots(1, 2, figsize=(FS.FULL_W * 1.15, 3.7))
    ax = axes[0]
    for c in FS.COND_ORDER:
        M = tier_c.fold_matrix(seeds, c)
        m, s = M.mean(0), M.std(0)
        x = np.arange(1, len(m) + 1)
        ax.plot(x, m, "-o", ms=4, lw=1.8, color=FS.COND_COLOR[c])
        ax.fill_between(x, m - s, m + s, color=FS.COND_COLOR[c], alpha=0.10)
    ax.set_title("Per-fold AUC")
    ax.set_xlabel("Walk-forward fold"); ax.set_ylabel("Test AUC")
    ax = axes[1]
    for c in FS.COND_ORDER:
        rows = [s[c]["jaccard"]["per_fold_jaccard"] for s in seeds
                if isinstance(s[c].get("jaccard"), dict)]
        n = min(len(r) for r in rows)
        J = np.array([r[:n] for r in rows], float)
        x = np.arange(1, n + 1)
        ax.plot(x, J.mean(0), "-o", ms=4, lw=1.8, color=FS.COND_COLOR[c])
        ax.fill_between(x, J.mean(0) - J.std(0), J.mean(0) + J.std(0),
                        color=FS.COND_COLOR[c], alpha=0.10)
    ax.set_title("Jaccard similarity of consecutive folds")
    ax.set_xlabel("Fold pair $(i, i+1)$"); ax.set_ylabel("Jaccard similarity")
    ax.set_ylim(-0.02, 1.05)              # Jaccard is bounded; the s.d. band is not
    for ax, off in ((axes[0], 1.0), (axes[1], 0.5)):
        for b in d["break_folds"]:
            ax.axvline(b + off, color="red", ls="--", lw=1.3, alpha=0.8)
        for i, v in enumerate(d["trigger_fire_rate"]):
            if v > 0:
                ax.axvline(i + off, color="orange", ls=":", lw=2.0)
    h = _cond_handles() + [
        plt.Line2D([], [], color="red", ls="--", label="documented break"),
        plt.Line2D([], [], color="orange", ls=":", lw=2, label="detector fired")]
    fig.legend(handles=h, loc="lower center", ncol=7, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    FS.save(fig, "supp_fama_french", "TierC_FamaFrench")


def _straddle(level):
    ph = _seeds(level, "full_hybrid")[0]["fold_phase"]
    return ph.index("straddle") + 1          # 1-indexed fold


# ── S8: synthetic L4 per-fold AUC across the switch ──────────────────────────
def s8_level4():
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for c in FS.COND_ORDER:
        M = _fold_matrix("regime_switch", c)
        m, s = M.mean(0), M.std(0)
        x = np.arange(1, len(m) + 1)
        ax.plot(x, m, "-o", ms=4, lw=1.8, color=FS.COND_COLOR[c], label=FS.COND[c])
        ax.fill_between(x, m - s, m + s, color=FS.COND_COLOR[c], alpha=0.10)
    st = _straddle("regime_switch")
    ax.axvspan(st - 0.5, st + 0.5, color="red", alpha=0.08, label="straddle fold")
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("Walk-forward fold"); ax.set_ylabel("Test AUC (mean $\\pm$ s.d. over seeds)")
    ax.legend(fontsize=8.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    FS.save(fig, "supp_level4_per_fold", "level4Synthetic")


# ── S9: inter-fold Jaccard, L2 and L4 ────────────────────────────────────────
def s9_jaccard():
    fig, axes = plt.subplots(1, 2, figsize=(FS.FULL_W * 1.1, 3.6), sharey=True)
    for ax, lv in zip(axes, ["ar1", "regime_switch"]):
        for c in FS.COND_ORDER:
            rows = [s["jaccard"]["per_fold_jaccard"] for s in _seeds(lv, c)]
            n = min(len(r) for r in rows)
            J = np.array([r[:n] for r in rows], float)
            x = np.arange(1, n + 1)
            ax.plot(x, J.mean(0), "-o", ms=4, lw=1.8, color=FS.COND_COLOR[c])
            ax.fill_between(x, J.mean(0) - J.std(0), J.mean(0) + J.std(0),
                            color=FS.COND_COLOR[c], alpha=0.10)
        ax.set_title(FS.LEVEL[lv])
        ax.set_xlabel("Fold pair $(i, i+1)$")
        ax.set_ylim(0, 1.05)
    # the causally correct pair: straddle fold -> first post-switch fold
    st = _straddle("regime_switch")
    axes[1].axvline(st, color="red", ls="--", lw=1.3,
                    label="straddle $\\rightarrow$ first post-switch fold")
    axes[0].set_ylabel("Jaccard similarity")
    h = _cond_handles() + [plt.Line2D([], [], color="red", ls="--",
                           label="straddle $\\rightarrow$ first post-switch pair")]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    FS.save(fig, "supp_jaccard", "JaccardSim")


if __name__ == "__main__":
    for fn in (s1_mean_auc, s3_sector_etf, s4_fama_french, s8_level4, s9_jaccard):
        fn()
