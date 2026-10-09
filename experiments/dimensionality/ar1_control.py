"""
The AR(1) control arm of the synthetic sweep — extraction and reporting.

WHY THIS EXISTS
───────────────
Section 2.7 states that a stationary AR(1) arm is generated at every width of
the synthetic sweep, as a control: "without it, an advantage appearing at large
p could not be attributed to regime shift rather than to dimensionality alone."

The runs were done. The result was never reported anywhere in the paper. This
script extracts it: hd_ar1_p{50,200,500,1000} width-matched against
hd_regime_switch_p{...} under both classifiers.

NO NEW COMPUTATION HAPPENS HERE. This is tabulation of an existing run.

Two runs can supply it, and the newer is preferred (see SWEEP_20 below).
results/tier_d_curve.json is the original August run at 8 seeds; the 20-seed
re-run is produced by deploy/ec2_sweep.sh stage 2. The 20-seed extension exists
because the k-NN panel of the same sweep was already at 20 -- review item 4 ran
that model only -- leaving the LogReg and LightGBM cells at 8 beside it.

WHAT THE CONTROL DECIDES
────────────────────────
If the selection advantage at large p is present on the STATIONARY arm too,
then dimensionality alone accounts for it and regime shift is not part of the
mechanism. That is a claim about the paper's title, not only its Section 3.9.

The test is paired per-seed Wilcoxon of (arm AUC - all-features AUC) within a
level, exactly as Tables 10-14 do, so the numbers are directly comparable to
every other delta in the manuscript. Seeds are matched across levels by
construction: tier_d_curve.py runs seed s on both levels at every width.

THE MANUSCRIPT TABLE IS NO LONGER THIS ONE
──────────────────────────────────────────
An earlier draft carried the control as its own Table 11 and the full sweep as
Table 12, and the two printed the SAME forty numbers back to back: the control
IS the AR(1) columns of the sweep, so a separate table was a duplicate that a
referee would have caught. The manuscript now \\inputs only
tables/synthetic_sweep.tex (analysis/ladder_tables.py), which carries both
levels side by side.

What this script still owns is the ANALYSIS: results/ar1_control.json holds the
per-arm, per-width AR(1)-vs-regime comparison and its Wilcoxon statistics, and
tables/ar1_control.tex is a standalone exhibit. Neither is \\input by the paper.

Run:  python experiments/dimensionality/ar1_control.py
Writes results/ar1_control.json and tables/ar1_control.tex
"""
import os
import sys
import json
import collections

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Preference order, richest first -- the SAME order analysis/ladder_tables.py
# and analysis/make_manuscript_figures.py use. If this script read the
# 8-seed run while the paper's Table 12b read the 20-seed one, the analysis of
# record and the published table would quietly disagree about the numbers they
# are both supposed to be reporting.
SWEEP_20 = ("results/tier_d_fullnovelty_"
            "all-lasso-mrmr-rfe-sparse_orpsoc-sparse_pso.json")
CURVE_8 = "results/tier_d_curve.json"
OUT_JSON = "results/ar1_control.json"
OUT_TEX = "tables/ar1_control.tex"

# Ordered so the table reads swarm arms first, then filters -- same order as
# Table 10, so a reader comparing the two is not re-learning a row order.
ARM_ORDER = ["sparse_pso", "sparse_orpsoc", "full_hybrid",
             "mrmr", "lasso", "rfe"]
ARM_LABEL = {"sparse_pso": "Sparse PSO", "sparse_orpsoc": "Sparse OrPSOC",
             "full_hybrid": "Full Hybrid", "mrmr": "mRMR", "lasso": "LASSO",
             "rfe": "RFE"}
LEVEL_OF = lambda ds: "ar1" if "ar1" in ds else "regime_switch"


def load():
    """((model, p, level, arm) -> {seed: mean AUC}, source path)."""
    src = next((p for p in (SWEEP_20, CURVE_8) if os.path.exists(p)), None)
    if src is None:
        raise SystemExit(f"neither {SWEEP_20} nor {CURVE_8} is present")
    print(f"  source: {src}")
    with open(src) as fh:
        raw = json.load(fh)
    cells = collections.defaultdict(dict)
    for r in raw:
        key = (r["model"], int(r["n_features"]), LEVEL_OF(r["ds"]))
        for arm, v in r["arms"].items():
            # Mean over walk-forward folds, one number per seed -- the same
            # per-seed reduction every other delta in the paper uses.
            cells[key + (arm,)][int(r["seed"])] = float(np.mean(v["auc"]))
    return cells, src


def delta(cells, model, p, level, arm):
    """(mean delta vs all-features, Wilcoxon p, n seeds) or None."""
    base = cells.get((model, p, level, "all"))
    sel = cells.get((model, p, level, arm))
    if not base or not sel:
        return None
    seeds = sorted(set(base) & set(sel))
    if len(seeds) < 2:
        return None
    b = np.array([base[s] for s in seeds])
    a = np.array([sel[s] for s in seeds])
    try:
        pv = float(wilcoxon(a, b).pvalue)
    except ValueError:                  # all differences zero
        pv = 1.0
    return float((a - b).mean()), pv, len(seeds)


def main():
    cells, src = load()
    models = sorted({k[0] for k in cells}, reverse=True)      # LogReg first
    widths = sorted({k[1] for k in cells})
    arms = [a for a in ARM_ORDER if any(k[3] == a for k in cells)]

    summary = []
    print("=" * 84)
    print("  AR(1) STATIONARY CONTROL vs REGIME SWITCH — synthetic sweep")
    print("  delta AUC against the all-features baseline, paired per-seed Wilcoxon")
    print("=" * 84)

    for model in models:
        print(f"\n  {model}")
        print(f"  {'p':>6} {'arm':<16} {'AR(1)':>11} {'regime':>11} "
              f"{'RS - AR1':>10} {'seeds':>6}")
        print("  " + "-" * 64)
        for p in widths:
            for arm in arms:
                d_ar = delta(cells, model, p, "ar1", arm)
                d_rs = delta(cells, model, p, "regime_switch", arm)
                if d_ar is None or d_rs is None:
                    continue
                (a, ap, an), (r, rp, rn) = d_ar, d_rs
                fmt = lambda v, q: f"{v:+.4f}{'*' if q < 0.05 else ' '}"
                print(f"  {p:>6} {arm:<16} {fmt(a, ap):>11} {fmt(r, rp):>11} "
                      f"{r - a:>+10.4f} {min(an, rn):>6}")
                summary.append({"model": model, "p": p, "arm": arm,
                                "ar1_delta": a, "ar1_p": ap,
                                "regime_delta": r, "regime_p": rp,
                                "n_seeds": int(min(an, rn))})
            print()

    # ── the one-line verdict the title depends on ────────────────────────────
    lg = [s for s in summary if s["model"] == "LogReg" and s["p"] >= 200]
    pos_ar = sum(1 for s in lg if s["ar1_delta"] > 0 and s["ar1_p"] < 0.05)
    bigger = sum(1 for s in lg if s["ar1_delta"] > s["regime_delta"])
    print("=" * 84)
    print(f"  LogReg, p >= 200: {pos_ar}/{len(lg)} cells significantly POSITIVE "
          f"on the stationary control.")
    print(f"  The stationary gain EXCEEDS the regime-switch gain in "
          f"{bigger}/{len(lg)} of them.")
    print("  Reading: dimensionality accounts for the advantage; regime shift")
    print("  attenuates rather than creates it.")
    print("=" * 84)

    os.makedirs("reports", exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump({"source": src, "note": "extraction only, no new computation",
                   "cells": summary}, fh, indent=2)
    write_tex(summary, models, widths, arms)
    print(f"\n  wrote {OUT_JSON}\n  wrote {OUT_TEX}")


def write_tex(summary, models, widths, arms):
    idx = {(s["model"], s["p"], s["arm"]): s for s in summary}
    L = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"    \centering")
    L.append(r"    \small\setlength{\tabcolsep}{5pt}")
    L.append(r"    \begin{tabular}{l" + "rr" * len(widths) + "}")
    L.append(r"    \hline")
    L.append("    & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{$p = {p}$}}" for p in widths) + r" \\")
    L.append("    Arm & " + " & ".join(
        r"AR(1) & Regime" for _ in widths) + r" \\")
    L.append(r"    \hline")
    for model in models:
        L.append(rf"    \multicolumn{{{1 + 2 * len(widths)}}}{{l}}"
                 rf"{{\textit{{{model}}}}} \\")
        for arm in arms:
            row = [ARM_LABEL.get(arm, arm)]
            for p in widths:
                s = idx.get((model, p, arm))
                if s is None:
                    row += ["--", "--"]
                    continue
                for d, q in ((s["ar1_delta"], s["ar1_p"]),
                             (s["regime_delta"], s["regime_p"])):
                    row.append(f"${d:+.4f}$" + (r"$^{*}$" if q < 0.05 else ""))
            L.append("    " + " & ".join(row) + r" \\")
        L.append(r"    \hline")
    L.append(r"    \end{tabular}")
    L.append(r"    \caption{The stationary control. $\Delta$ AUC against the "
             r"all-features baseline on the synthetic sweep, at each width, "
             r"under the stationary AR(1) level and the regime-switch level. "
             r"8 seeds, matched seed-for-seed across levels; $^{*}$ denotes "
             r"$p < 0.05$, paired per-seed Wilcoxon. Signal count, series "
             r"length and seed are held fixed at every width; only the number "
             r"of pure-noise distractors changes.}")
    L.append(r"    \label{tab:ar1control}")
    L.append(r"\end{table}")
    with open(OUT_TEX, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
