"""
Item 7 — cross-seed Jaccard at matched folds.

THE DISTINCTION
───────────────
The paper reports INTER-FOLD Jaccard: for one seed, how much does the selected
subset change from fold t to fold t+1? That measures whether the selector tracks
the data over time, and it is the right measure for the adaptivity claim.

It says nothing about REPRODUCIBILITY. For a stochastic search, the more basic
question is: run the same selector on the SAME fold with a different random
seed, and do you get the same features? That is cross-seed Jaccard at matched
folds, and for a swarm method it is arguably the more important of the two --- a
selector whose output changes wholesale with the seed is not identifying
structure, it is sampling from a large set of roughly equivalent subsets.

The two can diverge sharply and in an informative direction. High inter-fold
with low cross-seed means "stable over time, but arbitrary": the run is
self-consistent yet the particular features are not reproducible.

NO NEW COMPUTATION IS REQUIRED
──────────────────────────────
An earlier assessment recorded this item as blocked, on the grounds that
selected feature sets are not persisted. That is true of the Tier D ladder
checkpoints, which store only auc/k/recall. It is NOT true of the synthetic
study: results/step7_ablation_v2.json carries `fold_selected` --- the full
feature names, per fold, for all 30 seeds, for every condition. The inter-fold
Jaccard already reported in the paper is computed from exactly that field.

So this script is extraction, not computation.

WHAT IS COMPARED
────────────────
For each (level, condition, fold): the mean pairwise Jaccard over all
seed pairs, |A n B| / |A u B|. Reported against the inter-fold value already in
the paper, and against a random-subset baseline, since with k of D features drawn
at random the expected Jaccard is not zero and a raw number is uninterpretable
without it.

Run:  python experiments/adaptation/crossseed_jaccard.py
Writes results/crossseed_jaccard.json and tables/crossseed_jaccard.tex
"""
import os
import sys
import json
import itertools
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SRC = "results/step7_ablation_v2.json"
OUT_JSON = "results/crossseed_jaccard.json"
OUT_TEX = "tables/crossseed_jaccard.tex"

LEVEL_LABEL = {"v2_null": "L0 null", "v2_white_noise": "L1 white noise",
               "v2_ar1": "L2 AR(1)", "v2_drift": "L3 drift",
               "v2_regime_switch": "L4 regime switch"}
COND_LABEL = {"standard_orpsoc": "Standard OrPSOC", "apsoll": "+APSOLL",
              "full_hybrid": "Full Hybrid", "full_hybrid_noimp": "Hybrid (no imp.)"}


def jac(a, b):
    A, B = set(a), set(b)
    u = A | B
    return len(A & B) / len(u) if u else float("nan")


def main():
    with open(SRC) as fh:
        blob = json.load(fh)
    D = blob.get("config", {}).get("n_features") or 50

    rows = []
    print("=" * 92)
    print("  CROSS-SEED JACCARD AT MATCHED FOLDS  (reproducibility of a stochastic selector)")
    print("=" * 92)
    print(f"  {'level':<18}{'condition':<20}{'cross-seed':>12}{'inter-fold':>12}"
          f"{'random':>10}{'mean k':>8}{'seeds':>7}")
    print("  " + "-" * 88)

    for level, lv in blob["full_results"].items():
        conds = lv.get("conditions", {})
        for cond, seedruns in conds.items():
            if cond == "baseline" or not isinstance(seedruns, list):
                continue
            runs = [r for r in seedruns
                    if isinstance(r, dict) and r.get("fold_selected")]
            if len(runs) < 3:
                continue
            n_folds = min(len(r["fold_selected"]) for r in runs)

            # cross-seed: fix the fold, vary the seed
            per_fold = []
            for f in range(n_folds):
                sels = [r["fold_selected"][f] for r in runs]
                pairs = [jac(a, b) for a, b in itertools.combinations(sels, 2)]
                if pairs:
                    per_fold.append(statistics.fmean(pairs))
            cross = statistics.fmean(per_fold) if per_fold else float("nan")

            # inter-fold: fix the seed, vary the fold (as already reported)
            inter = []
            for r in runs:
                fs = r["fold_selected"]
                inter += [jac(fs[i], fs[i + 1]) for i in range(len(fs) - 1)]
            interv = statistics.fmean(inter) if inter else float("nan")

            ks = [len(s) for r in runs for s in r["fold_selected"]]
            k = statistics.fmean(ks) if ks else float("nan")
            # expected Jaccard for two independent uniform k-subsets of D
            rand = (k / D) / (2 - k / D) if k and D else float("nan")

            rows.append({"level": level, "condition": cond,
                         "cross_seed_jaccard": cross, "inter_fold_jaccard": interv,
                         "random_baseline": rand, "mean_k": k,
                         "n_seeds": len(runs), "n_folds": n_folds})
            print(f"  {LEVEL_LABEL.get(level,level):<18}{COND_LABEL.get(cond,cond):<20}"
                  f"{cross:>12.3f}{interv:>12.3f}{rand:>10.3f}{k:>8.1f}{len(runs):>7}")

    with open(OUT_JSON, "w") as fh:
        json.dump({"source": SRC, "note": "extraction only; fold_selected already stored",
                   "D": D, "rows": rows}, fh, indent=2)

    # table: the regime-switch level, which is the one the paper argues from
    L = [r"\begin{table}[htbp]", r"    \centering", r"    \small",
         r"    \begin{tabular}{lrrrr}", r"    \hline",
         r"    Condition & Cross-seed & Inter-fold & Random & Mean $k$ \\",
         r"    \hline"]
    for r in rows:
        if r["level"] != "v2_regime_switch":
            continue
        L.append(f"    {COND_LABEL.get(r['condition'], r['condition'])} & "
                 f"{r['cross_seed_jaccard']:.3f} & {r['inter_fold_jaccard']:.3f} & "
                 f"{r['random_baseline']:.3f} & {r['mean_k']:.1f} \\\\")
    L += [r"    \hline", r"    \end{tabular}",
          r"    \caption{Cross-seed Jaccard at matched folds on v2 Level~4, 30 "
          r"seeds. \emph{Cross-seed} fixes the fold and varies the random seed: "
          r"it measures whether the selector is reproducible at all. "
          r"\emph{Inter-fold} fixes the seed and varies the fold, and is the "
          r"quantity reported in Figure~\ref{fig:jaccard}; it measures whether "
          r"the subset tracks the data over time. \emph{Random} is the expected "
          r"Jaccard for two independent uniform $k$-subsets of $D = 50$, without "
          r"which neither column is interpretable. The two questions are "
          r"different and the answers differ.}",
          r"    \label{tab:crossseed}", r"\end{table}"]
    os.makedirs("tables", exist_ok=True)
    with open(OUT_TEX, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n  wrote {OUT_JSON}\n  wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
