"""
Item 5 — quantify collinearity at each ladder width, and relate it to the gain.

WHY
───
Footnote 1 makes this point informally for the sector ETFs: 58 features occupy
roughly 13 effective dimensions. The collinearity account of Section 3.9 rests
on that observation, but it has never been measured across the ladder, so the
account is currently supported by one anecdote at one width.

TWO MEASURES, AND WHY BOTH
──────────────────────────
1. EFFECTIVE RANK (Roy & Vetterli). Take the correlation matrix's eigenvalues,
   normalise them to a distribution p_i = lambda_i / sum(lambda), and report
   exp(H(p)) where H is Shannon entropy. It answers "how many dimensions does
   this design matrix really occupy?" and is well behaved when the matrix is
   near-singular, which every wide width here is.

2. MEAN VIF. The conventional econometric measure, computed as the mean of
   diag(R^-1) -- algebraically identical to regressing each column on all the
   others, but one pseudo-inverse instead of p regressions. It is the quantity a
   referee from a finance background will look for.

VIF is reported with a caveat and effective rank is led with, deliberately. At
p = 491 with n = 6501 the correlation matrix is severely ill-conditioned, so
diag(R^-1) is numerically delicate and individual VIFs can be enormous and
unstable. Effective rank uses the whole spectrum and degrades gracefully. Where
the two disagree, trust the rank.

WHAT IS AND IS NOT CLAIMED
──────────────────────────
The ladder has eight widths, so any relationship between collinearity and the
logistic gain rests on n = 8 points. This is reported as a descriptive plot with
a correlation coefficient, NOT as a fitted model with a p-value. Eight points
cannot support an inferential claim and the manuscript must not make one.

Run:  python experiments/dimensionality/collinearity.py
Writes results/collinearity.json and tables/collinearity.tex
"""
import os
import sys
import json
import pickle
import collections

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.folds import scored   # fold 0 on the real data is a placeholder

STORE = "results/checkpoints/tier_d_fullnovelty"
OUT_JSON = "results/collinearity.json"
OUT_TEX = "tables/collinearity.tex"

LADDER = [("ind5", 51), ("ind10", 101), ("ind12", 121), ("ind17", 171),
          ("ind30", 301), ("ind38", 361), ("ind48", 481), ("ff49ind", 491)]
CONTEXT = [("sector_etf", None), ("fama_french", None)]
GAIN_ARM = "lasso"      # significant at the most widths under LogReg


def effective_rank(R):
    """exp(Shannon entropy of the normalised eigenvalue spectrum)."""
    w = np.linalg.eigvalsh(R)
    w = np.clip(w, 0, None)
    s = w.sum()
    if s <= 0:
        return float("nan")
    p = w / s
    p = p[p > 1e-300]
    return float(np.exp(-(p * np.log(p)).sum()))


def mean_vif(R):
    """mean of diag(R^-1); pinv because these matrices are near-singular."""
    try:
        Ri = np.linalg.pinv(R, rcond=1e-10)
        v = np.diag(Ri)
        v = v[np.isfinite(v) & (v > 0)]
        return float(np.mean(v)) if len(v) else float("nan")
    except np.linalg.LinAlgError:
        return float("nan")


def measure(ds):
    path = f"data/{ds}.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        X = pickle.load(fh)["X"]
    A = np.asarray(X, dtype=float)
    A = A[:, np.nanstd(A, axis=0) > 0]              # drop constant columns
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    R = np.corrcoef(A, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    p = R.shape[0]
    er = effective_rank(R)
    return {"ds": ds, "p": int(p), "n": int(A.shape[0]),
            "effective_rank": er, "compression": float(p / er) if er else None,
            "mean_vif": mean_vif(R),
            "cond": float(np.linalg.cond(R))}


def logistic_gain(arm=GAIN_ARM):
    """(ds) -> mean delta AUC vs all-features under LogReg, from the merged
    ladder -- the same records as the ladder table, so the column matches it.
    (Reading the newest checkpoint generation by mtime picked a LightGBM-only
    generation once EC2 untars reset the directory times.)"""
    with open("results/tier_d_ladder_merged.json") as fh:
        recs = json.load(fh)
    cells = collections.defaultdict(dict)
    for r in recs:
        if r.get("model") != "LogReg":
            continue
        for a, v in r["arms"].items():
            cells[(r["ds"], a)][int(r["seed"])] = float(np.mean(scored(v["auc"], r)))
    out = {}
    for ds, _ in LADDER:
        b, s = cells.get((ds, "all")), cells.get((ds, arm))
        if not b or not s:
            continue
        k = sorted(set(b) & set(s))
        if len(k) < 3:
            continue
        bb = np.array([b[i] for i in k]); ss = np.array([s[i] for i in k])
        try:
            pv = float(wilcoxon(ss, bb).pvalue)
        except ValueError:
            pv = 1.0
        out[ds] = (float((ss - bb).mean()), pv, len(k))
    return out


def main():
    gains = logistic_gain()
    rows = []
    print("=" * 88)
    print("  COLLINEARITY ACROSS THE LADDER")
    print("=" * 88)
    print(f"  {'dataset':<12}{'p':>5}{'eff.rank':>10}{'p/eff':>8}"
          f"{'meanVIF':>12}{'cond':>12}{'gain('+GAIN_ARM+')':>14}")
    print("  " + "-" * 84)
    for ds, _ in LADDER + CONTEXT:
        m = measure(ds)
        if m is None:
            print(f"  {ds:<12}  (no data)")
            continue
        g = gains.get(ds)
        m["gain"] = g[0] if g else None
        m["gain_p"] = g[1] if g else None
        rows.append(m)
        gs = (f"{g[0]:+.4f}{'*' if g[1] < 0.05 else ' '}") if g else "   --   "
        print(f"  {ds:<12}{m['p']:>5}{m['effective_rank']:>10.2f}"
              f"{m['compression']:>8.1f}{m['mean_vif']:>12.1f}"
              f"{m['cond']:>12.2e}{gs:>14}")

    lad = [r for r in rows if r["gain"] is not None
           and any(r["ds"] == d for d, _ in LADDER)]
    if len(lad) >= 3:
        er = np.array([r["effective_rank"] for r in lad])
        cp = np.array([r["compression"] for r in lad])
        gn = np.array([r["gain"] for r in lad])
        print("\n" + "=" * 88)
        print(f"  n = {len(lad)} widths. Descriptive only -- eight points cannot")
        print(f"  support an inferential claim, and none is made.")
        print(f"    corr(effective rank, gain)      = {np.corrcoef(er, gn)[0,1]:+.3f}")
        print(f"    corr(p / effective rank, gain)  = {np.corrcoef(cp, gn)[0,1]:+.3f}")
        print("=" * 88)

    os.makedirs("tables", exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump({"gain_arm": GAIN_ARM, "rows": rows}, fh, indent=2)

    L = [r"\begin{table}[htbp]", r"    \centering",
         r"    \small\setlength{\tabcolsep}{5pt}",
         r"    \begin{tabular}{lrrrrr}", r"    \hline",
         r"    Dataset & $p$ & Effective rank & $p/\mathrm{eff}$ & Mean VIF"
         r" & $\Delta$ AUC (LASSO) \\", r"    \hline"]
    for r in rows:
        g = (f"${r['gain']:+.4f}$" + (r"$^{*}$" if r["gain_p"] and r["gain_p"] < .05 else "")) \
            if r["gain"] is not None else "--"
        # Dataset names go into \texttt verbatim, and two of them --
        # sector_etf and fama_french -- carry an underscore, which LaTeX reads
        # as a subscript operator outside math mode. Unescaped, it aborts the
        # compile with "Missing $ inserted" and takes the whole document with
        # it. The caption below always escaped them; the rows did not.
        ds = r["ds"].replace("_", r"\_")
        L.append(f"    \\texttt{{{ds}}} & {r['p']} & "
                 f"{r['effective_rank']:.1f} & {r['compression']:.1f} & "
                 f"{r['mean_vif']:.1f} & {g} \\\\")
    L += [r"    \hline", r"    \end{tabular}",
          r"    \caption{Collinearity across the ladder. Effective rank is "
          r"$\exp(H)$ for the Shannon entropy $H$ of the normalised eigenvalue "
          r"spectrum of the feature correlation matrix; $p/\mathrm{eff}$ is the "
          r"resulting compression factor. Mean VIF is the mean of "
          r"$\mathrm{diag}(R^{-1})$, computed by pseudo-inverse because these "
          r"matrices are severely ill-conditioned at the wider widths --- where "
          r"the two measures disagree, the effective rank is the more stable "
          r"and should be preferred. The final column is the logistic-regression "
          r"gain for LASSO at that width, from Table~\ref{tab:dimreal}. "
          r"\texttt{sector\_etf} and \texttt{fama\_french} are included for "
          r"context and are not part of the ladder.}",
          r"    \label{tab:collinearity}", r"\end{table}"]
    with open(OUT_TEX, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n  wrote {OUT_JSON}\n  wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
