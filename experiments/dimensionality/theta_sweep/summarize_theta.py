"""Table for the Supplement: per theta x dataset, sparse_pso vs all-features and matched filters."""
import glob, json, re, sys
import numpy as np
from scipy.stats import wilcoxon
rows = {}
for f in sorted(glob.glob("results/theta_sweep/fn_theta*_*_LogReg.json")):
    th = float(re.search(r"theta([0-9.]+)_", f).group(1))
    for r in json.load(open(f)):
        if len(r["arms"]["all"]["auc"]) < 2:
            continue
        rows.setdefault((th, r["ds"]), {})[r["seed"]] = r["arms"]
print(f"{'theta':>5} {'data':<7}{'n':>3} {'all':>7} {'pso':>7} {'dPSO':>8} {'p':>7} {'k':>6} {'mrmr':>7} {'lasso':>7} {'rfe':>7}  %all")
for (th, ds), per in sorted(rows.items()):
    if len(per) < 6:
        print(f"{th:5.2f} {ds:<7}{len(per):>3}  (too few seeds, skipped)")
        continue
    def sm(arm):                       # per-seed mean AUC over scoreable folds (drop the unscoreable fold 0)
        return np.array([np.mean(per[s][arm]["auc"][1:]) for s in sorted(per)])
    allv, pso = sm("all"), sm("sparse_pso")
    d = pso - allv
    p = wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
    k = np.mean([np.mean(per[s]["sparse_pso"]["k"][1:]) for s in per])
    print(f"{th:5.2f} {ds:<7}{len(per):>3} {allv.mean():7.4f} {pso.mean():7.4f} {d.mean():+8.4f} {p:7.4f} {k:6.1f} "
          f"{sm('mrmr').mean():7.4f} {sm('lasso').mean():7.4f} {sm('rfe').mean():7.4f}  {100*pso.mean()/allv.mean():4.0f}")
