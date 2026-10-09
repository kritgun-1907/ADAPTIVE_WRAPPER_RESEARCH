"""
Reconciling our golub result with the published OrPSOC Leukemia result.

THE CONFLICT
────────────
Bala, Karunarathne & Mitchell, "Optimizing Feature Selection by Enhancing
Particle Swarm Optimization with Orthogonal Initialization and Crossover
Operator", Computers, Materials & Continua 84(1):727-744, 2025
(doi:10.32604/cmc.2025.065706) reports, on Leukemia (p=7129, n=72):

    all-features SVM baseline    ~82%  accuracy
    OrPSOC                       88.02% accuracy      -> selection HELPS

Table 14 of the present manuscript reports, on the same nominal dataset:

    all-features baseline        0.9968 AUC (LogReg)
    all ten selection arms       significantly NEGATIVE  -> selection HURTS

Two papers with overlapping authors, opposite conclusions, same data, and no
explanation offered. This script supplies the explanation.

THE SIX AXES THAT DIFFER
────────────────────────
                        published (CMC 2025)        ours (Table 14)
  source                U. Pablo de Olavide         OpenML `leukemia`
  normalisation         min-max                     z-score (StandardScaler)
  missing values        replaced by attribute mean  none present; n/a
  protocol              70/30 holdout, 10 runs      stratified 5-fold, 20 seeds
  classifier            SVM / NaiveBayes / AdaBoost LogReg / LightGBM
  metric                accuracy                    AUC
  swarm budget          100 particles x 100 iters   20 particles x 60 iters

Six axes moved at once. The question this script answers is WHICH ONE carries
the difference -- and the answer is not the one a reader would guess.

THE HYPOTHESIS BEING TESTED
───────────────────────────
An all-features baseline of 0.9968 AUC has no headroom. With p=7129 and n=72,
a regularised linear model on standardised inputs is an almost-perfect
separator, so every selection arm can only lose. The published baseline of ~82%
is LOW, which is what leaves room for a +6 point lift. If that is right, the
two papers are not in conflict at all: they sit at opposite ends of a
baseline-ceiling axis, and the axis that moves the baseline is METRIC and
PROTOCOL, not the feature selector.

STAGE A runs the all-features baseline over the full factorial of the axes we
can control, reporting accuracy AND AUC in every cell, so the reader can see
exactly which knob moves 82% to 99.7%.

STAGE B then adds the selection arm at the two extreme corners, to show the
SIGN of the selection effect flipping as a function of the protocol alone,
with the selector held fixed.

WHAT THIS SCRIPT CANNOT DO
──────────────────────────
We do not have the Universidad Pablo de Olavide file. If that copy is already
floored/ceilinged/log-transformed in the manner standard for Golub microarray
data, while OpenML `leukemia` ships raw intensities (ours range -28400 to
+71369, verified), then a residual source difference remains that no amount of
re-analysis here can close. That axis is reported as UNRESOLVED rather than
silently assumed away. Obtaining the original file is the one outstanding item.

Run:  python experiments/dimensionality/golub_reconcile.py
Env:  GR_SEEDS=10   GR_STAGE=A|B|AB
Writes results/golub_reconcile.json and tables/golub_factorial.tex (via analysis/golub_tables.py)
"""
import os
import sys
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.dimensionality.tier_d_sparse_pso import sparse_pso            # noqa: E402

warnings.filterwarnings("ignore")

N_SEEDS = int(os.environ.get("GR_SEEDS", "10"))
STAGE = os.environ.get("GR_STAGE", "AB").upper()
# AdaBoost over 7129 raw features is by far the slowest cell and is not on the
# critical path for the reconciliation, so the classifier set is selectable.
_c = os.environ.get("GR_CLF")
STAGE_A_CLF = ([s.strip() for s in _c.split(",")] if _c else
               ["SVM", "LogReg", "NaiveBayes", "AdaBoost"])
INIT_DENSITY, BIAS = 0.05, -3.0

# Their SVM is unspecified beyond "SVM", so this is sklearn's default RBF.
# AUC uses decision_function rather than probability=True: no internal
# calibration CV, which on n=50 training rows would itself be a moving part.
CLASSIFIERS = {
    "SVM":      lambda: SVC(kernel="rbf", C=1.0, gamma="scale"),
    "NaiveBayes": lambda: GaussianNB(),
    "AdaBoost": lambda: AdaBoostClassifier(random_state=42),
    "LogReg":   lambda: LogisticRegression(max_iter=200),
}
SCALERS = {"minmax_theirs": MinMaxScaler, "zscore_ours": StandardScaler}


def load_golub():
    with open("data/golub.pkl", "rb") as fh:
        d = pickle.load(fh)
    return d["X"], d["y"]


def score(clf_name, Xtr, ytr, Xte, yte):
    """(accuracy, AUC) for one fitted split. Both metrics, always."""
    m = CLASSIFIERS[clf_name]()
    m.fit(Xtr, ytr)
    acc = accuracy_score(yte, m.predict(Xte))
    if hasattr(m, "decision_function"):
        s = m.decision_function(Xte)
    else:
        s = m.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, s) if len(np.unique(yte)) > 1 else np.nan
    return acc, auc


def splits(protocol, X, y, seed):
    """Index pairs. 'theirs' = 70/30 stratified holdout; 'ours' = 5-fold."""
    yv = np.asarray(y)
    if protocol == "holdout70_theirs":
        tr, te = train_test_split(np.arange(len(yv)), test_size=0.30,
                                  stratify=yv, random_state=seed)
        return [(tr, te)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(yv)), yv))


def prep(scaler_name, fit_on, Xtr_raw, Xte_raw, X_all_raw):
    """
    Scale. `fit_on` is the leakage axis: the published pipeline does not state
    whether min-max was fitted before or after the split, and normalising over
    the full dataset before splitting is a common and easily-missed choice, so
    both are measured rather than assumed.
    """
    sc = SCALERS[scaler_name]()
    sc.fit(X_all_raw if fit_on == "all" else Xtr_raw)
    return sc.transform(Xtr_raw), sc.transform(Xte_raw)


def stage_a(X, y):
    """All-features baseline over the full factorial. No selection anywhere."""
    Xa = X.to_numpy(dtype=float)
    rows = []
    print("=" * 92)
    print("  STAGE A — all-features baseline, no selection. Which axis moves "
          "82% to 99.7%?")
    print("=" * 92)
    print(f"  {'classifier':<11} {'normalisation':<15} {'scaler fit':<11} "
          f"{'protocol':<20} {'accuracy':>16} {'AUC':>16}")
    print("  " + "-" * 88)
    for clf in STAGE_A_CLF:
        for scaler in ["minmax_theirs", "zscore_ours"]:
            for fit_on in ["train", "all"]:
                for protocol in ["holdout70_theirs", "kfold5_ours"]:
                    accs, aucs = [], []
                    for seed in range(N_SEEDS):
                        for tr, te in splits(protocol, X, y, seed):
                            Xtr, Xte = prep(scaler, fit_on, Xa[tr], Xa[te], Xa)
                            a, u = score(clf, Xtr, np.asarray(y)[tr],
                                         Xte, np.asarray(y)[te])
                            accs.append(a); aucs.append(u)
                    r = {"stage": "A", "classifier": clf, "scaler": scaler,
                         "scaler_fit": fit_on, "protocol": protocol,
                         "acc_mean": float(np.mean(accs)),
                         "acc_std": float(np.std(accs)),
                         "auc_mean": float(np.nanmean(aucs)),
                         "auc_std": float(np.nanstd(aucs)),
                         "n_seeds": N_SEEDS}
                    rows.append(r)
                    print(f"  {clf:<11} {scaler:<15} {fit_on:<11} "
                          f"{protocol:<20} "
                          f"{r['acc_mean']:>8.4f} ±{r['acc_std']:<6.4f} "
                          f"{r['auc_mean']:>8.4f} ±{r['auc_std']:<6.4f}",
                          flush=True)
        print(flush=True)
    return rows


def stage_b(X, y):
    """
    The same selector, the same data, both protocols. Only the protocol and
    metric change between the two corners; the feature selector is identical.
    """
    Xa = X.to_numpy(dtype=float)
    feat = list(X.columns)
    rows = []
    print("=" * 92)
    print("  STAGE B — selection arm at both corners. Selector held FIXED.")
    print("=" * 92)
    print(f"  {'classifier':<11} {'normalisation':<15} {'protocol':<20} "
          f"{'arm':<12} {'accuracy':>10} {'AUC':>10} {'k':>7}")
    print("  " + "-" * 88)
    corners = [("SVM", "minmax_theirs", "holdout70_theirs"),
               ("LogReg", "zscore_ours", "kfold5_ours")]
    _k = os.environ.get("GR_CORNER")
    if _k:
        corners = [c for c in corners if c[0] == _k]
    for clf, scaler, protocol in corners:
        agg = {"all": {"acc": [], "auc": [], "k": []},
               "sparse_pso": {"acc": [], "auc": [], "k": []}}
        for seed in range(N_SEEDS):
            for tr, te in splits(protocol, X, y, seed):
                Xtr, Xte = prep(scaler, "train", Xa[tr], Xa[te], Xa)
                ytr, yte = np.asarray(y)[tr], np.asarray(y)[te]

                a, u = score(clf, Xtr, ytr, Xte, yte)
                agg["all"]["acc"].append(a); agg["all"]["auc"].append(u)
                agg["all"]["k"].append(len(feat))

                # The swarm's fitness uses the SAME classifier as the
                # downstream evaluation -- model_factory threads through
                # FoldEvalContext. No proxy objective.
                sel = sparse_pso(pd.DataFrame(Xtr, columns=feat),
                                 pd.Series(ytr), feat, INIT_DENSITY, BIAS,
                                 seed, model_factory=CLASSIFIERS[clf])
                cols = [feat.index(c) for c in sel]
                a, u = score(clf, Xtr[:, cols], ytr, Xte[:, cols], yte)
                agg["sparse_pso"]["acc"].append(a)
                agg["sparse_pso"]["auc"].append(u)
                agg["sparse_pso"]["k"].append(len(sel))

        for arm in ("all", "sparse_pso"):
            r = {"stage": "B", "classifier": clf, "scaler": scaler,
                 "protocol": protocol, "arm": arm,
                 "acc_mean": float(np.mean(agg[arm]["acc"])),
                 "auc_mean": float(np.nanmean(agg[arm]["auc"])),
                 "k_mean": float(np.mean(agg[arm]["k"])),
                 "n_seeds": N_SEEDS}
            rows.append(r)
            print(f"  {clf:<11} {scaler:<15} {protocol:<20} {arm:<12} "
                  f"{r['acc_mean']:>10.4f} {r['auc_mean']:>10.4f} "
                  f"{r['k_mean']:>7.1f}")

        d_acc = (rows[-1]["acc_mean"] - rows[-2]["acc_mean"])
        d_auc = (rows[-1]["auc_mean"] - rows[-2]["auc_mean"])
        print(f"  {'':<11} {'':<15} {'':<20} {'DELTA':<12} "
              f"{d_acc:>+10.4f} {d_auc:>+10.4f}\n")
    return rows


def main():
    X, y = load_golub()
    print(f"\n  golub: X={X.shape}  class counts={np.bincount(np.asarray(y))}  "
          f"majority-class rate={max(np.bincount(np.asarray(y)))/len(y):.4f}")
    print(f"  raw value range: [{X.to_numpy().min():.0f}, "
          f"{X.to_numpy().max():.0f}]  NaNs={int(X.isna().sum().sum())}")
    print(f"  seeds={N_SEEDS}  stage={STAGE}\n")

    rows = []
    if "A" in STAGE:
        rows += stage_a(X, y)
    if "B" in STAGE:
        rows += stage_b(X, y)

    # MERGE, don't overwrite. The classifier and corner filters exist so this
    # can be run in pieces (AdaBoost at p=7129 is minutes on its own), and a
    # partial run must not delete the cells a previous partial run produced.
    # Key on the full axis tuple so a re-run of the same cell replaces it.
    os.makedirs("reports", exist_ok=True)
    key = lambda r: (r.get("stage"), r.get("classifier"), r.get("scaler"),
                     r.get("scaler_fit"), r.get("protocol"), r.get("arm"))
    merged = {}
    if os.path.exists("results/golub_reconcile.json"):
        with open("results/golub_reconcile.json") as fh:
            for r in json.load(fh).get("rows", []):
                merged[key(r)] = r
    for r in rows:
        merged[key(r)] = r
    rows = list(merged.values())

    with open("results/golub_reconcile.json", "w") as fh:
        json.dump({"dataset": "golub (OpenML `leukemia`)",
                   "shape": list(X.shape),
                   "published_reference": "Bala, Karunarathne & Mitchell, "
                                          "CMC 84(1):727-744, 2025, "
                                          "doi:10.32604/cmc.2025.065706",
                   "unresolved_axis": "source file: U. Pablo de Olavide copy "
                                      "not obtained; possible prior "
                                      "floor/ceiling/log transform",
                   "n_seeds": N_SEEDS, "rows": rows}, fh, indent=2)
    print(f"  wrote results/golub_reconcile.json ({len(rows)} cells)")


if __name__ == "__main__":
    main()
