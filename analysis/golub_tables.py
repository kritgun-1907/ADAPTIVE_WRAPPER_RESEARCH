"""
Render results/golub_reconcile.json as the Stage A factorial table.

Separated from golub_reconcile.py so the table can be regenerated without
re-running any of the fits, and so a partially-populated JSON (the runner takes
GR_CLF and GR_CORNER filters) renders whatever cells exist rather than failing.

Run:  python analysis/golub_tables.py
Writes tables/golub_factorial.tex
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC = "results/golub_reconcile.json"
OUT = "tables/golub_factorial.tex"

CLF_ORDER = ["SVM", "NaiveBayes", "AdaBoost", "LogReg"]
CLF_LABEL = {"SVM": "SVM (RBF)", "NaiveBayes": "Na\\\"ive Bayes",
             "AdaBoost": "AdaBoost", "LogReg": "Logistic regression"}
SCALER_LABEL = {"minmax_theirs": "min--max", "zscore_ours": "$z$-score"}
PROTO_LABEL = {"holdout70_theirs": "70/30 holdout", "kfold5_ours": "5-fold CV"}


def main():
    with open(SRC) as fh:
        blob = json.load(fh)
    rows = [r for r in blob["rows"] if r.get("stage") == "A"]
    if not rows:
        raise SystemExit(f"{SRC} has no Stage A cells; run golub_reconcile.py")

    # Only the scaler-fitted-on-train cells go in the paper table. The
    # fit-on-all cells exist to show the leakage axis is small; that is a
    # sentence in the text, not four more columns.
    rows = [r for r in rows if r["scaler_fit"] == "train"]
    clfs = [c for c in CLF_ORDER if any(r["classifier"] == c for r in rows)]
    idx = {(r["classifier"], r["scaler"], r["protocol"]): r for r in rows}

    L = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"    \centering")
    L.append(r"    \small\setlength{\tabcolsep}{5pt}")
    L.append(r"    \begin{tabular}{llrrrr}")
    L.append(r"    \hline")
    L.append(r"    & & \multicolumn{2}{c}{70/30 holdout} "
             r"& \multicolumn{2}{c}{5-fold CV} \\")
    L.append(r"    Classifier & Norm. & Accuracy & AUC & Accuracy & AUC \\")
    L.append(r"    \hline")
    for clf in clfs:
        for i, scaler in enumerate(["minmax_theirs", "zscore_ours"]):
            cells = []
            for proto in ["holdout70_theirs", "kfold5_ours"]:
                r = idx.get((clf, scaler, proto))
                if r is None:
                    cells += ["--", "--"]
                else:
                    cells.append(f"${r['acc_mean']:.4f}$")
                    cells.append(f"${r['auc_mean']:.4f}$")
            name = CLF_LABEL.get(clf, clf) if i == 0 else ""
            L.append(f"    {name} & {SCALER_LABEL[scaler]} & "
                     + " & ".join(cells) + r" \\")
        L.append(r"    \hline")
    L.append(r"    \end{tabular}")
    L.append(r"    \caption{The golub factorial, all-features baseline only "
             r"--- no selection is applied anywhere in this table. Four axes "
             r"are varied; both metrics are reported in every cell. "
             r"10 repetitions; the scaler is fitted on the training split "
             r"throughout (fitting it on the full dataset instead moves every "
             r"cell by less than one point and is omitted). The published "
             r"configuration of Bala et al.~\cite{bala2025orpsoc} is the "
             r"SVM / min--max / 70/30 cell, and it reproduces their reported "
             r"baseline. The classifier axis moves the baseline by fourteen "
             r"to seventeen points; no other axis moves it by more than "
             r"three.}")
    L.append(r"    \label{tab:golubfactorial}")
    L.append(r"\end{table}")

    os.makedirs("tables", exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"  wrote {OUT}  ({len(rows)} cells, classifiers: {', '.join(clfs)})")


if __name__ == "__main__":
    main()
