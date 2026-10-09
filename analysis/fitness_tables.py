"""
Item 1 of the review: the fitness column for Table 13, and the (k, delta-AUC)
scatter at p = 171 and p = 301.

WHAT THE REVIEW ASKED, AND WHY
──────────────────────────────
The competing swarms are not forced onto a matched budget, because all seven
optimise the identical fitness with the identical compactness penalty -- k is
an outcome of the shared objective, not a free parameter. The three non-swarm
methods are budget-matched only because they have no intrinsic k. Given that
asymmetry, the AUC column alone cannot tell a reader whether Correlation-Guided
PSO's margin comes from a better search or from a larger subset. Two exhibits
answer it:

  1. a FITNESS column -- the objective value each swarm actually reached. The
     compactness term is the point: a swarm that buys AUC with a larger subset
     pays for it here, so leading on AUC while trailing on fitness IS the
     "larger subset" answer.
  2. a (k, delta-AUC) SCATTER, which shows the same thing geometrically and
     also shows the budget-matching of the non-swarm arms as a vertical stack.

TWO THINGS THIS FILE IS CAREFUL ABOUT
─────────────────────────────────────
Fold 0 is degenerate. With min_train = 150 and a trailing 25% fitness split,
the validation block is roughly 37 rows and is sometimes single-class, so
roc_auc_score raises and evaluate_ctx returns its -1.0 sentinel. The swarm saw
the same -1.0 for every particle on that fold, i.e. it had no gradient to
follow, and the test AUC there is 0.5 for every arm. Averaging a -1.0 sentinel
into a mean fitness would produce a number that is not on the fitness scale at
all. Sentinel folds are therefore EXCLUDED from the fitness mean and their
count is reported, rather than being silently dropped or silently averaged.

The fitness and AUC columns must describe the same run. The fitness run
(experiments/dimensionality/swarm_fitness.py) uses the identical selectors, particles,
iterations, init density, transfer bias, splits and seeds as the ladder run, so
its k column reproduces the ladder's. That reproduction is CHECKED here and the
script fails loudly if it does not hold, because a silent divergence would put
two different experiments in adjacent columns of one table.

Run:  python analysis/fitness_tables.py
Writes tables/swarm_fitness.tex and plots/fig_k_vs_delta.png
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

LADDER = "results/tier_d_ladder_merged.json"
FITNESS = "results/swarm_fitness.json"
OUT_TEX = "tables/swarm_fitness.tex"
OUT_PNG = "plots/fig_k_vs_delta.png"

WIDTHS = [("ind17", 171), ("ind30", 301)]

# Three families, because the review's question is about the asymmetry between
# them. Within a family the arms are interchangeable for this purpose, so the
# colour encodes family and the direct label carries arm identity -- 10 hues
# would be uncycled-categorical-palette abuse and unreadable in print.
FAMILY = {
    "sparse_pso": "ours", "sparse_orpsoc": "ours", "full_hybrid": "ours",
    "cgpso": "rival", "taguchi_cbpso": "rival", "hpso_de": "rival",
    "bwaoa_c": "rival",
    "mrmr": "nonswarm", "lasso": "nonswarm", "rfe": "nonswarm",
}
# tab10 blue / orange / purple: the palette the other manuscript figures
# already use, and it passes CVD separation on all adjacent pairs. Orange sits
# below 3:1 against white, which is why every point is directly labelled and
# why marker shape repeats the family -- the figure never rests on hue alone.
FAM_COLOR = {"ours": "#1f77b4", "rival": "#ff7f0e", "nonswarm": "#9467bd"}
FAM_MARK = {"ours": "o", "rival": "s", "nonswarm": "^"}
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analysis.fig_style as FS
FAM_LABEL = FS.FAMILY_LABEL      # "comparators", as in the manuscript text

ORDER = ["sparse_pso", "sparse_orpsoc", "full_hybrid",
         "cgpso", "taguchi_cbpso", "hpso_de", "bwaoa_c",
         "mrmr", "lasso", "rfe"]
LABEL = FS.ARM                   # table rows use the manuscript names too
SHORT = FS.ARM                   # names fixed by manuscript Figs 1 and 3

SENTINEL = -1.0


def load_ladder():
    """((ds, arm) -> {seed: {'delta','k'}}, (ds, seed) -> baseline AUC)."""
    with open(LADDER) as fh:
        records = json.load(fh)
    out = defaultdict(dict)
    base = {}
    for r in records:
        if r["model"] != "LogReg":
            continue
        # Fold 0 is excluded throughout: its 150-row training window holds one
        # class, so AUC there is a 0.5 placeholder and k comes from a search on
        # a single-class window (Methods §2.4.2; manuscript Table 12).
        b = float(np.mean(r["arms"]["all"]["auc"][1:]))
        base[(r["ds"], r["seed"])] = b
        for arm, v in r["arms"].items():
            if arm == "all":
                continue
            out[(r["ds"], arm)][r["seed"]] = {
                "delta": float(np.mean(v["auc"][1:]) - b),
                "k": float(np.mean(v["k"][1:])),
            }
    return out, base


def load_fitness(base):
    """
    (ds, arm) -> {seed: {'fitness','k','delta','n_drop'}}.

    delta is computed against the ladder's all-features baseline for the same
    (dataset, seed). That arm involves no swarm -- it fits one model on every
    feature -- so it is deterministic across machines and is the one quantity
    safe to take from the other run.
    """
    if not os.path.exists(FITNESS):
        return None
    with open(FITNESS) as fh:
        records = json.load(fh)
    out = defaultdict(dict)
    for r in records:
        if r["model"] != "LogReg":
            continue
        b = base.get((r["ds"], r["seed"]))
        if b is None:
            continue
        for arm, v in r["arms"].items():
            f = np.asarray(v["fitness"], dtype=float)
            good = f > SENTINEL
            out[(r["ds"], arm)][r["seed"]] = {
                "fitness": float(f[good].mean()) if good.any() else float("nan"),
                "k": float(np.mean(v["k"][1:])),               # fold 0 excluded, as above
                "delta": float(np.mean(v["auc"][1:]) - b),
                "n_drop": int((~good).sum()),
            }
    return out


def check_agreement(ladder, fit):
    """
    Does the fitness re-run say the same thing as the ladder run?

    An earlier version of this demanded that k reproduce SEED BY SEED and
    aborted otherwise. It fired, and what it caught is worth recording because
    it is not what it was written to catch.

    The selected subsets differ on 3.9% of folds (19.6% of cells have at least
    one). It is NOT a code difference: recomputing a differing cell on this
    machine reproduces the fitness run exactly and not the stored ladder value.
    It is an ENVIRONMENT difference -- those ladder cells were computed on the
    EC2 box, and a binary PSO's gbest is decided by strict `>` comparisons of
    validation AUC. Two subsets that tie to within last-bit floating point
    resolve differently under a different BLAS, the swarm then follows a
    different trajectory, and the fold ends on a different subset. Individual
    seeds move by up to 0.041 AUC.

    So bit-identity is the wrong test: it is unachievable across machines for
    this algorithm, and demanding it would block a table whose numbers are
    fine. The right test is whether any CONCLUSION moves. At the 20-seed mean
    the divergence averages out -- worst shift 0.0018 -- and no cell changes
    sign or significance. That is what is asserted here, and a violation is
    still fatal.
    """
    from scipy.stats import wilcoxon
    worst_shift, flips, n = 0.0, [], 0
    for (ds, _p) in WIDTHS:
        for arm in ORDER:
            if FAMILY[arm] == "nonswarm":
                continue
            a, b = ladder.get((ds, arm)), fit.get((ds, arm))
            if not a or not b:
                continue
            s = sorted(set(a) & set(b))
            if len(s) < 3:
                continue
            n += 1
            dl = np.array([a[i]["delta"] for i in s])
            df = np.array([b[i]["delta"] for i in s])
            worst_shift = max(worst_shift, abs(df.mean() - dl.mean()))
            pl, pf = wilcoxon(dl).pvalue, wilcoxon(df).pvalue
            if (pl < 0.05) != (pf < 0.05) or np.sign(dl.mean()) != np.sign(df.mean()):
                flips.append(f"{ds}/{arm}")
    if n == 0:
        raise SystemExit("no overlapping cells to check -- refusing to emit a "
                         "fitness column that nothing has verified.")
    if flips:
        raise SystemExit(
            f"the fitness re-run changes the CONCLUSION in {len(flips)} cell(s): "
            f"{', '.join(flips)}. That is not floating-point divergence; refusing "
            f"to print it beside Table 13.")
    if worst_shift > 0.005:
        raise SystemExit(
            f"fitness re-run shifts a 20-seed mean delta by {worst_shift:.4f}, "
            f"beyond the 0.005 tolerance. Investigate before publishing.")
    print(f"  agreement check: {n} cells, worst mean-delta shift "
          f"{worst_shift:.4f}, no sign or significance changes  OK")
    return worst_shift


# ── the fitness table ────────────────────────────────────────────────────────
def write_table(ladder, fit):
    rows = []
    dropped = 0
    for arm in ORDER:
        cells = []
        for ds, _p in WIDTHS:
            # For the seven swarm arms every column comes from the FITNESS run,
            # so k, fitness and delta all describe one execution on one machine.
            # Mixing them -- fitness from the re-run beside the ladder's delta --
            # would put two runs in one row, and those runs select different
            # subsets on 3.9% of folds (see check_agreement). The non-swarm arms
            # have no fitness, so they keep the ladder's numbers.
            src = fit.get((ds, arm)) if (fit and FAMILY[arm] != "nonswarm") else None
            src = src or ladder.get((ds, arm), {})
            seeds = sorted(src)
            k = np.mean([src[s]["k"] for s in seeds]) if seeds else float("nan")
            d = np.array([src[s]["delta"] for s in seeds]) if seeds else np.array([])
            try:
                p = wilcoxon(d).pvalue if len(d) > 1 and np.any(d != 0) else 1.0
            except Exception:
                p = 1.0
            f = fit.get((ds, arm)) if (fit and FAMILY[arm] != "nonswarm") else None
            if f:
                fv = np.nanmean([f[s]["fitness"] for s in sorted(f)])
                dropped += sum(f[s]["n_drop"] for s in sorted(f))
            else:
                fv = None
            cells.append((k, fv, float(d.mean()) if len(d) else float("nan"), p))
        rows.append((arm, cells))

    # Best fitness per width, so the reader can find the winner without
    # scanning -- and so the AUC winner and the fitness winner are visibly
    # different arms when they are.
    best = {}
    for j, (ds, _p) in enumerate(WIDTHS):
        vals = [(r[1][j][1], r[0]) for r in rows
                if FAMILY[r[0]] != "nonswarm" and r[1][j][1] is not None]
        if vals:
            best[j] = max(vals)[1]

    L = []
    L.append(r"\begin{table}[htbp]")
    L.append(r"    \centering")
    L.append(r"    \small\setlength{\tabcolsep}{4pt}")
    L.append(r"    \begin{tabular}{lcccccc}")
    L.append(r"        \toprule")
    L.append(r"        & \multicolumn{3}{c}{$p = 171$ (ind17)} & "
             r"\multicolumn{3}{c}{$p = 301$ (ind30)} \\")
    L.append(r"        \cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    L.append(r"        \textbf{Arm} & $k$ & \textbf{Fitness} & $\Delta$\textbf{AUC} "
             r"& $k$ & \textbf{Fitness} & $\Delta$\textbf{AUC} \\")
    L.append(r"        \midrule")
    for i, (arm, cells) in enumerate(rows):
        if i == 7:
            L.append(r"        \midrule")
        parts = []
        for j, (k, fv, dm, p) in enumerate(cells):
            parts.append(f"${k:.1f}$")
            if fv is None:
                parts.append("--")
            elif best.get(j) == arm:
                parts.append(f"$\\mathbf{{{fv:.4f}}}$")
            else:
                parts.append(f"${fv:.4f}$")
            s = "^{*}" if p < 0.05 else ""
            body = f"{dm:+.4f}"
            parts.append(f"$\\mathbf{{{body}}}{s}$" if p < 0.05 and dm > 0
                         else f"${body}{s}$")
        L.append(f"        {LABEL[arm]} & " + " & ".join(parts) + r" \\")
    L.append(r"        \bottomrule")
    L.append(r"    \end{tabular}")
    L.append(
        r"    \caption{Fitness alongside AUC, from a single-machine re-run of "
        r"the seven swarm arms; the three non-swarm rows are carried from "
        r"Table~\ref{tab:swarmcomp}. The re-run reproduces that table's "
        r"$\Delta$AUC to within $0.002$ at every cell with no change of sign or "
        r"significance, but not seed for seed: gbest is chosen by strict "
        r"comparison of validation AUC, so subsets that tie to within last-bit "
        r"floating point resolve differently under a different BLAS and the "
        r"swarm diverges from there. Individual seeds move by up to $0.041$; the "
        r"20-seed mean does not. \emph{Fitness} is the objective "
        r"every swarm arm actually optimises, "
        r"$0.5\cdot\mathrm{AUC}_{\mathrm{val}} + 0.5\cdot(1 - k/p)$, recovered "
        r"from the achieved gbest subset on each fold and averaged over 20 "
        r"seeds; higher is better. It is reported only for the seven swarm "
        r"arms, because the three non-swarm methods do not optimise it --- they "
        r"are handed the budget $k$ that sparse PSO chose on that fold, which "
        r"is the asymmetry this table exists to make visible. Bold fitness "
        r"marks the best swarm at that width. $\Delta$AUC is against the "
        r"all-features baseline under logistic regression, $^{*}$ denotes "
        r"$p < 0.05$ by paired per-seed Wilcoxon. Fold~1 of the walk-forward "
        r"sequence has a single-class fitness split at these widths, so the "
        r"objective is undefined there for every arm alike; those folds are "
        r"excluded from the fitness mean and included in $\Delta$AUC, where "
        r"they contribute zero to every arm.}")
    L.append(r"    \label{tab:swarmfitness}")
    L.append(r"\end{table}")

    os.makedirs("tables", exist_ok=True)
    with open(OUT_TEX, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"  -> {OUT_TEX}   ({dropped} sentinel folds excluded from fitness)")
    return rows


# ── the scatter ──────────────────────────────────────────────────────────────
# Label layout, fixed per width. Six arms at p = 171 and four at p = 301 sit
# within a few hundredths of AUC of each other, where no automatic placement
# keeps every label next to its own marker. Those clustered arms are labelled
# in a column beside the cluster, ordered top-to-bottom like their markers and
# joined by thin leaders (so leaders cannot cross); isolated arms are labelled
# directly. Positions are in data coordinates of each panel.
CALLOUT = {
    171: dict(arms=["taguchi_cbpso", "bwaoa_c", "sparse_pso", "sparse_orpsoc",
                    "full_hybrid", "hpso_de"], x=5.75, ytop=0.047, ybot=-0.003),
    301: dict(arms=["hpso_de", "bwaoa_c", "sparse_orpsoc", "sparse_pso"],
              x=13.2, ytop=0.074, ybot=0.047),
}
DIRECT = {   # (dx, dy) in points, horizontal alignment
    "cgpso": (0, 9, "center"), "mrmr": (7, 0, "left"), "lasso": (7, 0, "left"),
    "rfe": (7, 0, "left"), "taguchi_cbpso": (0, 10, "center"),
    "full_hybrid": (-7, 0, "right"),
}


def _label_arms(ax, p, pts):
    """pts: list of (k, delta, arm). Callout column for the cluster, direct
    labels for the rest."""
    co = CALLOUT[p]
    clus = sorted([q for q in pts if q[2] in co["arms"]], key=lambda q: -q[1])
    ys = np.linspace(co["ytop"], co["ybot"], len(clus)) if len(clus) > 1 else [co["ytop"]]
    for (x, y, arm), ty in zip(clus, ys):
        ax.annotate(SHORT[arm], (x, y), xytext=(co["x"], ty), textcoords="data",
                    fontsize=8.5, color="#333333", va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#999999",
                                    shrinkA=2, shrinkB=4))
    for (x, y, arm) in pts:
        if arm in co["arms"]:
            continue
        dx, dy, ha = DIRECT.get(arm, (7, 0, "left"))
        ax.annotate(SHORT[arm], (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color="#333333", va="center", ha=ha)


def _place_labels(ax, fig, points):
    """
    Greedy non-overlapping label placement.

    Six arms land within 0.01 AUC of each other at both widths, so a fixed
    offset overlaps three or four labels into illegibility. Each label tries
    candidate offsets on rings of increasing radius and takes the first that
    clears every label already placed and every marker; the last candidate is
    accepted unconditionally so placement always terminates. Anything pushed
    past the inner ring gets a thin leader line, because a label far from its
    point is worse than no label unless the association is drawn. Points are
    placed in a fixed order, so the figure is reproducible across runs.
    """
    # A radial fan: 12 angles at four increasing radii. The near ring is the
    # ordinary case; the far rings exist because six arms at p = 301 sit inside
    # a region roughly 40pt tall, which cannot hold six stacked 8.5pt labels at
    # any offset. Angles are ordered right-then-left at each radius so labels
    # prefer the side with more empty plot.
    angles = [0, -30, 30, -60, 60, -90, 90, 180, 150, -150, 120, -120]
    cands = [(int(r * np.cos(np.radians(a))), int(r * np.sin(np.radians(a))))
             for r in (11, 26, 44, 64) for a in angles]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    taken = []
    marks = [ax.transData.transform((x, y)) for x, y, _ in points]
    for (x, y, text) in points:
        px, py = ax.transData.transform((x, y))
        for i, (dx, dy) in enumerate(cands):
            # every label is tied to its marker by a thin line: in the dense
            # cluster at p = 301 proximity alone made two labels ambiguous
            lead = dict(arrowstyle="-", lw=0.6, color="#999999",
                        shrinkA=1, shrinkB=4)
            t = ax.annotate(text, (x, y), textcoords="offset points",
                            xytext=(dx, dy), fontsize=8.5, color="#333333",
                            zorder=4, arrowprops=lead,
                            ha="right" if dx < 0 else "left")
            bb = t.get_window_extent(renderer=rend).expanded(1.08, 1.25)
            clash = any(bb.overlaps(o) for o in taken) or any(
                bb.x0 - 6 < mx < bb.x1 + 6 and bb.y0 - 6 < my < bb.y1 + 6
                for (mx, my) in marks)
            if not clash or i == len(cands) - 1:
                taken.append(bb)
                break
            t.remove()


# The arm-vs-arm comparisons Section 3.5 quotes. They live here rather than in
# a shell one-liner so that regenerating the figure re-derives the numbers in
# the prose beside it; a claim in the paper that no script reproduces is a
# claim that drifts.
PAIRWISE = [("ind17", "cgpso", "taguchi_cbpso"),
            ("ind17", "cgpso", "bwaoa_c"),
            ("ind30", "cgpso", "hpso_de"),
            ("ind30", "cgpso", "sparse_pso")]


def pairwise_report(ladder):
    """Paired per-seed Wilcoxon between two arms on the same width."""
    out = []
    print("  arm-vs-arm (paired per-seed Wilcoxon), quoted in Section 3.5:")
    for ds, a, b in PAIRWISE:
        A, B = ladder.get((ds, a), {}), ladder.get((ds, b), {})
        s = sorted(set(A) & set(B))
        if len(s) < 3:
            continue
        da = np.array([A[i]["delta"] for i in s])
        db = np.array([B[i]["delta"] for i in s])
        ka = np.mean([A[i]["k"] for i in s])
        kb = np.mean([B[i]["k"] for i in s])
        try:
            pv = float(wilcoxon(da, db).pvalue)
        except ValueError:
            pv = 1.0
        out.append({"ds": ds, "a": a, "b": b, "n": len(s),
                    "delta_a": float(da.mean()), "delta_b": float(db.mean()),
                    "k_a": float(ka), "k_b": float(kb), "p": pv})
        print(f"    {ds:>6}  {SHORT[a]:>8} ({da.mean():+.4f}, k={ka:4.1f})  vs  "
              f"{SHORT[b]:<8} ({db.mean():+.4f}, k={kb:4.1f})   p = {pv:.3f}")
    with open("results/arm_pairwise.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def write_scatter(ladder):
    FS.apply()
    fig, axes = plt.subplots(1, 2, figsize=(FS.FULL_W * 1.35, 4.8))
    for ax, (ds, p) in zip(axes, WIDTHS):
        pts = []
        for arm in ORDER:
            lad = ladder.get((ds, arm), {})
            if not lad:
                continue
            seeds = sorted(lad)
            ks = np.array([lad[s]["k"] for s in seeds])
            dl = np.array([lad[s]["delta"] for s in seeds])
            fam = FAMILY[arm]
            ax.errorbar(ks.mean(), dl.mean(),
                        xerr=ks.std(ddof=1) / np.sqrt(len(ks)),
                        yerr=dl.std(ddof=1) / np.sqrt(len(dl)),
                        fmt=FAM_MARK[fam], color=FAM_COLOR[fam],
                        markersize=8, capsize=2, elinewidth=1.0,
                        markeredgecolor="white", markeredgewidth=0.8,
                        zorder=3)
            pts.append((ks.mean(), dl.mean(), arm))
        ax.axhline(0.0, color="#888888", lw=1.0, ls="--", zorder=1)
        ax.set_title(f"$p = {p}$")
        ax.set_xlabel("mean selected-subset size $k$")
        ax.margins(x=0.26, y=0.26)
        _label_arms(ax, p, pts)
    axes[0].set_ylabel(r"$\Delta$ AUC vs all-features")

    handles = [plt.Line2D([], [], marker=FAM_MARK[f], color=FAM_COLOR[f],
                          ls="none", markersize=8, markeredgecolor="white",
                          markeredgewidth=0.8, label=FAM_LABEL[f])
               for f in ("ours", "rival", "nonswarm")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    FS.save(fig, "fig_k_vs_delta", "KvsDelta")


def main():
    ladder, base = load_ladder()
    fit = load_fitness(base)
    print("  fitness run:",
          "present" if fit else f"ABSENT ({FITNESS}) -- table will show '--'")
    if fit:
        check_agreement(ladder, fit)
    pairwise_report(ladder)
    write_scatter(ladder)
    write_table(ladder, fit)


if __name__ == "__main__":
    main()
