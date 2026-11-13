#!/usr/bin/env bash
# Rebuild everything that regenerates from the stored results plus the Zenodo
# checkpoint archives, in one command. It unpacks the archives in the right
# order, rebuilds results/tier_d_ladder_merged.json and re-renders every table
# and figure. It never re-runs an experiment and needs no datasets.
#
#   bash scripts/rebuild_from_deposit.sh /path/to/folder/with/the/zenodo/files
#
# The folder must hold checkpoints.tgz, ladder_penalty_c7a_box_20261113.tgz,
# knn_synthetic_m4_20261112.tgz and theta_sweep_raw.tgz. PY overrides the
# interpreter (default python3). Run it on a clean checkout: it refuses if
# results/checkpoints/ exists or results/ or tables/ have local changes. It ends
# by listing any regenerated file that differs from the committed one; on a
# clean run that list is empty and it exits 0.
#
# Not covered: Table S5 (collinearity.py needs the datasets) and the legacy
# analysis/nested_cv_tables.py (not in the paper; exits 1 by design).
set -euo pipefail

DEP=$(cd "${1:?usage: bash scripts/rebuild_from_deposit.sh DEPOSIT_DIR}" && pwd)
cd "$(dirname "$0")/.."
PY=${PY:-python3}
export MPLBACKEND=Agg ORPSOC_GEN=aad95a66b882
G=results/checkpoints/tier_d_fullnovelty
ARCHIVES="checkpoints.tgz ladder_penalty_c7a_box_20261113.tgz knn_synthetic_m4_20261112.tgz theta_sweep_raw.tgz"

sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

for a in $ARCHIVES; do
  [ -f "$DEP/$a" ] || { echo "missing $DEP/$a"; exit 1; }
done
[ -e results/checkpoints ] && { echo "results/checkpoints/ already exists; remove it first (stale units would win the merge)"; exit 1; }
[ -n "$(git status --porcelain -- results tables)" ] && { echo "results/ or tables/ have local changes; commit or stash them first"; exit 1; }

echo "== unpacking"
tar xzf "$DEP/checkpoints.tgz"
# The stored units were written under the pre-release code hash 29a57cced36f;
# this release's code hashes to aad95a66b882. cp -p keeps their file times, so
# the 2026-11 units unpacked next are newer and win per arm in build_ladder.py.
mkdir -p "$G/aad95a66b882"
cp -p "$G"/29a57cced36f/*.pkl "$G/aad95a66b882/"
for a in ladder_penalty_c7a_box_20261113.tgz knn_synthetic_m4_20261112.tgz; do
  tar xzf "$DEP/$a" --exclude README.txt --exclude UNITS.sha256
  tar xzf "$DEP/$a" -O UNITS.sha256 | sha256 -c - > /dev/null || { echo "checksum failure in $a"; exit 1; }
  echo "  $a: units verified"
done
tar xzf "$DEP/theta_sweep_raw.tgz"

echo "== rebuilding"
run() { echo "-- $*"; "$PY" "$@" > /dev/null; }
run experiments/dimensionality/build_ladder.py                 # results/tier_d_ladder_merged.json
run analysis/ladder_tables.py                                  # Tables 6, 7, 8b, S2, S4
run analysis/knn_synthetic_tables.py                           # Table 8a (both panels)
run analysis/ridge_tables.py                                   # Table S6
run experiments/penalty/nested_cv_stored.py --table-only       # Table 9
run analysis/fitness_tables.py                                 # Table 12 fitness column, Fig. S10
run analysis/golub_tables.py                                   # Table S9
run experiments/adaptation/crossseed_jaccard.py                # Table S11
run analysis/make_manuscript_figures.py                        # Figs 3, 5, S2, S5, S6, S7
run analysis/nested_cv_figure_ref.py                           # Fig. 4
run analysis/make_supplementary_figures.py                     # Figs S1, S3, S4, S8, S9
echo "-- Table S13 (printed from the raw theta-sweep outputs):"
"$PY" experiments/dimensionality/theta_sweep/summarize_theta.py

echo "== regenerated files that differ from the committed ones:"
if [ -n "$(git status --porcelain -- results tables)" ]; then
  git status --short -- results tables
  exit 1
fi
echo "   none. Figures are in figures/ and plots/ (not committed, so not compared)."
