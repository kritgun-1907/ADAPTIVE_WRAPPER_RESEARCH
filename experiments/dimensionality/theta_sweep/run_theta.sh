#!/usr/bin/env bash
# Theta sweep on the EC2 box. Start with:  nohup bash ~/run_theta.sh > ~/theta.out 2>&1 &
set -u
DEADLINE_MIN=${DEADLINE_MIN:-240}
if [ "$DEADLINE_MIN" -gt 0 ]; then sudo shutdown -c 2>/dev/null; sudo shutdown -h +$DEADLINE_MIN; fi   # armed FIRST
TOOLS=${TOOLS:-$HOME}
cd ${ROOT:-$HOME/orpsoc_research}
PY=${PY:-$HOME/orpsoc-venv/bin/python}
SRC=${SRC:-experiments/tier_d_fullnovelty.py}; DST=${DST:-experiments/tier_d_theta.py}
export ORPSOC_N_JOBS=${ORPSOC_N_JOBS:-8}
export FN_MODELS=LogReg FN_ARMS=all,sparse_pso,mrmr,lasso,rfe
$PY $TOOLS/make_theta_script.py $SRC $DST || { echo "PATCH FAILED - nothing was run"; exit 1; }
echo "[$(date -u +%H:%M)] stage A: theta=0.5 replication, ind17, 2 seeds"
FN_THETA=0.5 FN_DATASETS=ind17 FN_SEEDS=2 $PY -u $DST || { echo "STAGE A FAILED"; exit 1; }
$PY $TOOLS/check_theta05.py $TOOLS/ref_ind17_logreg_theta05.json results/theta_sweep/fn_theta0.50_ind17_LogReg.json
for TH in ${THETAS:-0.7 0.9 0.5 0.3 1.0}; do
  echo "[$(date -u +%H:%M)] theta=$TH  ${DATASETS:-ind17,ind30}  ${SEEDS:-20} seeds"
  FN_THETA=$TH FN_DATASETS=${DATASETS:-ind17,ind30} FN_SEEDS=${SEEDS:-20} $PY -u $DST || echo "theta=$TH FAILED"
  $PY $TOOLS/summarize_theta.py
done
echo "[$(date -u +%H:%M)] ALL DONE"
