# Fitness-weight (theta) sweep, Supplementary Table S13

Sparse PSO and its matched-size filters (mRMR, LASSO, RFE) plus the all-features baseline, logistic regression,
ind17 and ind30, 20 seeds, theta in {0.3, 0.5, 0.7, 0.9, 1.0}.

## How it was produced
1. `make_theta_script.py` builds `experiments/tier_d_theta.py` from `experiments/tier_d_fullnovelty.py` with five
   exact-match edits (the original file is not modified, so its provenance hash and checkpoints stay valid):
   theta read from `FN_THETA` with no default; separate checkpoint store `theta_sweep`; theta set inside each
   worker on the sparse-PSO module (loky workers re-import it at 0.5); outputs to `results/theta_sweep/`.
2. `run_theta.sh` runs stage A (theta = 0.5, ind17, 2 seeds), compares it with `ref_ind17_logreg_theta05.json` via
   `check_theta05.py`, then runs the five thetas and calls `summarize_theta.py`.
3. `results/theta_sweep_per_seed.csv` holds the per-seed AUCs and subset sizes (200 rows) behind S13.

## Environment of the run (2026-11-12)
EC2 c7a.2xlarge, 8 workers, Ubuntu 24.04.4 LTS, Python 3.12.3, numpy 2.2.6, pandas 3.0.1, scipy 1.17.1,
scikit-learn 1.8.0, lightgbm 4.6.0. Code provenance hash 29a57cced36f at theta = 0.5.

## Known limits
- At theta = 0.5 the rerun reproduces the stored paper values for all-features, sparse PSO, mRMR and LASSO to
  four decimals. RFE does not (environment-sensitive; see the RFE note after Table S13), so RFE is omitted from S13.
- Raw per-unit JSONs (`results/theta_sweep/fn_theta*_ind17-ind30_LogReg.json`) live on the EC2 box only.
- The scripts assume `~/orpsoc_research` and `~/orpsoc-venv`; override with ROOT, PY, TOOLS, SRC, DST.
