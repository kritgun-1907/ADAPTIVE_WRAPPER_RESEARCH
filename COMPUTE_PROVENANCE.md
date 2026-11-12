# Where each committed result was computed

This inventory records, for every file in `results/` (and for the one table
whose underlying records survive only as LaTeX), the platform it was computed
on, the environment where one is on record, the date, and the evidence. It was
compiled on 2026-11-12 from the stored results, the surviving machine bundles
and checkpoints, and reruns on an Apple M4. Anything that could not be
determined is marked **UNKNOWN**.

The results were computed on three platforms. No single environment produced
all of them.

## Platforms

| Id | Machine | OS | Python | Libraries | BLAS | Active |
|---|---|---|---|---|---|---|
| **MAC** | Apple silicon laptop (arm64). The code names its reference machine as a MacBook Air M4 (`orpsoc/runner.py`). Whether it is the same physical machine as VER below is UNKNOWN. | macOS (version UNKNOWN) | 3.13 (patch UNKNOWN) | Pinned in `requirements.txt` from 2026-08-07: NumPy 2.2.6, pandas 3.0.1, SciPy 1.17.1, scikit-learn 1.8.0, LightGBM 4.6.0. Installed versions were not logged. | Accelerate (inferred from the macOS NumPy 2.x wheels; `data_prep/make_benchmark_v2.py` documents the Accelerate dispatch) | through 2026-10 |
| **C6I** | Amazon EC2 c6i.8xlarge, 32 vCPU, Intel, in a previous AWS account | Ubuntu 24.04 (glibc 2.39) | 3.12.3 (3.12.3-1ubuntu0.15) | LightGBM 4.6.0 (import check, 2026-08-06). Intended pins as for MAC (the 2026-08-07 requirements commit says every pin was checked against "Ubuntu 24.04 x86_64, Python 3.12"). Installed versions were not logged. scikit-learn was upgraded to 1.9.0 by the OpenML fetch before the 2026-08-18 ladder batch. | not recorded (UNKNOWN) | 2026-08-06 to about 2026-09-01 (end date as stated by the author) |
| **C7A** | Amazon EC2 c7a.2xlarge, 8 vCPU, AMD EPYC 9R14 (`i-05c80ac43b80be33f`) | Ubuntu 24.04.4 | 3.12.3 | `environment-lock.txt` (pip freeze taken 2026-11-12): NumPy 2.2.6, pandas 3.0.1, SciPy 1.17.1, scikit-learn 1.8.0, LightGBM 4.6.0. The code records the scikit-learn 1.8.0 pin as verified for units written from 2026-09-08. | OpenBLAS 0.3.29 (`scipy-openblas`, DYNAMIC_ARCH) | from 2026-08-25 |
| **VER** | Verification only, not a source of any committed result: Apple MacBook Air Mac16,12, Apple M4 (4P + 6E), 16 GB | macOS 26.2 (25C56), Darwin 25.2.0, arm64 | 3.13.7 (python.org build) | NumPy 2.2.6, pandas 3.0.1, SciPy 1.17.1, scikit-learn 1.8.0, LightGBM 4.6.0, joblib 1.5.3, threadpoolctl 3.8.0 | Accelerate (NumPy and SciPy); OpenMP via libomp | 2026-11-12 |

## How platform was established

- **Dated machine bundles.** A byte-identical copy inside a bundle written on
  the machine. The c6i bundles are `orpsoc_results_20260807T081801Z.tar.gz`
  (its tar mtimes are exact) and `ec2_results/` (zip mtimes are shifted by a
  constant 12 h 30 min; use only their order). The c7a bundle is
  `orpsoc_out.tgz` (2026-08-26) plus the files left on the instance.
- **Instance lifetimes.** A unit written before 2026-08-25 cannot come from
  the c7a. A unit written after about 2026-09-01 cannot come from the c6i.
- **Platform fingerprint (reruns on VER).** For the same unit, Apple silicon
  and x86-64 give different AUCs in many arms. The all-features arm of the
  ind10 LogReg ladder, for example, differs by 3.8 × 10⁻⁴, while two x86
  machines (c6i and c7a) agree to 5 × 10⁻¹¹. So:
  - a VER rerun that **differs** shows the result was not computed on Apple silicon;
  - a VER rerun that is **bit-identical**, where the same quantity is known to
    differ on x86, shows Apple silicon;
  - a VER rerun that is bit-identical with no x86 contrast available is
    **inconclusive**: some computations (small p, synthetic data) give the
    same bits on both platforms.
- **Derived files.** A file computed only from other stored results inherits
  their platform. Each such file was rebuilt on VER and compared byte for byte.

## Inventory

"Rerun on VER" gives the result of re-running one or more units on the
verification Mac and comparing them with the committed values: **differs**
(not Apple silicon), **identical, x86 differs** (Apple silicon) or
**identical** (inconclusive on its own). Dates are UTC. Where only the date of
first commit to this repository is known (2026-10-09), the computation date
is UNKNOWN.

### Adaptation study (D = 50 synthetic and real markets)

| File | Platform | Date | Evidence | Rerun on VER |
|---|---|---|---|---|
| `step7_ablation_v2.json` | C6I | 2026-08-06/07 | Byte-identical to the member of the c6i tarball `orpsoc_results_20260807T081801Z` (tar mtime 2026-08-07 08:18Z); c6i log `10_step7_ablation`; units `step7/4e388f80e5ac` | not rerun (tests/test_equivalence.py passes) |
| `step8_statistical_summary.json` | C6I (derived from step7) | 2026-08-07 | Byte-identical to the c6i tarball member; c6i log `11_step8_results` | — |
| `step9_real_data.json` | C6I | 2026-08-07 | Byte-identical to the c6i tarball member; units `step9/c4e9cd394b46` written during c6i run `20260807T044327Z`; c6i log `12_step9_realdata` | — |
| `tier_c.json` | C6I (derived from step9 units) | 2026-08-07/08 | Copy in c6i `ec2_results`; rebuilt byte-identical on VER from `step9/c4e9cd394b46` | rebuilt identical |
| `tier_a.json` | C6I (derived; "analysis only, zero re-runs") | 2026-08-07/08 | `provenance` field `4e388f80e5ac` = the c6i step7 units; A4 runtimes are c6i timings; copy in c6i `ec2_results` | — |
| `crossseed_jaccard.json` | C6I (extracted from `step7_ablation_v2.json`) | extraction first committed 2026-10-09 | rebuilt byte-identical on VER | rebuilt identical |
| `apsoll_sweep.json` (the patience-and-rearm result, Section 3.1.3) | C6I, verified | 2026-08-07 | `apsoll_sweep_log.txt` with `workers=30`; units `apsoll_sweep/f122edc43c7c` in the c6i bundle | identical (both platforms give these bits) |
| `elite_frac_sweep.json` | C6I | 2026-08-07/08 | units `elite_frac/a98ad52b61dc` in the c6i bundle, written after the apsoll units | identical |
| `lag_factorial.json` | C6I | 2026-08-08 | units `lag_factorial/833ec4b5f541` in the c6i bundle | identical |
| `gated_memory.json` | C6I | 2026-08-08 | units `gated_memory/0f154d7a01dd` in the c6i bundle | identical |
| `pareto_knee.json` (knee rule, Section 3.2.1; Table S10) | C6I, probable (UNVERIFIED) | by 2026-08-08 | Copy in c6i `ec2_results`; the script's own run line is `ORPSOC_N_JOBS=30`; no log or units | not rerun |
| `detector_multidraw.json`, `detector_pca.json`, `detector_supervised.json`, `detector_supervised_real.json` | C6I, probable (UNVERIFIED) | by 2026-08-08 | Copies in c6i `ec2_results`; no logs or units | all four rebuilt byte-identical (both platforms give these bits) |
| `adapt_4markets.json` ("46 of 48", Section 3.1.3) | MAC | 2026-08-06 (14:06Z) | The copy in the c6i tarball keeps mtime 2026-08-06 14:06Z, before the c6i pipeline started (19:40Z); committed to `orpsoc_project` in `89b79f8` (16:20Z) before EC2 deployment. The c6i's own run of `ff_adaptation.py` (log `14_adaptation`) wrote to the script's default `/tmp/ff.json`, not to `results/` | not rerun |
| `phase3_mediank.json` | MAC | 2026-08-06 (14:14Z) | As above (tar mtime 14:14Z; `89b79f8`); no stage of the c6i pipeline runs it. No paper use found | not rerun |
| `compass_ceiling.json` (Section 3.1.3) | MAC | 2026-08-06 | Committed file = `orpsoc_project` `89b79f8` (before EC2). The c6i re-ran it on 2026-08-07 (log `15_compass`; 187,649 bytes against 187,640). The three v2_regime_switch blocks are bit-identical in the two versions, so the numbers the paper reports (Spearman–Brown ceiling +0.718, criterion +0.201, best alternative 55% of the ceiling) are the same in both. The three sector-ETF blocks differ in 309 numeric fields (test AUC by up to 1.4e-2, univariate scores by up to 2.9e-6). The c6i rebuilt `sector_etf.pkl` from cached raw prices; whether the difference comes from the platform or from a different raw-data copy is UNKNOWN | not rerun |
| `warm_gate.json` (detector-gated warm start; not used in the paper) | UNKNOWN (C6I or MAC) | UNKNOWN; first committed 2026-10-09 | The c6i queued a 30-seed run after the 2026-08-18 batch (`run_warmgate_after.sh`, 30 workers), but that output was not retained, and the file is not in any bundle | seed 0 identical (both platforms give these bits) |

### Dimensionality study: synthetic sweep

| File | Platform | Date | Evidence | Rerun on VER |
|---|---|---|---|---|
| `tier_d_curve.json` (Table 7, 8 seeds) | C6I | 2026-08-12/13 | Byte-identical to the c6i `ec2_results` copy; units `tier_d_curve/8d2b587bfb5c` in the c6i bundle; c6i log `20260811T155629Z_07_hd_sweep` | — |
| `tier_d_fullnovelty_full_hybrid-sparse_orpsoc.json` (p = 2000, ff100) | C6I | 2026-08-13/14 | Byte-identical to the c6i `ec2_results` copy; units `6e48c607d548` | — |
| `tier_d_fullnovelty_all-lasso-mrmr-rfe-sparse_orpsoc-sparse_pso.json` (20 seeds, LogReg and LightGBM) | C7A | 2026-09-10 | Same size as the copy on the c7a (1,147,160 bytes, 2026-09-10 13:25); c7a logs `02_synthetic_20260910*`; its 320 units on the c7a are dated 2026-09-10 | all-features arm identical (inconclusive) |
| `ar1_control.json` | C7A (extracted from the file above) | extraction first committed 2026-10-09 | rebuilt byte-identical on VER | rebuilt identical |
| `tier_d_ksweep.json` (p = 2000; not used in the paper) | UNKNOWN (consistent with MAC) | UNKNOWN; first committed 2026-10-09 | not in any EC2 bundle or on the c7a | all-features baseline identical (inconclusive) |
| `tier_d_fullnovelty_all-cgpso-sparse_pso.json` (1 record, LinearSVM `hd_ar1_p200` s0) | C6I (by date only) | 2026-08-18 | its unit in `29a57cced36f` is dated 2026-08-18, before the c7a existed | not rerun |
| Table 8a, panel (a): synthetic k-NN (exists only as `tables/knn_irrelevance.tex`) | MAC | 2026-09-08 (per the archive notes) | No units survive. Attributed to Apple silicon because its p = 50 column reproduces all 10 cells exactly on VER | p = 50 reproduced; p = 200, 500, 1000 re-run in progress |

### Dimensionality study: real ladder

`tier_d_ladder_merged.json` (586 records) mixes platforms by record group and,
within the main ladder, by arm:

| Records | Non-RFE arms | RFE arm | Date of units | Evidence |
|---|---|---|---|---|
| ind10, ind12, ind17, ind30 × LogReg, LightGBM, k-NN (240) | C6I | MAC | 2026-08-18 to 08-20 | Units predate the c7a. On VER the all-features arm differs (ind10 LogReg 3.8e-4, ind12 LightGBM 3.7e-2, ind17 k-NN 3.7e-4), so x86. The RFE arm regenerates bit for bit on VER for all 240 records (`rfe_repin.py` and the 49-unit budget rerun), so Apple silicon, under scikit-learn 1.8.0 |
| LinearSVM ind10, ind12, ind17 (60) | C6I, every arm including RFE | (same unit) | 2026-08-18 to 08-20 | On VER the all-features arm differs (ind10 7.3e-4). RFE inferred to be scikit-learn **1.9.0**: the units were written by the c6i batch's stage 3 (one process, 400 units, all five estimators), after the stage-1 OpenML fetch that the supplement (Section S3) identifies as the upgrade, and the LogReg and LightGBM RFE from that process carry the version-change signature (64-cell check below). No unit records a library version, so this is an inference. **Never re-pinned**: the merged LinearSVM RFE equals the original multi-arm unit in 80 of 80 records (ind30 included). This contradicts S3's statement that RFE was re-run under 1.8.0 "for every estimator at every width". These are the three non-significant or negative RFE cells behind "37 of 40" (Table 4, Section 3.2.1, Table S3 caption): ind10 −0.0233\*, ind12 −0.0082\*, ind17 +0.0059 |
| LinearSVM ind30 (20) | C7A | (same unit) | 2026-08-26 | c7a log `05_linearsvm_20260825T211528`; on VER the all-features arm differs (3.9e-3). scikit-learn version UNKNOWN (before the 2026-09-08 pin check) |
| RandomForest ind10 (20), ind12 (6) | C6I | (same unit) | 2026-08-19/20 | On VER the all-features arm differs (ind10 3.2e-3) |
| ind5 × LogReg, LightGBM, k-NN (60) | C7A, every arm including RFE | (same unit) | LogReg 2026-09-07; LightGBM and k-NN 2026-09-08 00:00–06:00Z | Units postdate the c6i and are on the c7a. On VER the all-features arm differs for k-NN, so x86; it is identical (inconclusive) for LogReg and LightGBM. RFE scikit-learn version **UNKNOWN**: the LogReg units predate the 2026-09-08 pin date that `build_ladder.py` uses, the others sit at that boundary, and RFE was **never re-pinned** (merged RFE equals the original unit in 60 of 60 records; `rfe_repin.py` covered ind10–ind30 only) |
| ind38, ind48, ff49ind × LogReg, LightGBM, k-NN (180) | C7A, every arm including RFE | (same unit) | 2026-09-08 to 09-17 | Units postdate the c6i and are on the c7a. On VER the all-features arm differs for ind38 LogReg and LightGBM, ind48 LogReg, k-NN and LightGBM, and ff49ind LogReg, so x86; it is identical (inconclusive) for ff49ind k-NN and LightGBM. scikit-learn 1.8.0 according to the code's pin record (verified from 2026-09-08); no unit records a version |
| ind5 LightGBM seeds 12–19 (8, inside the ind5 row) | C7A by date and grouping (no unit survives) | (same unit) | 2026-09-08 | Re-run on VER in progress at the time of writing |

Other ladder files:

| File | Platform | Date | Evidence |
|---|---|---|---|
| `tier_d_fullnovelty_rfe.json` (160 RFE records, LogReg and k-NN) | MAC | after 2026-09-08 | All 160 equal the merged ladder's RFE arm, which regenerates bit for bit on VER |
| `tier_d_fullnovelty.json` (ind48 and ff49ind LightGBM, 20 seeds) | C7A | 2026-09-16/17 | Same size as the copy on the c7a (257,047 bytes, 2026-09-17 17:12); c7a log `01_ladder_lgbm_20260916T134248`; on VER ind48's all-features arm differs (2.2e-2) |
| `tier_d_fullnovelty_8seed_20260814.json` | C6I | 2026-08-13/14 | Byte-identical to the c6i `ec2_results` `tier_d_fullnovelty.json`; on VER the all-features arm differs (3.8e-4). Not used in the paper: only `build_ladder.py` reads it, as a fallback, and the merged ladder takes 0 of its 586 records from it |
| `tier_d_LinearSVM.json` | C6I (ind10–ind17) and C7A (ind30) | 2026-08-18 to 08-26 | Byte-identical to the c7a bundle `orpsoc_out.tgz` (2026-08-26); dates as in the ladder rows above |
| `arm_pairwise.json` | derived from the merged ladder | first committed 2026-10-09 | rebuilt byte-identical on VER |
| `collinearity.json` | derived (data statistics and the ladder) | first committed 2026-10-09 | rebuilt byte-identical on VER |
| `nonfinancial_check.json` (madelon, golub) | C7A | 2026-08-25 | Byte-identical to the c7a bundle `orpsoc_out.tgz`; units `nonfinancial/64e74faa1d9e` dated 2026-08-25; c7a log `03_nonfinancial_20260825T150719`. VER: madelon LogReg all-features identical (inconclusive) |
| `golub_reconcile.json` (Section 3.7.1, Table S9) | UNKNOWN (consistent with MAC) | UNKNOWN; first committed 2026-10-09 | Not in any EC2 bundle or on the c7a. VER: all 8 stage-A LogReg rows bit-identical (inconclusive) |
| θ sweep (`experiments/dimensionality/theta_sweep/results/theta_sweep_per_seed.csv`, Table S13) | C7A | 2026-11 (run recorded in its README as 2026-11-12) | package README; raw outputs and units come from the c7a |

### Penalty study (Table 9, Fig. 4, Table 12 fitness column, Fig. S10, Table S6)

| File | Platform | Date | Evidence |
|---|---|---|---|
| `ridge_sweep.json` | MAC | UNKNOWN; first committed 2026-10-09 | Its C = 1 all-features AUCs (the same estimator as the ladder's LogReg) differ from the x86 ladder's and equal VER's bit for bit (ind17 and ind30, seeds 0–2). |
| `swarm_fitness.json` | MAC | UNKNOWN; first committed 2026-10-09 | Sparse-PSO AUCs differ from the x86 ladder's and equal VER's bit for bit (ind17 s0: VER rerun and earlier VER ladder rerun) |
| `nested_cv_c.json` | MAC | UNKNOWN; first committed 2026-10-09 | Its fixed-C all-features AUCs equal the Apple-silicon ridge values and differ from x86 (ind17, ind30, seeds 0–1); VER rerun of ind17 s0 bit-identical in every field |
| `nested_cv_stored.json` | MAC | UNKNOWN; first committed 2026-10-09 | As `nested_cv_c.json`; VER rerun of ind17 s0 bit-identical. It re-scores the subsets stored in `swarm_fitness.json` (MAC) |

## Library-version checks

- **The 64-cell check (Supplementary Section S3) is exact.** The 8-seed run
  (`6e48c607d548`, 2026-08-13/14) and the 20-seed run (`29a57cced36f`,
  2026-08-18 to 08-20), both on the c6i, share 64 cells: ind10, ind12, ind17,
  ind30 × LogReg, LightGBM × seeds 0–7. In all 64, every arm (all features,
  sparse PSO, sparse OrPSOC, Full Hybrid, mRMR, LASSO) is bit-identical except
  RFE, which differs in all 64.
- **RFE never re-pinned under 1.8.0:** LinearSVM at every width (ind10–ind17
  inferred 1.9.0; ind30 version UNKNOWN), and ind5 for LogReg, LightGBM and k-NN
  (version UNKNOWN).
- **Input data changed between the 8-seed and 20-seed runs for ind5 and
  ff49ind.** Where the two runs share cells (seeds 0–7, LogReg and LightGBM),
  ind38 and ind48 are bit-identical in every arm across the c6i and the c7a,
  but ind5 and ff49ind differ in every arm, with different subset sizes. That is
  a data difference, not a platform one. It affects only the 8-seed file, which
  the paper does not use.

## What this means for the computing-environment paragraph

- The results come from three environments: Apple-silicon macOS (Python
  3.13), and two x86-64 Linux EC2 instances (Ubuntu 24.04, Python 3.12.3,
  Intel c6i and AMD c7a). The library pins (NumPy 2.2.6, SciPy 1.17.1, pandas
  3.0.1, scikit-learn 1.8.0, LightGBM 4.6.0) are common to all three, apart
  from the c6i ladder batch of 2026-08-18 to 08-20, which ran after an
  upgrade to scikit-learn 1.9.0. Its RFE was re-run under 1.8.0 for LogReg,
  LightGBM and k-NN, but not for LinearSVM; ind5's RFE version is UNKNOWN.
- Apple silicon produced: the RFE arm of the main ladder (ind10–ind30, LogReg,
  LightGBM, k-NN), the whole penalty study, `compass_ceiling`,
  `adapt_4markets`, `phase3_mediank`, and the synthetic k-NN panel of Table 8a.
- The c6i produced: the D = 50 adaptation study (step7 to step9 and its
  sweeps, including the APSOLL patience-and-rearm sweep), the 8-seed synthetic
  sweep and 8-seed ladder, and the 2026-08 ladder widths (ind10 to ind30,
  non-RFE arms; LinearSVM ind10–ind17; RandomForest).
- The c7a produced: madelon and golub, the 20-seed synthetic sweep, ind30
  LinearSVM, the 2026-09 ladder widths (ind5, ind38, ind48, ff49ind), and the
  θ sweep.
- Platform UNKNOWN: `golub_reconcile.json` (used in Section 3.7.1 and Table
  S9; consistent with Apple silicon), and `warm_gate.json` and
  `tier_d_ksweep.json` (neither used in the paper). Probable but unverified
  c6i: `pareto_knee.json` (Section 3.2.1, Table S10) and the four detector
  files (Section 3.1.2, Fig. S2, Table S14).
- For the adaptation study and the detectors, platform does not change the
  numbers. Every rerun on Apple silicon (apsoll, elite fraction, lag
  factorial, gated memory, all four detector files, and seed 0 of the warm
  gate) gave the same bits as the stored results, and apsoll is known to come
  from the c6i. For the ladder, the penalty study and RFE, platform does
  change the bits: each arm reproduces exactly only on the platform it was
  computed on.
