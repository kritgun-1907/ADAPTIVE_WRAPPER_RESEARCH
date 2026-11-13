# Where each committed result was computed

This inventory records, for every file in `results/` (and for the one table
whose underlying records survive only as LaTeX), the platform it was computed
on, the environment where one is on record, the date, and the evidence. It was
compiled on 2026-11-12 from the stored results, the surviving machine bundles
and checkpoints, and reruns on an Apple M4, and updated on 2026-11-13 after
the RFE arm, ind5 LogReg and LightGBM, and the penalty study were re-run on
the c7a with the canonical data copy. Anything that could not be determined is
marked **UNKNOWN**.

The results were computed on three platforms and from two downloads of the
Ken French industry files (the canonical copy of 2026-08-26 and the laptop
copy of 2026-09-06; `data/README.md`). No single environment produced all of
them.

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
| Table 8a, panel (a): synthetic k-NN (`tables/knn_irrelevance.tex`) | MAC (UNVERIFIED) | 2026-09-08 (per the archive notes) | The original units survive in no copy. All four widths reproduce all 40 cells exactly on VER (2026-11-12); those VER units are `knn_synthetic_m4_20261112.tgz`. Whether x86 gives the same bits was not tested, so the reproduction does not by itself establish the platform | all 40 cells identical |

### Dimensionality study: real ladder

`tier_d_ladder_merged.json` (586 records) mixes platforms by record group and,
within the main ladder, by arm. "Canonical copy" is the Ken French download of
2026-08-26; "laptop copy" the one of 2026-09-06 (`data/README.md`).

| Records | Non-RFE arms | RFE arm | Evidence |
|---|---|---|---|
| ind10, ind12, ind17, ind30 × LogReg, LightGBM, k-NN (240) | C6I, 2026-08-18 to 08-20 | C7A, 2026-11-12, scikit-learn 1.8.0, canonical copy, stored sparse-PSO budgets (`rfe_repin.py`) | Non-RFE units predate the c7a; on VER the all-features arm differs (ind10 LogReg 3.8e-4, ind12 LightGBM 3.7e-2, ind17 k-NN 3.7e-4), so x86. The RFE units carry a `meta` block. The RFE they replace was computed on Apple silicon with the laptop copy: all 240 regenerate bit for bit on VER with that copy (`verification_reruns_rfe.tgz`) |
| LinearSVM ind10, ind12, ind17 (60) | C6I, 2026-08-18 to 08-20 | C7A, 2026-11-12, as above | On VER the all-features arm differs (ind10 7.3e-4). The RFE they replace came from the c6i batch, inferred to be scikit-learn 1.9.0 (stage 3 of that batch ran after the stage-1 OpenML fetch that Supplementary Section S3 identifies as the upgrade; no unit records a version) |
| LinearSVM ind30 (20) | C7A, 2026-08-26 | C7A; the 1.8.0 re-pin is bit-identical to it in 20 of 20 records | c7a log `05_linearsvm_20260825T211528`; on VER the all-features arm differs (3.9e-3) |
| RandomForest ind10 (20), ind12 (6) | C6I, 2026-08-19/20 | C6I (same unit), inferred 1.9.0, not re-pinned | On VER the all-features arm differs (ind10 3.2e-3). Not used in the paper |
| ind5 k-NN (20) | C7A by store location and date (UNVERIFIED), 2026-09-08, canonical copy | C7A; the 1.8.0 re-pin is bit-identical in 20 of 20 records | The all-features arm (seed 0) reproduces exactly on VER from the canonical copy, so these records used it. At ind5 that arm gives the same bits on both platforms, so it does not identify the platform |
| ind5 LogReg, LightGBM (40) | C7A, 2026-11-12/13, canonical copy, every arm in one unit | (same unit) | LightGBM seeds 12–19 from 2026-11-12, the rest 2026-11-12/13; each unit carries a `meta` block. The values they replace were computed from the laptop copy, every arm: on VER their all-features arm (seed 0) reproduces exactly from that copy and not from the canonical one, and their RFE reproduces bit for bit with it (LogReg 20 of 20, LightGBM seeds 0–11 12 of 12). Their platform is UNKNOWN: at ind5 the all-features arm gives the same bits on VER and the c7a, ind5 RFE was never compared across platforms on one data copy, and the units are in the c7a's own store (`checkpoints_box_only`), dated 2026-09-07/08 |
| ind38, ind48, ff49ind × LogReg, LightGBM, k-NN (180) | C7A, 2026-09-08 to 09-17, every arm including RFE | (same unit) | Units postdate the c6i and are on the c7a. On VER the all-features arm differs for ind38 LogReg and LightGBM, ind48 LogReg, k-NN and LightGBM, and ff49ind LogReg, so x86; it is identical (inconclusive) for ff49ind k-NN and LightGBM. scikit-learn 1.8.0 according to the code's pin record (verified from 2026-09-08); the c7a re-pin of ind38 LogReg seeds 0–1 is bit-identical |

The units of every C7A 2026-11 entry are in `ladder_penalty_c7a_box_20261113.tgz`.
The RFE values replaced in the first two rows are kept in
`tier_d_ladder_rfe_superseded.json` (300 records, both versions per record).

Other ladder files:

| File | Platform | Date | Evidence |
|---|---|---|---|
| `tier_d_ladder_rfe_superseded.json` | as the rows above | 2026-11-12 | Written when the RFE arm was re-pinned. No script reads it |
| `tier_d_fullnovelty_rfe.json` (160 RFE records, LogReg and k-NN) | MAC, laptop copy | after 2026-09-08 | All 160 equal `rfe_superseded` in the file above, i.e. the RFE values before the re-pin. No script reads it |
| `tier_d_fullnovelty.json` (ind48 and ff49ind LightGBM, 20 seeds) | C7A | 2026-09-16/17 | Same size as the copy on the c7a (257,047 bytes, 2026-09-17 17:12); c7a log `01_ladder_lgbm_20260916T134248`; on VER ind48's all-features arm differs (2.2e-2) |
| `tier_d_fullnovelty_8seed_20260814.json` | C6I | 2026-08-13/14 | Byte-identical to the c6i `ec2_results` `tier_d_fullnovelty.json`; on VER the all-features arm differs (3.8e-4). Not used in the paper: only `build_ladder.py` reads it, as a fallback, and the merged ladder takes 0 of its 586 records from it |
| `tier_d_LinearSVM.json` (SHA-256 `0884c626b696de77a53cea6c4474f2edced46d33604e1574e9c5ddcab78debcf`) | C6I (ind10–ind17) and C7A (ind30) | 2026-08-18 to 08-26 | Byte-identical to the c7a bundle `orpsoc_out.tgz` (2026-08-26). A corroboration file: no script reads it, and all 80 records equal the merged ladder's LinearSVM records in every arm, RFE included. Its ind10–ind17 RFE therefore already held the values that the 2026-11-12 re-pin produced, while the ladder held different ones |
| `arm_pairwise.json` | derived from the merged ladder | first committed 2026-10-09 | rebuilt byte-identical from the current ladder |
| `collinearity.json` | derived (data statistics and the ladder's LogReg LASSO gain) | first committed 2026-10-09 | rebuilt byte-identical on VER before the ind5 re-run. Its data statistics come from the laptop copy: they reproduce exactly from it (ind5, ind10, ind12, ind17, ind38) and differ on the canonical copy by up to 5e-3, which does not change the table. Not rebuilt since (it needs the datasets): from the current ladder the ind5 gain is 0.0249406 against the stored 0.0249355, the same in `collinearity.tex` (+0.0249) |
| `nonfinancial_check.json` (madelon, golub) | C7A | 2026-08-25 | Byte-identical to the c7a bundle `orpsoc_out.tgz`; units `nonfinancial/64e74faa1d9e` dated 2026-08-25; c7a log `03_nonfinancial_20260825T150719`. The c7a 1.8.0 re-pin of its RFE and its mRMR and LASSO anchors is bit-identical in 80 of 80 records |
| `golub_reconcile.json` (Section 3.7.1, Table S9) | UNKNOWN (consistent with MAC) | UNKNOWN; first committed 2026-10-09 | Not in any EC2 bundle or on the c7a. VER: all 8 stage-A LogReg rows bit-identical (inconclusive) |
| θ sweep (`experiments/dimensionality/theta_sweep/results/theta_sweep_per_seed.csv`, Table S13) | C7A | 2026-11 (run recorded in its README as 2026-11-12) | package README; raw outputs and units come from the c7a |

### Penalty study (Table 9, Fig. 4, Table 12 fitness column, Fig. S10, Table S6)

| File | Platform | Date | Evidence |
|---|---|---|---|
| `swarm_fitness.json` | C7A, canonical copy | 2026-11-13 (ind30 seed 0 on 2026-11-12) | Units `swarm_fitness/85802cd6be89` in `ladder_penalty_c7a_box_20261113.tgz`. Its swarm-arm ΔAUC now reproduces the ladder's exactly (`fitness_tables.py` agreement check 0.0000). The version it replaced was MAC, laptop copy (Sparse-PSO AUCs equal VER's bit for bit, ind17 s0) |
| `nested_cv_stored.json` | C7A, canonical copy | 2026-11-13 | Re-scores the subsets stored in `swarm_fitness.json` and recomputes mRMR, LASSO and RFE from the stored budgets. Its C = 1 all-features AUCs (0.7493, 0.6917) now equal the ladder's. The version it replaced was MAC, laptop copy (VER rerun of ind17 s0 bit-identical) |
| `ridge_sweep.json` (Table S6) | MAC, laptop copy | UNKNOWN; first committed 2026-10-09 | Its C = 1 all-features AUCs (the same estimator as the ladder's LogReg) differ from the x86 ladder's and equal VER's bit for bit (ind17 and ind30, seeds 0–2). Not re-run on the canonical copy |
| `nested_cv_c.json` (re-swarm nested CV, not in the paper) | MAC, laptop copy | UNKNOWN; first committed 2026-10-09 | Its fixed-C all-features AUCs equal the Apple-silicon ridge values and differ from x86 (ind17, ind30, seeds 0–1); VER rerun of ind17 s0 bit-identical in every field. Not re-run. `analysis/nested_cv_tables.py` compares its C = 1 column with the ladder and now exits 1 on RFE (−0.0250 against −0.0411 at ind17, +0.0534 against +0.0717 at ind30) |

## Library-version checks

- **The 64-cell check (Supplementary Section S3) is exact.** The 8-seed run
  (`6e48c607d548`, 2026-08-13/14) and the 20-seed run (`29a57cced36f`,
  2026-08-18 to 08-20), both on the c6i, share 64 cells: ind10, ind12, ind17,
  ind30 × LogReg, LightGBM × seeds 0–7. In all 64, every arm (all features,
  sparse PSO, sparse OrPSOC, Full Hybrid, mRMR, LASSO) is bit-identical except
  RFE, which differs in all 64.
- **RFE now under 1.8.0 everywhere the paper uses it.** Every RFE value of the
  c6i batch that the paper uses has been replaced by the c7a 1.8.0 re-pin.
  Only RandomForest (ind10, ind12; not in the paper) keeps the batch's RFE.
  The c7a was on-pin before 2026-09-08 as well: its LinearSVM ind30
  (2026-08-26), ind5 k-NN (2026-09-08) and nonfinancial (2026-08-25) RFE
  re-pin bit-identically.
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
  upgrade to scikit-learn 1.9.0. Only its RFE is affected, and every RFE
  value of that batch the paper uses was re-run under 1.8.0.
- They also come from two downloads of the Ken French industry files. The
  ladder at ind5 to ind48 and the penalty study are on the canonical copy
  (2026-08-26), except Table S6 (ridge sweep), which is on the laptop copy.
  The c6i's 2026-08 non-RFE arms predate that download; their all-features
  baselines reproduce from it to within the platform residual (about 1e-5 for
  LogReg). For ff49ind's raw data see `data/README.md`.
- Apple silicon produced: the penalty study's ridge sweep (Table S6) and
  `nested_cv_c` (not in the paper), `compass_ceiling`, `adapt_4markets`,
  `phase3_mediank`, and probably the synthetic k-NN panel of Table 8a
  (UNVERIFIED; its units are lost and were recomputed on VER).
- The c6i produced: the D = 50 adaptation study (step7 to step9 and its
  sweeps, including the APSOLL patience-and-rearm sweep), the 8-seed synthetic
  sweep and 8-seed ladder, and the 2026-08 ladder widths (ind10 to ind30,
  non-RFE arms; RandomForest).
- The c7a produced: madelon and golub, the 20-seed synthetic sweep, ind30
  LinearSVM, ind5 k-NN, the 2026-09 ladder widths (ind38, ind48, ff49ind), the
  θ sweep, and on 2026-11-12/13 the ladder's RFE arm, every arm of ind5 LogReg
  and LightGBM, `swarm_fitness` and `nested_cv_stored`.
- Platform UNKNOWN: `golub_reconcile.json` (used in Section 3.7.1 and Table
  S9; consistent with Apple silicon), and `warm_gate.json` and
  `tier_d_ksweep.json` (neither used in the paper). Probable but unverified
  c6i: `pareto_knee.json` (Section 3.2.1, Table S10) and the four detector
  files (Section 3.1.2, Fig. S2, Table S14).
- For the adaptation study and the detectors, platform does not change the
  numbers. Every rerun on Apple silicon (apsoll, elite fraction, lag
  factorial, gated memory, all four detector files, and seed 0 of the warm
  gate) gave the same bits as the stored results, and apsoll is known to come
  from the c6i. For the ladder and the penalty study, platform changes the
  bits slightly (all-features cell means by at most about 4e-4 on the same
  data copy). RFE is not portable at all: its ΔAUC moves by up to 0.056
  between platforms and by up to 0.068 between data copies (README, "Known
  reproducibility limitation: RFE").
