# Datasets

Datasets are built locally from public sources and are not committed. Every
file is a pickle of `{"X": DataFrame, "y": Series, "base": ...}`. Run the
builders from the repository root, in this order, since later ones read files
written by earlier ones:

| Step | Command | Writes | Source |
|---|---|---|---|
| 1 | `python data_prep/make_benchmark_v1.py` | `white_noise`, `ar1`, `drift`, `regime_switch` | synthetic |
| 2 | `python data_prep/make_level0_null.py` | `null` (L0, needs step 1) | synthetic |
| 3 | `python data_prep/make_benchmark_v2.py` | `v2_{null,white_noise,ar1,drift,regime_switch}` — the benchmark used in the paper | synthetic |
| 4 | `python data_prep/make_highdim_sweep.py` | `hd_<level>_p<p>` (needs step 3) | synthetic |
| 5 | `python pipeline/step9_real_data.py` with `PREP_ONLY = True` set near the top | `sector_etf`, `fama_french`, raw caches | Yahoo Finance, Ken French Data Library |
| 6 | `python data_prep/build_industry_ladder.py` | `ind5` … `ind48` (needs step 5) | Ken French Data Library |
| 7 | `python data_prep/fetch_ff_wide_raw.py` then `python data_prep/build_highdim_real.py` | `ff49ind`, `ff100` | Ken French Data Library |
| 8 | `python data_prep/build_extra_markets.py` | `bonds`, `commodities` | Yahoo Finance |
| 9 | `python data_prep/build_nonfinancial.py` | `madelon`, `gisette`, `golub` | OpenML / UCI |

Then check everything with `python data_prep/verify_data.py`.

Raw downloads are cached as `data/raw_*.pkl`. Keep them: Yahoo Finance and the
Ken French library revise past values, so a later download can differ slightly
from the series the stored results were produced on.

Step 5 needs a one-line edit: set `PREP_ONLY = True` near the top of
`pipeline/step9_real_data.py` (line 99), run it, then restore the file exactly
(`git checkout pipeline/step9_real_data.py`). The script's own source is part of
its provenance hash, so while the edit is in place its checkpoints go to a
different generation.

## Sources and terms

No raw or derived dataset is redistributed in this repository or in the Zenodo
deposit. `results/` and the archived checkpoints hold only model outputs
(per-fold AUCs, subset sizes, selected-feature indices), not data values.

| Dataset(s) | Source | Exact files / identifiers | Terms |
|---|---|---|---|
| `fama_french` | Kenneth R. French Data Library, `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/` | `F-F_Research_Data_5_Factors_2x3_daily_CSV.zip`, from 2000-01-01 | No licence or terms of use are stated. The library carries a copyright notice for Eugene F. Fama and Kenneth R. French. Cite the library when using the data. |
| `ind5` … `ind48` | same | `{5,10,12,17,30,38,48}_Industry_Portfolios_daily_CSV.zip`, from 2000-01-01 | as above |
| `ff49ind`, `ff100` | same | `49_Industry_Portfolios_daily_CSV.zip`, `100_Portfolios_10x10_Daily_CSV.zip`, from 2000-01-01 | as above |
| `sector_etf` | Yahoo Finance via `yfinance` (adjusted close, `auto_adjust=True`) | XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, from 2000-01-01 | Yahoo's Terms of Service forbid reproducing or distributing Yahoo content without written permission. That is why prices are not shipped here; each user downloads them. |
| `bonds` | same | TLT, IEF, SHY, LQD, AGG, TIP, from 2003-01-01 | as above |
| `commodities` | same | GLD, SLV, USO, DBC, DBA, UNG, from 2007-01-01 | as above |
| `madelon` | OpenML, `fetch_openml("madelon", version=1)` | OpenML data id 1485 (originally UCI; NIPS 2003 feature selection challenge) | OpenML licence field: "Public". Cite Guyon, Gunn, Ben-Hur & Dror (2004), *Result analysis of the NIPS 2003 feature selection challenge*. |
| `golub` | OpenML, `fetch_openml("leukemia", version=1)` | OpenML data id 1104 | OpenML licence field: "Public". Cite Golub et al. (1999), *Science* 286:531–537. |
| `gisette` | OpenML, `fetch_openml("gisette", version=1)` | UNVERIFIED: OpenML currently lists `gisette` only as version 2 (id 41026), so this call may fail | Not used in the paper. |

The sources' terms were last checked on 2026-11-12.

When the stored results were downloaded: UNVERIFIED. The dates of the original
`data/raw_*.pkl` caches on the EC2 instance were not recorded here.
`yfinance` 1.8.0 (see `requirements.txt`) is the version that resolved when this
release was prepared. It is not necessarily the version used for the original
download.
