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
| `gisette` | OpenML, `fetch_openml("gisette", version=1)` | The call resolved when the cache was built (2026-08-25), but OpenML now lists `gisette` only as version 2 (id 41026), so a fresh fetch may fail | Not used in the paper. |

The sources' terms were last checked on 2026-11-12.

### When the data behind the stored results were downloaded

| Raw cache | Downloaded | Evidence |
|---|---|---|
| `raw_bonds_prices.pkl`, `raw_commodities_prices.pkl` (Yahoo) | 2026-08-06 | Prices end on 2026-08-06, and the caches already existed when the c6i.8xlarge run log of 2026-08-06 19:40 UTC started |
| `raw_sector_prices.pkl` (Yahoo) | 2026-06-14 | Prices end on Friday 2026-06-12; cache file time 2026-06-14 |
| `raw_fama_french.pkl` (five factors) | 2026-06-14 (file time) | Factors end on 2026-04-30 (the library's publication lag); same file time on every copy |
| `raw_ind{5,10,12,17,30,38,48}.pkl` | 2026-08-26 | Cache file times on the c7a.2xlarge |
| `raw_madelon.pkl`, `raw_golub.pkl` (OpenML) | 2026-08-25 | Cache file times on the c7a.2xlarge |
| `raw_ff49ind.pkl`, `raw_ff100.pkl` | 2026-08-10 (file time, laptop copy) | Not on the c7a.2xlarge, which holds only the built `ff49ind.pkl` (2026-09-08) and `ff100.pkl` (2026-09-09); those builds are byte-identical to the laptop's |

### Checksums, and the sources revise history

[`CHECKSUMS.sha256`](CHECKSUMS.sha256) gives the SHA-256, size and file date
of every raw cache behind the stored results. The canonical copy is the one on
the c7a.2xlarge. Five raw caches exist only in the laptop copy (Yahoo bond,
commodity and sector-ETF prices, and the 49-industry and 100-portfolio files),
and are listed separately. Check a rebuilt cache with
`cd data && sha256sum -c CHECKSUMS.sha256` (`shasum -a 256 -c` on macOS).

**Both sources revise past values.** The Ken French library and Yahoo Finance
both change historical data after the fact, so a fresh download is not the
series the results were computed on. This happened within the project. The
industry files (`{5,10,12,17,30,38,48}_Industry_Portfolios`) were downloaded
twice: on 2026-08-26 (the canonical copy) and again on 2026-09-06 into the
laptop copy. The two differ in about 1% of cells, spread across 2000–2026,
with identical dates and labels. The second download's hashes are listed in
`CHECKSUMS.sha256` as superseded, so results computed from it can be
identified. If a rebuilt cache fails the checksum, the results will not
reproduce bit for bit from it.

Dates taken from file times are upper bounds, because a copy can postdate the
download. `yfinance` 1.8.0 (see
`requirements.txt`) is the version that resolved when this release was
prepared; the version used for the original Yahoo downloads was not recorded.
