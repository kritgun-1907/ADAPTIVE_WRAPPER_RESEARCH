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
