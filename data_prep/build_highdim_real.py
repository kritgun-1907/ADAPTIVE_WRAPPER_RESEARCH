"""
Tier D / D2.a — REAL HIGH-DIMENSIONAL DATASETS.

The synthetic sweep (make_highdim_sweep.py) can locate a crossover but it is our
own generator. This supplies the real-data counterpart at comparable p, so the
dimensionality claim does not rest on synthetic data alone.

    ff49ind    49 industry portfolios  x 10 features + market  =  491 features
    ff100      95 size/BM portfolios   x 10 features + market  =  951 features

Both come from the SAME Dartmouth source already used for the 50-feature
`fama_french` dataset, over the SAME 2000+ window, with the SAME 10 rolling
transforms and the SAME volatility-regime target. That is deliberate: against
the existing `fama_french` result these differ in dimensionality and in
essentially nothing else, so a change in outcome is attributable to p rather
than to asset class, period, or target definition. The documented
2001 / 2008 / 2020 breaks therefore apply unchanged.

Why 95 and not 100: five of the 10x10 size/BM portfolios have gaps in the modern
window (thinly populated corner buckets). They are dropped rather than
forward-filled, because a forward-filled return is a fabricated observation and
would flow straight into the volatility features. See fetch_ff_wide_raw.py.

Feature engineering is copied from step9_real_data.build_fama_french_dataset by
EXECUTING that file rather than re-typing it -- the same trick
build_extra_markets.py uses. A re-implementation that silently drifted would
break the comparison this whole experiment depends on.

Run:  python data_prep/build_highdim_real.py
      (expects data/raw_ff100.pkl and data/raw_ff49ind.pkl; run
       data_prep/fetch_ff_wide_raw.py first)
Writes data/ff49ind.pkl, data/ff100.pkl
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOURCES = {
    "ff49ind": "data/raw_ff49ind.pkl",
    "ff100":   "data/raw_ff100.pkl",
}
HORIZON = 5          # same forecast horizon as sector_etf / fama_french


def build_wide_dataset(rets: pd.DataFrame, _align, horizon: int = HORIZON):
    """
    N return series x 10 rolling features, plus an equal-weight market return.

    The 10 transforms are exactly those in build_fama_french_dataset. The
    leading `mkt_ret_1d` column mirrors build_sector_etf_dataset: step7/step9's
    get_hmm_trigger keys off feat_names[0] and requires a RETURN series there,
    not a volatility or momentum column.
    """
    mkt = rets.mean(axis=1)                     # equal-weight market proxy

    feats = {"mkt_ret_1d": mkt}                 # <- col 0 (HMM driver)
    for c in rets.columns:
        s = rets[c]
        feats[f"{c}_lvl"]    = s
        feats[f"{c}_vol20"]  = s.rolling(20).std()
        feats[f"{c}_vol60"]  = s.rolling(60).std()
        feats[f"{c}_mom20"]  = s.rolling(20).sum()
        feats[f"{c}_mom60"]  = s.rolling(60).sum()
        feats[f"{c}_ma20"]   = s.rolling(20).mean()
        feats[f"{c}_ma60"]   = s.rolling(60).mean()
        feats[f"{c}_z20"]    = (s - s.rolling(20).mean()) / (s.rolling(20).std() + 1e-8)
        feats[f"{c}_skew60"] = s.rolling(60).skew()
        feats[f"{c}_kurt60"] = s.rolling(60).kurt()

    X = pd.DataFrame(feats, index=rets.index)

    # TARGET: "will realized vol exceed its 1-year median?" -- identical to both
    # existing real builders, so the label definition is not a moving part.
    rolling_vol = mkt.rolling(20).std()
    future_vol  = rolling_vol.shift(-horizon)
    median_vol  = rolling_vol.rolling(252).median()
    y    = (future_vol > median_vol).astype(int)
    base = mkt

    return _align(X, y, base, warmup=120, trim=horizon)


def main():
    # Pull _align out of step9 so warm-up trimming, NaN handling and date
    # preservation are bit-identical to the published datasets.
    src = open("pipeline/step9_real_data.py").read().split("#  ABLATION LOOP")[0]
    ns = {}
    exec(compile(src, "s9", "exec"), ns)
    _align = ns["_align"]

    print("=" * 68)
    print("  TIER D — REAL HIGH-DIMENSIONAL DATASETS")
    print("=" * 68, flush=True)

    for name, raw_path in SOURCES.items():
        if not os.path.exists(raw_path):
            print(f"  SKIP {name}: {raw_path} missing — run "
                  f"data_prep/fetch_ff_wide_raw.py first", flush=True)
            continue

        rets = pd.read_pickle(raw_path)
        X, y, base, dates = build_wide_dataset(rets, _align)

        out = f"data/{name}.pkl"
        with open(out, "wb") as fh:
            pickle.dump({"X": X, "y": y, "base": base, "dates": dates}, fh)

        print(f"  {name:<10} series={rets.shape[1]:<4} X={X.shape}  "
              f"y-mean={y.mean():.3f}  "
              f"{pd.Timestamp(dates.iloc[0]).date()}..{pd.Timestamp(dates.iloc[-1]).date()}"
              f"  -> {out}", flush=True)

        # Sanity: col 0 must be a return series, or the HMM keys off the wrong
        # thing and every trigger result becomes meaningless.
        assert X.columns[0] == "mkt_ret_1d", X.columns[0]
        finite = np.isfinite(X.to_numpy()).all()
        print(f"  {'':<10} col0={X.columns[0]}  all-finite={finite}  "
              f"class balance={y.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
