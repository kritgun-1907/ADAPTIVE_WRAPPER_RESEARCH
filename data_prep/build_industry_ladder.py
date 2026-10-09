"""
A REAL-DATA dimensionality curve: Ken French industry portfolios at increasing
granularity.

WHY THIS RATHER THAN SUBSAMPLING ff100
──────────────────────────────────────
The synthetic sweep varies p with signal count fixed at 5. The real-data
analogue needs the same property: the underlying prediction problem held
constant while the number of features describing it grows. Ken French publishes
the SAME US equity market partitioned at 5, 10, 12, 17, 30, 38, 48 and 49
industries -- the same economy at increasing resolution. Adding industries adds
descriptive columns, not new information about a different market.

The alternative (randomly subsampling columns out of ff100) gives strictly
nested feature sets but arbitrary subsets, and would need several draws per
width to avoid a lucky one. The industry ladder is published, deterministic and
citable.

    5 ind ->  51 feats     30 ind -> 301        49 ind -> 491  (already built)
   10 ind -> 101           38 ind -> 381        95 sz/BM -> 951 (already built)
   12 ind -> 121           48 ind -> 481

THE TARGET IS HELD FIXED, AND THIS IS THE WHOLE POINT
─────────────────────────────────────────────────────
Each industry set implies a slightly different equal-weight market. Deriving the
target from each set's own market would change the PREDICTION PROBLEM as p grows
and the curve would confound dimensionality with a moving label. So the target
is built ONCE from the canonical `Mkt-RF` series in data/raw_fama_french.pkl --
the same volatility-regime definition used by every other real dataset -- and
reused unchanged at every width. Only the feature matrix widens.

CAVEATS TO CARRY INTO THE WRITE-UP
  - No ground truth, so recall is unmeasurable here (unlike the synthetic arm).
  - The industry sets are NOT nested: 10-industry is not a coarsening of
    17-industry. "Only p varies" holds in the 'same market, finer partition'
    sense, which is weaker than the synthetic guarantee. Say so.

Run:  python data_prep/build_industry_ladder.py
Writes data/ind{N}.pkl for N in the ladder, plus data/raw_ind{N}.pkl caches.
"""
import io
import os
import re
import ssl
import sys
import zipfile
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
WIDTHS = [5, 10, 12, 17, 30, 38, 48]        # 49 already exists as ff49ind
START_DATE = "2000-01-01"
HORIZON = 5
MISSING = (-99.99, -999.0)


def _get(url, timeout=180):
    headers = {"User-Agent": "Mozilla/5.0 (research; orpsoc)"}
    try:
        import requests
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception:
        pass
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                  timeout=timeout, context=ctx).read()


def parse_first_block(txt):
    """First stacked block only: average value-weighted returns."""
    lines = txt.splitlines()
    is_data = lambda l: re.match(r"^\s*\d{8}\s*,", l) is not None
    first = next(i for i, l in enumerate(lines) if is_data(l))
    hdr = next(i for i in range(first - 1, -1, -1) if re.match(r"^\s*,", lines[i]))
    end = first
    while end < len(lines) and is_data(lines[end]):
        end += 1
    df = pd.read_csv(io.StringIO("\n".join([lines[hdr]] + lines[first:end])))
    df = df.rename(columns={df.columns[0]: "Date"})
    df.columns = ["Date"] + [c.strip() for c in df.columns[1:]]
    df["Date"] = pd.to_datetime(df["Date"].astype(int).astype(str), format="%Y%m%d")
    df = df.set_index("Date").astype(float)
    for m in MISSING:
        df = df.mask(np.isclose(df, m))
    return df / 100.0


def main():
    src = open("pipeline/step9_real_data.py").read().split("#  ABLATION LOOP")[0]
    ns = {}
    exec(compile(src, "s9", "exec"), ns)
    _align = ns["_align"]

    from data_prep.build_highdim_real import build_wide_dataset

    if not os.path.exists("data/raw_fama_french.pkl"):
        raise SystemExit("data/raw_fama_french.pkl missing — needed for the "
                         "fixed target")
    ff = pd.read_pickle("data/raw_fama_french.pkl")
    mkt = ff["Mkt-RF"]

    print("=" * 78)
    print("  REAL-DATA DIMENSIONALITY LADDER (Ken French industries)")
    print(f"  target: vol-regime on Mkt-RF, IDENTICAL at every width")
    print("=" * 78, flush=True)

    for n in WIDTHS:
        cache = f"data/raw_ind{n}.pkl"
        if os.path.exists(cache):
            rets = pd.read_pickle(cache)
            print(f"  [cache] ind{n} <- {cache}  {rets.shape}", flush=True)
        else:
            fname = f"{n}_Industry_Portfolios_daily_CSV.zip"
            print(f"  downloading {fname} ...", flush=True)
            zf = zipfile.ZipFile(io.BytesIO(_get(BASE + fname)))
            rets = parse_first_block(zf.read(zf.namelist()[0]).decode("latin-1"))
            rets = rets[rets.index >= pd.Timestamp(START_DATE)]
            before = rets.shape[1]
            rets = rets.dropna(axis=1, how="any")
            if rets.shape[1] < before:
                print(f"    dropped {before - rets.shape[1]} with gaps", flush=True)
            rets.to_pickle(cache)

        X, y, base, dates = build_wide_dataset(rets, _align, horizon=HORIZON)

        # Overwrite the target with the CANONICAL one so the label is identical
        # across widths; build_wide_dataset derives it from the set's own market.
        rv = mkt.rolling(20).std()
        y_fixed = (rv.shift(-HORIZON) > rv.rolling(252).median()).astype(int)
        y_fixed = y_fixed.reindex(pd.DatetimeIndex(dates)).reset_index(drop=True)
        keep = y_fixed.notna()
        X, y, dates = X[keep.values], y_fixed[keep].astype(int), dates[keep.values]
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)

        out = f"data/ind{n}.pkl"
        import pickle
        with open(out, "wb") as fh:
            pickle.dump({"X": X, "y": y, "base": np.asarray(base)[:len(X)],
                         "dates": dates.reset_index(drop=True)}, fh)
        print(f"  ind{n:<3} series={rets.shape[1]:<3} X={X.shape}  "
              f"y-mean={y.mean():.3f}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
