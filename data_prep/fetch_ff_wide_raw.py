"""
Fetch the two WIDE Ken French daily files used by Tier D (D2.a) and cache them
as data/raw_*.pkl, exactly as step9 caches raw_fama_french.pkl.

Why cache the raw block rather than re-download per run: Dartmouth restates and
extends these files, so a re-download on a different day yields a different
series and nothing stays comparable. Same reasoning as the yfinance caches --
see deploy/README.md.

These files carry SEVERAL stacked blocks (value-weighted returns, equal-weighted
returns, number of firms, average firm size, ...). Only the FIRST block --
"Average Value Weighted Returns -- Daily" -- is taken. Missing observations are
coded -99.99 / -999 and become NaN.

Run:  python data_prep/fetch_ff_wide_raw.py
Writes data/raw_ff100.pkl  (100 size/BM portfolios)
       data/raw_ff49ind.pkl (49 industry portfolios)
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
SOURCES = {
    "ff100":  "100_Portfolios_10x10_Daily_CSV.zip",
    "ff49ind": "49_Industry_Portfolios_daily_CSV.zip",
}
# Match step9: the other real datasets start here, so the documented
# 2001/2008/2020 breaks land in a comparable window.
START_DATE = "2000-01-01"
MISSING = (-99.99, -999.0, -99.99e2)


def _http_get_bytes(url: str, timeout: int = 300) -> bytes:
    """Same fetch ladder as step9_real_data._http_get_bytes."""
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
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout, context=ctx).read()


def parse_first_block(txt: str) -> pd.DataFrame:
    """
    Extract the first stacked data block: the header row (starts with a comma,
    naming the portfolios) followed by the contiguous run of YYYYMMDD rows.
    """
    lines = txt.splitlines()
    is_data = lambda l: re.match(r"^\s*\d{8}\s*,", l) is not None

    # Header = last comma-leading row before the first data row.
    first_data = next(i for i, l in enumerate(lines) if is_data(l))
    hdr = next(i for i in range(first_data - 1, -1, -1)
               if re.match(r"^\s*,", lines[i]))

    end = first_data
    while end < len(lines) and is_data(lines[end]):
        end += 1

    block = "\n".join([lines[hdr]] + lines[first_data:end])
    df = pd.read_csv(io.StringIO(block))
    df = df.rename(columns={df.columns[0]: "Date"})
    df.columns = ["Date"] + [c.strip() for c in df.columns[1:]]
    df["Date"] = pd.to_datetime(df["Date"].astype(int).astype(str), format="%Y%m%d")
    df = df.set_index("Date").astype(float)

    for m in MISSING:
        df = df.mask(np.isclose(df, m))
    return df / 100.0                      # percent -> decimal


def main():
    os.makedirs("data", exist_ok=True)
    for key, fname in SOURCES.items():
        cache = f"data/raw_{key}.pkl"
        if os.path.exists(cache):
            d = pd.read_pickle(cache)
            print(f"  [cache] {key} <- {cache}  shape={d.shape}", flush=True)
            continue

        print(f"  Downloading {fname} ...", flush=True)
        raw = _http_get_bytes(BASE + fname)
        zf = zipfile.ZipFile(io.BytesIO(raw))
        txt = zf.read(zf.namelist()[0]).decode("latin-1")

        df = parse_first_block(txt)
        df = df[df.index >= pd.Timestamp(START_DATE)]

        # Portfolios with any gap in the window are dropped rather than filled:
        # a forward-filled return series is a fabricated observation.
        before = df.shape[1]
        df = df.dropna(axis=1, how="any")
        if df.shape[1] < before:
            print(f"    dropped {before - df.shape[1]} portfolios with gaps",
                  flush=True)

        df.to_pickle(cache)
        print(f"  Saved {key} -> {cache}  shape={df.shape}  "
              f"{df.index.min().date()}..{df.index.max().date()}", flush=True)


if __name__ == "__main__":
    main()
