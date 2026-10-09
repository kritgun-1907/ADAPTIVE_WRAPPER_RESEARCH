"""
Non-financial high-dimensional datasets: the stationary check.

WHY (review, 2026-08-18): every dataset in the paper is financial, and the four
asset classes in the frozen-vs-refit replication are all built by the same ten
rolling transforms, so "46 of 48 cells" is less independent than it looks. The
real ladder is one market partitioned more finely. But the Conclusion makes a
general machine-learning claim -- deploy against an estimator without embedded
selection -- and a general claim validated on one market plus one generator is
the gap a reviewer presses hardest.

The classifier-capacity mechanism is not about non-stationarity at all: if it is
real, it should hold on IID data. Testing it there ISOLATES it rather than
complicating it, so these are run under plain stratified cross-validation with
NO walk-forward and no regime structure, and are reported as the stationary
check.

  madelon   500 features, 2600 rows. XOR-like, 5 informative + 15 redundant
            linear combinations + 480 noise. Purpose-built for selection, and
            its redundancy structure is exactly the collinearity axis the
            earlier round identified.
  gisette   5000 features, 7000 rows. Digit 4-vs-9 with 2500 injected noise
            features -- a genuinely high-p case an order above our ladder.
  golub     7129 features, 72 rows. Leukaemia expression; p >> n, a regime the
            paper never otherwise touches (all financial sets have n >> p).

Sources are UCI / OpenML and are cached to data/raw_*.pkl on first fetch, for
the same reason the yfinance and Dartmouth caches exist: a re-download later
must not silently change the data underneath a published number.

Run:  python data_prep/build_nonfinancial.py
Writes data/madelon.pkl, data/gisette.pkl, data/golub.pkl
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"


def _install_certifi_ssl():
    """sklearn's fetch_openml goes through urllib, which on macOS uses the
    system trust store and fails with CERTIFICATE_VERIFY_FAILED. step9 already
    hit this for the Dartmouth download and solved it by falling back to a
    certifi-backed context; the same fix is applied globally here so the
    OpenML fetch works on a laptop as well as on the instance."""
    import ssl
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl._create_unverified_context()
    ssl._create_default_https_context = lambda *a, **k: ctx


_install_certifi_ssl()


def _fetch_openml(name, version=1):
    """as_frame=True fails on OpenML's sparse-ARFF copies (gisette v1 is one),
    so fall back to the array form and densify. Densifying is safe at these
    sizes -- gisette is 7000x5000 -- and keeps every downstream path identical
    to the dense financial sets."""
    from sklearn.datasets import fetch_openml
    try:
        d = fetch_openml(name, version=version, as_frame=True, parser="auto")
        X = d.data.apply(pd.to_numeric, errors="coerce")
    except Exception:
        d = fetch_openml(name, version=version, as_frame=False, parser="auto")
        arr = d.data
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        X = pd.DataFrame(np.asarray(arr, dtype=float),
                         columns=[f"f{i}" for i in range(np.shape(arr)[1])])
    y = pd.Series(pd.factorize(np.asarray(d.target).ravel())[0], name="target")
    return X, y


def build_madelon():
    cache = "data/raw_madelon.pkl"
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    X, y = _fetch_openml("madelon")
    df = X.copy(); df["__y__"] = y.values
    df.to_pickle(cache)
    return df


def build_gisette():
    cache = "data/raw_gisette.pkl"
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    X, y = _fetch_openml("gisette")
    df = X.copy(); df["__y__"] = y.values
    df.to_pickle(cache)
    return df


def build_golub():
    cache = "data/raw_golub.pkl"
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    X, y = _fetch_openml("leukemia")
    df = X.copy(); df["__y__"] = y.values
    df.to_pickle(cache)
    return df


BUILDERS = {"madelon": build_madelon, "gisette": build_gisette,
            "golub": build_golub}


def main():
    os.makedirs("data", exist_ok=True)
    print("=" * 74)
    print("  NON-FINANCIAL HIGH-DIMENSIONAL DATASETS (stationary check)")
    print("=" * 74, flush=True)

    for name, fn in BUILDERS.items():
        out = f"data/{name}.pkl"
        if os.path.exists(out):
            with open(out, "rb") as fh:
                d = pickle.load(fh)
            print(f"  [have] {name:<10} X={d['X'].shape}  y-mean={d['y'].mean():.3f}",
                  flush=True)
            continue
        try:
            df = fn()
        except Exception as e:
            print(f"  FAIL {name}: {type(e).__name__}: {e}", flush=True)
            continue

        y = pd.Series(df["__y__"].astype(int).values, name="target")
        X = df.drop(columns="__y__")
        # Numeric, finite, non-constant columns only -- a zero-variance column
        # cannot be informative and breaks the scaler.
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.loc[:, np.isfinite(X.to_numpy()).all(axis=0)]
        X = X.loc[:, X.std(axis=0) > 0]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        with open(out, "wb") as fh:
            pickle.dump({"X": X, "y": y, "base": np.zeros(len(X))}, fh)
        print(f"  built  {name:<10} X={X.shape}  y-mean={y.mean():.3f}  -> {out}",
              flush=True)

    print("\n  NOTE: these are IID. They are evaluated with stratified k-fold,")
    print("  not walk-forward, and must be reported as the stationary check.",
          flush=True)


if __name__ == "__main__":
    main()
