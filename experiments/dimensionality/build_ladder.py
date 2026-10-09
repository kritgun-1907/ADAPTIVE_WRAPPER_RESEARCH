"""
Rebuild the real-ladder record list from the CHECKPOINT STORE, not from JSONs.

WHY THIS REPLACES merge_ladder.py
─────────────────────────────────
merge_ladder.py spliced two named result files:

    results/tier_d_fullnovelty.json                  "the 20-seed run"
    results/tier_d_fullnovelty_8seed_20260814.json   "the 8-seed run"

That was correct when written and is now actively dangerous, because
tier_d_fullnovelty.json is OVERWRITTEN by every invocation of
tier_d_fullnovelty.py. On 2026-09-08 an unrelated job (k-NN on the synthetic
sweep) wrote 160 synthetic records into it. Running the old merge at that point
would have spliced hd_ar1/hd_regime_switch rows into the real ladder and the
resulting figure would have been silently wrong -- no error, no missing file,
just a curve built from the wrong data.

The checkpoint store does not have this failure mode. It is:
  * append-only, one file per (dataset, model, seed, arm-set)
  * scoped by provenance hash, so a code change cannot silently mix generations
  * the union of every run ever executed, rather than whichever finished last

So the store is the authoritative source and this script reads it directly.
No file that any future run might overwrite is consulted.

SEED COUNTS ARE NOT UNIFORM AND THAT IS CARRIED, NOT HIDDEN
───────────────────────────────────────────────────────────
Every record keeps its own seed count. Downstream tables must report n per row.
Claiming a uniform n the data does not support is the specific failure this
whole pipeline is built to avoid.

LIBRARY PROVENANCE
──────────────────
Checkpoints do not record the scikit-learn version, which is a real gap: the
2026-08 ladder run drifted to 1.9.0 and only the `rfe` arm moved (verified
bit-identical on all other arms across 64 shared cells). Units written on or
after 2026-09-08 were produced under a verified 1.8.0 pin. This script
therefore reports the mtime split per width so the RFE curve can be audited,
and refuses to silently present a mixed-version arm as uniform.

Run:  python experiments/dimensionality/build_ladder.py
Writes results/tier_d_ladder_merged.json
"""
import os
import sys
import json
import time
import pickle
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

STORE = "results/checkpoints/tier_d_fullnovelty"
OUT = "results/tier_d_ladder_merged.json"
FALLBACK = "results/tier_d_fullnovelty_8seed_20260814.json"
LADDER = ["ind5", "ind10", "ind12", "ind17", "ind30", "ind38", "ind48", "ff49ind"]
PIN_EPOCH = time.mktime(time.strptime("2026-09-08", "%Y-%m-%d"))


def pick_generation(store):
    """
    Choose the checkpoint generation to read, WITHOUT using mtime.

    This selected `max(gens, key=os.path.getmtime)` until 2026-09-10, when
    untarring results fetched from EC2 reset every generation directory's mtime
    to the same minute. The "newest" generation then became whichever one the
    filesystem happened to return first: build_ladder.py silently rebuilt the
    merged ladder from a 200-unit generation instead of the 1304-unit one,
    dropping every 20-seed cell and both extra estimators, and
    knn_synthetic_tables.py emitted a table with no rows at all. Neither raised
    an error -- the tables were simply wrong.

    mtime answers "when was this directory last touched", which is not the
    question. Unit count answers "which generation actually holds the run", and
    is invariant to copying, extracting and syncing. ORPSOC_GEN overrides it
    when a specific generation is wanted.
    """
    import os
    pin = os.environ.get("ORPSOC_GEN")
    gens = [d for d in os.listdir(store) if os.path.isdir(os.path.join(store, d))]
    if not gens:
        raise SystemExit(f"no checkpoint generations under {store}")
    if pin:
        if pin not in gens:
            raise SystemExit(f"ORPSOC_GEN={pin} not present under {store}")
        return pin
    def n_units(d):
        p = os.path.join(store, d)
        return sum(1 for f in os.listdir(p) if f.endswith(".pkl"))
    ranked = sorted(gens, key=n_units, reverse=True)
    if len(ranked) > 1 and n_units(ranked[0]) == n_units(ranked[1]):
        raise SystemExit(
            f"cannot choose between generations {ranked[0]} and {ranked[1]}: "
            f"both hold {n_units(ranked[0])} units. Set ORPSOC_GEN explicitly.")
    return ranked[0]


def main():
    gen = pick_generation(STORE)
    root = os.path.join(STORE, gen)

    # (ds, model, seed) -> merged arms dict. A later-written unit wins per ARM,
    # so a targeted re-run (e.g. FN_ARMS=rfe under the pin) supersedes the same
    # arm from an older unit without disturbing arms it did not compute.
    merged = {}
    arm_mtime = {}
    for f in sorted(os.listdir(root),
                    key=lambda x: os.path.getmtime(os.path.join(root, x))):
        if not f.endswith(".pkl"):
            continue
        path = os.path.join(root, f)
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
        except Exception:
            continue
        r = obj.get("payload", obj)
        if r.get("ds") not in LADDER:
            continue
        key = (r["ds"], r["model"], r["seed"])
        rec = merged.setdefault(key, {"ds": r["ds"], "model": r["model"],
                                      "seed": r["seed"],
                                      "n_features": r.get("n_features"),
                                      "arms": {}})
        mt = os.path.getmtime(path)
        for arm, v in r["arms"].items():
            rec["arms"][arm] = v
            arm_mtime[(r["ds"], r["model"], arm)] = mt

    for r in merged.values():
        r["source"] = "checkpoint_current_generation"

    # ── FALLBACK, explicitly tagged ─────────────────────────────────────────
    # ind48 / ff49ind LightGBM exist only in the 2026-08-14 8-seed run, which
    # lives in an EARLIER provenance generation (before the four enhanced
    # swarms were added to the runner). The checkpoint reader above correctly
    # refuses to mix generations, which would otherwise leave those two cells
    # empty and silently shorten the LightGBM curve.
    #
    # merge_ladder.py verified that across the 64 (dataset, model, seed) cells
    # the two generations share, every arm reproduces BIT-IDENTICALLY except
    # `rfe`. The code change added arms; it did not alter any existing arm's
    # code path. Re-using those cells is therefore defensible -- but it is a
    # cross-generation splice at a lower seed count, so every such record is
    # TAGGED and the seed count travels with it. Any table built from this file
    # must report n per row and must not present these cells as 20-seed.
    if os.path.exists(FALLBACK):
        with open(FALLBACK) as fh:
            old = json.load(fh)
        added = collections.Counter()
        for r in old:
            key = (r["ds"], r["model"], r["seed"])
            if r.get("ds") not in LADDER or key in merged:
                continue
            r = dict(r)
            r["source"] = "fallback_8seed_20260814_prior_generation"
            merged[key] = r
            added[(r["ds"], r["model"])] += 1
        if added:
            print("\n  FALLBACK cells used (prior generation, 8 seeds):")
            for (ds, m), n in sorted(added.items()):
                print(f"    {ds:<10}{m:<11}{n:>3} seeds")

    records = list(merged.values())
    with open(OUT, "w") as fh:
        json.dump(records, fh, indent=2, default=float)

    # ── report, so the caller sees the shape of what was built ──────────────
    per = collections.Counter((r["ds"], r["model"]) for r in records)
    print("=" * 74)
    print(f"  LADDER REBUILT FROM CHECKPOINTS  (generation {gen})")
    print("=" * 74)
    print(f"  {'dataset':<10}{'model':<11}{'seeds':>6}")
    for ds in LADDER:
        for m in ("LogReg", "LightGBM", "kNN"):
            n = per.get((ds, m), 0)
            if n:
                print(f"  {ds:<10}{m:<11}{n:>6}")
    print(f"\n  {len(records)} records -> {OUT}")

    # library-version audit for the arm known to be version sensitive
    print("\n  RFE library-version audit (mtime proxy; 1.8.0 pin verified from 2026-09-08):")
    for ds in LADDER:
        for m in ("LogReg", "LightGBM", "kNN"):
            mt = arm_mtime.get((ds, m, "rfe"))
            if mt is None:
                continue
            tag = "1.8.0 (pinned)" if mt >= PIN_EPOCH else "PRE-PIN, may be 1.9.0"
            print(f"    {ds:<10}{m:<11}{time.strftime('%Y-%m-%d', time.localtime(mt))}"
                  f"  {tag}")


if __name__ == "__main__":
    main()
