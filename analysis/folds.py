"""
Which walk-forward folds count when a stored run is averaged.

On the real datasets run with a 150-row first training window, fold 0's
training rows all belong to one class, so the model cannot be fitted and every
arm -- the all-features baseline included -- is stored with the placeholder
AUC 0.5. Averaging it in scales every real-data delta by 7/8. The paper reports
the seven scoreable folds, and so do all tables and figures.

The test is per record, not per arm: a synthetic run can genuinely score 0.5 on
one arm at fold 0, and that fold must be kept. Fold 0 is dropped only when
every arm in the record sits at exactly 0.5 there.
"""


def dead_fold0(record, key="auc"):
    """True if fold 0 of this record is the unscoreable placeholder."""
    series = [v[key] for v in record["arms"].values() if v.get(key)]
    return bool(series) and all(s[0] == 0.5 for s in series)


def scored(values, record, key="auc"):
    """Per-fold values with the placeholder fold removed when present."""
    return values[1:] if dead_fold0(record, key) else values
