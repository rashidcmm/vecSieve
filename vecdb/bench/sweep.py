# vecdb/bench/sweep.py
from __future__ import annotations
import numpy as np
from vecdb.store.metadata import MetaStore

SELECTIVITY_GRID: list[float] = [0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]


def predicate_for_selectivity(meta: MetaStore, target_s: float, col: str = "score") -> dict:
    """A `score < t` predicate where t is the empirical target_s-quantile of `col`,
    so the true selectivity lands within noise of target_s regardless of col's
    exact distribution."""
    if target_s >= 1.0:
        return {"op": "gte", "col": col, "val": float(meta.columns[col].min()) - 1.0}
    t = float(np.quantile(meta.columns[col], target_s))
    return {"op": "lt", "col": col, "val": t}
