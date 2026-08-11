from __future__ import annotations
import numpy as np
from vecdb.store.metadata import MetaStore


def _frac_below(stats, x: float) -> float:
    """Fraction of values < x, via linear interpolation within the histogram bin
    containing x. Reads only stats.hist_edges / stats.hist_counts / stats.n."""
    edges, counts, n = stats.hist_edges, stats.hist_counts, stats.n
    if x <= edges[0]:
        return 0.0
    if x >= edges[-1]:
        return 1.0
    idx = int(np.searchsorted(edges, x, side="right") - 1)
    idx = min(max(idx, 0), len(counts) - 1)
    bin_lo, bin_hi = edges[idx], edges[idx + 1]
    bin_frac = (x - bin_lo) / (bin_hi - bin_lo) if bin_hi > bin_lo else 0.0
    prior_count = counts[:idx].sum()
    within_bin = bin_frac * counts[idx]
    return float((prior_count + within_bin) / n)


def estimate_selectivity(pred: dict, meta: MetaStore) -> float:
    op = pred["op"]
    if op == "and":
        result = 1.0
        for clause in pred["clauses"]:
            result *= estimate_selectivity(clause, meta)
        return result
    if op == "or":
        result = 0.0
        for clause in pred["clauses"]:
            s = estimate_selectivity(clause, meta)
            result = result + s - result * s
        return result
    if op == "not":
        return 1.0 - estimate_selectivity(pred["clause"], meta)

    stats = meta.stats[pred["col"]]
    val = pred["val"]
    if stats.kind == "categorical":
        if op == "eq":
            return stats.value_counts.get(val, 0) / stats.n
        if op == "ne":
            return 1.0 - stats.value_counts.get(val, 0) / stats.n
        if op == "in":
            return sum(stats.value_counts.get(v, 0) for v in val) / stats.n
        raise ValueError(f"unsupported categorical op for estimation: {op!r}")

    if op in ("lt", "lte"):
        return min(max(_frac_below(stats, val), 0.0), 1.0)
    if op in ("gt", "gte"):
        return min(max(1.0 - _frac_below(stats, val), 0.0), 1.0)
    raise ValueError(f"unsupported numeric op for estimation: {op!r}")
