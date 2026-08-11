from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Columns whose dtype is one of these are treated as categorical, not numeric.
_CATEGORICAL_DTYPES = (np.int32,)


@dataclass
class ColumnStats:
    kind: str  # "categorical" | "numeric"
    n: int
    value_counts: dict | None = None
    hist_edges: np.ndarray | None = None
    hist_counts: np.ndarray | None = None


def _compute_stats(values: np.ndarray) -> ColumnStats:
    n = values.shape[0]
    if values.dtype == np.int32:
        keys, counts = np.unique(values, return_counts=True)
        return ColumnStats(kind="categorical", n=n, value_counts=dict(zip(keys.tolist(), counts.tolist())))
    counts, edges = np.histogram(values, bins=64)
    return ColumnStats(kind="numeric", n=n, hist_edges=edges, hist_counts=counts)


class MetaStore:
    """Columnar attributes, one np.ndarray per column, row-aligned with VectorStore.
    'category' (int32) is treated as categorical; everything else (year, score, ...)
    is treated as numeric with a 64-bin histogram."""

    def __init__(self, columns: dict[str, np.ndarray]):
        self.columns: dict[str, np.ndarray] = columns
        lengths = {len(v) for v in columns.values()}
        assert len(lengths) <= 1, "all columns must have the same length"
        self.n: int = next(iter(lengths), 0)
        self.stats: dict[str, ColumnStats] = {name: _compute_stats(col) for name, col in columns.items()}
