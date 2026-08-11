from __future__ import annotations
import numpy as np
import pandas as pd
from vecdb.index.base import Index


def recall_at_k(returned_ids: np.ndarray, true_ids: np.ndarray, k: int) -> float:
    """|returned ∩ true| / min(k, len(true)). Using min(k, len(true)) as the
    denominator (not len(true_ids[:k]) blindly, and not a naive len(true)) is what
    prevents recall from exceeding 1.0 when a highly selective filter has fewer
    than k true survivors to begin with."""
    denom = min(k, len(true_ids))
    if denom == 0:
        return 1.0
    returned_set = set(np.asarray(returned_ids[:k]).tolist())
    true_set = set(np.asarray(true_ids[:k]).tolist())
    return len(returned_set & true_set) / denom


def run_benchmark(index: Index, queries: np.ndarray, groundtruth: list[np.ndarray], k: int,
                   masks: list[np.ndarray] | None = None, params: dict | None = None,
                   warmup: int = 0) -> pd.DataFrame:
    rows = []
    for i, q in enumerate(queries):
        mask = masks[i] if masks is not None else None
        result = index.search(q, k, mask=mask, params=params)
        if i < warmup:
            continue
        rows.append({
            "query_idx": i,
            "recall": recall_at_k(result.ids, groundtruth[i], k),
            "underfill": result.n_returned < k,
            "latency_ms": result.latency_ms,
            "dist_ops": result.n_distance_ops,
            "n_returned": result.n_returned,
            "strategy": result.strategy,
        })
    return pd.DataFrame(rows)
