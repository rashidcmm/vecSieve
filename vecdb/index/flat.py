from __future__ import annotations
import time
import numpy as np
from vecdb.store.vectors import VectorStore
from vecdb.index.base import Index, SearchResult


class FlatIndex(Index):
    """Exact search over all (or masked) rows. This is the ground-truth oracle
    every other index/strategy is measured against."""

    def __init__(self):
        self.store: VectorStore | None = None

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        assert np.array_equal(np.asarray(ids), np.arange(len(ids))), \
            "FlatIndex expects dense internal ids 0..N-1"
        self.store = VectorStore(vectors)

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None,
                params: dict | None = None) -> SearchResult:
        assert self.store is not None, "call add() first"
        t0 = time.perf_counter()
        rows = np.nonzero(mask)[0] if mask is not None else np.arange(len(self.store))
        if rows.size == 0:
            return SearchResult(ids=np.array([], dtype=np.int64), distances=np.array([], dtype=np.float32),
                                 n_distance_ops=0, strategy="flat",
                                 latency_ms=(time.perf_counter() - t0) * 1000, n_returned=0)
        d = self.store.distances(q, rows)
        k_eff = min(k, rows.size)
        part = np.argpartition(d, k_eff - 1)[:k_eff]
        order = part[np.argsort(d[part])]
        result_ids = rows[order]
        result_d = d[order]
        return SearchResult(ids=result_ids, distances=result_d, n_distance_ops=int(rows.size),
                             strategy="flat", latency_ms=(time.perf_counter() - t0) * 1000,
                             n_returned=int(result_ids.size))
