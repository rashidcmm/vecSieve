"""The three filtered-search strategies, built on top of an already-constructed
FlatIndex / HNSWIndex. None of these implement add() — they wrap indexes built
elsewhere so Flat/HNSW storage is never duplicated across strategies."""
from __future__ import annotations
import time
import numpy as np
from vecdb.index.base import Index, SearchResult
from vecdb.index.flat import FlatIndex
from vecdb.index.hnsw import HNSWIndex


class PreFilterStrategy(Index):
    """Strategy A: materialise the masked rows, exact-scan them. Correctness: exact,
    recall = 1.0 always. Cost: O(N * s * d). This IS FlatIndex's masked search —
    the strategy wrapper exists so the benchmark harness can label it distinctly and
    the planner can address it uniformly alongside the other two strategies."""

    def __init__(self, flat_index: FlatIndex):
        self.flat_index = flat_index

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        raise NotImplementedError("PreFilterStrategy wraps an already-built FlatIndex")

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None,
                params: dict | None = None) -> SearchResult:
        result = self.flat_index.search(q, k, mask=mask, params=params)
        result.strategy = "pre_filter"
        return result
