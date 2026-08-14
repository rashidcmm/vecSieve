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


class PostFilterStrategy(Index):
    """Strategy B: run HNSW with a widened beam, discard non-matches, take top-k.
    Expected survivors from a top-ef list is ef * s, so ef must grow as s shrinks —
    ef = clamp(alpha * k / s, ef_min, N). This is *probabilistic*: it can under-fill.
    Under-fill is tracked explicitly (never silently returned as a short list), retried
    once with a wider beam, and if still short, handed off to the exact fallback."""

    def __init__(self, hnsw_index: HNSWIndex, fallback: Index, alpha: float = 4.0,
                 ef_min: int = 16, max_retries: int = 1, retry_multiplier: float = 4.0):
        self.hnsw = hnsw_index
        self.fallback = fallback
        self.alpha = alpha
        self.ef_min = ef_min
        self.max_retries = max_retries
        self.retry_multiplier = retry_multiplier
        self.fallback_count = 0
        self.query_count = 0

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        raise NotImplementedError("PostFilterStrategy wraps an already-built HNSWIndex")

    def _ef_for(self, k: int, sel_hat: float) -> int:
        sel_hat = max(sel_hat, 1e-6)
        ef = self.alpha * k / sel_hat
        return int(np.clip(ef, self.ef_min, len(self.hnsw.store)))

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / self.query_count if self.query_count else 0.0

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None,
                params: dict | None = None) -> SearchResult:
        self.query_count += 1
        params = params or {}
        sel_hat = params.get("selectivity_hat", 1.0)
        ef = self._ef_for(k, sel_hat)
        t0 = time.perf_counter()
        ops_before = self.hnsw.store.n_distance_ops

        raw = self.hnsw.search(q, k=ef, params={"ef": ef})
        if mask is None:
            latency_ms = (time.perf_counter() - t0) * 1000
            raw.strategy = "post_filter"
            raw.latency_ms = latency_ms
            return raw

        keep = mask[raw.ids]
        filtered_ids, filtered_d = raw.ids[keep], raw.distances[keep]

        attempts = 0
        while filtered_ids.size < k and attempts < self.max_retries and ef < len(self.hnsw.store):
            attempts += 1
            ef = int(min(ef * self.retry_multiplier, len(self.hnsw.store)))
            raw = self.hnsw.search(q, k=ef, params={"ef": ef})
            keep = mask[raw.ids]
            filtered_ids, filtered_d = raw.ids[keep], raw.distances[keep]

        if filtered_ids.size < k:
            self.fallback_count += 1
            fb = self.fallback.search(q, k, mask=mask)
            fb.strategy = "post_filter_fallback"
            fb.n_distance_ops += self.hnsw.store.n_distance_ops - ops_before
            return fb

        order = np.argsort(filtered_d)[:k]
        ids, dists = filtered_ids[order], filtered_d[order]
        latency_ms = (time.perf_counter() - t0) * 1000
        return SearchResult(ids=ids, distances=dists,
                             n_distance_ops=self.hnsw.store.n_distance_ops - ops_before,
                             strategy="post_filter", latency_ms=latency_ms, n_returned=int(ids.size))
