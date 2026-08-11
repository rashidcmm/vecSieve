from __future__ import annotations
import time
import numpy as np
import faiss
from vecdb.index.base import Index, SearchResult


class FaissFlatIndex(Index):
    """Sanity baseline: FAISS's own exact search. Recall against it should be 1.0."""

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None,
                params: dict | None = None) -> SearchResult:
        assert mask is None, "FaissFlatIndex baseline is unfiltered-only"
        t0 = time.perf_counter()
        distances, ids = self.index.search(q.reshape(1, -1).astype(np.float32), k)
        latency_ms = (time.perf_counter() - t0) * 1000
        return SearchResult(ids=ids[0], distances=distances[0], n_distance_ops=self.index.ntotal,
                             strategy="faiss_flat", latency_ms=latency_ms, n_returned=k)


class FaissHNSWIndex(Index):
    """The real comparison target for Phase 4's Pareto curve and Phase 7's dist_ops table."""

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200):
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = ef_construction

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def set_ef_search(self, ef: int) -> None:
        self.index.hnsw.efSearch = ef

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None,
                params: dict | None = None) -> SearchResult:
        assert mask is None, "FaissHNSWIndex baseline is unfiltered-only"
        if params and "ef" in params:
            self.set_ef_search(params["ef"])
        faiss.cvar.hnsw_stats.reset()
        t0 = time.perf_counter()
        distances, ids = self.index.search(q.reshape(1, -1).astype(np.float32), k)
        latency_ms = (time.perf_counter() - t0) * 1000
        return SearchResult(ids=ids[0], distances=distances[0],
                             n_distance_ops=int(faiss.cvar.hnsw_stats.ndis),
                             strategy="faiss_hnsw", latency_ms=latency_ms, n_returned=k)
