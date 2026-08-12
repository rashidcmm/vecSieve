from __future__ import annotations
import heapq
import math
import pickle
import random
import time
from pathlib import Path
import numpy as np
from vecdb.store.vectors import VectorStore
from vecdb.index.base import Index, SearchResult


class HNSWIndex(Index):
    """Hand-written Hierarchical Navigable Small World index (Malkov & Yashunin, 2016).
    Built up across Tasks 14-18: this task gives the skeleton, level assignment, and
    layer-list growth. Later tasks add search_layer, the neighbour heuristic, the full
    insert() algorithm, and persistence."""

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, seed: int = 42):
        self.dim = dim
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(M)
        self.entry_point: int = -1
        self.max_level: int = -1
        self.levels: list[int] = []
        self.graph: list[dict[int, list[int]]] = []  # graph[layer][node_id] -> neighbour ids
        self.store: VectorStore | None = None
        self.ef_search_default = 50
        self._rng = random.Random(seed)

    def _assign_level(self) -> int:
        return int(-math.log(self._rng.random()) * self.mL)

    def _ensure_layers(self, level: int) -> None:
        while len(self.graph) <= level:
            self.graph.append({})

    def _search_layer(self, q: np.ndarray, entry_points: list[int], ef: int, layer: int) -> list[tuple[float, int]]:
        """Greedy beam search on one layer. Three collections: a min-heap of candidates
        to expand, a max-heap of the best `ef` results found so far (so the worst can be
        evicted cheaply), and a visited set. The early-break when the nearest unexplored
        candidate is already worse than our worst kept result is what makes this sub-linear
        instead of a full traversal — see source plan Day 2 debugging checklist if search
        ends up slow despite good recall."""
        visited = set(entry_points)
        candidates: list[tuple[float, int]] = []   # min-heap by distance
        results: list[tuple[float, int]] = []       # max-heap via negated distance

        if entry_points:
            dists = self.store.distances(q, np.array(entry_points, dtype=np.int64))
            for ep, d in zip(entry_points, dists):
                d = float(d)
                heapq.heappush(candidates, (d, ep))
                heapq.heappush(results, (-d, ep))

        while candidates:
            d_c, c = heapq.heappop(candidates)
            worst_d = -results[0][0]
            if d_c > worst_d and len(results) >= ef:
                break
            neighbours = [n for n in self.graph[layer].get(c, []) if n not in visited]
            if not neighbours:
                continue
            visited.update(neighbours)
            dists = self.store.distances(q, np.array(neighbours, dtype=np.int64))
            for n, d in zip(neighbours, dists):
                d = float(d)
                worst_d = -results[0][0] if results else float("inf")
                if len(results) < ef:
                    heapq.heappush(candidates, (d, n))
                    heapq.heappush(results, (-d, n))
                elif d < worst_d:
                    heapq.heappush(candidates, (d, n))
                    heapq.heappush(results, (-d, n))
                    heapq.heappop(results)

        return sorted((-nd, node) for nd, node in results)

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Stub: will be implemented in later tasks."""
        raise NotImplementedError("HNSW insert() will be implemented in a later task")

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None,
                params: dict | None = None) -> SearchResult:
        """Stub: will be implemented in later tasks."""
        raise NotImplementedError("HNSW search() will be implemented in a later task")
