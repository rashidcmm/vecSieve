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

    def _select_neighbors_heuristic(self, candidates: list[tuple[float, int]], M: int) -> list[tuple[float, int]]:
        """Algorithm 4 (Malkov & Yashunin). Keep candidate c only if it is closer to
        the query than to every neighbour already selected — this enforces a relative-
        neighbourhood-graph property (diverse directions) instead of M clustered points.
        `candidates` is (distance_to_query, node_id) pairs, any order."""
        candidates = sorted(candidates)
        selected: list[tuple[float, int]] = []
        for d_qc, c in candidates:
            if len(selected) >= M:
                break
            c_vec = self.store.vector(c)
            keep = True
            for _, r in selected:
                d_cr = float(np.sum((c_vec - self.store.vector(r)) ** 2))
                if d_cr < d_qc:
                    keep = False
                    break
            if keep:
                selected.append((d_qc, c))
        return selected

    def _insert(self, node_id: int) -> None:
        level = self._assign_level()
        self._ensure_layers(level)
        while len(self.levels) <= node_id:
            self.levels.append(-1)
        self.levels[node_id] = level

        if self.entry_point == -1:
            for layer in range(level + 1):
                self.graph[layer].setdefault(node_id, [])
            self.entry_point = node_id
            self.max_level = level
            return

        q = self.store.vector(node_id)
        ep = [self.entry_point]
        for layer in range(self.max_level, level, -1):
            nearest = self._search_layer(q, ep, ef=1, layer=layer)
            if nearest:
                ep = [nearest[0][1]]

        for layer in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(q, ep, self.ef_construction, layer)
            cap = self.M0 if layer == 0 else self.M
            neighbours = self._select_neighbors_heuristic(candidates, cap)

            self.graph[layer].setdefault(node_id, [])
            self.graph[layer][node_id] = [n for _, n in neighbours]

            for _, n in neighbours:
                self.graph[layer].setdefault(n, [])
                if node_id not in self.graph[layer][n]:
                    self.graph[layer][n].append(node_id)
                if len(self.graph[layer][n]) > cap:
                    nb_ids = self.graph[layer][n]
                    nb_vec = self.store.vector(n)
                    nb_candidates = [(float(np.sum((self.store.vector(x) - nb_vec) ** 2)), x) for x in nb_ids]
                    pruned = self._select_neighbors_heuristic(nb_candidates, cap)
                    self.graph[layer][n] = [x for _, x in pruned]

            ep = [n for _, n in neighbours] if neighbours else ep

        if level > self.max_level:
            self.max_level = level
            self.entry_point = node_id

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        assert np.array_equal(np.asarray(ids), np.arange(len(ids))), \
            "HNSWIndex expects dense internal ids 0..N-1"
        self.store = VectorStore(vectors)
        for node_id in range(len(ids)):
            self._insert(node_id)

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None,
                params: dict | None = None) -> SearchResult:
        assert mask is None, "HNSWIndex.search is unfiltered; filtered strategies live in vecdb.index.strategies"
        assert self.store is not None and self.entry_point != -1, "index is empty"
        ef = (params or {}).get("ef", max(self.ef_search_default, k))
        ops_before = self.store.n_distance_ops
        t0 = time.perf_counter()
        ep = [self.entry_point]
        for layer in range(self.max_level, 0, -1):
            nearest = self._search_layer(q, ep, ef=1, layer=layer)
            if nearest:
                ep = [nearest[0][1]]
        candidates = self._search_layer(q, ep, ef=max(ef, k), layer=0)[:k]
        latency_ms = (time.perf_counter() - t0) * 1000
        ids = np.array([n for _, n in candidates], dtype=np.int64)
        dists = np.array([d for d, _ in candidates], dtype=np.float32)
        return SearchResult(ids=ids, distances=dists,
                             n_distance_ops=self.store.n_distance_ops - ops_before,
                             strategy="hnsw", latency_ms=latency_ms, n_returned=int(ids.size))
