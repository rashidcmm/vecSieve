import numpy as np
from vecdb.index.hnsw import HNSWIndex
from vecdb.store.vectors import VectorStore

def test_heuristic_prefers_diverse_directions_over_pure_nearest():
    # Query at origin. c0 is nearest, c1 is second-nearest but nearly collinear with c0
    # (so it's much closer to c0 than to the query), and c2 is farthest of the three but
    # in a diverse direction (closer to the query than to c0). Sorted-by-distance order
    # is c0, c1, c2, so the M=2 loop keeps c0, then actually evaluates c1 next (it hasn't
    # hit the M cap yet) and must reject it via the relative-neighbourhood check
    # (d(c1,c0)^2=0.0025 << d(query,c1)^2=1.1025) before reaching c2, which it keeps
    # (d(c2,c0)^2=2.44 >= d(query,c2)^2=1.44). A naive "keep the M closest" shortcut would
    # instead pick {c0, c1} and fail this test.
    points = np.array([
        [1.0, 0.0],   # c0: dist^2 to origin = 1.0 (nearest, kept)
        [1.05, 0.0],  # c1: dist^2 to origin = 1.1025 (second-nearest, but rejected: too close to c0)
        [0.0, 1.2],   # c2: dist^2 to origin = 1.44 (farthest, but kept: diverse direction from c0)
    ], dtype=np.float32)
    idx = HNSWIndex(dim=2, seed=0)
    idx.store = VectorStore(points)
    q = np.array([0.0, 0.0], dtype=np.float32)
    candidates = [(float(np.sum((points[i] - q) ** 2)), i) for i in range(3)]
    selected = idx._select_neighbors_heuristic(candidates, M=2)
    selected_ids = {node for _, node in selected}
    assert selected_ids == {0, 2}
    assert len(selected) == 2

def test_heuristic_respects_m_cap_with_many_diverse_candidates():
    rng = np.random.default_rng(0)
    points = rng.random((20, 4)).astype(np.float32)
    idx = HNSWIndex(dim=4, seed=0)
    idx.store = VectorStore(points)
    q = np.zeros(4, dtype=np.float32)
    candidates = [(float(np.sum((points[i] - q) ** 2)), i) for i in range(20)]
    selected = idx._select_neighbors_heuristic(candidates, M=5)
    assert len(selected) <= 5

def test_heuristic_returns_sorted_by_distance_ascending():
    rng = np.random.default_rng(1)
    points = rng.random((10, 4)).astype(np.float32)
    idx = HNSWIndex(dim=4, seed=0)
    idx.store = VectorStore(points)
    q = np.zeros(4, dtype=np.float32)
    candidates = [(float(np.sum((points[i] - q) ** 2)), i) for i in range(10)]
    selected = idx._select_neighbors_heuristic(candidates, M=10)
    dists = [d for d, _ in selected]
    assert dists == sorted(dists)
