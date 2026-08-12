import numpy as np
from vecdb.index.hnsw import HNSWIndex
from vecdb.store.vectors import VectorStore

def test_heuristic_prefers_diverse_directions_over_pure_nearest():
    # Query at origin. Two candidates are close together in the same direction (c0, c1,
    # nearly collinear with the query) and one is farther but in a different direction (c2).
    # Naive top-2-nearest would pick {c0, c1}; the heuristic should reject c1 because it's
    # closer to c0 than to the query, and pick c2 instead for diversity.
    points = np.array([
        [1.0, 0.0],   # c0: dist^2 to origin = 1
        [1.1, 0.0],   # c1: dist^2 to origin = 1.21, but very close to c0
        [0.0, 1.05],  # c2: dist^2 to origin = 1.1025, different direction from c0
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
