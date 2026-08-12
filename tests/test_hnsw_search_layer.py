import numpy as np
from vecdb.index.hnsw import HNSWIndex

def _build_line_graph(idx: HNSWIndex, points: np.ndarray) -> None:
    """A toy graph: node i connects to i-1 and i+1 (a chain), all on layer 0."""
    idx.store = None
    from vecdb.store.vectors import VectorStore
    idx.store = VectorStore(points)
    idx._ensure_layers(0)
    n = points.shape[0]
    for i in range(n):
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < n]
        idx.graph[0][i] = neighbours

def test_search_layer_finds_exact_nearest_on_a_chain():
    # points laid out on a 1D line: 0,1,2,...,9 (as (x,0) vectors) with chain edges
    points = np.array([[float(i), 0.0] for i in range(10)], dtype=np.float32)
    idx = HNSWIndex(dim=2, seed=0)
    _build_line_graph(idx, points)
    q = np.array([7.4, 0.0], dtype=np.float32)
    result = idx._search_layer(q, entry_points=[0], ef=3, layer=0)
    assert result[0][1] == 7  # nearest point to x=7.4 is node 7
    assert len(result) == 3

def test_search_layer_returns_sorted_ascending_by_distance():
    points = np.array([[float(i), 0.0] for i in range(10)], dtype=np.float32)
    idx = HNSWIndex(dim=2, seed=0)
    _build_line_graph(idx, points)
    result = idx._search_layer(np.array([3.0, 0.0], dtype=np.float32), entry_points=[0], ef=5, layer=0)
    dists = [d for d, _ in result]
    assert dists == sorted(dists)

def test_search_layer_respects_ef_budget():
    points = np.array([[float(i), 0.0] for i in range(10)], dtype=np.float32)
    idx = HNSWIndex(dim=2, seed=0)
    _build_line_graph(idx, points)
    result = idx._search_layer(np.array([5.0, 0.0], dtype=np.float32), entry_points=[0], ef=2, layer=0)
    assert len(result) <= 2

def test_search_layer_single_isolated_entry_point_returns_itself():
    points = np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
    idx = HNSWIndex(dim=2, seed=0)
    from vecdb.store.vectors import VectorStore
    idx.store = VectorStore(points)
    idx._ensure_layers(0)
    idx.graph[0][0] = []  # no neighbours at all
    result = idx._search_layer(np.array([0.0, 0.0], dtype=np.float32), entry_points=[0], ef=5, layer=0)
    assert result == [(0.0, 0)]
