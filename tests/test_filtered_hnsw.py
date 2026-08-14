import numpy as np
from vecdb.index.hnsw import HNSWIndex
from vecdb.index.flat import FlatIndex
from vecdb.index.strategies import PreFilterStrategy, FilteredHNSWStrategy
from vecdb.store.vectors import VectorStore

def _chain_index(n=20, d=2):
    """A chain graph 0-1-2-...-(n-1) on layer 0, so traversal must cross several
    non-matching nodes to reach a distant match."""
    points = np.array([[float(i), 0.0] for i in range(n)], dtype=np.float32)
    idx = HNSWIndex(dim=d, seed=0)
    idx.store = VectorStore(points)
    idx._visited_stamp = np.zeros(n, dtype=np.uint32)
    idx._visit_generation = 0
    idx._ensure_layers(0)
    idx.levels = [0] * n
    for i in range(n):
        idx.graph[0][i] = [j for j in (i - 1, i + 1) if 0 <= j < n]
    idx.entry_point = 0
    idx.max_level = 0
    return idx, points

def test_two_tier_admission_only_returns_matching_nodes():
    idx, points = _chain_index()
    mask = np.zeros(20, dtype=bool)
    mask[[15, 16, 17]] = True  # far from entry point 0; must be crossed-through to reach
    results, ops_used = idx._search_layer_filtered(
        np.array([16.0, 0.0], dtype=np.float32), entry_points=[0], ef=3, layer=0, mask=mask,
    )
    returned_ids = {node for _, node in results}
    assert returned_ids.issubset({15, 16, 17})
    assert ops_used > 3  # had to traverse through non-matching nodes to get there

def test_search_finds_matches_via_seeded_entry_point_even_when_far_from_global_entry():
    idx, points = _chain_index(n=50)
    flat = FlatIndex()
    flat.add(points, np.arange(50))
    strategy = FilteredHNSWStrategy(idx, fallback=PreFilterStrategy(flat), n_seed_matches=4, seed=1)
    mask = np.zeros(50, dtype=bool)
    mask[45:50] = True  # far from entry_point=0
    result = strategy.search(np.array([47.0, 0.0], dtype=np.float32), k=3, mask=mask,
                               params={"selectivity_hat": 0.1})
    assert all(mask[i] for i in result.ids)
    assert result.n_returned == 3

def test_ef_eff_widens_as_selectivity_drops():
    idx, points = _chain_index()
    flat = FlatIndex(); flat.add(points, np.arange(20))
    strategy = FilteredHNSWStrategy(idx, fallback=PreFilterStrategy(flat))
    assert strategy._ef_eff(sel_hat=0.5) < strategy._ef_eff(sel_hat=0.05)
