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

def test_two_hop_expansion_reaches_matches_two_hops_from_the_nearest_neighbour():
    # star graph: hub 0 connects to 1..5; node 1 (far away) connects to match 6,
    # while nodes 2-5 (close) connect to non-matches 7-10. Natural traversal must
    # expand nodes 2-5 before node 1 (due to distance ordering), consuming ops.
    # With tight budget, the natural path exhausts ops expanding dead-ends before
    # reaching node 6. But two-hop expansion finds all neighbors-of-neighbors (including
    # node 6 via node 1) in iteration 1, finding node 6 without wasting ops on 2-5.
    idx = HNSWIndex(dim=2, seed=0)
    points = np.zeros((11, 2), dtype=np.float32)
    points[1] = [10.0, 0.0]  # node 1 is far
    points[6] = [0.01, 0.0]  # node 6 nearly coincides with query; clearly best match
    from vecdb.store.vectors import VectorStore
    idx.store = VectorStore(points)
    idx._visited_stamp = np.zeros(11, dtype=np.uint32)
    idx._visit_generation = 0
    idx._ensure_layers(0)
    idx.graph[0] = {
        0: [1, 2, 3, 4, 5],   # hub
        1: [0, 6],            # far node 1 -> match 6
        2: [0, 7],            # close node 2 -> non-match 7
        3: [0, 8],            # close node 3 -> non-match 8
        4: [0, 9],            # close node 4 -> non-match 9
        5: [0, 10],           # close node 5 -> non-match 10
        6: [1], 7: [2], 8: [3], 9: [4], 10: [5]
    }
    idx.entry_point = 0
    idx.max_level = 0

    mask = np.zeros(11, dtype=bool)
    mask[6] = True  # only node 6 matches
    q = np.array([0.0, 0.0], dtype=np.float32)

    # With two-hop enabled and budget=11: entry (1) + expand from 0 (5) + two-hop (5) = 11 ops.
    # Two-hop block's condition is ops_used=6 < 11? Yes, so it executes and finds node 6.
    results, ops_used = idx._search_layer_filtered(q, entry_points=[0], ef=1, layer=0,
                                                      mask=mask, budget_remaining=11, two_hop_threshold=0.5)
    assert results and results[0][1] == 6, f"Expected node 6 with two-hop, got {results}"

    # With two-hop disabled and same budget=11: entry (1) + expand from 0 (5 ops) = 6 ops.
    # Iteration 2 pops nodes 2,3,4,5 (closer), expanding to 7,8,9,10 (4 ops): total 10 ops.
    # Iteration 3 check: 10 < 11? Yes, but this time we pop node 1. Expanding to node 6 would
    # make ops_used=11, finding the match. So actually budget=11 is just enough for both paths.
    # We need a tighter budget. With budget=10: after nodes 2-5 expansion (10 ops total), no ops left.
    idx._visited_stamp = np.zeros(11, dtype=np.uint32)
    idx._visit_generation = 0
    results_disabled, ops_used_disabled = idx._search_layer_filtered(q, entry_points=[0], ef=1, layer=0,
                                                                       mask=mask, budget_remaining=10, two_hop_threshold=-1.0)
    assert not results_disabled, f"Expected no results with two-hop disabled and budget=10, got {results_disabled}"
