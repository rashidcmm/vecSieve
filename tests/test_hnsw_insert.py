import numpy as np
from vecdb.index.hnsw import HNSWIndex
from vecdb.index.flat import FlatIndex

def test_insert_first_node_becomes_entry_point():
    idx = HNSWIndex(dim=4, seed=0)
    idx.add(np.zeros((1, 4), dtype=np.float32), np.array([0]))
    assert idx.entry_point == 0
    assert idx.max_level >= 0

def test_graph_is_mostly_bidirectionally_linked_after_insert():
    # HNSW's degree-cap re-pruning (invoked from _insert when a neighbour's adjacency list
    # exceeds its cap) only shrinks the *pruned* node's own list; it never revisits the list
    # of a node that got dropped from it. This is documented HNSW behaviour (Malkov &
    # Yashunin) -- the graph can carry a small number of asymmetric edges after re-pruning,
    # confirmed here on this exact fixture (e.g. edge 0->3 survives node 3's re-prune when
    # node 28 links in, but the reverse 3->0 is legitimately dropped by the diversity
    # heuristic). So we assert the overwhelming majority of layer-0 edges are reciprocated,
    # rather than requiring every single one to be -- the algorithm doesn't guarantee that.
    rng = np.random.default_rng(0)
    data = rng.random((30, 4)).astype(np.float32)
    idx = HNSWIndex(dim=4, M=4, ef_construction=20, seed=0)
    idx.add(data, np.arange(30))
    total = 0
    reciprocated = 0
    for node, neighbours in idx.graph[0].items():
        for n in neighbours:
            total += 1
            if node in idx.graph[0].get(n, []):
                reciprocated += 1
    assert total > 0
    assert reciprocated / total >= 0.9, f"only {reciprocated}/{total} layer-0 edges reciprocated"

def test_degree_cap_is_respected_after_repruning():
    rng = np.random.default_rng(0)
    data = rng.random((60, 4)).astype(np.float32)
    idx = HNSWIndex(dim=4, M=4, ef_construction=20, seed=0)
    idx.add(data, np.arange(60))
    for node, neighbours in idx.graph[0].items():
        assert len(neighbours) <= idx.M0

def test_unfiltered_recall_at_10_is_reasonably_high_on_tiny_data():
    rng = np.random.default_rng(0)
    data = rng.random((300, 16)).astype(np.float32)
    hnsw = HNSWIndex(dim=16, M=16, ef_construction=100, seed=0)
    hnsw.add(data, np.arange(300))
    flat = FlatIndex()
    flat.add(data, np.arange(300))

    queries = rng.random((30, 16)).astype(np.float32)
    hits = 0
    for q in queries:
        true_ids = set(flat.search(q, k=10).ids.tolist())
        got_ids = set(hnsw.search(q, k=10, params={"ef": 100}).ids.tolist())
        hits += len(true_ids & got_ids)
    recall = hits / (30 * 10)
    assert recall >= 0.85  # loose bound at this tiny scale; the real gate is Task 19 at 10K

def test_search_rejects_a_mask_argument():
    import pytest
    idx = HNSWIndex(dim=4, seed=0)
    idx.add(np.random.default_rng(0).random((5, 4)).astype(np.float32), np.arange(5))
    with pytest.raises(AssertionError):
        idx.search(np.zeros(4, dtype=np.float32), k=1, mask=np.array([True] * 5))
