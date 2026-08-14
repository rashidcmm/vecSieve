import numpy as np
from vecdb.index.hnsw import HNSWIndex

def test_approx_size_bytes_counts_adjacency_entries_only():
    idx = HNSWIndex(dim=4, seed=0)
    idx.store = None
    from vecdb.store.vectors import VectorStore
    idx.store = VectorStore(np.zeros((3, 4), dtype=np.float32))
    idx._ensure_layers(0)
    idx.graph[0] = {0: [1, 2], 1: [0], 2: [0]}  # 4 total adjacency entries
    assert idx.approx_size_bytes() == 4 * 4  # 4 entries * 4 bytes (int32)

def test_approx_size_bytes_grows_with_a_real_build():
    rng = np.random.default_rng(0)
    data = rng.random((100, 8)).astype(np.float32)
    idx = HNSWIndex(dim=8, M=8, ef_construction=50, seed=0)
    idx.add(data, np.arange(100))
    assert idx.approx_size_bytes() > 0
