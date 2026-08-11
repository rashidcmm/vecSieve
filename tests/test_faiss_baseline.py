import numpy as np
from vecdb.index.faiss_baseline import FaissFlatIndex, FaissHNSWIndex

def test_faiss_flat_matches_brute_force_exactly():
    rng = np.random.default_rng(0)
    data = rng.random((300, 16)).astype(np.float32)
    idx = FaissFlatIndex(dim=16)
    idx.add(data, np.arange(300))
    q = rng.random(16).astype(np.float32)
    result = idx.search(q, k=5)
    expected_order = np.argsort(np.sum((data - q) ** 2, axis=1))[:5]
    np.testing.assert_array_equal(result.ids, expected_order)
    assert result.n_distance_ops == 300

def test_faiss_hnsw_returns_k_results_and_positive_dist_ops():
    rng = np.random.default_rng(0)
    data = rng.random((500, 16)).astype(np.float32)
    idx = FaissHNSWIndex(dim=16, M=16, ef_construction=100)
    idx.add(data, np.arange(500))
    idx.set_ef_search(64)
    q = rng.random(16).astype(np.float32)
    result = idx.search(q, k=10)
    assert result.n_returned == 10
    assert result.n_distance_ops > 0
    assert result.strategy == "faiss_hnsw"
