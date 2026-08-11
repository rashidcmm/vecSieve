import numpy as np
from vecdb.store.vectors import VectorStore

def test_distances_match_brute_force_l2_squared():
    rng = np.random.default_rng(0)
    data = rng.random((50, 8)).astype(np.float32)
    store = VectorStore(data)
    q = rng.random(8).astype(np.float32)
    rows = np.array([0, 10, 25, 49])
    got = store.distances(q, rows)
    expected = np.sum((data[rows] - q) ** 2, axis=1)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

def test_distances_increments_counter():
    data = np.zeros((10, 4), dtype=np.float32)
    store = VectorStore(data)
    assert store.n_distance_ops == 0
    store.distances(np.zeros(4, dtype=np.float32), np.array([0, 1, 2]))
    assert store.n_distance_ops == 3
    store.distances(np.zeros(4, dtype=np.float32), np.array([3]))
    assert store.n_distance_ops == 4

def test_distances_empty_rows_returns_empty_array_and_no_crash():
    data = np.ones((5, 4), dtype=np.float32)
    store = VectorStore(data)
    result = store.distances(np.zeros(4, dtype=np.float32), np.array([], dtype=np.int64))
    assert result.shape == (0,)
    assert store.n_distance_ops == 0

def test_vector_returns_single_row():
    data = np.arange(20, dtype=np.float32).reshape(5, 4)
    store = VectorStore(data)
    np.testing.assert_array_equal(store.vector(2), data[2])
