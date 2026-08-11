import numpy as np
from vecdb.index.flat import FlatIndex

def test_unfiltered_search_matches_brute_force_ranking():
    rng = np.random.default_rng(0)
    data = rng.random((200, 16)).astype(np.float32)
    idx = FlatIndex()
    idx.add(data, np.arange(200))
    q = rng.random(16).astype(np.float32)
    result = idx.search(q, k=5)
    expected_order = np.argsort(np.sum((data - q) ** 2, axis=1))[:5]
    np.testing.assert_array_equal(result.ids, expected_order)
    assert result.n_returned == 5
    assert result.n_distance_ops == 200

def test_masked_search_only_returns_matching_rows():
    data = np.arange(40, dtype=np.float32).reshape(10, 4)
    idx = FlatIndex()
    idx.add(data, np.arange(10))
    mask = np.zeros(10, dtype=bool)
    mask[[2, 5, 7]] = True
    result = idx.search(np.zeros(4, dtype=np.float32), k=10, mask=mask)
    assert set(result.ids.tolist()) == {2, 5, 7}
    assert result.n_returned == 3  # fewer than k because only 3 rows match
    assert result.n_distance_ops == 3

def test_empty_mask_returns_empty_result_without_crash():
    data = np.ones((5, 4), dtype=np.float32)
    idx = FlatIndex()
    idx.add(data, np.arange(5))
    mask = np.zeros(5, dtype=bool)
    result = idx.search(np.zeros(4, dtype=np.float32), k=3, mask=mask)
    assert result.n_returned == 0
    assert result.ids.shape == (0,)
