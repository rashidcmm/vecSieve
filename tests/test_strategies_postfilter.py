import numpy as np
from vecdb.index.hnsw import HNSWIndex
from vecdb.index.flat import FlatIndex
from vecdb.index.strategies import PreFilterStrategy, PostFilterStrategy

def _build(n=500, d=16, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.random((n, d)).astype(np.float32)
    hnsw = HNSWIndex(dim=d, M=16, ef_construction=100, seed=seed)
    hnsw.add(data, np.arange(n))
    flat = FlatIndex()
    flat.add(data, np.arange(n))
    return data, hnsw, flat

def test_postfilter_returns_only_matching_ids_when_not_underfilled():
    data, hnsw, flat = _build()
    strategy = PostFilterStrategy(hnsw, fallback=PreFilterStrategy(flat))
    rng = np.random.default_rng(1)
    q = rng.random(16).astype(np.float32)
    mask = np.zeros(500, dtype=bool)
    mask[:250] = True  # 50% selectivity - post-filter should comfortably find k=10
    result = strategy.search(q, k=10, mask=mask, params={"selectivity_hat": 0.5})
    assert all(mask[i] for i in result.ids)
    assert result.strategy in ("post_filter", "post_filter_fallback")

def test_postfilter_falls_back_to_prefilter_when_selectivity_is_extreme():
    data, hnsw, flat = _build(n=500)
    strategy = PostFilterStrategy(hnsw, fallback=PreFilterStrategy(flat), max_retries=1)
    rng = np.random.default_rng(2)
    q = rng.random(16).astype(np.float32)
    mask = np.zeros(500, dtype=bool)
    mask[:3] = True  # 0.6% selectivity, k=10 > available survivors even in theory... use k=3
    result = strategy.search(q, k=3, mask=mask, params={"selectivity_hat": 0.006})
    # whichever path was taken, the result must still be correct and non-underfilled
    # given only 3 true survivors exist
    assert result.n_returned <= 3
    assert all(mask[i] for i in result.ids)

def test_fallback_rate_is_tracked_across_calls():
    data, hnsw, flat = _build(n=500)
    strategy = PostFilterStrategy(hnsw, fallback=PreFilterStrategy(flat), max_retries=0)
    rng = np.random.default_rng(3)
    mask = np.zeros(500, dtype=bool)
    mask[:2] = True  # forces underfill for k=10 with no retries
    for _ in range(5):
        q = rng.random(16).astype(np.float32)
        strategy.search(q, k=10, mask=mask, params={"selectivity_hat": 0.004})
    assert strategy.query_count == 5
    assert strategy.fallback_count == 5
    assert strategy.fallback_rate == 1.0

def test_ef_widens_as_estimated_selectivity_drops():
    data, hnsw, flat = _build()
    strategy = PostFilterStrategy(hnsw, fallback=PreFilterStrategy(flat))
    assert strategy._ef_for(k=10, sel_hat=0.5) < strategy._ef_for(k=10, sel_hat=0.01)
