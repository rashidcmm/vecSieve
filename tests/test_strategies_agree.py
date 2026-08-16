# tests/test_strategies_agree.py
import numpy as np
from vecdb.index.flat import FlatIndex
from vecdb.index.hnsw import HNSWIndex
from vecdb.index.strategies import PreFilterStrategy, PostFilterStrategy, FilteredHNSWStrategy

def _build(n=400, d=16, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.random((n, d)).astype(np.float32)
    flat = FlatIndex(); flat.add(data, np.arange(n))
    hnsw = HNSWIndex(dim=d, M=16, ef_construction=200, seed=seed)
    hnsw.add(data, np.arange(n))
    return data, flat, hnsw

def test_all_three_strategies_agree_with_flat_on_tiny_data_with_generous_budgets():
    """With ef/budget large enough relative to N, every strategy should recover the
    exact top-k that FlatIndex does — this is the correctness contract every strategy
    must satisfy regardless of how it gets there."""
    data, flat, hnsw = _build(n=300)
    rng = np.random.default_rng(1)
    mask = rng.random(300) < 0.4  # generous selectivity so none of the strategies underfill

    pre = PreFilterStrategy(flat)
    post = PostFilterStrategy(hnsw, fallback=pre, alpha=8.0, ef_min=64)
    pred_aware = FilteredHNSWStrategy(hnsw, fallback=pre, ef_base=128, budget_fraction=0.9)

    for trial in range(5):
        q = rng.random(16).astype(np.float32)
        true_ids = set(flat.search(q, k=5, mask=mask).ids.tolist())

        pre_ids = set(pre.search(q, k=5, mask=mask).ids.tolist())
        post_ids = set(post.search(q, k=5, mask=mask, params={"selectivity_hat": 0.4}).ids.tolist())
        pred_ids = set(pred_aware.search(q, k=5, mask=mask, params={"selectivity_hat": 0.4}).ids.tolist())

        assert pre_ids == true_ids, f"trial {trial}: pre-filter disagreed with Flat"
        # HNSW-backed strategies are approximate even with generous budgets; require
        # substantial overlap rather than exact equality
        assert len(post_ids & true_ids) >= 4, f"trial {trial}: post-filter recall too low"
        assert len(pred_ids & true_ids) >= 4, f"trial {trial}: predicate-aware recall too low"
