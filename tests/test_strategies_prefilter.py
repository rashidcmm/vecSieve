# tests/test_strategies_prefilter.py
import numpy as np
from vecdb.index.flat import FlatIndex
from vecdb.index.strategies import PreFilterStrategy

def test_prefilter_matches_flat_index_exactly_and_relabels_strategy():
    rng = np.random.default_rng(0)
    data = rng.random((200, 8)).astype(np.float32)
    flat = FlatIndex()
    flat.add(data, np.arange(200))
    strategy = PreFilterStrategy(flat)

    q = rng.random(8).astype(np.float32)
    mask = rng.random(200) < 0.2
    direct = flat.search(q, k=10, mask=mask)
    via_strategy = strategy.search(q, k=10, mask=mask)

    np.testing.assert_array_equal(direct.ids, via_strategy.ids)
    assert via_strategy.strategy == "pre_filter"
    assert via_strategy.n_distance_ops == direct.n_distance_ops
