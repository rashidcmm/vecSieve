import math
import numpy as np
from vecdb.index.hnsw import HNSWIndex

def test_init_sets_expected_defaults():
    idx = HNSWIndex(dim=8, M=16, ef_construction=200, seed=0)
    assert idx.M == 16
    assert idx.M0 == 32
    assert idx.entry_point == -1
    assert idx.max_level == -1
    assert idx.graph == []
    assert idx.mL == pytest_approx(1.0 / math.log(16))

def pytest_approx(x, tol=1e-9):
    class _Approx(float):
        def __eq__(self, other):
            return abs(other - x) < tol
    return _Approx(x)

def test_assign_level_distribution_matches_geometric_decay():
    # With M=16, P(level >= 1) should be roughly 1/16 ~ 6%. Sample many draws.
    idx = HNSWIndex(dim=8, M=16, seed=0)
    levels = [idx._assign_level() for _ in range(20_000)]
    frac_at_least_1 = sum(1 for l in levels if l >= 1) / len(levels)
    assert 0.03 < frac_at_least_1 < 0.10  # loose bound around the theoretical 1/16
    assert min(levels) == 0  # level 0 must be the overwhelming majority

def test_ensure_layers_grows_graph_list_to_cover_level():
    idx = HNSWIndex(dim=8, seed=0)
    idx._ensure_layers(3)
    assert len(idx.graph) == 4  # layers 0,1,2,3
    assert all(isinstance(layer, dict) for layer in idx.graph)
