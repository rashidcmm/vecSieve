# tests/test_sweep.py
import numpy as np
from vecdb.store.metadata import MetaStore
from vecdb.bench.sweep import predicate_for_selectivity, SELECTIVITY_GRID
from vecdb.predicate.compile import compile as compile_pred

def test_predicate_for_selectivity_hits_target_within_tolerance():
    rng = np.random.default_rng(0)
    meta = MetaStore({"score": rng.uniform(0, 1, size=20_000).astype(np.float32)})
    for target in SELECTIVITY_GRID:
        pred = predicate_for_selectivity(meta, target)
        mask = compile_pred(pred, meta)
        actual = mask.sum() / meta.n
        assert abs(actual - target) < 0.02, f"target={target} got={actual}"

def test_selectivity_grid_is_the_spec_defined_values():
    assert SELECTIVITY_GRID == [0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
