import numpy as np
import pytest
from vecdb.store.metadata import MetaStore
from vecdb.predicate.selectivity import estimate_selectivity

@pytest.fixture
def meta():
    rng = np.random.default_rng(0)
    return MetaStore({
        "category": rng.integers(0, 10, size=10_000).astype(np.int32),
        "score": rng.uniform(0, 1, size=10_000).astype(np.float32),
    })

def test_categorical_eq_matches_value_count_fraction(meta):
    s = estimate_selectivity({"op": "eq", "col": "category", "val": 3}, meta)
    expected = meta.stats["category"].value_counts.get(3, 0) / meta.n
    assert s == pytest.approx(expected)

def test_categorical_eq_missing_value_is_zero(meta):
    s = estimate_selectivity({"op": "eq", "col": "category", "val": 999}, meta)
    assert s == 0.0

def test_numeric_range_is_close_to_true_uniform_fraction(meta):
    # score ~ U(0,1) with 10k samples: P(score < 0.3) should estimate near 0.3
    s = estimate_selectivity({"op": "lt", "col": "score", "val": 0.3}, meta)
    assert abs(s - 0.3) < 0.03

def test_and_uses_independence_assumption(meta):
    pred = {"op": "and", "clauses": [
        {"op": "eq", "col": "category", "val": 3},
        {"op": "lt", "col": "score", "val": 0.5},
    ]}
    s = estimate_selectivity(pred, meta)
    s1 = estimate_selectivity(pred["clauses"][0], meta)
    s2 = estimate_selectivity(pred["clauses"][1], meta)
    assert s == pytest.approx(s1 * s2, rel=1e-6)

def test_or_uses_inclusion_exclusion(meta):
    pred = {"op": "or", "clauses": [
        {"op": "eq", "col": "category", "val": 1},
        {"op": "eq", "col": "category", "val": 2},
    ]}
    s = estimate_selectivity(pred, meta)
    s1 = estimate_selectivity(pred["clauses"][0], meta)
    s2 = estimate_selectivity(pred["clauses"][1], meta)
    assert s == pytest.approx(s1 + s2 - s1 * s2, rel=1e-6)

def test_not_is_one_minus_selectivity(meta):
    inner = {"op": "eq", "col": "category", "val": 3}
    s_not = estimate_selectivity({"op": "not", "clause": inner}, meta)
    s_inner = estimate_selectivity(inner, meta)
    assert s_not == pytest.approx(1.0 - s_inner)

def test_estimate_never_touches_columns_directly(meta):
    # sabotage the raw column so a correct implementation (stats-only) is unaffected
    meta.columns["category"] = None
    s = estimate_selectivity({"op": "eq", "col": "category", "val": 3}, meta)
    assert 0.0 <= s <= 1.0
