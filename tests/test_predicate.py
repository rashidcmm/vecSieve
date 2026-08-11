import numpy as np
import pytest
from vecdb.store.metadata import MetaStore
from vecdb.predicate.dsl import validate_predicate
from vecdb.predicate.compile import compile as compile_pred

@pytest.fixture
def meta():
    return MetaStore({
        "category": np.array([0, 1, 1, 2, 0], dtype=np.int32),
        "year": np.array([2018, 2020, 2021, 2019, 2022], dtype=np.int16),
    })

def test_eq_leaf(meta):
    mask = compile_pred({"op": "eq", "col": "category", "val": 1}, meta)
    np.testing.assert_array_equal(mask, [False, True, True, False, False])

def test_gt_leaf(meta):
    mask = compile_pred({"op": "gt", "col": "year", "val": 2019}, meta)
    np.testing.assert_array_equal(mask, [False, True, True, False, True])

def test_and_combines_with_logical_and(meta):
    pred = {"op": "and", "clauses": [
        {"op": "eq", "col": "category", "val": 1},
        {"op": "gt", "col": "year", "val": 2019},
    ]}
    mask = compile_pred(pred, meta)
    np.testing.assert_array_equal(mask, [False, True, True, False, False])

def test_or_combines_with_logical_or(meta):
    pred = {"op": "or", "clauses": [
        {"op": "eq", "col": "category", "val": 2},
        {"op": "eq", "col": "category", "val": 0},
    ]}
    mask = compile_pred(pred, meta)
    np.testing.assert_array_equal(mask, [True, False, False, True, True])

def test_not_inverts(meta):
    mask = compile_pred({"op": "not", "clause": {"op": "eq", "col": "category", "val": 1}}, meta)
    np.testing.assert_array_equal(mask, [True, False, False, True, True])

def test_in_op(meta):
    mask = compile_pred({"op": "in", "col": "category", "val": [0, 2]}, meta)
    np.testing.assert_array_equal(mask, [True, False, False, True, True])

def test_empty_mask_when_no_rows_match(meta):
    mask = compile_pred({"op": "eq", "col": "category", "val": 999}, meta)
    assert mask.sum() == 0
    assert mask.shape == (5,)

def test_full_mask_for_always_true_predicate(meta):
    mask = compile_pred({"op": "gte", "col": "year", "val": 0}, meta)
    assert mask.sum() == 5

def test_validate_predicate_rejects_unknown_op():
    with pytest.raises(ValueError):
        validate_predicate({"op": "xor", "col": "category", "val": 1})

def test_validate_predicate_accepts_nested_valid_predicate():
    pred = {"op": "and", "clauses": [
        {"op": "eq", "col": "category", "val": 1},
        {"op": "not", "clause": {"op": "lt", "col": "year", "val": 2000}},
    ]}
    validate_predicate(pred)  # should not raise
