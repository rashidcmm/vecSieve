# tests/test_planner_lookup.py
"""Task 38 follow-up: Planner.from_lookup_table, the decile lookup-table fallback
(spec §8's "cost model doesn't fit" risk) — additive to the Task 36 cost-model
Planner, tested separately in tests/test_planner.py (unmodified)."""
from vecdb.planner.planner import Planner


def make_lookup_planner():
    return Planner.from_lookup_table({0.001: "pre_filter", 0.1: "post_filter", 0.5: "predicate_aware"})


def test_from_lookup_table_returns_exact_match_strategy_at_a_table_key():
    plan = make_lookup_planner().plan(k=10, sel_hat=0.1)
    assert plan.strategy == "post_filter"


def test_from_lookup_table_returns_nearest_strategy_between_two_keys():
    # 0.06 is closer to 0.1 (dist 0.04) than to 0.001 (dist 0.059)
    plan = make_lookup_planner().plan(k=10, sel_hat=0.06)
    assert plan.strategy == "post_filter"

    # 0.02 is closer to 0.001 (dist 0.019) than to 0.1 (dist 0.08)
    plan = make_lookup_planner().plan(k=10, sel_hat=0.02)
    assert plan.strategy == "pre_filter"


def test_from_lookup_table_reason_names_lookup_table_provenance_not_a_cost_model():
    plan = make_lookup_planner().plan(k=10, sel_hat=0.001)
    assert "lookup_table" in plan.reason


def test_from_lookup_table_costs_is_empty_dict():
    plan = make_lookup_planner().plan(k=10, sel_hat=0.001)
    assert plan.costs == {}


def test_from_lookup_table_records_the_selectivity_it_was_given():
    plan = make_lookup_planner().plan(k=10, sel_hat=0.02)
    assert plan.sel_hat == 0.02
