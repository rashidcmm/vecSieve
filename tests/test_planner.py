# tests/test_planner.py
from vecdb.planner.cost_model import CostModelParams
from vecdb.planner.planner import Planner

def make_planner():
    params = CostModelParams(c_scan=1.0, c_hop=0.01, alpha=4.0, beta=5.0,
                               ef_min=16, ef_base=64, M=16, N=100_000)
    return Planner(params)

def test_pre_filter_wins_at_very_low_selectivity():
    plan = make_planner().plan(k=10, sel_hat=0.0005)
    assert plan.strategy == "pre_filter"

def test_plan_reason_names_the_chosen_strategy_and_its_cost():
    plan = make_planner().plan(k=10, sel_hat=0.0005)
    assert plan.strategy in plan.reason
    assert "ŝ=" in plan.reason

def test_plan_reports_all_three_costs():
    plan = make_planner().plan(k=10, sel_hat=0.1)
    assert set(plan.costs) == {"pre_filter", "post_filter", "predicate_aware"}
    assert plan.costs[plan.strategy] == min(plan.costs.values())

def test_plan_records_the_selectivity_it_was_given():
    plan = make_planner().plan(k=10, sel_hat=0.02)
    assert plan.sel_hat == 0.02
