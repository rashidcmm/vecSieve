import pandas as pd
import pytest
from vecdb.bench.regret import compute_regret, regret_summary

def test_regret_is_zero_when_planner_always_picks_the_best():
    df = pd.DataFrame({
        "pre_filter_latency_ms": [10.0, 5.0, 8.0],
        "post_filter_latency_ms": [20.0, 3.0, 9.0],
        "predicate_aware_latency_ms": [15.0, 4.0, 7.0],
        "planner_latency_ms": [10.0, 3.0, 7.0],  # always the min of the three
    })
    regret = compute_regret(df, ["pre_filter", "post_filter", "predicate_aware"], "planner_latency_ms")
    assert (regret == 0).all()

def test_regret_is_positive_when_planner_picks_worse_option():
    df = pd.DataFrame({
        "pre_filter_latency_ms": [10.0],
        "post_filter_latency_ms": [20.0],
        "predicate_aware_latency_ms": [15.0],
        "planner_latency_ms": [20.0],  # picked the worst of the three
    })
    regret = compute_regret(df, ["pre_filter", "post_filter", "predicate_aware"], "planner_latency_ms")
    assert regret.iloc[0] == pytest.approx(10.0)  # 20 - min(10,20,15)

def test_regret_summary_reports_mean_and_p95():
    df = pd.DataFrame({
        "pre_filter_latency_ms": [10.0] * 100,
        "post_filter_latency_ms": [10.0] * 100,
        "predicate_aware_latency_ms": [10.0] * 100,
        "planner_latency_ms": [10.0] * 99 + [50.0],  # one bad outlier
    })
    regret = compute_regret(df, ["pre_filter", "post_filter", "predicate_aware"], "planner_latency_ms")
    summary = regret_summary(regret)
    assert summary["mean_regret_ms"] > 0
    assert summary["p95_regret_ms"] >= 0
