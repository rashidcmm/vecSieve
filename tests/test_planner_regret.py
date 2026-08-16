# tests/test_planner_regret.py
from pathlib import Path
import pandas as pd
import pytest
from vecdb.bench.regret import compute_regret, regret_summary

SWEEP_PATH = Path("results/sweep_uncorrelated.csv")

@pytest.mark.skipif(not SWEEP_PATH.exists(), reason="requires scripts/run_full_sweep.py to have been run")
def test_mean_regret_is_bounded_relative_to_oracle():
    df = pd.read_csv(SWEEP_PATH)
    wide = (
        df[df["strategy"].isin(["pre_filter", "post_filter", "predicate_aware", "planner"])]
        .groupby(["query_idx", "target_selectivity", "strategy"])["latency_ms"]
        .mean()
        .unstack("strategy")
        .rename(columns={
            "pre_filter": "pre_filter_latency_ms",
            "post_filter": "post_filter_latency_ms",
            "predicate_aware": "predicate_aware_latency_ms",
            "planner": "planner_latency_ms",
        })
        .dropna()
    )
    regret = compute_regret(wide, ["pre_filter", "post_filter", "predicate_aware"], "planner_latency_ms")
    summary = regret_summary(regret)
    oracle_mean = wide[["pre_filter_latency_ms", "post_filter_latency_ms", "predicate_aware_latency_ms"]].min(axis=1).mean()
    # spec §5 target: mean regret < 15% of the oracle's mean latency
    assert summary["mean_regret_ms"] < 0.15 * oracle_mean, (
        f"mean regret {summary['mean_regret_ms']:.3f}ms exceeds 15% of oracle mean {oracle_mean:.3f}ms — "
        "this is a real finding, report it in the milestone/README rather than loosening the threshold"
    )
