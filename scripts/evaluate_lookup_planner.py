# scripts/evaluate_lookup_planner.py
"""Task 38 follow-up: fit and evaluate the decile lookup-table planner fallback
(spec §8's documented "cost model doesn't fit" risk) purely from data already on
disk (results/sweep_uncorrelated.csv, results/sweep_correlated.csv) — no new
100K-scale benchmark run.

Calibration/evaluation split (by query_idx, disjoint, so the table is never fit
and evaluated on the same queries):
    query_idx <  110  ->  calibration
    query_idx >= 110  ->  evaluation

The full query_idx range present in the sweep CSVs is 20-199 (180 unique
values, uniform across strategies and selectivities). The split above yields
90 calibration query_idx values (20..109) and 90 evaluation query_idx values
(110..199).

The lookup table is keyed by `sel_hat` (the estimated selectivity a real
Planner.plan() call receives at query time) rather than `target_selectivity`
(the sweep's nominal knob, not something the planner observes in production).
Within one metadata variant, sel_hat is a deterministic function of
target_selectivity (one predicate per grid point), so this is equivalent to
keying by target_selectivity for the 8 grid points evaluated here, but it
keeps the table's key space aligned with Planner.plan()'s actual input
contract.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from vecdb.bench.regret import compute_regret, regret_summary

FIXED_STRATEGIES = ["pre_filter", "post_filter", "predicate_aware"]
CALIBRATION_MAX_QUERY_IDX = 110  # query_idx < 110 -> calibration, >= 110 -> evaluation


def fit_lookup_table(calib: pd.DataFrame) -> dict[float, str]:
    """For each target_selectivity grid point, the empirically fastest fixed
    strategy on the calibration half, keyed by that grid point's sel_hat."""
    table: dict[float, str] = {}
    for target_s, group in calib.groupby("target_selectivity"):
        mean_latency = group.groupby("strategy")["latency_ms"].mean()
        best_strategy = mean_latency.idxmin()
        sel_hat = group["sel_hat"].iloc[0]
        table[float(sel_hat)] = best_strategy
    return table


def realized_planner_latencies(df_half: pd.DataFrame, lookup_table: dict[float, str]) -> pd.DataFrame:
    """For each row of df_half (a fixed-strategy measurement), attach the strategy
    the lookup table would have picked for that row's sel_hat, then keep only the
    rows whose own strategy matches that pick — i.e. the "planner" realized
    latencies are exactly the chosen strategy's own measured latency_ms values."""
    def pick(sel_hat: float) -> str:
        return min(lookup_table, key=lambda point: abs(point - sel_hat))

    chosen = df_half["sel_hat"].map(lambda s: lookup_table[pick(s)])
    return df_half[df_half["strategy"] == chosen]


def gate_check(eval_half: pd.DataFrame, lookup_table: dict[float, str], label: str) -> dict:
    mean_fixed = {s: eval_half[eval_half["strategy"] == s]["latency_ms"].mean() for s in FIXED_STRATEGIES}
    planner_rows = realized_planner_latencies(eval_half, lookup_table)
    mean_planner = planner_rows["latency_ms"].mean()
    gate_holds = mean_planner < min(mean_fixed.values())

    print(f"\n--- gate check: {label} ---")
    print("mean latency (ms) per fixed strategy on evaluation half:")
    for s, v in mean_fixed.items():
        print(f"  {s}: {v:.6f}")
    print(f"mean lookup-table 'planner' latency (ms) on evaluation half: {mean_planner:.6f}")
    print(f"min fixed mean: {min(mean_fixed.values()):.6f}")
    print(f"Phase 7 gate (mean_planner_eval < min(mean_fixed_eval)): {gate_holds}")

    return {"mean_fixed": mean_fixed, "mean_planner": mean_planner, "gate_holds": gate_holds}


def regret_on_half(eval_half: pd.DataFrame, lookup_table: dict[float, str], label: str) -> dict:
    wide = (
        eval_half.groupby(["query_idx", "target_selectivity", "strategy"])["latency_ms"]
        .mean()
        .unstack("strategy")
    )
    wide = wide.rename(columns={s: f"{s}_latency_ms" for s in FIXED_STRATEGIES})

    # sel_hat is constant per target_selectivity within a variant; pick the strategy
    # the lookup table selects for each target_selectivity's sel_hat, then read that
    # strategy's own latency out of the same wide row.
    sel_hat_by_target = eval_half.groupby("target_selectivity")["sel_hat"].first()

    def pick(sel_hat: float) -> str:
        nearest = min(lookup_table, key=lambda point: abs(point - sel_hat))
        return lookup_table[nearest]

    planner_latency_ms = []
    for (_, target_s), row in wide.iterrows():
        strategy = pick(sel_hat_by_target[target_s])
        planner_latency_ms.append(row[f"{strategy}_latency_ms"])
    wide["planner_latency_ms"] = planner_latency_ms

    regret = compute_regret(wide, FIXED_STRATEGIES, "planner_latency_ms")
    summary = regret_summary(regret)
    print(f"\n--- regret summary: {label} ---")
    print(summary)
    return summary


def load_fixed_strategy_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["strategy"].isin(FIXED_STRATEGIES)].copy()


def main() -> None:
    uncorrelated = load_fixed_strategy_rows(Path("results/sweep_uncorrelated.csv"))

    calib = uncorrelated[uncorrelated["query_idx"] < CALIBRATION_MAX_QUERY_IDX]
    evaluation = uncorrelated[uncorrelated["query_idx"] >= CALIBRATION_MAX_QUERY_IDX]
    print(f"calibration query_idx range: {calib['query_idx'].min()}..{calib['query_idx'].max()} "
          f"({calib['query_idx'].nunique()} unique)")
    print(f"evaluation query_idx range: {evaluation['query_idx'].min()}..{evaluation['query_idx'].max()} "
          f"({evaluation['query_idx'].nunique()} unique)")

    lookup_table = fit_lookup_table(calib)
    print("\nfitted lookup table (sel_hat -> strategy):")
    for sel_hat, strategy in sorted(lookup_table.items()):
        print(f"  {sel_hat:.6f} -> {strategy}")

    # Primary gate check: held-out evaluation half of the uncorrelated sweep — the
    # same data the lookup table was calibrated from, but disjoint queries.
    primary = gate_check(evaluation, lookup_table, "uncorrelated evaluation half (primary)")
    primary_regret = regret_on_half(evaluation, lookup_table, "uncorrelated evaluation half (primary)")

    # Secondary, exploratory check: does the uncorrelated-fitted table generalize to
    # the correlated metadata variant? Same query_idx split, same fitted table, but
    # this is NOT the primary gate — the table was never fit on this data at all.
    correlated = load_fixed_strategy_rows(Path("results/sweep_correlated.csv"))
    correlated_eval = correlated[correlated["query_idx"] >= CALIBRATION_MAX_QUERY_IDX]
    secondary = gate_check(correlated_eval, lookup_table, "correlated evaluation half (secondary, exploratory)")
    secondary_regret = regret_on_half(correlated_eval, lookup_table, "correlated evaluation half (secondary, exploratory)")

    out_path = Path("results/lookup_table.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in sorted(lookup_table.items())}, f, indent=2)
    print(f"\nwrote {out_path}")

    print("\n=== SUMMARY ===")
    print(f"Primary gate (uncorrelated, held-out eval half) holds: {primary['gate_holds']}")
    print(f"Secondary check (correlated, held-out eval half) holds: {secondary['gate_holds']}")


if __name__ == "__main__":
    main()
