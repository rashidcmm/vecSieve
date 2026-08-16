# scripts/generate_figures.py
"""Phase 7 headline figures. crossover.png is THE deliverable — it goes at the top
of the README, above the fold."""
from pathlib import Path
import pandas as pd
from vecdb.bench.plots import plot_lines

STRATEGY_LABELS = {
    "pre_filter": "pre-filter", "post_filter": "post-filter",
    "predicate_aware": "predicate-aware", "planner": "planner (chosen)",
}

def _series_by_strategy(df: pd.DataFrame, value_col: str, agg: str) -> dict[str, list[tuple[float, float]]]:
    grouped = df.groupby(["strategy", "target_selectivity"])[value_col]
    agg_df = grouped.quantile(0.95) if agg == "p95" else grouped.mean()
    series: dict[str, list[tuple[float, float]]] = {}
    for (strategy, s), value in agg_df.items():
        if strategy not in STRATEGY_LABELS:
            continue
        series.setdefault(STRATEGY_LABELS[strategy], []).append((s, value))
    return series

def crossover(df: pd.DataFrame, out_path: Path, title: str) -> None:
    series = _series_by_strategy(df, "latency_ms", "p95")
    plot_lines(series, out_path, xlabel="selectivity (log)", ylabel="p95 latency (ms, log)",
               title=title, xscale="log", yscale="log")

def recall_vs_selectivity(df: pd.DataFrame, out_path: Path) -> None:
    series = _series_by_strategy(df, "recall", "mean")
    plot_lines(series, out_path, xlabel="selectivity (log)", ylabel="mean recall@10",
               title="Recall vs selectivity", xscale="log")

def underfill(df: pd.DataFrame, out_path: Path) -> None:
    post = df[df["strategy"] == "post_filter"]
    grouped = post.groupby("target_selectivity")["underfill"].mean()
    series = {"post-filter underfill rate": list(grouped.items())}
    plot_lines(series, out_path, xlabel="selectivity (log)", ylabel="underfill rate",
               title="Post-filter under-fill rate (the correctness bug, visualised)", xscale="log")

def dist_ops(df: pd.DataFrame, out_path: Path) -> None:
    series = _series_by_strategy(df, "dist_ops", "mean")
    plot_lines(series, out_path, xlabel="selectivity (log)", ylabel="mean dist_ops (log)",
               title="Distance computations vs selectivity (hardware-independent)",
               xscale="log", yscale="log")

def main() -> None:
    uncorr = pd.read_csv("results/sweep_uncorrelated.csv")
    corr = pd.read_csv("results/sweep_correlated.csv")

    crossover(uncorr, Path("results/figures/crossover.png"),
              "Crossover: winning strategy by selectivity (uncorrelated metadata)")
    crossover(corr, Path("results/figures/crossover_correlated.png"),
              "Crossover: winning strategy by selectivity (correlated metadata)")
    recall_vs_selectivity(uncorr, Path("results/figures/recall_vs_selectivity.png"))
    underfill(uncorr, Path("results/figures/underfill.png"))
    dist_ops(uncorr, Path("results/figures/dist_ops.png"))
    print("wrote crossover.png, crossover_correlated.png, recall_vs_selectivity.png, underfill.png, dist_ops.png")

if __name__ == "__main__":
    main()
