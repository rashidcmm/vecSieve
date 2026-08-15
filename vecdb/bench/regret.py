from __future__ import annotations
import numpy as np
import pandas as pd


def compute_regret(df: pd.DataFrame, fixed_strategies: list[str], chosen_col: str) -> pd.Series:
    """regret_ms = chosen strategy's latency - best fixed strategy's latency, per
    query, in hindsight. df needs one f'{strategy}_latency_ms' column per fixed
    strategy plus chosen_col holding the planner's realized latency."""
    best = df[[f"{s}_latency_ms" for s in fixed_strategies]].min(axis=1)
    return df[chosen_col] - best


def regret_summary(regret: pd.Series) -> dict[str, float]:
    return {
        "mean_regret_ms": float(np.mean(regret)),
        "p95_regret_ms": float(np.quantile(regret, 0.95)),
    }
