# scripts/calibrate.py
"""Phase 7: fit c_scan, c_hop, beta by least squares against Task 32's measured
dist_ops on the uncorrelated sweep. alpha/ef_min/ef_base/M are read from the
strategy defaults already in use, not fit."""
from pathlib import Path
import numpy as np
import pandas as pd
from vecdb.planner.cost_model import CostModelParams

ALPHA = 4.0     # PostFilterStrategy default
EF_MIN = 16     # PostFilterStrategy default
EF_BASE = 64    # FilteredHNSWStrategy default
M = 16          # HNSW build parameter (Phase 3/4)
K = 10

def fit_c_scan(df: pd.DataFrame, N: int) -> float:
    pre = df[df["strategy"] == "pre_filter"]
    x = pre["true_selectivity"].to_numpy() * N
    y = pre["dist_ops"].to_numpy()
    return float(np.sum(x * y) / np.sum(x * x))

def fit_c_hop(df: pd.DataFrame, N: int) -> float:
    post = df[df["strategy"] == "post_filter"]
    s = post["true_selectivity"].to_numpy()
    ef_req = np.clip(ALPHA * K / np.maximum(s, 1e-6), EF_MIN, N)
    x = M * ef_req
    y = post["dist_ops"].to_numpy()
    return float(np.sum(x * y) / np.sum(x * x))

def fit_beta(df: pd.DataFrame, c_hop: float) -> float:
    pred = df[df["strategy"] == "predicate_aware"]
    s = np.maximum(pred["true_selectivity"].to_numpy(), 1e-6)
    x = (1.0 / s) - 1.0
    y = pred["dist_ops"].to_numpy() / max(c_hop * M * EF_BASE, 1e-9)
    design = np.column_stack([np.ones_like(x), x])
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(b / a) if a != 0 else 0.0

def main() -> None:
    N = 100_000
    df = pd.read_csv("results/sweep_uncorrelated.csv")
    c_scan = fit_c_scan(df, N)
    c_hop = fit_c_hop(df, N)
    beta = fit_beta(df, c_hop)
    params = CostModelParams(c_scan=c_scan, c_hop=c_hop, alpha=ALPHA, beta=beta,
                               ef_min=EF_MIN, ef_base=EF_BASE, M=M, N=N)
    params.save(Path("results/calibration.json"))
    print(params)

if __name__ == "__main__":
    main()
