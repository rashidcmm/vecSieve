# scripts/run_full_sweep.py
"""Phase 7 gate: the full 8 selectivities x 2 metadata variants x 4 executors
(3 fixed strategies + planner) x 200 queries sweep, plus the regret report."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd
from vecdb.io.dataset import load
from vecdb.io.metadata_gen import load_metadata
from vecdb.store.metadata import MetaStore
from vecdb.predicate.compile import compile as compile_pred
from vecdb.predicate.selectivity import estimate_selectivity
from vecdb.index.flat import FlatIndex
from vecdb.index.hnsw import HNSWIndex
from vecdb.index.strategies import PreFilterStrategy, PostFilterStrategy, FilteredHNSWStrategy
from vecdb.bench.sweep import SELECTIVITY_GRID, predicate_for_selectivity
from vecdb.bench.groundtruth import compute_filtered_groundtruth, cache_groundtruth, load_groundtruth
from vecdb.bench.harness import run_benchmark
from vecdb.bench.regret import compute_regret, regret_summary
from vecdb.planner.cost_model import CostModelParams
from vecdb.planner.planner import Planner

N_QUERIES = 200
WARMUP = 20

def run_variant(variant: str, bundle, hnsw: HNSWIndex, flat: FlatIndex, planner: Planner) -> pd.DataFrame:
    cols = load_metadata(Path(f"data/sift1m_100k_meta_{variant}.npz"))
    meta = MetaStore(cols)
    queries = bundle.queries[:N_QUERIES]
    rows = []
    for target_s in SELECTIVITY_GRID:
        pred = predicate_for_selectivity(meta, target_s)
        mask = compile_pred(pred, meta)
        true_s = mask.sum() / meta.n
        sel_hat = estimate_selectivity(pred, meta)
        masks = [mask] * len(queries)

        gt_path = Path(f"results/gt_cache/{variant}_{target_s}.npy")
        gt = load_groundtruth(gt_path) if gt_path.exists() else compute_filtered_groundtruth(flat, queries, masks, k=10)
        if not gt_path.exists():
            cache_groundtruth(gt_path, gt)

        plan = planner.plan(k=10, sel_hat=sel_hat)
        strategies = {
            "pre_filter": PreFilterStrategy(flat),
            "post_filter": PostFilterStrategy(hnsw, fallback=PreFilterStrategy(flat)),
            "predicate_aware": FilteredHNSWStrategy(hnsw, fallback=PreFilterStrategy(flat)),
        }
        per_strategy = {}
        for name, strat in strategies.items():
            df = run_benchmark(strat, queries, gt, k=10, masks=masks,
                                 params={"selectivity_hat": sel_hat}, warmup=WARMUP)
            df["target_selectivity"] = target_s
            df["true_selectivity"] = true_s
            df["sel_hat"] = sel_hat
            df["metadata_variant"] = variant
            per_strategy[name] = df
            rows.append(df)

        planner_df = per_strategy[plan.strategy].copy()
        planner_df["strategy"] = "planner"
        planner_df["planner_reason"] = plan.reason
        rows.append(planner_df)
        print(f"[{variant}] s={target_s} planner picked {plan.strategy} ({plan.reason})")

    return pd.concat(rows, ignore_index=True)

def main() -> None:
    bundle = load("sift1m_100k", cache_dir=Path("data"))
    flat = FlatIndex()
    flat.add(bundle.base, np.arange(len(bundle.base)))
    hnsw = HNSWIndex.load(Path("data/hnsw_100k"))
    params = CostModelParams.load(Path("results/calibration.json"))
    planner = Planner(params)

    for variant, out_name in [("uncorrelated", "sweep_uncorrelated.csv"), ("correlated", "sweep_correlated.csv")]:
        df = run_variant(variant, bundle, hnsw, flat, planner)
        df.to_csv(f"results/{out_name}", index=False)
        print(f"wrote results/{out_name}")

    # Regret, computed on the uncorrelated sweep (the planner's calibration target)
    df = pd.read_csv("results/sweep_uncorrelated.csv")
    wide = df.groupby(["query_idx", "target_selectivity", "strategy"])["latency_ms"].mean().unstack("strategy")
    wide = wide.rename(columns={s: f"{s}_latency_ms" for s in ["pre_filter", "post_filter", "predicate_aware"]})
    wide = wide.rename(columns={"planner": "planner_latency_ms"})
    regret = compute_regret(wide, ["pre_filter", "post_filter", "predicate_aware"], "planner_latency_ms")
    print("regret:", regret_summary(regret))

    mean_fixed = {s: df[df["strategy"] == s]["latency_ms"].mean() for s in ["pre_filter", "post_filter", "predicate_aware"]}
    mean_planner = df[df["strategy"] == "planner"]["latency_ms"].mean()
    print("mean latency — fixed strategies:", mean_fixed, " planner:", mean_planner)
    print("Phase 7 planner-beats-fixed gate:", mean_planner < min(mean_fixed.values()))

if __name__ == "__main__":
    main()
