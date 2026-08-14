# scripts/run_sweep_all.py
"""Phase 6 gate: full selectivity grid x both metadata variants for Strategies A, B,
and C. Supersedes scripts/run_sweep_ab.py (delete that file)."""
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

N_QUERIES = 200
WARMUP = 20

def run_variant(variant: str, bundle, hnsw: HNSWIndex, flat: FlatIndex) -> pd.DataFrame:
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
        if gt_path.exists():
            gt = load_groundtruth(gt_path)
        else:
            gt = compute_filtered_groundtruth(flat, queries, masks, k=10)
            cache_groundtruth(gt_path, gt)

        strategies = {
            "pre_filter": PreFilterStrategy(flat),
            "post_filter": PostFilterStrategy(hnsw, fallback=PreFilterStrategy(flat)),
            "predicate_aware": FilteredHNSWStrategy(hnsw, fallback=PreFilterStrategy(flat)),
        }
        for name, strat in strategies.items():
            df = run_benchmark(strat, queries, gt, k=10, masks=masks,
                                 params={"selectivity_hat": sel_hat}, warmup=WARMUP)
            df["target_selectivity"] = target_s
            df["true_selectivity"] = true_s
            df["sel_hat"] = sel_hat
            df["metadata_variant"] = variant
            rows.append(df)
            print(f"[{variant}] s={target_s} {name}: recall={df['recall'].mean():.3f}"
                  f" p95={df['latency_ms'].quantile(0.95):.2f}ms dist_ops_mean={df['dist_ops'].mean():.0f}")
    return pd.concat(rows, ignore_index=True)

def main() -> None:
    bundle = load("sift1m_100k", cache_dir=Path("data"))
    flat = FlatIndex()
    flat.add(bundle.base, np.arange(len(bundle.base)))
    hnsw = HNSWIndex.load(Path("data/hnsw_100k"))

    for variant, out_name in [("uncorrelated", "sweep_uncorrelated.csv"), ("correlated", "sweep_correlated.csv")]:
        df = run_variant(variant, bundle, hnsw, flat)
        df.to_csv(f"results/{out_name}", index=False)
        print(f"wrote results/{out_name}")

if __name__ == "__main__":
    main()
