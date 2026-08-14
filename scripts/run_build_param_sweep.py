"""Phase 4: M x efConstruction sweep on siftsmall. Table of build time / index size /
recall@10 at a fixed efSearch, so the operating point can be picked and justified."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from vecdb.io.dataset import load
from vecdb.index.hnsw import HNSWIndex
from vecdb.bench.harness import run_benchmark

M_VALUES = [8, 16, 32]
EFC_VALUES = [100, 200, 400]
FIXED_EF_SEARCH = 128

def main() -> None:
    bundle = load("siftsmall", cache_dir=Path("data"))
    gt = [bundle.groundtruth[i][:10] for i in range(len(bundle.queries))]
    rows = []
    for M in M_VALUES:
        for efc in EFC_VALUES:
            idx = HNSWIndex(dim=bundle.base.shape[1], M=M, ef_construction=efc, seed=42)
            t0 = time.perf_counter()
            idx.add(bundle.base, np.arange(len(bundle.base)))
            build_s = time.perf_counter() - t0
            df = run_benchmark(idx, bundle.queries, gt, k=10, params={"ef": FIXED_EF_SEARCH})
            rows.append({
                "M": M, "ef_construction": efc, "build_time_s": build_s,
                "index_bytes": idx.approx_size_bytes(), "recall_at_10": df["recall"].mean(),
            })
            print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv("results/build_param_sweep.csv", index=False)
    print("wrote results/build_param_sweep.csv")

if __name__ == "__main__":
    main()
