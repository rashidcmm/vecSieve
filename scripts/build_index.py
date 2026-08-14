# scripts/build_index.py
"""Phase 3 gate: build HNSW on siftsmall, confirm build time and recall@10 vs FlatIndex."""
import time
from pathlib import Path
import numpy as np
from vecdb.io.dataset import load
from vecdb.index.hnsw import HNSWIndex
from vecdb.bench.harness import run_benchmark

def main() -> None:
    bundle = load("siftsmall", cache_dir=Path("data"))

    idx = HNSWIndex(dim=bundle.base.shape[1], M=16, ef_construction=200, seed=42)
    t0 = time.perf_counter()
    idx.add(bundle.base, np.arange(len(bundle.base)))
    build_s = time.perf_counter() - t0
    print(f"build time: {build_s:.1f}s")
    assert build_s < 120, "Phase 3 gate requires siftsmall build under 2 minutes"

    gt = [bundle.groundtruth[i][:10] for i in range(len(bundle.queries))]
    df = run_benchmark(idx, bundle.queries, gt, k=10, params={"ef": 100})
    recall = df["recall"].mean()
    print(f"recall@10 at efSearch=100: {recall:.4f}")
    assert recall >= 0.95, f"Phase 3 gate requires recall@10 >= 0.95, got {recall:.4f}"

    idx.save(Path("data/hnsw_siftsmall"))
    print("Phase 3 gate: PASS")

if __name__ == "__main__":
    main()
