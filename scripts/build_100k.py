# scripts/build_100k.py
"""Phase 4 gate: build HNSW on the 100K SIFT subset, confirm it completes in a
reasonable window, persist it for Phases 5-7 to load."""
import time
from pathlib import Path
import numpy as np
from vecdb.io.dataset import load
from vecdb.index.hnsw import HNSWIndex

def main() -> None:
    bundle = load("sift1m_100k", cache_dir=Path("data"))
    idx = HNSWIndex(dim=bundle.base.shape[1], M=16, ef_construction=200, seed=42)
    t0 = time.perf_counter()
    idx.add(bundle.base, np.arange(len(bundle.base)))
    build_s = time.perf_counter() - t0
    print(f"100K build time: {build_s / 60:.1f} min")
    idx.save(Path("data/hnsw_100k"))
    print(f"index size (adjacency only): {idx.approx_size_bytes() / 1e6:.1f} MB")
    print("saved to data/hnsw_100k")

if __name__ == "__main__":
    main()
