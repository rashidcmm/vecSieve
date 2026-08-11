"""Phase 2 gate: FlatIndex unfiltered recall@100 must be exactly 1.000 against the
shipped SIFT ground truth, and FAISS-Flat must match it. Prints a metrics table."""
from pathlib import Path
import numpy as np
from vecdb.io.dataset import load
from vecdb.index.flat import FlatIndex
from vecdb.index.faiss_baseline import FaissFlatIndex
from vecdb.bench.harness import run_benchmark

def main() -> None:
    bundle = load("siftsmall", cache_dir=Path("data"))
    k = 100

    flat = FlatIndex()
    flat.add(bundle.base, np.arange(len(bundle.base)))
    gt = [bundle.groundtruth[i][:k] for i in range(len(bundle.queries))]
    df_flat = run_benchmark(flat, bundle.queries, gt, k=k)

    faiss_flat = FaissFlatIndex(dim=bundle.base.shape[1])
    faiss_flat.add(bundle.base, np.arange(len(bundle.base)))
    df_faiss = run_benchmark(faiss_flat, bundle.queries, gt, k=k)

    print("FlatIndex      recall@100:", df_flat["recall"].mean())
    print("FaissFlatIndex recall@100:", df_faiss["recall"].mean())
    assert df_flat["recall"].mean() == 1.0, "FlatIndex must be exact"
    assert df_faiss["recall"].mean() == 1.0, "FaissFlatIndex must be exact"
    print("Phase 2 gate: PASS")

if __name__ == "__main__":
    main()
