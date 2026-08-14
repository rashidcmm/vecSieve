"""Phase 4: efSearch sweep on siftsmall for hand-written HNSW vs FAISS HNSW, same M/efConstruction."""
from pathlib import Path
import numpy as np
from vecdb.io.dataset import load
from vecdb.index.hnsw import HNSWIndex
from vecdb.index.faiss_baseline import FaissHNSWIndex
from vecdb.bench.harness import run_benchmark
from vecdb.bench.plots import plot_lines

EF_VALUES = [10, 20, 40, 80, 160, 320]

def main() -> None:
    bundle = load("siftsmall", cache_dir=Path("data"))
    gt = [bundle.groundtruth[i][:10] for i in range(len(bundle.queries))]

    ours = HNSWIndex(dim=bundle.base.shape[1], M=16, ef_construction=200, seed=42)
    ours.add(bundle.base, np.arange(len(bundle.base)))

    theirs = FaissHNSWIndex(dim=bundle.base.shape[1], M=16, ef_construction=200)
    theirs.add(bundle.base, np.arange(len(bundle.base)))

    series = {"hand-written HNSW": [], "FAISS HNSW": []}
    for ef in EF_VALUES:
        df_ours = run_benchmark(ours, bundle.queries, gt, k=10, params={"ef": ef})
        series["hand-written HNSW"].append((df_ours["latency_ms"].quantile(0.5), df_ours["recall"].mean()))
        df_theirs = run_benchmark(theirs, bundle.queries, gt, k=10, params={"ef": ef})
        series["FAISS HNSW"].append((df_theirs["latency_ms"].quantile(0.5), df_theirs["recall"].mean()))
        print(f"ef={ef:4d}  ours: recall={df_ours['recall'].mean():.3f} p50={df_ours['latency_ms'].quantile(0.5):.3f}ms"
              f"  |  faiss: recall={df_theirs['recall'].mean():.3f} p50={df_theirs['latency_ms'].quantile(0.5):.3f}ms")

    plot_lines(series, Path("results/figures/pareto_unfiltered.png"),
               xlabel="p50 latency (ms)", ylabel="recall@10", title="Unfiltered recall/latency Pareto curve")
    print("wrote results/figures/pareto_unfiltered.png")

if __name__ == "__main__":
    main()
