# tests/test_hnsw_correctness.py
from pathlib import Path
import pytest
from vecdb.io.dataset import load
from vecdb.index.hnsw import HNSWIndex
from vecdb.bench.harness import run_benchmark

SIFTSMALL_HNSW_PATH = Path("data/hnsw_siftsmall")

@pytest.mark.skipif(not SIFTSMALL_HNSW_PATH.exists(), reason="requires scripts/build_index.py to have been run")
def test_recall_at_10_floor_against_flat_ground_truth():
    idx = HNSWIndex.load(SIFTSMALL_HNSW_PATH)
    bundle = load("siftsmall", cache_dir=Path("data"))
    gt = [bundle.groundtruth[i][:10] for i in range(len(bundle.queries))]
    df = run_benchmark(idx, bundle.queries, gt, k=10, params={"ef": 100})
    assert df["recall"].mean() >= 0.95, (
        "HNSW recall floor regressed below the Phase 3 gate — this must be treated as "
        "a real regression, not a threshold to relax"
    )
