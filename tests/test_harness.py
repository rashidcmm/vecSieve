import numpy as np
import pytest
from pathlib import Path
from vecdb.bench.harness import recall_at_k, run_benchmark
from vecdb.bench.groundtruth import compute_filtered_groundtruth, cache_groundtruth, load_groundtruth
from vecdb.index.flat import FlatIndex

def test_recall_at_k_perfect_match_is_one():
    assert recall_at_k(np.array([1, 2, 3]), np.array([1, 2, 3]), k=3) == 1.0

def test_recall_at_k_partial_overlap():
    assert recall_at_k(np.array([1, 2, 9]), np.array([1, 2, 3]), k=3) == pytest.approx(2 / 3)

def test_recall_at_k_does_not_exceed_one_when_true_set_shorter_than_k():
    # true set has only 2 survivors (a highly selective filter) but k=5 was requested
    r = recall_at_k(np.array([1, 2, 3, 4, 5]), np.array([1, 2]), k=5)
    assert r <= 1.0
    assert r == 1.0  # both true survivors were returned

def test_recall_at_k_both_empty_is_vacuously_one():
    assert recall_at_k(np.array([]), np.array([]), k=5) == 1.0

def test_run_benchmark_produces_expected_columns_and_skips_warmup():
    rng = np.random.default_rng(0)
    data = rng.random((100, 8)).astype(np.float32)
    idx = FlatIndex()
    idx.add(data, np.arange(100))
    queries = rng.random((10, 8)).astype(np.float32)
    groundtruth = [idx.search(q, k=5).ids for q in queries]
    df = run_benchmark(idx, queries, groundtruth, k=5, warmup=2)
    assert len(df) == 8  # 10 queries - 2 warmup
    assert set(df.columns) == {"query_idx", "recall", "underfill", "latency_ms", "dist_ops", "n_returned", "strategy"}
    assert (df["recall"] == 1.0).all()  # FlatIndex vs its own ground truth is exact

def test_groundtruth_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.random((50, 8)).astype(np.float32)
    idx = FlatIndex()
    idx.add(data, np.arange(50))
    queries = rng.random((5, 8)).astype(np.float32)
    masks = [rng.random(50) < 0.3 for _ in range(5)]
    gt = compute_filtered_groundtruth(idx, queries, masks, k=5)
    path = tmp_path / "gt.npy"
    cache_groundtruth(path, gt)
    loaded = load_groundtruth(path)
    for original, restored in zip(gt, loaded):
        np.testing.assert_array_equal(np.sort(original), np.sort(restored))
