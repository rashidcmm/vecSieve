import numpy as np
from vecdb.store.metadata import MetaStore

def test_categorical_stats_are_exact_value_counts():
    columns = {"category": np.array([0, 0, 1, 2, 2, 2], dtype=np.int32)}
    store = MetaStore(columns)
    stats = store.stats["category"]
    assert stats.kind == "categorical"
    assert stats.value_counts == {0: 2, 1: 1, 2: 3}
    assert stats.n == 6

def test_numeric_stats_build_64_bin_histogram_covering_full_range():
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, size=1000).astype(np.float32)
    store = MetaStore({"score": values})
    stats = store.stats["score"]
    assert stats.kind == "numeric"
    assert len(stats.hist_edges) == 65  # 64 bins -> 65 edges
    assert stats.hist_counts.sum() == 1000
    assert stats.hist_edges[0] <= values.min()
    assert stats.hist_edges[-1] >= values.max()

def test_metastore_exposes_row_count_and_raw_columns():
    columns = {"category": np.array([0, 1, 2], dtype=np.int32), "year": np.array([2020, 2021, 2022], dtype=np.int16)}
    store = MetaStore(columns)
    assert store.n == 3
    np.testing.assert_array_equal(store.columns["year"], columns["year"])
