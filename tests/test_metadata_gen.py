import numpy as np
from vecdb.io.metadata_gen import generate_uncorrelated, generate_correlated, save_metadata, load_metadata

def test_uncorrelated_has_expected_columns_and_ranges():
    cols = generate_uncorrelated(1000, seed=0)
    assert set(cols) == {"category", "year", "score"}
    assert cols["category"].shape == (1000,)
    assert cols["category"].min() >= 0 and cols["category"].max() <= 99
    assert cols["year"].min() >= 2000 and cols["year"].max() <= 2025
    assert cols["score"].min() >= 0.0 and cols["score"].max() <= 1.0

def test_correlated_category_tracks_kmeans_cluster_mostly():
    rng = np.random.default_rng(0)
    # two well-separated blobs so k-means clustering is unambiguous
    blob_a = rng.normal(loc=0.0, scale=0.1, size=(200, 8))
    blob_b = rng.normal(loc=10.0, scale=0.1, size=(200, 8))
    vectors = np.vstack([blob_a, blob_b]).astype(np.float32)
    cols = generate_correlated(vectors, n_clusters=2, agree_prob=1.0, seed=0)
    # with agree_prob=1.0, every point's category must exactly equal its cluster id
    assert len(set(cols["category"][:200])) == 1
    assert len(set(cols["category"][200:])) == 1
    assert cols["category"][0] != cols["category"][200]

def test_save_and_load_metadata_roundtrip(tmp_path):
    cols = generate_uncorrelated(50, seed=1)
    path = tmp_path / "meta.npz"
    save_metadata(path, cols)
    loaded = load_metadata(path)
    for key in cols:
        np.testing.assert_array_equal(loaded[key], cols[key])
