import numpy as np
from vecdb.index.hnsw import HNSWIndex

def test_save_then_load_produces_identical_search_results(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.random((100, 8)).astype(np.float32)
    idx = HNSWIndex(dim=8, M=8, ef_construction=50, seed=0)
    idx.add(data, np.arange(100))

    q = rng.random(8).astype(np.float32)
    before = idx.search(q, k=5, params={"ef": 50})

    save_path = tmp_path / "index"
    idx.save(save_path)
    reloaded = HNSWIndex.load(save_path)
    after = reloaded.search(q, k=5, params={"ef": 50})

    np.testing.assert_array_equal(before.ids, after.ids)
    np.testing.assert_allclose(before.distances, after.distances, rtol=1e-5)
    assert reloaded.entry_point == idx.entry_point
    assert reloaded.max_level == idx.max_level
    assert reloaded.graph == idx.graph
