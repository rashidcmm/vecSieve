import numpy as np
from vecdb.io.dataset import DatasetBundle, _subset_100k

def test_dataset_bundle_is_dataclass_with_expected_fields():
    bundle = DatasetBundle(
        base=np.zeros((10, 4), dtype=np.float32),
        queries=np.zeros((2, 4), dtype=np.float32),
        groundtruth=np.zeros((2, 3), dtype=np.int32),
    )
    assert bundle.base.shape == (10, 4)
    assert bundle.queries.shape == (2, 4)
    assert bundle.groundtruth.shape == (2, 3)

def test_subset_100k_truncates_base_and_remaps_groundtruth():
    # groundtruth ids >= 100k must be dropped from each row, not just clipped
    base = np.arange(150_000 * 4, dtype=np.float32).reshape(150_000, 4)
    gt = np.array([[0, 99_999, 100_000, 149_999]], dtype=np.int32)
    new_base, new_gt = _subset_100k(base, gt, n=100_000)
    assert new_base.shape == (100_000, 4)
    assert list(new_gt[0]) == [0, 99_999]  # ids >= 100k dropped, order preserved

def test_subset_100k_pads_short_rows_instead_of_truncating_all_rows_to_shortest():
    base = np.arange(150_000 * 4, dtype=np.float32).reshape(150_000, 4)
    gt = np.array([
        [0, 99_999, 100_000, 149_999],   # 2 ids survive
        [100_000, 100_001, 100_002, 100_003],  # 0 ids survive
    ], dtype=np.int32)
    new_base, new_gt = _subset_100k(base, gt, n=100_000)
    assert new_gt.shape == (2, 2)          # width = the longest surviving row (2), not the shortest (0)
    assert list(new_gt[0]) == [0, 99_999]  # first row keeps its real ids
    assert list(new_gt[1]) == [-1, -1]     # second row is entirely padding, not silently dropped
