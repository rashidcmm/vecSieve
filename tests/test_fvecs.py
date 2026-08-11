import numpy as np
from vecdb.io.fvecs import read_fvecs, read_ivecs, write_fvecs

def test_write_then_read_roundtrip(tmp_path):
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    path = tmp_path / "test.fvecs"
    write_fvecs(path, arr)
    result = read_fvecs(path)
    np.testing.assert_array_equal(result, arr)

def test_read_ivecs_matches_fvecs_layout(tmp_path):
    # ivecs uses the identical layout with int32 payloads instead of float32
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    path = tmp_path / "test.ivecs"
    with open(path, "wb") as f:
        for row in arr:
            np.array([len(row)], dtype=np.int32).tofile(f)
            row.tofile(f)
    result = read_ivecs(path)
    np.testing.assert_array_equal(result, arr)
