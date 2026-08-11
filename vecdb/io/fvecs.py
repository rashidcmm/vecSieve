from pathlib import Path
import numpy as np


def _read_vecs(path: str | Path, dtype: np.dtype) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=dtype)
    d = raw[0]
    row_stride = d + 1  # 1 int32 dimension header + d payload values
    raw = raw.reshape(-1, row_stride)
    assert np.all(raw[:, 0] == d), "inconsistent dimension across rows"
    payload = raw[:, 1:].view(dtype).astype(dtype, copy=False)
    return payload


def read_fvecs(path: str | Path) -> np.ndarray:
    return _read_vecs(path, np.float32)


def read_ivecs(path: str | Path) -> np.ndarray:
    return _read_vecs(path, np.int32)


def write_fvecs(path: str | Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n, d = arr.shape
    with open(path, "wb") as f:
        dims = np.full((n, 1), d, dtype=np.int32)
        header_and_body = np.hstack([dims, arr.view(np.int32)])
        header_and_body.astype(np.int32).tofile(f)
