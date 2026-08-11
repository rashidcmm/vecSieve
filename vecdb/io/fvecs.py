from pathlib import Path
import numpy as np


def _read_vecs(path: str | Path, dtype: np.dtype) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=dtype)
    d = raw[0]
    row_stride = d + 1  # 1 int32 dimension header + d payload values
    raw = raw.reshape(-1, row_stride)
    if not np.all(raw[:, 0] == d):
        raise ValueError(f"inconsistent dimension across rows: expected {d}, found {np.unique(raw[:, 0])}")
    payload = raw[:, 1:].view(dtype).astype(dtype, copy=False)
    return payload


def read_fvecs(path: str | Path) -> np.ndarray:
    """Read float32 vectors from .fvecs binary file.

    Format: no file header; each vector is stored as little-endian int32 dimension d,
    followed by d little-endian float32 values, back-to-back with no padding.

    Args:
        path: Path to .fvecs file.

    Returns:
        Array of shape (n, d) with dtype float32, where n is number of vectors and d is dimension.
    """
    return _read_vecs(path, np.float32)


def read_ivecs(path: str | Path) -> np.ndarray:
    """Read int32 vectors from .ivecs binary file.

    Format: no file header; each vector is stored as little-endian int32 dimension d,
    followed by d little-endian int32 values, back-to-back with no padding.

    Args:
        path: Path to .ivecs file.

    Returns:
        Array of shape (n, d) with dtype int32, where n is number of vectors and d is dimension.
    """
    return _read_vecs(path, np.int32)


def write_fvecs(path: str | Path, arr: np.ndarray) -> None:
    """Write float32 vectors to .fvecs binary file.

    Format: no file header; each vector is stored as little-endian int32 dimension d,
    followed by d little-endian float32 values, back-to-back with no padding.

    Args:
        path: Path to write .fvecs file to.
        arr: Array of shape (n, d) with float32 dtype. n is number of vectors, d is dimension.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n, d = arr.shape
    with open(path, "wb") as f:
        dims = np.full((n, 1), d, dtype=np.int32)
        header_and_body = np.hstack([dims, arr.view(np.int32)])
        header_and_body.tofile(f)
