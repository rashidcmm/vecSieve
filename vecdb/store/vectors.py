from __future__ import annotations
import numpy as np


class VectorStore:
    """Row-major float32 vectors. Batched squared-L2 distance is the hot path:
    ||q-v||^2 = ||q||^2 - 2 q.v + ||v||^2, computed as one matrix-vector product
    rather than a Python loop over rows."""

    def __init__(self, data: np.ndarray):
        self.data: np.ndarray = np.ascontiguousarray(data, dtype=np.float32)
        self.sq_norms: np.ndarray = np.sum(self.data * self.data, axis=1)
        self.n_distance_ops: int = 0

    def distances(self, q: np.ndarray, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return np.empty(0, dtype=np.float32)
        q = np.asarray(q, dtype=np.float32)
        vecs = self.data[rows]                       # (k, d)
        dot = vecs @ q                                # (k,)
        q_sq = float(q @ q)
        d = q_sq - 2.0 * dot + self.sq_norms[rows]
        self.n_distance_ops += rows.shape[0]
        return np.maximum(d, 0.0)  # clamp tiny negative values from float error

    def vector(self, row: int) -> np.ndarray:
        return self.data[row]

    def __len__(self) -> int:
        return self.data.shape[0]
