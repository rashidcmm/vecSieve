from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans


def generate_uncorrelated(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "category": rng.integers(0, 100, size=n).astype(np.int32),
        "year": rng.integers(2000, 2026, size=n).astype(np.int16),
        "score": rng.random(size=n).astype(np.float32),
    }


def generate_correlated(
    vectors: np.ndarray, n_clusters: int = 100, agree_prob: float = 0.85, seed: int = 0
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = vectors.shape[0]
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_ids = km.fit_predict(vectors).astype(np.int32)
    random_category = rng.integers(0, n_clusters, size=n).astype(np.int32)
    use_cluster = rng.random(size=n) < agree_prob
    category = np.where(use_cluster, cluster_ids, random_category)
    return {
        "category": category,
        "year": rng.integers(2000, 2026, size=n).astype(np.int16),
        "score": rng.random(size=n).astype(np.float32),
    }


def save_metadata(path: Path, columns: dict[str, np.ndarray]) -> None:
    np.savez(path, **columns)


def load_metadata(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
