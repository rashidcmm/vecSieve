from __future__ import annotations
from pathlib import Path
import numpy as np
from vecdb.index.flat import FlatIndex


def compute_filtered_groundtruth(flat_index: FlatIndex, queries: np.ndarray,
                                   masks: list[np.ndarray], k: int) -> list[np.ndarray]:
    """Exact top-k ids per query under its mask via brute force. There is no shortcut
    and no published ground truth for synthetic predicates."""
    return [flat_index.search(q, k, mask=mask).ids for q, mask in zip(queries, masks)]


def cache_groundtruth(path: Path, groundtruth: list[np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    max_len = max((len(g) for g in groundtruth), default=0)
    padded = np.full((len(groundtruth), max_len), -1, dtype=np.int64)
    for i, g in enumerate(groundtruth):
        padded[i, : len(g)] = g
    np.save(path, padded)


def load_groundtruth(path: Path) -> list[np.ndarray]:
    padded = np.load(path)
    return [row[row >= 0] for row in padded]
