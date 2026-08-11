from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class SearchResult:
    ids: np.ndarray
    distances: np.ndarray
    n_distance_ops: int   # hardware-independent cost metric
    strategy: str
    latency_ms: float
    n_returned: int        # < k means an under-fill


class Index(ABC):
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None,
                params: dict | None = None) -> SearchResult: ...
