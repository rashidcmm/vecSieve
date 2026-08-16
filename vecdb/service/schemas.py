from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class InsertRequest(BaseModel):
    id: str
    vector: list[float]
    metadata: dict[str, Any] = {}


class SearchRequest(BaseModel):
    vector: list[float]
    k: int = 10
    filter: dict[str, Any] | None = None


class SearchResultItem(BaseModel):
    id: str
    distance: float


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    strategy: str
    reason: str
    latency_ms: float
    n_distance_ops: int


class StatsResponse(BaseModel):
    n_vectors: int
    dim: int


class PersistResponse(BaseModel):
    path: str
