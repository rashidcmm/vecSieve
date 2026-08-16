from __future__ import annotations
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, Depends

from vecdb.io.dataset import load
from vecdb.io.metadata_gen import load_metadata
from vecdb.store.metadata import MetaStore
from vecdb.store.idmap import IdMap
from vecdb.index.flat import FlatIndex
from vecdb.index.hnsw import HNSWIndex
from vecdb.index.strategies import PreFilterStrategy, PostFilterStrategy, FilteredHNSWStrategy
from vecdb.predicate.compile import compile as compile_pred
from vecdb.predicate.selectivity import estimate_selectivity
from vecdb.planner.cost_model import CostModelParams
from vecdb.planner.planner import Planner
from vecdb.service.schemas import (
    InsertRequest, SearchRequest, SearchResponse, SearchResultItem, StatsResponse, PersistResponse,
)

DATA_DIR = Path("data")
DATA_ROOT = Path("data").resolve()


class VecDBService:
    def __init__(self, flat: FlatIndex, hnsw: HNSWIndex, meta: MetaStore, idmap: IdMap, planner: Planner):
        self.flat = flat
        self.hnsw = hnsw
        self.meta = meta
        self.idmap = idmap
        self.planner = planner
        self.staged_vectors: list[np.ndarray] = []
        self.staged_ids: list[str] = []
        self.staged_meta: list[dict] = []

    @classmethod
    def from_disk(cls, data_dir: Path = DATA_DIR) -> "VecDBService":
        bundle = load("sift1m_100k", cache_dir=data_dir)
        flat = FlatIndex()
        flat.add(bundle.base, np.arange(len(bundle.base)))
        hnsw = HNSWIndex.load(data_dir / "hnsw_100k")
        cols = load_metadata(data_dir / "sift1m_100k_meta_uncorrelated.npz")
        meta = MetaStore(cols)
        idmap = IdMap()
        for i in range(len(bundle.base)):
            idmap.add(f"base-{i}")
        params = CostModelParams.load(Path("results/calibration.json"))
        return cls(flat, hnsw, meta, idmap, Planner(params))

    def insert(self, req: InsertRequest) -> None:
        self.idmap.add(req.id)
        self.staged_vectors.append(np.asarray(req.vector, dtype=np.float32))
        self.staged_ids.append(req.id)
        self.staged_meta.append(req.metadata)

    def _staged_matches(self, filter_pred: dict | None) -> list[int]:
        if not self.staged_meta:
            return []
        if filter_pred is None:
            return list(range(len(self.staged_meta)))
        cols: dict[str, list] = {}
        for m in self.staged_meta:
            for key, val in m.items():
                cols.setdefault(key, []).append(val)
        # Match each staged column's dtype to what the base MetaStore already decided
        # for that column (int32 = categorical, else numeric) — forcing everything to
        # int32 would silently truncate numeric columns like "score" or "year".
        typed_cols = {}
        for key, values in cols.items():
            if key in self.meta.stats and self.meta.stats[key].kind == "categorical":
                typed_cols[key] = np.array(values, dtype=np.int32)
            else:
                typed_cols[key] = np.array(values, dtype=np.float32)
        staged_meta_store = MetaStore(typed_cols)
        mask = compile_pred(filter_pred, staged_meta_store)
        return list(np.nonzero(mask)[0])

    def search(self, req: SearchRequest) -> SearchResponse:
        q = np.asarray(req.vector, dtype=np.float32)
        mask = compile_pred(req.filter, self.meta) if req.filter else None
        sel_hat = estimate_selectivity(req.filter, self.meta) if req.filter else 1.0
        plan = self.planner.plan(k=req.k, sel_hat=sel_hat)

        strategies = {
            "pre_filter": PreFilterStrategy(self.flat),
            "post_filter": PostFilterStrategy(self.hnsw, fallback=PreFilterStrategy(self.flat)),
            "predicate_aware": FilteredHNSWStrategy(self.hnsw, fallback=PreFilterStrategy(self.flat)),
        }
        base_result = strategies[plan.strategy].search(q, req.k, mask=mask, params={"selectivity_hat": sel_hat})

        candidates = [(self.idmap.to_external(int(i)), float(d))
                      for i, d in zip(base_result.ids, base_result.distances)]
        for i in self._staged_matches(req.filter):
            d = float(np.sum((self.staged_vectors[i] - q) ** 2))
            candidates.append((self.staged_ids[i], d))
        candidates.sort(key=lambda pair: pair[1])
        top = candidates[: req.k]

        return SearchResponse(
            results=[SearchResultItem(id=cid, distance=dist) for cid, dist in top],
            strategy=plan.strategy, reason=plan.reason,
            latency_ms=base_result.latency_ms, n_distance_ops=base_result.n_distance_ops,
        )

    def stats(self) -> StatsResponse:
        return StatsResponse(n_vectors=len(self.idmap), dim=self.hnsw.dim)

    def persist(self, path: str) -> PersistResponse:
        self.hnsw.save(Path(path))
        return PersistResponse(path=path)


app = FastAPI(title="filtered-vecdb")
_service_singleton: VecDBService | None = None


def get_service() -> VecDBService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = VecDBService.from_disk()
    return _service_singleton


@app.post("/insert")
def insert(req: InsertRequest, service: VecDBService = Depends(get_service)):
    service.insert(req)
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, service: VecDBService = Depends(get_service)):
    try:
        return service.search(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/stats", response_model=StatsResponse)
def stats(service: VecDBService = Depends(get_service)):
    return service.stats()


@app.post("/persist", response_model=PersistResponse)
def persist(name: str = "hnsw_100k", service: VecDBService = Depends(get_service)):
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid name")
    target = (DATA_ROOT / name).resolve()
    if not str(target).startswith(str(DATA_ROOT) + "\\") and not str(target).startswith(str(DATA_ROOT) + "/"):
        raise HTTPException(status_code=400, detail="path escapes data dir")
    return service.persist(str(target))
