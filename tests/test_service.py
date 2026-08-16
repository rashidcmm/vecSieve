import numpy as np
from fastapi.testclient import TestClient
from vecdb.store.metadata import MetaStore
from vecdb.store.idmap import IdMap
from vecdb.index.flat import FlatIndex
from vecdb.index.hnsw import HNSWIndex
from vecdb.planner.cost_model import CostModelParams
from vecdb.planner.planner import Planner
from vecdb.service.app import app, get_service, VecDBService

def _small_service() -> VecDBService:
    rng = np.random.default_rng(0)
    data = rng.random((50, 4)).astype(np.float32)
    flat = FlatIndex(); flat.add(data, np.arange(50))
    hnsw = HNSWIndex(dim=4, M=8, ef_construction=50, seed=0)
    hnsw.add(data, np.arange(50))
    meta = MetaStore({"category": rng.integers(0, 3, size=50).astype(np.int32)})
    idmap = IdMap()
    for i in range(50):
        idmap.add(f"base-{i}")
    params = CostModelParams(c_scan=1.0, c_hop=1.0, alpha=4.0, beta=1.0, N=50)
    return VecDBService(flat, hnsw, meta, idmap, Planner(params))

def _client():
    service = _small_service()
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app), service

def test_search_endpoint_returns_k_results_and_a_reason():
    client, _ = _client()
    resp = client.post("/search", json={"vector": [0.1, 0.2, 0.3, 0.4], "k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 5
    assert body["strategy"] in {"pre_filter", "post_filter", "predicate_aware"}
    assert body["reason"]

def test_search_endpoint_applies_filter():
    client, service = _client()
    resp = client.post("/search", json={
        "vector": [0.1, 0.2, 0.3, 0.4], "k": 5,
        "filter": {"op": "eq", "col": "category", "val": 0},
    })
    assert resp.status_code == 200
    body = resp.json()
    returned_internal = [service.idmap.to_internal(item["id"]) for item in body["results"]]
    assert all(service.meta.columns["category"][i] == 0 for i in returned_internal)

def test_insert_then_search_can_return_the_new_vector():
    client, service = _client()
    client.post("/insert", json={"id": "new-1", "vector": [0.0, 0.0, 0.0, 0.0], "metadata": {"category": 0}})
    resp = client.post("/search", json={"vector": [0.0, 0.0, 0.0, 0.0], "k": 1})
    body = resp.json()
    assert body["results"][0]["id"] == "new-1"

def test_stats_endpoint_reports_vector_count_and_dim():
    client, _ = _client()
    resp = client.get("/stats")
    body = resp.json()
    assert body["n_vectors"] == 50
    assert body["dim"] == 4
