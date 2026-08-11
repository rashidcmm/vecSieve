# Filtered Vector Database Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a filtered approximate-nearest-neighbour vector database in Python/NumPy with a hand-written HNSW index, three filtered-search strategies (pre-filter, post-filter, predicate-aware traversal), a calibrated cost-based query planner that picks between them, a measured crossover benchmark, a FastAPI service layer, and a README that leads with the crossover chart.

**Architecture:** Layered — storage (memmapped vectors + columnar metadata) → predicate compiler/selectivity estimator → index (Flat ground-truth, hand-written HNSW) → three strategy implementations → cost-model-driven planner → FastAPI service. Benchmarking harness and figure generation sit alongside as `bench/`.

**Tech Stack:** Python 3.12, NumPy, FAISS (`faiss-cpu`, baseline comparison only), FastAPI + uvicorn + Pydantic v2, matplotlib, pandas, pytest, scikit-learn (k-means for correlated metadata only), tqdm, requests.

## Global Constraints

- Python 3.12 (already installed at `C:\Users\ASUS\AppData\Local\Programs\Python\Python312`). Create a project-local venv at `D:\vecdb\.venv`.
- OS is Windows 11; commands run through the Bash tool (Git Bash) unless noted. Use forward slashes and Python's `pathlib` everywhere in source — no hardcoded backslashes.
- Dependencies (pin no further than major version): `numpy`, `faiss-cpu`, `fastapi`, `uvicorn`, `pydantic>=2`, `matplotlib`, `pandas`, `pytest`, `tqdm`, `scikit-learn`, `requests`. No Numba (cut per spec §3).
- Benchmark scale is **100K SIFT subset only** — never build or sweep the full 1M set.
- Dataset source is `http://corpus-texmex.irisa.fr/` — plain HTTP only, the host's HTTPS cert does not validate from this environment.
- Hand-written files (must contain real, author-level algorithmic code, not scaffolding): `vecdb/index/hnsw.py`, `vecdb/index/strategies.py`, `vecdb/planner/cost_model.py`, `vecdb/planner/planner.py`, `vecdb/predicate/compile.py`, `vecdb/predicate/selectivity.py`, `vecdb/bench/harness.py` (recall/tie-handling logic specifically).
- Git: one commit per task (minimum), `git push origin main` at the end of every phase. Never force-push.
- Every "Done when" gate in a phase is a hard checkpoint — do not proceed past a failed gate without either fixing it or explicitly logging it as a documented limitation per spec §6.
- Source technical reference for full algorithm detail, pseudocode, and debugging checklists: `filtered-vector-db-7-day-plan (1).md` (repo root). This plan's code is authoritative for structure/signatures; that file is authoritative for exhaustive algorithm explanation when a task references it.

---

## Phase 1 — Setup

### Task 1: Project scaffold and environment

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `vecdb/__init__.py`, `vecdb/io/__init__.py`, `vecdb/store/__init__.py`, `vecdb/predicate/__init__.py`, `vecdb/index/__init__.py`, `vecdb/planner/__init__.py`, `vecdb/service/__init__.py`, `vecdb/bench/__init__.py`
- Create: `tests/__init__.py`
- Create: `scripts/` (empty dir placeholder via `.gitkeep`)
- Create: `results/.gitkeep`, `results/figures/.gitkeep`

**Interfaces:**
- Produces: an installable package `vecdb` importable after `pip install -e .`; nothing consumed (first task).

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "filtered-vecdb"
version = "0.1.0"
description = "A cost-based query planner for filtered approximate nearest-neighbour search"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "faiss-cpu>=1.8",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "pydantic>=2.6",
    "matplotlib>=3.8",
    "pandas>=2.2",
    "tqdm>=4.66",
    "scikit-learn>=1.4",
    "requests>=2.31",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["vecdb*"]
```

- [ ] **Step 2: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
data/
*.npz
*.npy
results/*.csv
results/*.json
results/figures/*.png
!results/.gitkeep
!results/figures/.gitkeep
.pytest_cache/
*.egg-info/
```

- [ ] **Step 3: Create package directories with empty `__init__.py` files, and empty `tests/__init__.py`, `scripts/.gitkeep`, `results/.gitkeep`, `results/figures/.gitkeep`**

Run: `mkdir -p vecdb/io vecdb/store vecdb/predicate vecdb/index vecdb/planner vecdb/service vecdb/bench tests scripts results/figures && touch vecdb/__init__.py vecdb/io/__init__.py vecdb/store/__init__.py vecdb/predicate/__init__.py vecdb/index/__init__.py vecdb/planner/__init__.py vecdb/service/__init__.py vecdb/bench/__init__.py tests/__init__.py scripts/.gitkeep results/.gitkeep results/figures/.gitkeep`

- [ ] **Step 4: Create venv and install in editable+dev mode**

```bash
cd D:/vecdb
python -m venv .venv
".venv/Scripts/python" -m pip install --upgrade pip
".venv/Scripts/python" -m pip install -e ".[dev]"
```

Expected: install completes with no errors; `faiss-cpu` has a prebuilt Windows wheel for Python 3.12, so this should not require a compiler. If `faiss-cpu` fails to find a wheel, note the exact error in the Phase 1 milestone report — do not silently skip FAISS, since it is needed as a baseline in Phase 2.

- [ ] **Step 5: Verify the package imports**

Run: `".venv/Scripts/python" -c "import vecdb, numpy, faiss, fastapi, pandas, matplotlib, sklearn; print('ok')"`
Expected: prints `ok`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore vecdb tests scripts results
git commit -m "chore: project scaffold, package layout, dependency install"
```

---

### Task 2: SIFT `.fvecs`/`.ivecs` readers

**Files:**
- Create: `vecdb/io/fvecs.py`
- Test: `tests/test_fvecs.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `read_fvecs(path: str | Path) -> np.ndarray` (float32, shape `(n, d)`), `read_ivecs(path: str | Path) -> np.ndarray` (int32, shape `(n, d)`), `write_fvecs(path, arr: np.ndarray) -> None` (used only by the test to build a fixture).

The `.fvecs`/`.ivecs` format: each vector is stored as a little-endian int32 dimension `d`, followed by `d` little-endian float32 (fvecs) or int32 (ivecs) values, back to back with no header.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fvecs.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_fvecs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vecdb.io.fvecs'`

- [ ] **Step 3: Implement**

```python
# vecdb/io/fvecs.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_fvecs.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add vecdb/io/fvecs.py tests/test_fvecs.py
git commit -m "feat: fvecs/ivecs binary format readers"
```

---

### Task 3: Dataset download and cache

**Files:**
- Create: `vecdb/io/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `read_fvecs`, `read_ivecs` from `vecdb.io.fvecs`.
- Produces: `download_siftsmall(cache_dir: Path) -> Path` (returns extracted dir), `download_sift1m(cache_dir: Path) -> Path`, `load(name: Literal["siftsmall", "sift1m_100k"], cache_dir: Path = Path("data")) -> DatasetBundle` where `DatasetBundle` is a `@dataclass` with fields `base: np.ndarray` (N×d float32), `queries: np.ndarray` (Q×d float32), `groundtruth: np.ndarray` (Q×k int32, unfiltered top-k ids from the shipped `.ivecs`).

- [ ] **Step 1: Write the failing test (network-free part first)**

```python
# tests/test_dataset.py
import numpy as np
from vecdb.io.dataset import DatasetBundle, _subset_100k

def test_dataset_bundle_is_dataclass_with_expected_fields():
    bundle = DatasetBundle(
        base=np.zeros((10, 4), dtype=np.float32),
        queries=np.zeros((2, 4), dtype=np.float32),
        groundtruth=np.zeros((2, 3), dtype=np.int32),
    )
    assert bundle.base.shape == (10, 4)
    assert bundle.queries.shape == (2, 4)
    assert bundle.groundtruth.shape == (2, 3)

def test_subset_100k_truncates_base_and_remaps_groundtruth():
    # groundtruth ids >= 100k must be dropped from each row, not just clipped
    base = np.arange(150_000 * 4, dtype=np.float32).reshape(150_000, 4)
    gt = np.array([[0, 99_999, 100_000, 149_999]], dtype=np.int32)
    new_base, new_gt = _subset_100k(base, gt, n=100_000)
    assert new_base.shape == (100_000, 4)
    assert list(new_gt[0]) == [0, 99_999]  # ids >= 100k dropped, order preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# vecdb/io/dataset.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import tarfile
import numpy as np
import requests
from tqdm import tqdm

from vecdb.io.fvecs import read_fvecs, read_ivecs

SIFTSMALL_URL = "http://corpus-texmex.irisa.fr/siftsmall.tar.gz"
SIFT1M_URL = "http://corpus-texmex.irisa.fr/sift.tar.gz"


@dataclass
class DatasetBundle:
    base: np.ndarray        # (N, d) float32
    queries: np.ndarray     # (Q, d) float32
    groundtruth: np.ndarray  # (Q, k) int32, unfiltered top-k neighbour ids


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    tmp.rename(dest)


def _extract(archive: Path, into: Path) -> None:
    if into.exists() and any(into.iterdir()):
        return
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(into)


def download_siftsmall(cache_dir: Path) -> Path:
    archive = cache_dir / "siftsmall.tar.gz"
    _download(SIFTSMALL_URL, archive)
    out = cache_dir / "siftsmall"
    _extract(archive, out)
    return out / "siftsmall"


def download_sift1m(cache_dir: Path) -> Path:
    archive = cache_dir / "sift.tar.gz"
    _download(SIFT1M_URL, archive)
    out = cache_dir / "sift1m"
    _extract(archive, out)
    return out / "sift"


def _subset_100k(base: np.ndarray, groundtruth: np.ndarray, n: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """Truncate base to the first n rows; drop groundtruth ids that fall outside the subset
    from each row (do NOT clip/relabel them — a dropped id must not silently become a wrong id)."""
    new_base = base[:n]
    new_gt = [row[row < n] for row in groundtruth]
    max_len = max(len(r) for r in new_gt)
    padded = np.full((groundtruth.shape[0], max_len), -1, dtype=np.int32)
    for i, row in enumerate(new_gt):
        padded[i, : len(row)] = row
    # trim to the shortest common length so the array stays rectangular and valid everywhere
    min_len = min(len(row) for row in new_gt)
    return new_base, padded[:, :min_len]


def load(name: str, cache_dir: Path = Path("data")) -> DatasetBundle:
    cache_dir = Path(cache_dir)
    if name == "siftsmall":
        root = download_siftsmall(cache_dir)
        base = read_fvecs(root / "siftsmall_base.fvecs")
        queries = read_fvecs(root / "siftsmall_query.fvecs")
        gt = read_ivecs(root / "siftsmall_groundtruth.ivecs")
        return DatasetBundle(base=base, queries=queries, groundtruth=gt)
    if name == "sift1m_100k":
        root = download_sift1m(cache_dir)
        base = read_fvecs(root / "sift_base.fvecs")
        queries = read_fvecs(root / "sift_query.fvecs")
        gt = read_ivecs(root / "sift_groundtruth.ivecs")
        base, gt = _subset_100k(base, gt, n=100_000)
        return DatasetBundle(base=base, queries=queries, groundtruth=gt)
    raise ValueError(f"unknown dataset name: {name!r}")
```

- [ ] **Step 4: Run the offline tests**

Run: `".venv/Scripts/python" -m pytest tests/test_dataset.py -v`
Expected: PASS (2 tests) — these do not touch the network.

- [ ] **Step 5: Live download smoke test (run manually, not part of `pytest`)**

Run: `".venv/Scripts/python" -c "from vecdb.io.dataset import load; d = load('siftsmall'); print(d.base.shape, d.queries.shape, d.groundtruth.shape)"`
Expected: `(10000, 128) (100, 128) (100, 100)`. This is the Phase 1 "Done when" gate from spec/source-plan Day 0 — record the actual printed shapes in the Phase 1 milestone report.

- [ ] **Step 6: Commit**

```bash
git add vecdb/io/dataset.py tests/test_dataset.py
git commit -m "feat: SIFT dataset download, cache, and 100K subset loader"
```

---

### Task 4: Synthetic metadata generator

**Files:**
- Create: `vecdb/io/metadata_gen.py`
- Test: `tests/test_metadata_gen.py`

**Interfaces:**
- Consumes: nothing beyond a `np.ndarray` of base vectors (for the correlated variant's k-means).
- Produces: `generate_uncorrelated(n: int, seed: int = 0) -> dict[str, np.ndarray]` with keys `category` (int32 codes 0-99), `year` (int16, 2000-2025), `score` (float32, U(0,1)); `generate_correlated(vectors: np.ndarray, n_clusters: int = 100, agree_prob: float = 0.85, seed: int = 0) -> dict[str, np.ndarray]` with the same keys, where `category` is k-means cluster id with probability `agree_prob` and a uniform-random category otherwise; `save_metadata(path: Path, columns: dict[str, np.ndarray]) -> None` / `load_metadata(path: Path) -> dict[str, np.ndarray]` via `np.savez`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metadata_gen.py
import numpy as np
from vecdb.io.metadata_gen import generate_uncorrelated, generate_correlated, save_metadata, load_metadata

def test_uncorrelated_has_expected_columns_and_ranges():
    cols = generate_uncorrelated(1000, seed=0)
    assert set(cols) == {"category", "year", "score"}
    assert cols["category"].shape == (1000,)
    assert cols["category"].min() >= 0 and cols["category"].max() <= 99
    assert cols["year"].min() >= 2000 and cols["year"].max() <= 2025
    assert cols["score"].min() >= 0.0 and cols["score"].max() <= 1.0

def test_correlated_category_tracks_kmeans_cluster_mostly():
    rng = np.random.default_rng(0)
    # two well-separated blobs so k-means clustering is unambiguous
    blob_a = rng.normal(loc=0.0, scale=0.1, size=(200, 8))
    blob_b = rng.normal(loc=10.0, scale=0.1, size=(200, 8))
    vectors = np.vstack([blob_a, blob_b]).astype(np.float32)
    cols = generate_correlated(vectors, n_clusters=2, agree_prob=1.0, seed=0)
    # with agree_prob=1.0, every point's category must exactly equal its cluster id
    assert len(set(cols["category"][:200])) == 1
    assert len(set(cols["category"][200:])) == 1
    assert cols["category"][0] != cols["category"][200]

def test_save_and_load_metadata_roundtrip(tmp_path):
    cols = generate_uncorrelated(50, seed=1)
    path = tmp_path / "meta.npz"
    save_metadata(path, cols)
    loaded = load_metadata(path)
    for key in cols:
        np.testing.assert_array_equal(loaded[key], cols[key])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_metadata_gen.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# vecdb/io/metadata_gen.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_metadata_gen.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add vecdb/io/metadata_gen.py tests/test_metadata_gen.py
git commit -m "feat: synthetic metadata generator (uncorrelated + k-means correlated)"
```

---

### Task 5: README skeleton and Phase 1 milestone report

**Files:**
- Create: `README.md`
- Create: `docs/superpowers/milestones/01-setup.md`
- Create: `scripts/download_data.py` (thin CLI wrapper so the source plan's "Definition of Done" clean-clone flow has a real entry point)

**Interfaces:**
- Consumes: `vecdb.io.dataset.load`, `vecdb.io.metadata_gen.generate_uncorrelated`, `generate_correlated`, `save_metadata`.
- Produces: nothing consumed by later tasks — this is a documentation/UX checkpoint.

- [ ] **Step 1: Write `scripts/download_data.py`**

```python
# scripts/download_data.py
"""Downloads siftsmall + the 100K SIFT subset and generates synthetic metadata for both."""
from pathlib import Path
from vecdb.io.dataset import load
from vecdb.io.metadata_gen import generate_uncorrelated, generate_correlated, save_metadata

DATA_DIR = Path("data")


def main() -> None:
    small = load("siftsmall", cache_dir=DATA_DIR)
    print(f"siftsmall: base={small.base.shape} queries={small.queries.shape} gt={small.groundtruth.shape}")

    full = load("sift1m_100k", cache_dir=DATA_DIR)
    print(f"sift1m_100k: base={full.base.shape} queries={full.queries.shape} gt={full.groundtruth.shape}")

    for name, bundle in [("siftsmall", small), ("sift1m_100k", full)]:
        uncorr = generate_uncorrelated(bundle.base.shape[0], seed=0)
        save_metadata(DATA_DIR / f"{name}_meta_uncorrelated.npz", uncorr)
        corr = generate_correlated(bundle.base, n_clusters=100, agree_prob=0.85, seed=0)
        save_metadata(DATA_DIR / f"{name}_meta_correlated.npz", corr)
        print(f"{name}: metadata generated (uncorrelated + correlated)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `".venv/Scripts/python" scripts/download_data.py`
Expected: prints shapes for both datasets and confirms metadata generation for both. This will take real wall-clock time for the 100K download+k-means — run it and wait for completion rather than backgrounding it, since Phase 2 needs the cached data on disk.

- [ ] **Step 3: Write `README.md` skeleton**

```markdown
# filtered-vecdb

> A cost-based query planner for approximate nearest-neighbour search that chooses
> between pre-filtering, post-filtering, and predicate-aware graph traversal using
> estimated predicate selectivity — with a measured crossover curve showing exactly
> where each strategy wins and why.

## Results

*(filled in during Phase 7 — crossover chart goes here, above the fold)*

## Architecture

*(filled in during Phase 8)*

## How to run

*(filled in during Phase 8 — must work from a clean clone)*

## What I got wrong / limitations

*(filled in during Phase 8)*

## What I'd do next

*(filled in during Phase 8)*

## References

- Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using
  Hierarchical Navigable Small World graphs" (2016)
- Gollapudi et al., "Filtered-DiskANN" (2023)
- Patel et al., "ACORN" (2024)
- Selinger et al., "Access Path Selection in a Relational Database Management System" (1979)
```

- [ ] **Step 4: Write the Phase 1 milestone report**

Create `docs/superpowers/milestones/01-setup.md` containing exactly these sections, filled with real values from this phase's runs (not placeholders):
- **What got built:** one paragraph, plain language.
- **Numbers:** the actual printed shapes from Task 3 Step 5 and this task's Step 2, plus how long the 100K download + correlated metadata generation took.
- **Gate status:** whether `data.base.shape == (10000, 128)` etc. held exactly.
- **Interview note:** 3-5 sentences explaining, in plain language, why SIFT + synthetic metadata (not real embeddings) is the right benchmark choice here, referencing source plan §4.1.

- [ ] **Step 5: Commit and push**

```bash
git add README.md scripts/download_data.py docs/superpowers/milestones/01-setup.md
git commit -m "docs: README skeleton, data download script, Phase 1 milestone report"
git push origin main
```

---

## Phase 2 — Storage, predicates, Flat baseline, benchmark harness

### Task 6: VectorStore

**Files:**
- Create: `vecdb/store/vectors.py`
- Test: `tests/test_vectors.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `class VectorStore` with `__init__(self, data: np.ndarray)` (copies into a `(N, d)` float32 memmap-backed array), `.data: np.ndarray`, `.sq_norms: np.ndarray` (N,), `.n_distance_ops: int`, `.distances(self, q: np.ndarray, rows: np.ndarray) -> np.ndarray` (returns squared L2 distances, one per row in `rows`, and increments `n_distance_ops` by `len(rows)`), `.vector(self, row: int) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vectors.py
import numpy as np
from vecdb.store.vectors import VectorStore

def test_distances_match_brute_force_l2_squared():
    rng = np.random.default_rng(0)
    data = rng.random((50, 8)).astype(np.float32)
    store = VectorStore(data)
    q = rng.random(8).astype(np.float32)
    rows = np.array([0, 10, 25, 49])
    got = store.distances(q, rows)
    expected = np.sum((data[rows] - q) ** 2, axis=1)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

def test_distances_increments_counter():
    data = np.zeros((10, 4), dtype=np.float32)
    store = VectorStore(data)
    assert store.n_distance_ops == 0
    store.distances(np.zeros(4, dtype=np.float32), np.array([0, 1, 2]))
    assert store.n_distance_ops == 3
    store.distances(np.zeros(4, dtype=np.float32), np.array([3]))
    assert store.n_distance_ops == 4

def test_distances_empty_rows_returns_empty_array_and_no_crash():
    data = np.ones((5, 4), dtype=np.float32)
    store = VectorStore(data)
    result = store.distances(np.zeros(4, dtype=np.float32), np.array([], dtype=np.int64))
    assert result.shape == (0,)
    assert store.n_distance_ops == 0

def test_vector_returns_single_row():
    data = np.arange(20, dtype=np.float32).reshape(5, 4)
    store = VectorStore(data)
    np.testing.assert_array_equal(store.vector(2), data[2])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_vectors.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# vecdb/store/vectors.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_vectors.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add vecdb/store/vectors.py tests/test_vectors.py
git commit -m "feat: VectorStore with vectorised batched L2 distance and dist_ops counter"
```

---

### Task 7: MetaStore and column stats

**Files:**
- Create: `vecdb/store/metadata.py`
- Test: `tests/test_metastore.py`

**Interfaces:**
- Consumes: `dict[str, np.ndarray]` columns as produced by `vecdb.io.metadata_gen`.
- Produces: `@dataclass ColumnStats` with `kind: Literal["categorical", "numeric"]`, `value_counts: dict | None`, `hist_edges: np.ndarray | None`, `hist_counts: np.ndarray | None`, `n: int`; `class MetaStore` with `__init__(self, columns: dict[str, np.ndarray])`, `.columns: dict[str, np.ndarray]`, `.stats: dict[str, ColumnStats]`, `.n: int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metastore.py
import numpy as np
from vecdb.store.metadata import MetaStore

def test_categorical_stats_are_exact_value_counts():
    columns = {"category": np.array([0, 0, 1, 2, 2, 2], dtype=np.int32)}
    store = MetaStore(columns)
    stats = store.stats["category"]
    assert stats.kind == "categorical"
    assert stats.value_counts == {0: 2, 1: 1, 2: 3}
    assert stats.n == 6

def test_numeric_stats_build_64_bin_histogram_covering_full_range():
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, size=1000).astype(np.float32)
    store = MetaStore({"score": values})
    stats = store.stats["score"]
    assert stats.kind == "numeric"
    assert len(stats.hist_edges) == 65  # 64 bins -> 65 edges
    assert stats.hist_counts.sum() == 1000
    assert stats.hist_edges[0] <= values.min()
    assert stats.hist_edges[-1] >= values.max()

def test_metastore_exposes_row_count_and_raw_columns():
    columns = {"category": np.array([0, 1, 2], dtype=np.int32), "year": np.array([2020, 2021, 2022], dtype=np.int16)}
    store = MetaStore(columns)
    assert store.n == 3
    np.testing.assert_array_equal(store.columns["year"], columns["year"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_metastore.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# vecdb/store/metadata.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Columns whose dtype is one of these are treated as categorical, not numeric.
_CATEGORICAL_DTYPES = (np.int32,)


@dataclass
class ColumnStats:
    kind: str  # "categorical" | "numeric"
    n: int
    value_counts: dict | None = None
    hist_edges: np.ndarray | None = None
    hist_counts: np.ndarray | None = None


def _compute_stats(values: np.ndarray) -> ColumnStats:
    n = values.shape[0]
    if values.dtype == np.int32:
        keys, counts = np.unique(values, return_counts=True)
        return ColumnStats(kind="categorical", n=n, value_counts=dict(zip(keys.tolist(), counts.tolist())))
    counts, edges = np.histogram(values, bins=64)
    return ColumnStats(kind="numeric", n=n, hist_edges=edges, hist_counts=counts)


class MetaStore:
    """Columnar attributes, one np.ndarray per column, row-aligned with VectorStore.
    'category' (int32) is treated as categorical; everything else (year, score, ...)
    is treated as numeric with a 64-bin histogram."""

    def __init__(self, columns: dict[str, np.ndarray]):
        self.columns: dict[str, np.ndarray] = columns
        lengths = {len(v) for v in columns.values()}
        assert len(lengths) <= 1, "all columns must have the same length"
        self.n: int = next(iter(lengths), 0)
        self.stats: dict[str, ColumnStats] = {name: _compute_stats(col) for name, col in columns.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_metastore.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add vecdb/store/metadata.py tests/test_metastore.py
git commit -m "feat: MetaStore with categorical value-count and numeric histogram stats"
```

---

### Task 8: IdMap

**Files:**
- Create: `vecdb/store/idmap.py`
- Test: `tests/test_idmap.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `class IdMap` with `.add(self, external_id) -> int` (returns assigned internal row index, sequential from 0), `.to_internal(self, external_id) -> int`, `.to_external(self, internal_row: int)`, `.__len__(self) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_idmap.py
import pytest
from vecdb.store.idmap import IdMap

def test_add_assigns_sequential_internal_rows():
    m = IdMap()
    assert m.add("a") == 0
    assert m.add("b") == 1
    assert m.add("c") == 2
    assert len(m) == 3

def test_round_trip_external_to_internal_and_back():
    m = IdMap()
    m.add("x")
    m.add("y")
    assert m.to_internal("y") == 1
    assert m.to_external(1) == "y"

def test_duplicate_external_id_raises():
    m = IdMap()
    m.add("a")
    with pytest.raises(ValueError):
        m.add("a")

def test_unknown_external_id_raises_keyerror():
    m = IdMap()
    with pytest.raises(KeyError):
        m.to_internal("missing")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_idmap.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# vecdb/store/idmap.py
from __future__ import annotations


class IdMap:
    """External id <-> internal dense row index, assigned sequentially on add()."""

    def __init__(self):
        self._ext_to_int: dict = {}
        self._int_to_ext: list = []

    def add(self, external_id) -> int:
        if external_id in self._ext_to_int:
            raise ValueError(f"duplicate external id: {external_id!r}")
        internal = len(self._int_to_ext)
        self._ext_to_int[external_id] = internal
        self._int_to_ext.append(external_id)
        return internal

    def to_internal(self, external_id) -> int:
        return self._ext_to_int[external_id]

    def to_external(self, internal_row: int):
        return self._int_to_ext[internal_row]

    def __len__(self) -> int:
        return len(self._int_to_ext)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_idmap.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add vecdb/store/idmap.py tests/test_idmap.py
git commit -m "feat: IdMap for external/internal id translation"
```

---

### Task 9: Predicate DSL and compiler

**Files:**
- Create: `vecdb/predicate/dsl.py`
- Create: `vecdb/predicate/compile.py`
- Test: `tests/test_predicate.py`

**Interfaces:**
- Consumes: `vecdb.store.metadata.MetaStore`.
- Produces: `validate_predicate(pred: dict) -> None` (raises `ValueError` on malformed input), `compile(pred: dict, meta: MetaStore) -> np.ndarray` (bool mask, shape `(meta.n,)`).

Wire format (JSON-compatible nested dict): `{"op": "eq"|"ne"|"lt"|"lte"|"gt"|"gte"|"in", "col": str, "val": Any}` for leaves, `{"op": "and"|"or", "clauses": [pred, ...]}`, `{"op": "not", "clause": pred}`. Categorical columns (`category`) compare against integer codes directly — this project does not implement a string-label-to-code table, since the synthetic metadata generator already emits integer codes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_predicate.py
import numpy as np
import pytest
from vecdb.store.metadata import MetaStore
from vecdb.predicate.dsl import validate_predicate
from vecdb.predicate.compile import compile as compile_pred

@pytest.fixture
def meta():
    return MetaStore({
        "category": np.array([0, 1, 1, 2, 0], dtype=np.int32),
        "year": np.array([2018, 2020, 2021, 2019, 2022], dtype=np.int16),
    })

def test_eq_leaf(meta):
    mask = compile_pred({"op": "eq", "col": "category", "val": 1}, meta)
    np.testing.assert_array_equal(mask, [False, True, True, False, False])

def test_gt_leaf(meta):
    mask = compile_pred({"op": "gt", "col": "year", "val": 2019}, meta)
    np.testing.assert_array_equal(mask, [False, True, True, False, True])

def test_and_combines_with_logical_and(meta):
    pred = {"op": "and", "clauses": [
        {"op": "eq", "col": "category", "val": 1},
        {"op": "gt", "col": "year", "val": 2019},
    ]}
    mask = compile_pred(pred, meta)
    np.testing.assert_array_equal(mask, [False, True, True, False, False])

def test_or_combines_with_logical_or(meta):
    pred = {"op": "or", "clauses": [
        {"op": "eq", "col": "category", "val": 2},
        {"op": "eq", "col": "category", "val": 0},
    ]}
    mask = compile_pred(pred, meta)
    np.testing.assert_array_equal(mask, [True, False, False, True, True])

def test_not_inverts(meta):
    mask = compile_pred({"op": "not", "clause": {"op": "eq", "col": "category", "val": 1}}, meta)
    np.testing.assert_array_equal(mask, [True, False, False, True, True])

def test_in_op(meta):
    mask = compile_pred({"op": "in", "col": "category", "val": [0, 2]}, meta)
    np.testing.assert_array_equal(mask, [True, False, False, True, True])

def test_empty_mask_when_no_rows_match(meta):
    mask = compile_pred({"op": "eq", "col": "category", "val": 999}, meta)
    assert mask.sum() == 0
    assert mask.shape == (5,)

def test_full_mask_for_always_true_predicate(meta):
    mask = compile_pred({"op": "gte", "col": "year", "val": 0}, meta)
    assert mask.sum() == 5

def test_validate_predicate_rejects_unknown_op():
    with pytest.raises(ValueError):
        validate_predicate({"op": "xor", "col": "category", "val": 1})

def test_validate_predicate_accepts_nested_valid_predicate():
    pred = {"op": "and", "clauses": [
        {"op": "eq", "col": "category", "val": 1},
        {"op": "not", "clause": {"op": "lt", "col": "year", "val": 2000}},
    ]}
    validate_predicate(pred)  # should not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_predicate.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `vecdb/predicate/dsl.py`**

```python
# vecdb/predicate/dsl.py
from __future__ import annotations

LEAF_OPS = {"eq", "ne", "lt", "lte", "gt", "gte", "in"}
COMBINATOR_OPS = {"and", "or"}


def validate_predicate(pred: dict) -> None:
    if not isinstance(pred, dict) or "op" not in pred:
        raise ValueError(f"predicate must be a dict with an 'op' key, got {pred!r}")
    op = pred["op"]
    if op in LEAF_OPS:
        if "col" not in pred or "val" not in pred:
            raise ValueError(f"leaf predicate {pred!r} must have 'col' and 'val'")
        return
    if op in COMBINATOR_OPS:
        if "clauses" not in pred or not isinstance(pred["clauses"], list) or not pred["clauses"]:
            raise ValueError(f"'{op}' predicate must have a non-empty 'clauses' list")
        for clause in pred["clauses"]:
            validate_predicate(clause)
        return
    if op == "not":
        if "clause" not in pred:
            raise ValueError("'not' predicate must have a 'clause'")
        validate_predicate(pred["clause"])
        return
    raise ValueError(f"unsupported predicate op: {op!r}")
```

- [ ] **Step 4: Implement `vecdb/predicate/compile.py`**

```python
# vecdb/predicate/compile.py
from __future__ import annotations
import numpy as np
from vecdb.store.metadata import MetaStore


def compile(pred: dict, meta: MetaStore) -> np.ndarray:
    op = pred["op"]
    if op == "and":
        result = compile(pred["clauses"][0], meta).copy()
        for clause in pred["clauses"][1:]:
            result &= compile(clause, meta)
        return result
    if op == "or":
        result = compile(pred["clauses"][0], meta).copy()
        for clause in pred["clauses"][1:]:
            result |= compile(clause, meta)
        return result
    if op == "not":
        return ~compile(pred["clause"], meta)

    col = meta.columns[pred["col"]]
    val = pred["val"]
    if op == "eq":
        return col == val
    if op == "ne":
        return col != val
    if op == "lt":
        return col < val
    if op == "lte":
        return col <= val
    if op == "gt":
        return col > val
    if op == "gte":
        return col >= val
    if op == "in":
        return np.isin(col, val)
    raise ValueError(f"unsupported op: {op!r}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_predicate.py -v`
Expected: PASS (10 tests)

- [ ] **Step 6: Commit**

```bash
git add vecdb/predicate/dsl.py vecdb/predicate/compile.py tests/test_predicate.py
git commit -m "feat: predicate DSL validation and AST-to-boolean-mask compiler"
```

---

### Task 10: Selectivity estimator

**Files:**
- Create: `vecdb/predicate/selectivity.py`
- Test: `tests/test_selectivity.py`

**Interfaces:**
- Consumes: `vecdb.store.metadata.MetaStore`, `ColumnStats`.
- Produces: `estimate_selectivity(pred: dict, meta: MetaStore) -> float`. Must read only `meta.stats` — never touch `meta.columns` or materialise a mask. This is the whole point: the planner decides before doing anything expensive.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_selectivity.py
import numpy as np
import pytest
from vecdb.store.metadata import MetaStore
from vecdb.predicate.selectivity import estimate_selectivity

@pytest.fixture
def meta():
    rng = np.random.default_rng(0)
    return MetaStore({
        "category": rng.integers(0, 10, size=10_000).astype(np.int32),
        "score": rng.uniform(0, 1, size=10_000).astype(np.float32),
    })

def test_categorical_eq_matches_value_count_fraction(meta):
    s = estimate_selectivity({"op": "eq", "col": "category", "val": 3}, meta)
    expected = meta.stats["category"].value_counts.get(3, 0) / meta.n
    assert s == pytest.approx(expected)

def test_categorical_eq_missing_value_is_zero(meta):
    s = estimate_selectivity({"op": "eq", "col": "category", "val": 999}, meta)
    assert s == 0.0

def test_numeric_range_is_close_to_true_uniform_fraction(meta):
    # score ~ U(0,1) with 10k samples: P(score < 0.3) should estimate near 0.3
    s = estimate_selectivity({"op": "lt", "col": "score", "val": 0.3}, meta)
    assert abs(s - 0.3) < 0.03

def test_and_uses_independence_assumption(meta):
    pred = {"op": "and", "clauses": [
        {"op": "eq", "col": "category", "val": 3},
        {"op": "lt", "col": "score", "val": 0.5},
    ]}
    s = estimate_selectivity(pred, meta)
    s1 = estimate_selectivity(pred["clauses"][0], meta)
    s2 = estimate_selectivity(pred["clauses"][1], meta)
    assert s == pytest.approx(s1 * s2, rel=1e-6)

def test_or_uses_inclusion_exclusion(meta):
    pred = {"op": "or", "clauses": [
        {"op": "eq", "col": "category", "val": 1},
        {"op": "eq", "col": "category", "val": 2},
    ]}
    s = estimate_selectivity(pred, meta)
    s1 = estimate_selectivity(pred["clauses"][0], meta)
    s2 = estimate_selectivity(pred["clauses"][1], meta)
    assert s == pytest.approx(s1 + s2 - s1 * s2, rel=1e-6)

def test_not_is_one_minus_selectivity(meta):
    inner = {"op": "eq", "col": "category", "val": 3}
    s_not = estimate_selectivity({"op": "not", "clause": inner}, meta)
    s_inner = estimate_selectivity(inner, meta)
    assert s_not == pytest.approx(1.0 - s_inner)

def test_estimate_never_touches_columns_directly(meta):
    # sabotage the raw column so a correct implementation (stats-only) is unaffected
    meta.columns["category"] = None
    s = estimate_selectivity({"op": "eq", "col": "category", "val": 3}, meta)
    assert 0.0 <= s <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_selectivity.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# vecdb/predicate/selectivity.py
from __future__ import annotations
import numpy as np
from vecdb.store.metadata import MetaStore


def _frac_below(stats, x: float) -> float:
    """Fraction of values < x, via linear interpolation within the histogram bin
    containing x. Reads only stats.hist_edges / stats.hist_counts / stats.n."""
    edges, counts, n = stats.hist_edges, stats.hist_counts, stats.n
    if x <= edges[0]:
        return 0.0
    if x >= edges[-1]:
        return 1.0
    idx = int(np.searchsorted(edges, x, side="right") - 1)
    idx = min(max(idx, 0), len(counts) - 1)
    bin_lo, bin_hi = edges[idx], edges[idx + 1]
    bin_frac = (x - bin_lo) / (bin_hi - bin_lo) if bin_hi > bin_lo else 0.0
    prior_count = counts[:idx].sum()
    within_bin = bin_frac * counts[idx]
    return float((prior_count + within_bin) / n)


def estimate_selectivity(pred: dict, meta: MetaStore) -> float:
    op = pred["op"]
    if op == "and":
        result = 1.0
        for clause in pred["clauses"]:
            result *= estimate_selectivity(clause, meta)
        return result
    if op == "or":
        result = 0.0
        for clause in pred["clauses"]:
            s = estimate_selectivity(clause, meta)
            result = result + s - result * s
        return result
    if op == "not":
        return 1.0 - estimate_selectivity(pred["clause"], meta)

    stats = meta.stats[pred["col"]]
    val = pred["val"]
    if stats.kind == "categorical":
        if op == "eq":
            return stats.value_counts.get(val, 0) / stats.n
        if op == "ne":
            return 1.0 - stats.value_counts.get(val, 0) / stats.n
        if op == "in":
            return sum(stats.value_counts.get(v, 0) for v in val) / stats.n
        raise ValueError(f"unsupported categorical op for estimation: {op!r}")

    if op in ("lt", "lte"):
        return min(max(_frac_below(stats, val), 0.0), 1.0)
    if op in ("gt", "gte"):
        return min(max(1.0 - _frac_below(stats, val), 0.0), 1.0)
    raise ValueError(f"unsupported numeric op for estimation: {op!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_selectivity.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add vecdb/predicate/selectivity.py tests/test_selectivity.py
git commit -m "feat: stats-only selectivity estimator with independence assumption for AND"
```

---

### Task 11: `Index` base interface and `FlatIndex`

**Files:**
- Create: `vecdb/index/base.py`
- Create: `vecdb/index/flat.py`
- Test: `tests/test_flat_index.py`

**Interfaces:**
- Consumes: `vecdb.store.vectors.VectorStore`.
- Produces: `@dataclass SearchResult(ids, distances, n_distance_ops, strategy, latency_ms, n_returned)`, `abstract class Index` with `add(vectors, ids)` and `search(q, k, mask=None, params=None) -> SearchResult`, `class FlatIndex(Index)` — exact brute-force search, optionally masked.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_flat_index.py
import numpy as np
from vecdb.index.flat import FlatIndex

def test_unfiltered_search_matches_brute_force_ranking():
    rng = np.random.default_rng(0)
    data = rng.random((200, 16)).astype(np.float32)
    idx = FlatIndex()
    idx.add(data, np.arange(200))
    q = rng.random(16).astype(np.float32)
    result = idx.search(q, k=5)
    expected_order = np.argsort(np.sum((data - q) ** 2, axis=1))[:5]
    np.testing.assert_array_equal(result.ids, expected_order)
    assert result.n_returned == 5
    assert result.n_distance_ops == 200

def test_masked_search_only_returns_matching_rows():
    data = np.arange(40, dtype=np.float32).reshape(10, 4)
    idx = FlatIndex()
    idx.add(data, np.arange(10))
    mask = np.zeros(10, dtype=bool)
    mask[[2, 5, 7]] = True
    result = idx.search(np.zeros(4, dtype=np.float32), k=10, mask=mask)
    assert set(result.ids.tolist()) == {2, 5, 7}
    assert result.n_returned == 3  # fewer than k because only 3 rows match
    assert result.n_distance_ops == 3

def test_empty_mask_returns_empty_result_without_crash():
    data = np.ones((5, 4), dtype=np.float32)
    idx = FlatIndex()
    idx.add(data, np.arange(5))
    mask = np.zeros(5, dtype=bool)
    result = idx.search(np.zeros(4, dtype=np.float32), k=3, mask=mask)
    assert result.n_returned == 0
    assert result.ids.shape == (0,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_flat_index.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `vecdb/index/base.py`**

```python
# vecdb/index/base.py
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
```

- [ ] **Step 4: Implement `vecdb/index/flat.py`**

```python
# vecdb/index/flat.py
from __future__ import annotations
import time
import numpy as np
from vecdb.store.vectors import VectorStore
from vecdb.index.base import Index, SearchResult


class FlatIndex(Index):
    """Exact search over all (or masked) rows. This is the ground-truth oracle
    every other index/strategy is measured against."""

    def __init__(self):
        self.store: VectorStore | None = None

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        assert np.array_equal(np.asarray(ids), np.arange(len(ids))), \
            "FlatIndex expects dense internal ids 0..N-1"
        self.store = VectorStore(vectors)

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None,
                params: dict | None = None) -> SearchResult:
        assert self.store is not None, "call add() first"
        t0 = time.perf_counter()
        rows = np.nonzero(mask)[0] if mask is not None else np.arange(len(self.store))
        if rows.size == 0:
            return SearchResult(ids=np.array([], dtype=np.int64), distances=np.array([], dtype=np.float32),
                                 n_distance_ops=0, strategy="flat",
                                 latency_ms=(time.perf_counter() - t0) * 1000, n_returned=0)
        d = self.store.distances(q, rows)
        k_eff = min(k, rows.size)
        part = np.argpartition(d, k_eff - 1)[:k_eff]
        order = part[np.argsort(d[part])]
        result_ids = rows[order]
        result_d = d[order]
        return SearchResult(ids=result_ids, distances=result_d, n_distance_ops=int(rows.size),
                             strategy="flat", latency_ms=(time.perf_counter() - t0) * 1000,
                             n_returned=int(result_ids.size))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_flat_index.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add vecdb/index/base.py vecdb/index/flat.py tests/test_flat_index.py
git commit -m "feat: Index interface, SearchResult, and exact FlatIndex ground-truth oracle"
```

---

### Task 12: Benchmark harness and ground-truth caching

**Files:**
- Create: `vecdb/bench/harness.py`
- Create: `vecdb/bench/groundtruth.py`
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: `vecdb.index.base.Index`, `vecdb.index.flat.FlatIndex`.
- Produces: `recall_at_k(returned_ids, true_ids, k) -> float`, `run_benchmark(index, queries, groundtruth, k, masks=None, params=None, warmup=0) -> pandas.DataFrame` (columns: `query_idx, recall, underfill, latency_ms, dist_ops, n_returned, strategy`), `compute_filtered_groundtruth(flat_index, queries, masks, k) -> list[np.ndarray]`, `cache_groundtruth(path, groundtruth) -> None`, `load_groundtruth(path) -> list[np.ndarray]`.

Spec §4.4's `qps` metric is deliberately not a separate column here — it's `1000 / latency_ms` per query, or `n_queries / total_wall_seconds` in batch mode, both trivially derivable from the `latency_ms` column already produced. Any milestone report that wants a qps number computes it from this DataFrame rather than the harness storing a redundant column.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness.py
import numpy as np
from pathlib import Path
from vecdb.bench.harness import recall_at_k, run_benchmark
from vecdb.bench.groundtruth import compute_filtered_groundtruth, cache_groundtruth, load_groundtruth
from vecdb.index.flat import FlatIndex

def test_recall_at_k_perfect_match_is_one():
    assert recall_at_k(np.array([1, 2, 3]), np.array([1, 2, 3]), k=3) == 1.0

def test_recall_at_k_partial_overlap():
    assert recall_at_k(np.array([1, 2, 9]), np.array([1, 2, 3]), k=3) == pytest_approx(2 / 3)

def pytest_approx(x, tol=1e-9):
    class _Approx(float):
        def __eq__(self, other):
            return abs(other - x) < tol
    return _Approx(x)

def test_recall_at_k_does_not_exceed_one_when_true_set_shorter_than_k():
    # true set has only 2 survivors (a highly selective filter) but k=5 was requested
    r = recall_at_k(np.array([1, 2, 3, 4, 5]), np.array([1, 2]), k=5)
    assert r <= 1.0
    assert r == 1.0  # both true survivors were returned

def test_recall_at_k_both_empty_is_vacuously_one():
    assert recall_at_k(np.array([]), np.array([]), k=5) == 1.0

def test_run_benchmark_produces_expected_columns_and_skips_warmup():
    rng = np.random.default_rng(0)
    data = rng.random((100, 8)).astype(np.float32)
    idx = FlatIndex()
    idx.add(data, np.arange(100))
    queries = rng.random((10, 8)).astype(np.float32)
    groundtruth = [idx.search(q, k=5).ids for q in queries]
    df = run_benchmark(idx, queries, groundtruth, k=5, warmup=2)
    assert len(df) == 8  # 10 queries - 2 warmup
    assert set(df.columns) == {"query_idx", "recall", "underfill", "latency_ms", "dist_ops", "n_returned", "strategy"}
    assert (df["recall"] == 1.0).all()  # FlatIndex vs its own ground truth is exact

def test_groundtruth_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.random((50, 8)).astype(np.float32)
    idx = FlatIndex()
    idx.add(data, np.arange(50))
    queries = rng.random((5, 8)).astype(np.float32)
    masks = [rng.random(50) < 0.3 for _ in range(5)]
    gt = compute_filtered_groundtruth(idx, queries, masks, k=5)
    path = tmp_path / "gt.npy"
    cache_groundtruth(path, gt)
    loaded = load_groundtruth(path)
    for original, restored in zip(gt, loaded):
        np.testing.assert_array_equal(np.sort(original), np.sort(restored))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_harness.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `vecdb/bench/harness.py`**

```python
# vecdb/bench/harness.py
from __future__ import annotations
import numpy as np
import pandas as pd
from vecdb.index.base import Index


def recall_at_k(returned_ids: np.ndarray, true_ids: np.ndarray, k: int) -> float:
    """|returned ∩ true| / min(k, len(true)). Using min(k, len(true)) as the
    denominator (not len(true_ids[:k]) blindly, and not a naive len(true)) is what
    prevents recall from exceeding 1.0 when a highly selective filter has fewer
    than k true survivors to begin with."""
    denom = min(k, len(true_ids))
    if denom == 0:
        return 1.0
    returned_set = set(np.asarray(returned_ids[:k]).tolist())
    true_set = set(np.asarray(true_ids[:k]).tolist())
    return len(returned_set & true_set) / denom


def run_benchmark(index: Index, queries: np.ndarray, groundtruth: list[np.ndarray], k: int,
                    masks: list[np.ndarray] | None = None, params: dict | None = None,
                    warmup: int = 0) -> pd.DataFrame:
    rows = []
    for i, q in enumerate(queries):
        mask = masks[i] if masks is not None else None
        result = index.search(q, k, mask=mask, params=params)
        if i < warmup:
            continue
        rows.append({
            "query_idx": i,
            "recall": recall_at_k(result.ids, groundtruth[i], k),
            "underfill": result.n_returned < k,
            "latency_ms": result.latency_ms,
            "dist_ops": result.n_distance_ops,
            "n_returned": result.n_returned,
            "strategy": result.strategy,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Implement `vecdb/bench/groundtruth.py`**

```python
# vecdb/bench/groundtruth.py
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_harness.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add vecdb/bench/harness.py vecdb/bench/groundtruth.py tests/test_harness.py
git commit -m "feat: benchmark harness with tie-safe recall@k and cached ground truth"
```

---

### Task 13: FAISS baselines, end-to-end verification, Phase 2 milestone report

**Files:**
- Create: `vecdb/index/faiss_baseline.py`
- Create: `scripts/verify_baseline.py`
- Test: `tests/test_faiss_baseline.py`
- Create: `docs/superpowers/milestones/02-storage-predicate-flat-baseline.md`

**Interfaces:**
- Consumes: `vecdb.index.base.Index`, `SearchResult`.
- Produces: `class FaissFlatIndex(Index)`, `class FaissHNSWIndex(Index)` (with `.set_ef_search(ef: int)`), both wrapping `faiss.IndexFlatL2` / `faiss.IndexHNSWFlat`. `n_distance_ops` for `FaissHNSWIndex` is read from FAISS's own `faiss.cvar.hnsw_stats.ndis` counter (reset before each search) so the dist_ops comparison in later phases is apples-to-apples.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_faiss_baseline.py
import numpy as np
from vecdb.index.faiss_baseline import FaissFlatIndex, FaissHNSWIndex

def test_faiss_flat_matches_brute_force_exactly():
    rng = np.random.default_rng(0)
    data = rng.random((300, 16)).astype(np.float32)
    idx = FaissFlatIndex(dim=16)
    idx.add(data, np.arange(300))
    q = rng.random(16).astype(np.float32)
    result = idx.search(q, k=5)
    expected_order = np.argsort(np.sum((data - q) ** 2, axis=1))[:5]
    np.testing.assert_array_equal(result.ids, expected_order)
    assert result.n_distance_ops == 300

def test_faiss_hnsw_returns_k_results_and_positive_dist_ops():
    rng = np.random.default_rng(0)
    data = rng.random((500, 16)).astype(np.float32)
    idx = FaissHNSWIndex(dim=16, M=16, ef_construction=100)
    idx.add(data, np.arange(500))
    idx.set_ef_search(64)
    q = rng.random(16).astype(np.float32)
    result = idx.search(q, k=10)
    assert result.n_returned == 10
    assert result.n_distance_ops > 0
    assert result.strategy == "faiss_hnsw"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_faiss_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# vecdb/index/faiss_baseline.py
from __future__ import annotations
import time
import numpy as np
import faiss
from vecdb.index.base import Index, SearchResult


class FaissFlatIndex(Index):
    """Sanity baseline: FAISS's own exact search. Recall against it should be 1.0."""

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None,
                params: dict | None = None) -> SearchResult:
        assert mask is None, "FaissFlatIndex baseline is unfiltered-only"
        t0 = time.perf_counter()
        distances, ids = self.index.search(q.reshape(1, -1).astype(np.float32), k)
        latency_ms = (time.perf_counter() - t0) * 1000
        return SearchResult(ids=ids[0], distances=distances[0], n_distance_ops=self.index.ntotal,
                             strategy="faiss_flat", latency_ms=latency_ms, n_returned=k)


class FaissHNSWIndex(Index):
    """The real comparison target for Phase 4's Pareto curve and Phase 7's dist_ops table."""

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200):
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = ef_construction

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def set_ef_search(self, ef: int) -> None:
        self.index.hnsw.efSearch = ef

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None,
                params: dict | None = None) -> SearchResult:
        assert mask is None, "FaissHNSWIndex baseline is unfiltered-only"
        if params and "ef" in params:
            self.set_ef_search(params["ef"])
        faiss.cvar.hnsw_stats.reset()
        t0 = time.perf_counter()
        distances, ids = self.index.search(q.reshape(1, -1).astype(np.float32), k)
        latency_ms = (time.perf_counter() - t0) * 1000
        return SearchResult(ids=ids[0], distances=distances[0],
                             n_distance_ops=int(faiss.cvar.hnsw_stats.ndis),
                             strategy="faiss_hnsw", latency_ms=latency_ms, n_returned=k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_faiss_baseline.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write `scripts/verify_baseline.py` and run the Phase 2 "Done when" gate**

```python
# scripts/verify_baseline.py
"""Phase 2 gate: FlatIndex unfiltered recall@100 must be exactly 1.000 against the
shipped SIFT ground truth, and FAISS-Flat must match it. Prints a metrics table."""
from pathlib import Path
import numpy as np
from vecdb.io.dataset import load
from vecdb.index.flat import FlatIndex
from vecdb.index.faiss_baseline import FaissFlatIndex
from vecdb.bench.harness import run_benchmark

def main() -> None:
    bundle = load("siftsmall", cache_dir=Path("data"))
    k = 100

    flat = FlatIndex()
    flat.add(bundle.base, np.arange(len(bundle.base)))
    gt = [bundle.groundtruth[i][:k] for i in range(len(bundle.queries))]
    df_flat = run_benchmark(flat, bundle.queries, gt, k=k)

    faiss_flat = FaissFlatIndex(dim=bundle.base.shape[1])
    faiss_flat.add(bundle.base, np.arange(len(bundle.base)))
    df_faiss = run_benchmark(faiss_flat, bundle.queries, gt, k=k)

    print("FlatIndex      recall@100:", df_flat["recall"].mean())
    print("FaissFlatIndex recall@100:", df_faiss["recall"].mean())
    assert df_flat["recall"].mean() == 1.0, "FlatIndex must be exact"
    assert df_faiss["recall"].mean() == 1.0, "FaissFlatIndex must be exact"
    print("Phase 2 gate: PASS")

if __name__ == "__main__":
    main()
```

Run: `".venv/Scripts/python" scripts/verify_baseline.py`
Expected: both recall lines print `1.0`, ending with `Phase 2 gate: PASS`. Record the actual printed values in the milestone report — do not paraphrase them as "1.0" without having seen the real output.

- [ ] **Step 6: Write the Phase 2 milestone report**

Create `docs/superpowers/milestones/02-storage-predicate-flat-baseline.md` with: **What got built** (VectorStore, MetaStore, IdMap, predicate DSL/compiler, selectivity estimator, FlatIndex, benchmark harness, FAISS baselines), **Numbers** (the exact recall values printed by `verify_baseline.py`, and `pytest` pass counts for Tasks 6-13), **Gate status**, **Interview note** — 3-5 sentences on why selectivity estimation must read only precomputed stats and never touch the mask (source plan §3.2), and why recall tie-handling needed `min(k, len(true_ids))` rather than a naive `len(true_ids)` denominator.

- [ ] **Step 7: Run the full test suite so far, commit, and push**

Run: `".venv/Scripts/python" -m pytest -v`
Expected: all tests across Tasks 2-13 PASS.

```bash
git add vecdb/index/faiss_baseline.py scripts/verify_baseline.py tests/test_faiss_baseline.py docs/superpowers/milestones/02-storage-predicate-flat-baseline.md
git commit -m "feat: FAISS Flat/HNSW baselines, Phase 2 gate verification, milestone report"
git push origin main
```

---

## Phase 3 — Hand-written HNSW

This is the longest and most defended phase. All four tasks below build up **one file**, `vecdb/index/hnsw.py`, method by method — read Malkov & Yashunin (2016) Algorithms 1-5 before starting (source plan §8), and use the debugging checklist in source plan §5 Day 2 if recall comes out low. Every method signature introduced here is final — later tasks and Phase 5/6 strategy code depend on these exact names.

### Task 14: HNSW class skeleton, level assignment, unlinked insert

**Files:**
- Create: `vecdb/index/hnsw.py`
- Test: `tests/test_hnsw_levels.py`

**Interfaces:**
- Consumes: `vecdb.store.vectors.VectorStore`, `vecdb.index.base.Index`, `SearchResult`.
- Produces: `class HNSWIndex(Index)` with `__init__(self, dim: int, M: int = 16, ef_construction: int = 200, seed: int = 42)`, attributes `.dim, .M, .M0, .ef_construction, .mL, .entry_point: int, .max_level: int, .levels: list[int], .graph: list[dict[int, list[int]]], .store: VectorStore | None`, method `._assign_level(self) -> int`, method `._ensure_layers(self, level: int) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hnsw_levels.py
import math
import numpy as np
from vecdb.index.hnsw import HNSWIndex

def test_init_sets_expected_defaults():
    idx = HNSWIndex(dim=8, M=16, ef_construction=200, seed=0)
    assert idx.M == 16
    assert idx.M0 == 32
    assert idx.entry_point == -1
    assert idx.max_level == -1
    assert idx.graph == []
    assert idx.mL == pytest_approx(1.0 / math.log(16))

def pytest_approx(x, tol=1e-9):
    class _Approx(float):
        def __eq__(self, other):
            return abs(other - x) < tol
    return _Approx(x)

def test_assign_level_distribution_matches_geometric_decay():
    # With M=16, P(level >= 1) should be roughly 1/16 ~ 6%. Sample many draws.
    idx = HNSWIndex(dim=8, M=16, seed=0)
    levels = [idx._assign_level() for _ in range(20_000)]
    frac_at_least_1 = sum(1 for l in levels if l >= 1) / len(levels)
    assert 0.03 < frac_at_least_1 < 0.10  # loose bound around the theoretical 1/16
    assert min(levels) == 0  # level 0 must be the overwhelming majority

def test_ensure_layers_grows_graph_list_to_cover_level():
    idx = HNSWIndex(dim=8, seed=0)
    idx._ensure_layers(3)
    assert len(idx.graph) == 4  # layers 0,1,2,3
    assert all(isinstance(layer, dict) for layer in idx.graph)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_hnsw_levels.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# vecdb/index/hnsw.py
from __future__ import annotations
import heapq
import math
import pickle
import random
import time
from pathlib import Path
import numpy as np
from vecdb.store.vectors import VectorStore
from vecdb.index.base import Index, SearchResult


class HNSWIndex(Index):
    """Hand-written Hierarchical Navigable Small World index (Malkov & Yashunin, 2016).
    Built up across Tasks 14-18: this task gives the skeleton, level assignment, and
    layer-list growth. Later tasks add search_layer, the neighbour heuristic, the full
    insert() algorithm, and persistence."""

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, seed: int = 42):
        self.dim = dim
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(M)
        self.entry_point: int = -1
        self.max_level: int = -1
        self.levels: list[int] = []
        self.graph: list[dict[int, list[int]]] = []  # graph[layer][node_id] -> neighbour ids
        self.store: VectorStore | None = None
        self.ef_search_default = 50
        self._rng = random.Random(seed)

    def _assign_level(self) -> int:
        return int(-math.log(self._rng.random()) * self.mL)

    def _ensure_layers(self, level: int) -> None:
        while len(self.graph) <= level:
            self.graph.append({})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_hnsw_levels.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add vecdb/index/hnsw.py tests/test_hnsw_levels.py
git commit -m "feat(hnsw): class skeleton, geometric level assignment"
```

---

### Task 15: `_search_layer` — the core greedy-beam routine

**Files:**
- Modify: `vecdb/index/hnsw.py`
- Test: `tests/test_hnsw_search_layer.py`

**Interfaces:**
- Consumes: `self.store.distances`, `self.graph[layer]` (a `dict[int, list[int]]` the test populates directly, since `insert()` doesn't exist yet).
- Produces: `._search_layer(self, q: np.ndarray, entry_points: list[int], ef: int, layer: int) -> list[tuple[float, int]]` — up to `ef` `(distance, node_id)` pairs, sorted ascending by distance.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hnsw_search_layer.py
import numpy as np
from vecdb.index.hnsw import HNSWIndex

def _build_line_graph(idx: HNSWIndex, points: np.ndarray) -> None:
    """A toy graph: node i connects to i-1 and i+1 (a chain), all on layer 0."""
    idx.store = None
    from vecdb.store.vectors import VectorStore
    idx.store = VectorStore(points)
    idx._ensure_layers(0)
    n = points.shape[0]
    for i in range(n):
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < n]
        idx.graph[0][i] = neighbours

def test_search_layer_finds_exact_nearest_on_a_chain():
    # points laid out on a 1D line: 0,1,2,...,9 (as (x,0) vectors) with chain edges
    points = np.array([[float(i), 0.0] for i in range(10)], dtype=np.float32)
    idx = HNSWIndex(dim=2, seed=0)
    _build_line_graph(idx, points)
    q = np.array([7.4, 0.0], dtype=np.float32)
    result = idx._search_layer(q, entry_points=[0], ef=3, layer=0)
    assert result[0][1] == 7  # nearest point to x=7.4 is node 7
    assert len(result) == 3

def test_search_layer_returns_sorted_ascending_by_distance():
    points = np.array([[float(i), 0.0] for i in range(10)], dtype=np.float32)
    idx = HNSWIndex(dim=2, seed=0)
    _build_line_graph(idx, points)
    result = idx._search_layer(np.array([3.0, 0.0], dtype=np.float32), entry_points=[0], ef=5, layer=0)
    dists = [d for d, _ in result]
    assert dists == sorted(dists)

def test_search_layer_respects_ef_budget():
    points = np.array([[float(i), 0.0] for i in range(10)], dtype=np.float32)
    idx = HNSWIndex(dim=2, seed=0)
    _build_line_graph(idx, points)
    result = idx._search_layer(np.array([5.0, 0.0], dtype=np.float32), entry_points=[0], ef=2, layer=0)
    assert len(result) <= 2

def test_search_layer_single_isolated_entry_point_returns_itself():
    points = np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
    idx = HNSWIndex(dim=2, seed=0)
    from vecdb.store.vectors import VectorStore
    idx.store = VectorStore(points)
    idx._ensure_layers(0)
    idx.graph[0][0] = []  # no neighbours at all
    result = idx._search_layer(np.array([0.0, 0.0], dtype=np.float32), entry_points=[0], ef=5, layer=0)
    assert result == [(0.0, 0)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_hnsw_search_layer.py -v`
Expected: FAIL — `AttributeError: 'HNSWIndex' object has no attribute '_search_layer'`

- [ ] **Step 3: Implement — add this method to `vecdb/index/hnsw.py`**

```python
    def _search_layer(self, q: np.ndarray, entry_points: list[int], ef: int, layer: int) -> list[tuple[float, int]]:
        """Greedy beam search on one layer. Three collections: a min-heap of candidates
        to expand, a max-heap of the best `ef` results found so far (so the worst can be
        evicted cheaply), and a visited set. The early-break when the nearest unexplored
        candidate is already worse than our worst kept result is what makes this sub-linear
        instead of a full traversal — see source plan Day 2 debugging checklist if search
        ends up slow despite good recall."""
        visited = set(entry_points)
        candidates: list[tuple[float, int]] = []   # min-heap by distance
        results: list[tuple[float, int]] = []       # max-heap via negated distance

        if entry_points:
            dists = self.store.distances(q, np.array(entry_points, dtype=np.int64))
            for ep, d in zip(entry_points, dists):
                d = float(d)
                heapq.heappush(candidates, (d, ep))
                heapq.heappush(results, (-d, ep))

        while candidates:
            d_c, c = heapq.heappop(candidates)
            worst_d = -results[0][0]
            if d_c > worst_d and len(results) >= ef:
                break
            neighbours = [n for n in self.graph[layer].get(c, []) if n not in visited]
            if not neighbours:
                continue
            visited.update(neighbours)
            dists = self.store.distances(q, np.array(neighbours, dtype=np.int64))
            for n, d in zip(neighbours, dists):
                d = float(d)
                worst_d = -results[0][0] if results else float("inf")
                if len(results) < ef:
                    heapq.heappush(candidates, (d, n))
                    heapq.heappush(results, (-d, n))
                elif d < worst_d:
                    heapq.heappush(candidates, (d, n))
                    heapq.heappush(results, (-d, n))
                    heapq.heappop(results)

        return sorted((-nd, node) for nd, node in results)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_hnsw_search_layer.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add vecdb/index/hnsw.py tests/test_hnsw_search_layer.py
git commit -m "feat(hnsw): search_layer greedy beam with early-break stopping rule"
```

---

### Task 16: `_select_neighbors_heuristic` — Algorithm 4

**Files:**
- Modify: `vecdb/index/hnsw.py`
- Test: `tests/test_hnsw_heuristic.py`

**Interfaces:**
- Consumes: `self.store.vector(node_id)`.
- Produces: `._select_neighbors_heuristic(self, candidates: list[tuple[float, int]], M: int) -> list[tuple[float, int]]`.

**Do not skip this and keep the M closest** — that gives clustered, low-diversity neighbours and measurably hurts recall (source plan §3.2 Day 2 Algorithm 4 note). Iterate candidates by ascending distance to the query; keep a candidate `c` only if it's closer to the query than to every already-kept neighbour.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hnsw_heuristic.py
import numpy as np
from vecdb.index.hnsw import HNSWIndex
from vecdb.store.vectors import VectorStore

def test_heuristic_prefers_diverse_directions_over_pure_nearest():
    # Query at origin. Two candidates are close together in the same direction (c0, c1,
    # nearly collinear with the query) and one is farther but in a different direction (c2).
    # Naive top-2-nearest would pick {c0, c1}; the heuristic should reject c1 because it's
    # closer to c0 than to the query, and pick c2 instead for diversity.
    points = np.array([
        [1.0, 0.0],   # c0: dist^2 to origin = 1
        [1.1, 0.0],   # c1: dist^2 to origin = 1.21, but very close to c0
        [0.0, 1.05],  # c2: dist^2 to origin = 1.1025, different direction from c0
    ], dtype=np.float32)
    idx = HNSWIndex(dim=2, seed=0)
    idx.store = VectorStore(points)
    q = np.array([0.0, 0.0], dtype=np.float32)
    candidates = [(float(np.sum((points[i] - q) ** 2)), i) for i in range(3)]
    selected = idx._select_neighbors_heuristic(candidates, M=2)
    selected_ids = {node for _, node in selected}
    assert selected_ids == {0, 2}
    assert len(selected) == 2

def test_heuristic_respects_m_cap_with_many_diverse_candidates():
    rng = np.random.default_rng(0)
    points = rng.random((20, 4)).astype(np.float32)
    idx = HNSWIndex(dim=4, seed=0)
    idx.store = VectorStore(points)
    q = np.zeros(4, dtype=np.float32)
    candidates = [(float(np.sum((points[i] - q) ** 2)), i) for i in range(20)]
    selected = idx._select_neighbors_heuristic(candidates, M=5)
    assert len(selected) <= 5

def test_heuristic_returns_sorted_by_distance_ascending():
    rng = np.random.default_rng(1)
    points = rng.random((10, 4)).astype(np.float32)
    idx = HNSWIndex(dim=4, seed=0)
    idx.store = VectorStore(points)
    q = np.zeros(4, dtype=np.float32)
    candidates = [(float(np.sum((points[i] - q) ** 2)), i) for i in range(10)]
    selected = idx._select_neighbors_heuristic(candidates, M=10)
    dists = [d for d, _ in selected]
    assert dists == sorted(dists)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_hnsw_heuristic.py -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Implement — add this method to `vecdb/index/hnsw.py`**

```python
    def _select_neighbors_heuristic(self, candidates: list[tuple[float, int]], M: int) -> list[tuple[float, int]]:
        """Algorithm 4 (Malkov & Yashunin). Keep candidate c only if it is closer to
        the query than to every neighbour already selected — this enforces a relative-
        neighbourhood-graph property (diverse directions) instead of M clustered points.
        `candidates` is (distance_to_query, node_id) pairs, any order."""
        candidates = sorted(candidates)
        selected: list[tuple[float, int]] = []
        for d_qc, c in candidates:
            if len(selected) >= M:
                break
            c_vec = self.store.vector(c)
            keep = True
            for _, r in selected:
                d_cr = float(np.sum((c_vec - self.store.vector(r)) ** 2))
                if d_cr < d_qc:
                    keep = False
                    break
            if keep:
                selected.append((d_qc, c))
        return selected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_hnsw_heuristic.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add vecdb/index/hnsw.py tests/test_hnsw_heuristic.py
git commit -m "feat(hnsw): neighbour-selection heuristic (Algorithm 4) for graph diversity"
```

---

### Task 17: `_insert` and public `add`/`search`

**Files:**
- Modify: `vecdb/index/hnsw.py`
- Test: `tests/test_hnsw_insert.py`

**Interfaces:**
- Consumes: `._search_layer`, `._select_neighbors_heuristic`, `._assign_level`, `._ensure_layers`, `VectorStore`.
- Produces: `._insert(self, node_id: int) -> None`, `.add(self, vectors: np.ndarray, ids: np.ndarray) -> None`, `.search(self, q: np.ndarray, k: int, mask=None, params=None) -> SearchResult` (asserts `mask is None` — filtered search lives in `vecdb/index/strategies.py`, Phase 5/6).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hnsw_insert.py
import numpy as np
from vecdb.index.hnsw import HNSWIndex
from vecdb.index.flat import FlatIndex

def test_insert_first_node_becomes_entry_point():
    idx = HNSWIndex(dim=4, seed=0)
    idx.add(np.zeros((1, 4), dtype=np.float32), np.array([0]))
    assert idx.entry_point == 0
    assert idx.max_level >= 0

def test_graph_is_bidirectionally_linked_after_insert():
    rng = np.random.default_rng(0)
    data = rng.random((30, 4)).astype(np.float32)
    idx = HNSWIndex(dim=4, M=4, ef_construction=20, seed=0)
    idx.add(data, np.arange(30))
    # every edge a->b at layer 0 must have a reciprocal b->a
    for node, neighbours in idx.graph[0].items():
        for n in neighbours:
            assert node in idx.graph[0].get(n, []), f"edge {node}->{n} is not reciprocated"

def test_degree_cap_is_respected_after_repruning():
    rng = np.random.default_rng(0)
    data = rng.random((60, 4)).astype(np.float32)
    idx = HNSWIndex(dim=4, M=4, ef_construction=20, seed=0)
    idx.add(data, np.arange(60))
    for node, neighbours in idx.graph[0].items():
        assert len(neighbours) <= idx.M0

def test_unfiltered_recall_at_10_is_reasonably_high_on_tiny_data():
    rng = np.random.default_rng(0)
    data = rng.random((300, 16)).astype(np.float32)
    hnsw = HNSWIndex(dim=16, M=16, ef_construction=100, seed=0)
    hnsw.add(data, np.arange(300))
    flat = FlatIndex()
    flat.add(data, np.arange(300))

    queries = rng.random((30, 16)).astype(np.float32)
    hits = 0
    for q in queries:
        true_ids = set(flat.search(q, k=10).ids.tolist())
        got_ids = set(hnsw.search(q, k=10, params={"ef": 100}).ids.tolist())
        hits += len(true_ids & got_ids)
    recall = hits / (30 * 10)
    assert recall >= 0.85  # loose bound at this tiny scale; the real gate is Task 19 at 10K

def test_search_rejects_a_mask_argument():
    import pytest
    idx = HNSWIndex(dim=4, seed=0)
    idx.add(np.random.default_rng(0).random((5, 4)).astype(np.float32), np.arange(5))
    with pytest.raises(AssertionError):
        idx.search(np.zeros(4, dtype=np.float32), k=1, mask=np.array([True] * 5))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_hnsw_insert.py -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Implement — add these methods to `vecdb/index/hnsw.py`**

```python
    def _insert(self, node_id: int) -> None:
        level = self._assign_level()
        self._ensure_layers(level)
        while len(self.levels) <= node_id:
            self.levels.append(-1)
        self.levels[node_id] = level

        if self.entry_point == -1:
            for layer in range(level + 1):
                self.graph[layer].setdefault(node_id, [])
            self.entry_point = node_id
            self.max_level = level
            return

        q = self.store.vector(node_id)
        ep = [self.entry_point]
        for layer in range(self.max_level, level, -1):
            nearest = self._search_layer(q, ep, ef=1, layer=layer)
            if nearest:
                ep = [nearest[0][1]]

        for layer in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(q, ep, self.ef_construction, layer)
            cap = self.M0 if layer == 0 else self.M
            neighbours = self._select_neighbors_heuristic(candidates, cap)

            self.graph[layer].setdefault(node_id, [])
            self.graph[layer][node_id] = [n for _, n in neighbours]

            for _, n in neighbours:
                self.graph[layer].setdefault(n, [])
                if node_id not in self.graph[layer][n]:
                    self.graph[layer][n].append(node_id)
                if len(self.graph[layer][n]) > cap:
                    nb_ids = self.graph[layer][n]
                    nb_vec = self.store.vector(n)
                    nb_candidates = [(float(np.sum((self.store.vector(x) - nb_vec) ** 2)), x) for x in nb_ids]
                    pruned = self._select_neighbors_heuristic(nb_candidates, cap)
                    self.graph[layer][n] = [x for _, x in pruned]

            ep = [n for _, n in neighbours] if neighbours else ep

        if level > self.max_level:
            self.max_level = level
            self.entry_point = node_id

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        assert np.array_equal(np.asarray(ids), np.arange(len(ids))), \
            "HNSWIndex expects dense internal ids 0..N-1"
        self.store = VectorStore(vectors)
        for node_id in range(len(ids)):
            self._insert(node_id)

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None,
                params: dict | None = None) -> SearchResult:
        assert mask is None, "HNSWIndex.search is unfiltered; filtered strategies live in vecdb.index.strategies"
        assert self.store is not None and self.entry_point != -1, "index is empty"
        ef = (params or {}).get("ef", max(self.ef_search_default, k))
        ops_before = self.store.n_distance_ops
        t0 = time.perf_counter()
        ep = [self.entry_point]
        for layer in range(self.max_level, 0, -1):
            nearest = self._search_layer(q, ep, ef=1, layer=layer)
            if nearest:
                ep = [nearest[0][1]]
        candidates = self._search_layer(q, ep, ef=max(ef, k), layer=0)[:k]
        latency_ms = (time.perf_counter() - t0) * 1000
        ids = np.array([n for _, n in candidates], dtype=np.int64)
        dists = np.array([d for d, _ in candidates], dtype=np.float32)
        return SearchResult(ids=ids, distances=dists,
                             n_distance_ops=self.store.n_distance_ops - ops_before,
                             strategy="hnsw", latency_ms=latency_ms, n_returned=int(ids.size))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_hnsw_insert.py -v`
Expected: PASS (5 tests). **If `test_unfiltered_recall_at_10_is_reasonably_high_on_tiny_data` fails**, work through source plan §5 Day 2's debugging checklist before touching anything else: recall stuck 0.3-0.6 usually means the heuristic isn't being used or neighbours aren't re-pruned after linking; recall high but slow means the early-break condition is wrong; nondeterministic recall across runs means the RNG isn't seeded somewhere.

- [ ] **Step 5: Commit**

```bash
git add vecdb/index/hnsw.py tests/test_hnsw_insert.py
git commit -m "feat(hnsw): insert() with bidirectional linking and re-pruning; add()/search()"
```

---

### Task 18: Persistence

**Files:**
- Modify: `vecdb/index/hnsw.py`
- Test: `tests/test_hnsw_persistence.py`

**Interfaces:**
- Consumes: `pickle`, `numpy.save`/`numpy.load`.
- Produces: `.save(self, path: str | Path) -> None`, `@classmethod .load(cls, path: str | Path) -> HNSWIndex`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hnsw_persistence.py
import numpy as np
from vecdb.index.hnsw import HNSWIndex

def test_save_then_load_produces_identical_search_results(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.random((100, 8)).astype(np.float32)
    idx = HNSWIndex(dim=8, M=8, ef_construction=50, seed=0)
    idx.add(data, np.arange(100))

    q = rng.random(8).astype(np.float32)
    before = idx.search(q, k=5, params={"ef": 50})

    save_path = tmp_path / "index"
    idx.save(save_path)
    reloaded = HNSWIndex.load(save_path)
    after = reloaded.search(q, k=5, params={"ef": 50})

    np.testing.assert_array_equal(before.ids, after.ids)
    np.testing.assert_allclose(before.distances, after.distances, rtol=1e-5)
    assert reloaded.entry_point == idx.entry_point
    assert reloaded.max_level == idx.max_level
    assert reloaded.graph == idx.graph
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_hnsw_persistence.py -v`
Expected: FAIL — `AttributeError: 'HNSWIndex' object has no attribute 'save'`

- [ ] **Step 3: Implement — add these methods to `vecdb/index/hnsw.py`**

```python
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.store.data)
        meta = {
            "dim": self.dim, "M": self.M, "M0": self.M0, "ef_construction": self.ef_construction,
            "entry_point": self.entry_point, "max_level": self.max_level, "levels": self.levels,
        }
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        with open(path / "graph.pkl", "wb") as f:
            pickle.dump(self.graph, f)

    @classmethod
    def load(cls, path: str | Path) -> "HNSWIndex":
        path = Path(path)
        vectors = np.load(path / "vectors.npy")
        with open(path / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        with open(path / "graph.pkl", "rb") as f:
            graph = pickle.load(f)
        idx = cls(dim=meta["dim"], M=meta["M"], ef_construction=meta["ef_construction"])
        idx.M0 = meta["M0"]
        idx.store = VectorStore(vectors)
        idx.entry_point = meta["entry_point"]
        idx.max_level = meta["max_level"]
        idx.levels = meta["levels"]
        idx.graph = graph
        return idx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_hnsw_persistence.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add vecdb/index/hnsw.py tests/test_hnsw_persistence.py
git commit -m "feat(hnsw): save/load persistence via npy vectors + pickled graph"
```

---

### Task 19: Build on siftsmall, validate the recall gate, Phase 3 milestone report

**Files:**
- Create: `scripts/build_index.py`
- Create: `tests/test_hnsw_correctness.py`
- Create: `docs/superpowers/milestones/03-hnsw.md`

**Interfaces:**
- Consumes: `vecdb.io.dataset.load`, `vecdb.index.hnsw.HNSWIndex`, `vecdb.bench.harness.run_benchmark`.
- Produces: a persisted HNSW index at `data/hnsw_siftsmall/`, used as a smoke artifact only (Phase 4 rebuilds at 100K) and reloaded by `tests/test_hnsw_correctness.py`.

- [ ] **Step 1: Write `scripts/build_index.py`**

```python
# scripts/build_index.py
"""Phase 3 gate: build HNSW on siftsmall, confirm build time and recall@10 vs FlatIndex."""
import time
from pathlib import Path
import numpy as np
from vecdb.io.dataset import load
from vecdb.index.hnsw import HNSWIndex
from vecdb.bench.harness import run_benchmark

def main() -> None:
    bundle = load("siftsmall", cache_dir=Path("data"))

    idx = HNSWIndex(dim=bundle.base.shape[1], M=16, ef_construction=200, seed=42)
    t0 = time.perf_counter()
    idx.add(bundle.base, np.arange(len(bundle.base)))
    build_s = time.perf_counter() - t0
    print(f"build time: {build_s:.1f}s")
    assert build_s < 120, "Phase 3 gate requires siftsmall build under 2 minutes"

    gt = [bundle.groundtruth[i][:10] for i in range(len(bundle.queries))]
    df = run_benchmark(idx, bundle.queries, gt, k=10, params={"ef": 100})
    recall = df["recall"].mean()
    print(f"recall@10 at efSearch=100: {recall:.4f}")
    assert recall >= 0.95, f"Phase 3 gate requires recall@10 >= 0.95, got {recall:.4f}"

    idx.save(Path("data/hnsw_siftsmall"))
    print("Phase 3 gate: PASS")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `".venv/Scripts/python" scripts/build_index.py`
Expected: prints build time (< 120s) and recall@10 (>= 0.95), ends with `Phase 3 gate: PASS`. **If the assertion fails**, this is the trigger condition from spec §3's locked fallback: debug against source plan §5 Day 2's checklist first; only fall back to `faiss.IndexHNSWFlat` wrapped behind `Index` if genuinely stuck, and document that plainly rather than quietly swapping it in.

- [ ] **Step 3: Write `tests/test_hnsw_correctness.py`, the officially named recall-floor test**

Spec §7's testing strategy names this file explicitly as a required deliverable. It loads the artifact Step 1 already built rather than rebuilding (keeps `pytest` fast), and skips cleanly if that artifact isn't present yet (a truly clean clone, before `scripts/build_index.py` has run):

```python
# tests/test_hnsw_correctness.py
from pathlib import Path
import pytest
from vecdb.io.dataset import load
from vecdb.index.hnsw import HNSWIndex
from vecdb.bench.harness import run_benchmark

SIFTSMALL_HNSW_PATH = Path("data/hnsw_siftsmall")

@pytest.mark.skipif(not SIFTSMALL_HNSW_PATH.exists(), reason="requires scripts/build_index.py to have been run")
def test_recall_at_10_floor_against_flat_ground_truth():
    idx = HNSWIndex.load(SIFTSMALL_HNSW_PATH)
    bundle = load("siftsmall", cache_dir=Path("data"))
    gt = [bundle.groundtruth[i][:10] for i in range(len(bundle.queries))]
    df = run_benchmark(idx, bundle.queries, gt, k=10, params={"ef": 100})
    assert df["recall"].mean() >= 0.95, (
        "HNSW recall floor regressed below the Phase 3 gate — this must be treated as "
        "a real regression, not a threshold to relax"
    )
```

Run: `".venv/Scripts/python" -m pytest tests/test_hnsw_correctness.py -v`
Expected: PASS (1 test), given Step 2 already built and saved the artifact this test loads.

- [ ] **Step 5: Write the Phase 3 milestone report**

Create `docs/superpowers/milestones/03-hnsw.md` with: **What got built** (the full hand-written HNSW: level assignment, search_layer, the diversity heuristic, insert with re-pruning, persistence), **Numbers** (the actual build time and recall@10 printed above, plus `pytest` results for Tasks 14-18), **Gate status**, **Interview note** — explain in plain language what Algorithm 4's diversity heuristic buys over naive top-M, and what the early-break in `_search_layer` buys over a full traversal (this maps directly to source plan §7 interview questions 4-6 — write real answers, not just "see the source plan").

- [ ] **Step 6: Run the full test suite, commit, and push**

Run: `".venv/Scripts/python" -m pytest -v`
Expected: all tests through Task 19 PASS (including the newly skippable `test_hnsw_correctness.py`, which should now actually run since Step 1 built the artifact it needs).

```bash
git add scripts/build_index.py tests/test_hnsw_correctness.py docs/superpowers/milestones/03-hnsw.md
git commit -m "feat: HNSW build/validation script, recall-floor test, Phase 3 gate pass, milestone report"
git push origin main
```

---

## Phase 4 — Search tuning and scale to 100K

### Task 20: Plotting utility and the unfiltered Pareto curve

**Files:**
- Create: `vecdb/bench/plots.py`
- Create: `scripts/run_ef_sweep.py`
- Test: `tests/test_plots.py`

**Interfaces:**
- Consumes: `vecdb.index.hnsw.HNSWIndex`, `vecdb.index.faiss_baseline.FaissHNSWIndex`, `vecdb.bench.harness.run_benchmark`.
- Produces: `pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]` (pure function: `(latency_ms, recall)` pairs → non-dominated subset), `plot_lines(series: dict[str, list[tuple[float, float]]], out_path, xlabel, ylabel, title, xscale="linear", yscale="linear") -> None` — the one reusable plotting function every later figure (Phases 4, 5, 6, 7) calls.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plots.py
from pathlib import Path
from vecdb.bench.plots import pareto_frontier, plot_lines

def test_pareto_frontier_drops_dominated_points():
    # (latency, recall): (10, 0.9) dominates (20, 0.85) — lower latency, higher recall
    points = [(10.0, 0.9), (20.0, 0.85), (5.0, 0.5), (30.0, 0.99)]
    frontier = pareto_frontier(points)
    assert (20.0, 0.85) not in frontier
    assert (10.0, 0.9) in frontier
    assert (5.0, 0.5) in frontier   # cheapest latency, nothing beats it on latency
    assert (30.0, 0.99) in frontier  # highest recall, nothing beats it on recall

def test_plot_lines_writes_a_png_file(tmp_path):
    out = tmp_path / "figures" / "test.png"
    plot_lines(
        series={"a": [(1.0, 0.5), (2.0, 0.8)], "b": [(1.0, 0.4), (2.0, 0.9)]},
        out_path=out, xlabel="x", ylabel="y", title="test",
    )
    assert out.exists()
    assert out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_plots.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# vecdb/bench/plots.py
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """points: (latency_ms, recall). Returns the subset not dominated by any other
    point (no other point has both lower-or-equal latency and higher-or-equal recall,
    strictly better in at least one)."""
    frontier = []
    for p in points:
        dominated = any(
            o != p and o[0] <= p[0] and o[1] >= p[1] and (o[0] < p[0] or o[1] > p[1])
            for o in points
        )
        if not dominated:
            frontier.append(p)
    return sorted(frontier)


def plot_lines(series: dict[str, list[tuple[float, float]]], out_path: Path, xlabel: str, ylabel: str,
                title: str, xscale: str = "linear", yscale: str = "linear") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, points in series.items():
        points = sorted(points)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_plots.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write `scripts/run_ef_sweep.py` and run it**

```python
# scripts/run_ef_sweep.py
"""Phase 4: efSearch sweep on siftsmall for hand-written HNSW vs FAISS HNSW, same M/efConstruction."""
from pathlib import Path
import numpy as np
from vecdb.io.dataset import load
from vecdb.index.hnsw import HNSWIndex
from vecdb.index.faiss_baseline import FaissHNSWIndex
from vecdb.bench.harness import run_benchmark
from vecdb.bench.plots import plot_lines

EF_VALUES = [10, 20, 40, 80, 160, 320]

def main() -> None:
    bundle = load("siftsmall", cache_dir=Path("data"))
    gt = [bundle.groundtruth[i][:10] for i in range(len(bundle.queries))]

    ours = HNSWIndex(dim=bundle.base.shape[1], M=16, ef_construction=200, seed=42)
    ours.add(bundle.base, np.arange(len(bundle.base)))

    theirs = FaissHNSWIndex(dim=bundle.base.shape[1], M=16, ef_construction=200)
    theirs.add(bundle.base, np.arange(len(bundle.base)))

    series = {"hand-written HNSW": [], "FAISS HNSW": []}
    for ef in EF_VALUES:
        df_ours = run_benchmark(ours, bundle.queries, gt, k=10, params={"ef": ef})
        series["hand-written HNSW"].append((df_ours["latency_ms"].quantile(0.5), df_ours["recall"].mean()))
        df_theirs = run_benchmark(theirs, bundle.queries, gt, k=10, params={"ef": ef})
        series["FAISS HNSW"].append((df_theirs["latency_ms"].quantile(0.5), df_theirs["recall"].mean()))
        print(f"ef={ef:4d}  ours: recall={df_ours['recall'].mean():.3f} p50={df_ours['latency_ms'].quantile(0.5):.3f}ms"
              f"  |  faiss: recall={df_theirs['recall'].mean():.3f} p50={df_theirs['latency_ms'].quantile(0.5):.3f}ms")

    plot_lines(series, Path("results/figures/pareto_unfiltered.png"),
               xlabel="p50 latency (ms)", ylabel="recall@10", title="Unfiltered recall/latency Pareto curve")
    print("wrote results/figures/pareto_unfiltered.png")

if __name__ == "__main__":
    main()
```

Run: `".venv/Scripts/python" scripts/run_ef_sweep.py`
Expected: a table of recall/latency per `ef` for both indexes, and `results/figures/pareto_unfiltered.png` created. Record the actual numbers — including the gap between "ours" and "faiss" latency at matched recall — in the Phase 4 milestone report (Task 23); this is the "2-5x FAISS, explained not hidden" number from spec §2.

- [ ] **Step 6: Commit**

```bash
git add vecdb/bench/plots.py tests/test_plots.py scripts/run_ef_sweep.py
git commit -m "feat: reusable plotting utility, unfiltered efSearch/Pareto sweep vs FAISS"
```

---

### Task 21: Build-parameter sweep

**Files:**
- Modify: `vecdb/index/hnsw.py` (add `.approx_size_bytes()`)
- Create: `scripts/run_build_param_sweep.py`
- Test: `tests/test_hnsw_size.py`

**Interfaces:**
- Consumes: `HNSWIndex.graph`, `HNSWIndex.store`.
- Produces: `.approx_size_bytes(self) -> int` — resident index size excluding raw vectors (source plan §4.4 `index_bytes` metric): 4 bytes per adjacency-list entry (int32 neighbour id) summed across all layers.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hnsw_size.py
import numpy as np
from vecdb.index.hnsw import HNSWIndex

def test_approx_size_bytes_counts_adjacency_entries_only():
    idx = HNSWIndex(dim=4, seed=0)
    idx.store = None
    from vecdb.store.vectors import VectorStore
    idx.store = VectorStore(np.zeros((3, 4), dtype=np.float32))
    idx._ensure_layers(0)
    idx.graph[0] = {0: [1, 2], 1: [0], 2: [0]}  # 4 total adjacency entries
    assert idx.approx_size_bytes() == 4 * 4  # 4 entries * 4 bytes (int32)

def test_approx_size_bytes_grows_with_a_real_build():
    rng = np.random.default_rng(0)
    data = rng.random((100, 8)).astype(np.float32)
    idx = HNSWIndex(dim=8, M=8, ef_construction=50, seed=0)
    idx.add(data, np.arange(100))
    assert idx.approx_size_bytes() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_hnsw_size.py -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Implement — add to `vecdb/index/hnsw.py`**

```python
    def approx_size_bytes(self) -> int:
        """Resident index size excluding raw vectors: 4 bytes per adjacency entry
        (int32 neighbour id), summed across every layer."""
        total_entries = sum(len(neighbours) for layer in self.graph for neighbours in layer.values())
        return total_entries * 4
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_hnsw_size.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write and run `scripts/run_build_param_sweep.py`**

```python
# scripts/run_build_param_sweep.py
"""Phase 4: M x efConstruction sweep on siftsmall. Table of build time / index size /
recall@10 at a fixed efSearch, so the operating point can be picked and justified."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from vecdb.io.dataset import load
from vecdb.index.hnsw import HNSWIndex
from vecdb.bench.harness import run_benchmark

M_VALUES = [8, 16, 32]
EFC_VALUES = [100, 200, 400]
FIXED_EF_SEARCH = 128

def main() -> None:
    bundle = load("siftsmall", cache_dir=Path("data"))
    gt = [bundle.groundtruth[i][:10] for i in range(len(bundle.queries))]
    rows = []
    for M in M_VALUES:
        for efc in EFC_VALUES:
            idx = HNSWIndex(dim=bundle.base.shape[1], M=M, ef_construction=efc, seed=42)
            t0 = time.perf_counter()
            idx.add(bundle.base, np.arange(len(bundle.base)))
            build_s = time.perf_counter() - t0
            df = run_benchmark(idx, bundle.queries, gt, k=10, params={"ef": FIXED_EF_SEARCH})
            rows.append({
                "M": M, "ef_construction": efc, "build_time_s": build_s,
                "index_bytes": idx.approx_size_bytes(), "recall_at_10": df["recall"].mean(),
            })
            print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv("results/build_param_sweep.csv", index=False)
    print("wrote results/build_param_sweep.csv")

if __name__ == "__main__":
    main()
```

Run: `".venv/Scripts/python" scripts/run_build_param_sweep.py`
Expected: 9 rows printed and written to `results/build_param_sweep.csv`. Pick the operating point for the rest of the project (default is `M=16, ef_construction=200`, already used throughout — confirm in the milestone report whether the sweep supports keeping that default or whether a different combination clearly dominates it, and justify the choice in one sentence).

- [ ] **Step 6: Commit**

```bash
git add vecdb/index/hnsw.py tests/test_hnsw_size.py scripts/run_build_param_sweep.py
git commit -m "feat: index size estimator, M x efConstruction build-parameter sweep"
```

---

### Task 22: Hot-path optimization — visited-set generation counter

**Files:**
- Modify: `vecdb/index/hnsw.py`
- Test: `tests/test_hnsw_search_layer.py` (extend)

**Interfaces:**
- Consumes/modifies: `._search_layer`. New attributes `._visited_stamp: np.ndarray` (uint32, one per node), `._visit_generation: int`.
- Produces: same `._search_layer` signature and return type as Task 15 — this is a pure performance change, not a behavioural one. A Python `set()` reallocated per call is replaced with a reusable stamp array + monotonically increasing generation counter, so membership checks become O(1) array reads instead of hashing, and no per-query allocation is needed.

- [ ] **Step 1: Write the failing test (append to `tests/test_hnsw_search_layer.py`)**

```python
def test_search_layer_visited_state_does_not_leak_between_calls():
    points = np.array([[float(i), 0.0] for i in range(10)], dtype=np.float32)
    idx = HNSWIndex(dim=2, seed=0)
    _build_line_graph(idx, points)
    idx._visited_stamp = np.zeros(10, dtype=np.uint32)
    idx._visit_generation = 0
    first = idx._search_layer(np.array([2.0, 0.0], dtype=np.float32), entry_points=[0], ef=3, layer=0)
    second = idx._search_layer(np.array([7.0, 0.0], dtype=np.float32), entry_points=[0], ef=3, layer=0)
    assert second[0][1] == 7  # second call must not treat nodes visited=stale from the first call
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_hnsw_search_layer.py -v`
Expected: FAIL — `AttributeError: 'HNSWIndex' object has no attribute '_visited_stamp'`

- [ ] **Step 3: Implement — replace `_search_layer` in `vecdb/index/hnsw.py`, and initialise the stamp array in `add()`**

```python
    def _search_layer(self, q: np.ndarray, entry_points: list[int], ef: int, layer: int) -> list[tuple[float, int]]:
        """Same contract as before. The visited set is now a reusable np.uint32 stamp
        array plus a generation counter, instead of a Python set() allocated per call —
        this avoids per-query allocation/hashing overhead, which matters once ef and the
        result set get large."""
        self._visit_generation += 1
        gen = self._visit_generation
        stamp = self._visited_stamp

        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []

        if entry_points:
            for ep in entry_points:
                stamp[ep] = gen
            dists = self.store.distances(q, np.array(entry_points, dtype=np.int64))
            for ep, d in zip(entry_points, dists):
                d = float(d)
                heapq.heappush(candidates, (d, ep))
                heapq.heappush(results, (-d, ep))

        while candidates:
            d_c, c = heapq.heappop(candidates)
            worst_d = -results[0][0]
            if d_c > worst_d and len(results) >= ef:
                break
            neighbours = [n for n in self.graph[layer].get(c, []) if stamp[n] != gen]
            if not neighbours:
                continue
            for n in neighbours:
                stamp[n] = gen
            dists = self.store.distances(q, np.array(neighbours, dtype=np.int64))
            for n, d in zip(neighbours, dists):
                d = float(d)
                worst_d = -results[0][0] if results else float("inf")
                if len(results) < ef:
                    heapq.heappush(candidates, (d, n))
                    heapq.heappush(results, (-d, n))
                elif d < worst_d:
                    heapq.heappush(candidates, (d, n))
                    heapq.heappush(results, (-d, n))
                    heapq.heappop(results)

        return sorted((-nd, node) for nd, node in results)
```

And in `add()`, right after `self.store = VectorStore(vectors)`, insert:

```python
        self._visited_stamp = np.zeros(len(vectors), dtype=np.uint32)
        self._visit_generation = 0
```

- [ ] **Step 4: Run the full HNSW test suite to verify nothing regressed**

Run: `".venv/Scripts/python" -m pytest tests/test_hnsw_levels.py tests/test_hnsw_search_layer.py tests/test_hnsw_heuristic.py tests/test_hnsw_insert.py tests/test_hnsw_persistence.py tests/test_hnsw_size.py -v`
Expected: all PASS, including the new leak-check test. `HNSWIndex.load()` (Task 18) must also set `._visited_stamp`/`._visit_generation` — add the same two lines there, after `idx.store = VectorStore(vectors)`.

- [ ] **Step 5: Profile and record findings (manual step, not a test)**

Run:
```bash
".venv/Scripts/python" -c "
import cProfile, pstats, numpy as np
from pathlib import Path
from vecdb.io.dataset import load
from vecdb.index.hnsw import HNSWIndex
bundle = load('siftsmall', cache_dir=Path('data'))
idx = HNSWIndex(dim=bundle.base.shape[1], M=16, ef_construction=200, seed=42)
idx.add(bundle.base, np.arange(len(bundle.base)))
profiler = cProfile.Profile()
profiler.enable()
for q in bundle.queries:
    idx.search(q, k=10, params={'ef': 128})
profiler.disable()
pstats.Stats(profiler).sort_stats('cumulative').print_stats(10)
"
```
Expected: a top-10 cumulative-time table. Record the top 2-3 hot functions in the Phase 4 milestone report (Task 23) — expect `distances`/`np.sum` and heap operations to dominate. If a single non-vectorised loop shows up unexpectedly high, that's a real bug to fix before moving on, not a footnote.

- [ ] **Step 6: Commit**

```bash
git add vecdb/index/hnsw.py tests/test_hnsw_search_layer.py
git commit -m "perf(hnsw): replace per-query visited set() with reusable stamp array + generation counter"
```

---

### Task 23: Scale to 100K, persist, Phase 4 milestone report

**Files:**
- Create: `scripts/build_100k.py`
- Create: `docs/superpowers/milestones/04-tuning-scale.md`

**Interfaces:**
- Consumes: `vecdb.io.dataset.load("sift1m_100k")`, `HNSWIndex`.
- Produces: a persisted 100K index at `data/hnsw_100k/`, used by every remaining phase.

- [ ] **Step 1: Write `scripts/build_100k.py`**

```python
# scripts/build_100k.py
"""Phase 4 gate: build HNSW on the 100K SIFT subset, confirm it completes in a
reasonable window, persist it for Phases 5-7 to load."""
import time
from pathlib import Path
import numpy as np
from vecdb.io.dataset import load
from vecdb.index.hnsw import HNSWIndex

def main() -> None:
    bundle = load("sift1m_100k", cache_dir=Path("data"))
    idx = HNSWIndex(dim=bundle.base.shape[1], M=16, ef_construction=200, seed=42)
    t0 = time.perf_counter()
    idx.add(bundle.base, np.arange(len(bundle.base)))
    build_s = time.perf_counter() - t0
    print(f"100K build time: {build_s / 60:.1f} min")
    idx.save(Path("data/hnsw_100k"))
    print(f"index size (adjacency only): {idx.approx_size_bytes() / 1e6:.1f} MB")
    print("saved to data/hnsw_100k")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it in the background and wait for completion**

Run (background): `".venv/Scripts/python" scripts/build_100k.py`
Expected: source plan §5 Day 3 estimates 15-40 minutes with vectorised distances. This is a genuine long-running background job — start it, continue other work if any remains queued, and check back on completion rather than polling. Record the actual wall-clock time; if it runs far outside that estimate (e.g. multiple hours), that itself is a milestone-report finding worth investigating before Phase 5 builds on top of it.

- [ ] **Step 3: Write the Phase 4 milestone report**

Create `docs/superpowers/milestones/04-tuning-scale.md` with: **What got built** (efSearch sweep + Pareto figure, build-param sweep, the visited-stamp optimization, the 100K build), **Numbers** (the actual efSearch sweep table, the build-param sweep table, the cProfile top hot functions from Task 22 Step 5, and the 100K build time/index size from this task), **Gate status** (`results/figures/pareto_unfiltered.png` exists; 100K index persisted to `data/hnsw_100k/`), **Interview note** — the "2-5x FAISS, explained not hidden" answer from source plan §7 Q10, using your own measured latency and dist_ops numbers, not the plan's placeholder language.

- [ ] **Step 4: Commit and push**

```bash
git add scripts/build_100k.py docs/superpowers/milestones/04-tuning-scale.md results/build_param_sweep.csv results/figures/pareto_unfiltered.png
git commit -m "feat: 100K HNSW build persisted, Phase 4 milestone report"
git push origin main
```

---

## Phase 5 — Pre-filter and post-filter strategies

`vecdb/index/strategies.py` is created in this phase and extended in Phase 6 — it is one of the hand-written ★ files (spec §3's Global Constraints).

### Task 24: `PreFilterStrategy`

**Files:**
- Create: `vecdb/index/strategies.py`
- Test: `tests/test_strategies_prefilter.py`

**Interfaces:**
- Consumes: `vecdb.index.flat.FlatIndex`, `vecdb.index.base.Index`, `SearchResult`.
- Produces: `class PreFilterStrategy(Index)` with `__init__(self, flat_index: FlatIndex)`. `search()` delegates to the wrapped `FlatIndex` (exact scan over the masked rows — this **is** Strategy A; there is nothing more to it) and relabels `strategy="pre_filter"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategies_prefilter.py
import numpy as np
from vecdb.index.flat import FlatIndex
from vecdb.index.strategies import PreFilterStrategy

def test_prefilter_matches_flat_index_exactly_and_relabels_strategy():
    rng = np.random.default_rng(0)
    data = rng.random((200, 8)).astype(np.float32)
    flat = FlatIndex()
    flat.add(data, np.arange(200))
    strategy = PreFilterStrategy(flat)

    q = rng.random(8).astype(np.float32)
    mask = rng.random(200) < 0.2
    direct = flat.search(q, k=10, mask=mask)
    via_strategy = strategy.search(q, k=10, mask=mask)

    np.testing.assert_array_equal(direct.ids, via_strategy.ids)
    assert via_strategy.strategy == "pre_filter"
    assert via_strategy.n_distance_ops == direct.n_distance_ops
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_strategies_prefilter.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# vecdb/index/strategies.py
"""The three filtered-search strategies, built on top of an already-constructed
FlatIndex / HNSWIndex. None of these implement add() — they wrap indexes built
elsewhere so Flat/HNSW storage is never duplicated across strategies."""
from __future__ import annotations
import time
import numpy as np
from vecdb.index.base import Index, SearchResult
from vecdb.index.flat import FlatIndex
from vecdb.index.hnsw import HNSWIndex


class PreFilterStrategy(Index):
    """Strategy A: materialise the masked rows, exact-scan them. Correctness: exact,
    recall = 1.0 always. Cost: O(N * s * d). This IS FlatIndex's masked search —
    the strategy wrapper exists so the benchmark harness can label it distinctly and
    the planner can address it uniformly alongside the other two strategies."""

    def __init__(self, flat_index: FlatIndex):
        self.flat_index = flat_index

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        raise NotImplementedError("PreFilterStrategy wraps an already-built FlatIndex")

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None,
                params: dict | None = None) -> SearchResult:
        result = self.flat_index.search(q, k, mask=mask, params=params)
        result.strategy = "pre_filter"
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_strategies_prefilter.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add vecdb/index/strategies.py tests/test_strategies_prefilter.py
git commit -m "feat(strategies): PreFilterStrategy (exact masked scan)"
```

---

### Task 25: `PostFilterStrategy` with adaptive ef, underfill tracking, and honest fallback

**Files:**
- Modify: `vecdb/index/strategies.py`
- Test: `tests/test_strategies_postfilter.py`

**Interfaces:**
- Consumes: `HNSWIndex.search`, `PreFilterStrategy` (used as the fallback target).
- Produces: `class PostFilterStrategy(Index)` with `__init__(self, hnsw_index: HNSWIndex, fallback: Index, alpha: float = 4.0, ef_min: int = 16, max_retries: int = 1, retry_multiplier: float = 4.0)`, properties `.fallback_count: int`, `.query_count: int`, `.fallback_rate: float`. `search(q, k, mask, params)` expects `params["selectivity_hat"]` (falls back to `1.0`, i.e. an unfiltered-width beam, if absent).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategies_postfilter.py
import numpy as np
from vecdb.index.hnsw import HNSWIndex
from vecdb.index.flat import FlatIndex
from vecdb.index.strategies import PreFilterStrategy, PostFilterStrategy

def _build(n=500, d=16, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.random((n, d)).astype(np.float32)
    hnsw = HNSWIndex(dim=d, M=16, ef_construction=100, seed=seed)
    hnsw.add(data, np.arange(n))
    flat = FlatIndex()
    flat.add(data, np.arange(n))
    return data, hnsw, flat

def test_postfilter_returns_only_matching_ids_when_not_underfilled():
    data, hnsw, flat = _build()
    strategy = PostFilterStrategy(hnsw, fallback=PreFilterStrategy(flat))
    rng = np.random.default_rng(1)
    q = rng.random(16).astype(np.float32)
    mask = np.zeros(500, dtype=bool)
    mask[:250] = True  # 50% selectivity - post-filter should comfortably find k=10
    result = strategy.search(q, k=10, mask=mask, params={"selectivity_hat": 0.5})
    assert all(mask[i] for i in result.ids)
    assert result.strategy in ("post_filter", "post_filter_fallback")

def test_postfilter_falls_back_to_prefilter_when_selectivity_is_extreme():
    data, hnsw, flat = _build(n=500)
    strategy = PostFilterStrategy(hnsw, fallback=PreFilterStrategy(flat), max_retries=1)
    rng = np.random.default_rng(2)
    q = rng.random(16).astype(np.float32)
    mask = np.zeros(500, dtype=bool)
    mask[:3] = True  # 0.6% selectivity, k=10 > available survivors even in theory... use k=3
    result = strategy.search(q, k=3, mask=mask, params={"selectivity_hat": 0.006})
    # whichever path was taken, the result must still be correct and non-underfilled
    # given only 3 true survivors exist
    assert result.n_returned <= 3
    assert all(mask[i] for i in result.ids)

def test_fallback_rate_is_tracked_across_calls():
    data, hnsw, flat = _build(n=500)
    strategy = PostFilterStrategy(hnsw, fallback=PreFilterStrategy(flat), max_retries=0)
    rng = np.random.default_rng(3)
    mask = np.zeros(500, dtype=bool)
    mask[:2] = True  # forces underfill for k=10 with no retries
    for _ in range(5):
        q = rng.random(16).astype(np.float32)
        strategy.search(q, k=10, mask=mask, params={"selectivity_hat": 0.004})
    assert strategy.query_count == 5
    assert strategy.fallback_count == 5
    assert strategy.fallback_rate == 1.0

def test_ef_widens_as_estimated_selectivity_drops():
    data, hnsw, flat = _build()
    strategy = PostFilterStrategy(hnsw, fallback=PreFilterStrategy(flat))
    assert strategy._ef_for(k=10, sel_hat=0.5) < strategy._ef_for(k=10, sel_hat=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_strategies_postfilter.py -v`
Expected: FAIL — `ImportError: cannot import name 'PostFilterStrategy'`

- [ ] **Step 3: Implement — add to `vecdb/index/strategies.py`**

```python
class PostFilterStrategy(Index):
    """Strategy B: run HNSW with a widened beam, discard non-matches, take top-k.
    Expected survivors from a top-ef list is ef * s, so ef must grow as s shrinks —
    ef = clamp(alpha * k / s, ef_min, N). This is *probabilistic*: it can under-fill.
    Under-fill is tracked explicitly (never silently returned as a short list), retried
    once with a wider beam, and if still short, handed off to the exact fallback."""

    def __init__(self, hnsw_index: HNSWIndex, fallback: Index, alpha: float = 4.0,
                 ef_min: int = 16, max_retries: int = 1, retry_multiplier: float = 4.0):
        self.hnsw = hnsw_index
        self.fallback = fallback
        self.alpha = alpha
        self.ef_min = ef_min
        self.max_retries = max_retries
        self.retry_multiplier = retry_multiplier
        self.fallback_count = 0
        self.query_count = 0

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        raise NotImplementedError("PostFilterStrategy wraps an already-built HNSWIndex")

    def _ef_for(self, k: int, sel_hat: float) -> int:
        sel_hat = max(sel_hat, 1e-6)
        ef = self.alpha * k / sel_hat
        return int(np.clip(ef, self.ef_min, len(self.hnsw.store)))

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / self.query_count if self.query_count else 0.0

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None,
                params: dict | None = None) -> SearchResult:
        self.query_count += 1
        params = params or {}
        sel_hat = params.get("selectivity_hat", 1.0)
        ef = self._ef_for(k, sel_hat)
        t0 = time.perf_counter()
        ops_before = self.hnsw.store.n_distance_ops

        raw = self.hnsw.search(q, k=ef, params={"ef": ef})
        if mask is None:
            latency_ms = (time.perf_counter() - t0) * 1000
            raw.strategy = "post_filter"
            raw.latency_ms = latency_ms
            return raw

        keep = mask[raw.ids]
        filtered_ids, filtered_d = raw.ids[keep], raw.distances[keep]

        attempts = 0
        while filtered_ids.size < k and attempts < self.max_retries and ef < len(self.hnsw.store):
            attempts += 1
            ef = int(min(ef * self.retry_multiplier, len(self.hnsw.store)))
            raw = self.hnsw.search(q, k=ef, params={"ef": ef})
            keep = mask[raw.ids]
            filtered_ids, filtered_d = raw.ids[keep], raw.distances[keep]

        if filtered_ids.size < k:
            self.fallback_count += 1
            fb = self.fallback.search(q, k, mask=mask)
            fb.strategy = "post_filter_fallback"
            fb.n_distance_ops += self.hnsw.store.n_distance_ops - ops_before
            return fb

        order = np.argsort(filtered_d)[:k]
        ids, dists = filtered_ids[order], filtered_d[order]
        latency_ms = (time.perf_counter() - t0) * 1000
        return SearchResult(ids=ids, distances=dists,
                             n_distance_ops=self.hnsw.store.n_distance_ops - ops_before,
                             strategy="post_filter", latency_ms=latency_ms, n_returned=int(ids.size))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_strategies_postfilter.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add vecdb/index/strategies.py tests/test_strategies_postfilter.py
git commit -m "feat(strategies): PostFilterStrategy with adaptive ef, tracked underfill, honest fallback"
```

---

### Task 26: Selectivity estimator validation

**Files:**
- Create: `scripts/run_selectivity_validation.py`

**Interfaces:**
- Consumes: `vecdb.predicate.selectivity.estimate_selectivity`, `vecdb.predicate.compile.compile`, `vecdb.store.metadata.MetaStore`, `vecdb.bench.plots.plot_lines` (reused in scatter mode via a thin wrapper — see Step 1).

- [ ] **Step 1: Write `scripts/run_selectivity_validation.py`**

```python
# scripts/run_selectivity_validation.py
"""Phase 5 headline figure: for 500 random predicates (single-clause, AND-of-2,
AND-of-3, OR), scatter estimated selectivity vs true selectivity on log-log axes,
for both uncorrelated and correlated metadata. The independence assumption for AND
should hold on uncorrelated columns and visibly fail on correlated ones."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vecdb.io.metadata_gen import load_metadata
from vecdb.store.metadata import MetaStore
from vecdb.predicate.compile import compile as compile_pred
from vecdb.predicate.selectivity import estimate_selectivity

def random_predicate(rng: np.random.Generator, meta: MetaStore) -> dict:
    shape = rng.choice(["single", "and2", "and3", "or2"])
    def leaf():
        col = rng.choice(["category", "year", "score"]) if "score" in meta.columns else rng.choice(["category", "year"])
        if meta.stats[col].kind == "categorical":
            val = int(rng.choice(list(meta.stats[col].value_counts.keys())))
            return {"op": "eq", "col": col, "val": val}
        lo = float(meta.stats[col].hist_edges[0])
        hi = float(meta.stats[col].hist_edges[-1])
        return {"op": rng.choice(["lt", "gt"]), "col": col, "val": float(rng.uniform(lo, hi))}
    if shape == "single":
        return leaf()
    if shape == "and2":
        return {"op": "and", "clauses": [leaf(), leaf()]}
    if shape == "and3":
        return {"op": "and", "clauses": [leaf(), leaf(), leaf()]}
    return {"op": "or", "clauses": [leaf(), leaf()]}

def collect(meta: MetaStore, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_s, est_s = [], []
    for _ in range(n):
        pred = random_predicate(rng, meta)
        mask = compile_pred(pred, meta)
        true = mask.sum() / meta.n
        est = estimate_selectivity(pred, meta)
        if true > 0:  # log-log plot can't show zero
            true_s.append(true)
            est_s.append(est if est > 0 else 1e-6)
    return np.array(true_s), np.array(est_s)

def plot_scatter(true_s, est_s, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_s, est_s, alpha=0.4, s=15)
    ax.plot([1e-4, 1], [1e-4, 1], "k--", linewidth=1, label="perfect estimate")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("true selectivity"); ax.set_ylabel("estimated selectivity (ŝ)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main() -> None:
    for variant in ["uncorrelated", "correlated"]:
        cols = load_metadata(Path(f"data/sift1m_100k_meta_{variant}.npz"))
        meta = MetaStore(cols)
        true_s, est_s = collect(meta, n=500, seed=0)
        err = np.abs(est_s - true_s) / true_s
        print(f"{variant}: median rel. error={np.median(err):.3f}  p95={np.quantile(err, 0.95):.3f}")
        plot_scatter(true_s, est_s, f"Selectivity estimation error ({variant})",
                     f"results/figures/selectivity_estimation_{variant}.png")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `".venv/Scripts/python" scripts/run_selectivity_validation.py`
Expected: two figures written (`selectivity_estimation_uncorrelated.png`, `selectivity_estimation_correlated.png`), and printed median/p95 relative error for each. **This is a headline figure, not a footnote** (source plan §3.2) — record both error numbers verbatim in the Phase 5 milestone report, and note whether the correlated-column error is visibly larger, which is the expected finding.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_selectivity_validation.py
git commit -m "feat: selectivity estimator validation scatter (uncorrelated vs correlated)"
```

---

### Task 27: Selectivity-grid sweep for Strategies A and B

**Files:**
- Create: `vecdb/bench/sweep.py`
- Create: `scripts/run_sweep_ab.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: `vecdb.store.metadata.MetaStore`, `vecdb.predicate.compile.compile`, `vecdb.bench.harness.run_benchmark`.
- Produces: `predicate_for_selectivity(meta: MetaStore, target_s: float, col: str = "score") -> dict` (builds a `score < quantile(target_s)` predicate hitting the target selectivity via the empirical quantile of the numeric `score` column), `SELECTIVITY_GRID: list[float]` = `[0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep.py
import numpy as np
from vecdb.store.metadata import MetaStore
from vecdb.bench.sweep import predicate_for_selectivity, SELECTIVITY_GRID
from vecdb.predicate.compile import compile as compile_pred

def test_predicate_for_selectivity_hits_target_within_tolerance():
    rng = np.random.default_rng(0)
    meta = MetaStore({"score": rng.uniform(0, 1, size=20_000).astype(np.float32)})
    for target in SELECTIVITY_GRID:
        pred = predicate_for_selectivity(meta, target)
        mask = compile_pred(pred, meta)
        actual = mask.sum() / meta.n
        assert abs(actual - target) < 0.02, f"target={target} got={actual}"

def test_selectivity_grid_is_the_spec_defined_values():
    assert SELECTIVITY_GRID == [0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_sweep.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `vecdb/bench/sweep.py`**

```python
# vecdb/bench/sweep.py
from __future__ import annotations
import numpy as np
from vecdb.store.metadata import MetaStore

SELECTIVITY_GRID: list[float] = [0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]


def predicate_for_selectivity(meta: MetaStore, target_s: float, col: str = "score") -> dict:
    """A `score < t` predicate where t is the empirical target_s-quantile of `col`,
    so the true selectivity lands within noise of target_s regardless of col's
    exact distribution."""
    if target_s >= 1.0:
        return {"op": "gte", "col": col, "val": float(meta.columns[col].min()) - 1.0}
    t = float(np.quantile(meta.columns[col], target_s))
    return {"op": "lt", "col": col, "val": t}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_sweep.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write and run `scripts/run_sweep_ab.py`**

```python
# scripts/run_sweep_ab.py
"""Phase 5 gate: full selectivity grid x both metadata variants for Strategies A
(pre-filter) and B (post-filter). 200 queries per cell, first 20 discarded as
warmup, per spec §5's selectivity grid."""
from pathlib import Path
import numpy as np
import pandas as pd
from vecdb.io.dataset import load
from vecdb.io.metadata_gen import load_metadata
from vecdb.store.metadata import MetaStore
from vecdb.predicate.compile import compile as compile_pred
from vecdb.predicate.selectivity import estimate_selectivity
from vecdb.index.flat import FlatIndex
from vecdb.index.hnsw import HNSWIndex
from vecdb.index.strategies import PreFilterStrategy, PostFilterStrategy
from vecdb.bench.sweep import SELECTIVITY_GRID, predicate_for_selectivity
from vecdb.bench.groundtruth import compute_filtered_groundtruth, cache_groundtruth, load_groundtruth
from vecdb.bench.harness import run_benchmark

N_QUERIES = 200
WARMUP = 20

def run_variant(variant: str, bundle, hnsw: HNSWIndex, flat: FlatIndex) -> pd.DataFrame:
    cols = load_metadata(Path(f"data/sift1m_100k_meta_{variant}.npz"))
    meta = MetaStore(cols)
    queries = bundle.queries[:N_QUERIES]
    rows = []
    for target_s in SELECTIVITY_GRID:
        pred = predicate_for_selectivity(meta, target_s)
        mask = compile_pred(pred, meta)
        true_s = mask.sum() / meta.n
        sel_hat = estimate_selectivity(pred, meta)

        gt_path = Path(f"results/gt_cache/{variant}_{target_s}.npy")
        if gt_path.exists():
            gt = load_groundtruth(gt_path)
        else:
            gt = compute_filtered_groundtruth(flat, queries, [mask] * len(queries), k=10)
            cache_groundtruth(gt_path, gt)

        masks = [mask] * len(queries)
        pre = PreFilterStrategy(flat)
        df_pre = run_benchmark(pre, queries, gt, k=10, masks=masks, warmup=WARMUP)
        df_pre["target_selectivity"] = target_s
        df_pre["true_selectivity"] = true_s
        df_pre["sel_hat"] = sel_hat
        df_pre["metadata_variant"] = variant

        post = PostFilterStrategy(hnsw, fallback=PreFilterStrategy(flat))
        df_post = run_benchmark(post, queries, gt, k=10, masks=masks,
                                  params={"selectivity_hat": sel_hat}, warmup=WARMUP)
        df_post["target_selectivity"] = target_s
        df_post["true_selectivity"] = true_s
        df_post["sel_hat"] = sel_hat
        df_post["metadata_variant"] = variant
        df_post["fallback_rate"] = post.fallback_rate

        rows.append(df_pre)
        rows.append(df_post)
        print(f"[{variant}] s={target_s}: pre recall={df_pre['recall'].mean():.3f} p95={df_pre['latency_ms'].quantile(0.95):.2f}ms"
              f" | post recall={df_post['recall'].mean():.3f} p95={df_post['latency_ms'].quantile(0.95):.2f}ms"
              f" fallback_rate={post.fallback_rate:.2f}")
    return pd.concat(rows, ignore_index=True)

def main() -> None:
    bundle = load("sift1m_100k", cache_dir=Path("data"))
    flat = FlatIndex()
    flat.add(bundle.base, np.arange(len(bundle.base)))
    hnsw = HNSWIndex.load(Path("data/hnsw_100k"))

    for variant, out_name in [("uncorrelated", "sweep_uncorrelated.csv"), ("correlated", "sweep_correlated.csv")]:
        df = run_variant(variant, bundle, hnsw, flat)
        df.to_csv(f"results/{out_name}", index=False)
        print(f"wrote results/{out_name}")

if __name__ == "__main__":
    main()
```

Run: `".venv/Scripts/python" scripts/run_sweep_ab.py`
Expected: per-selectivity print lines for both metadata variants, and `results/sweep_uncorrelated.csv` / `results/sweep_correlated.csv` written with complete rows for pre and post across all 8 selectivities. This reuses the same `results/gt_cache/` files in Phase 6 and Phase 7 — do not delete that cache between phases.

- [ ] **Step 6: Commit**

```bash
git add vecdb/bench/sweep.py tests/test_sweep.py scripts/run_sweep_ab.py
git commit -m "feat: selectivity-grid sweep driver for pre/post-filter strategies, both metadata variants"
```

---

### Task 28: Phase 5 milestone report

**Files:**
- Create: `docs/superpowers/milestones/05-pre-post-filter.md`

- [ ] **Step 1: Write the report**

Create `docs/superpowers/milestones/05-pre-post-filter.md` with: **What got built** (PreFilterStrategy, PostFilterStrategy with adaptive ef/retry/fallback, selectivity validation, the A/B sweep). **Numbers**: the actual `sweep_uncorrelated.csv`/`sweep_correlated.csv` summary (recall and p95 latency per selectivity for both strategies), the selectivity-estimation median/p95 error for both metadata variants from Task 26, and the observed underfill/fallback rate curve for post-filter as selectivity drops. **Gate status**: both CSVs have complete 8-selectivity rows for both strategies. **Interview note**: answer source plan §7 Q1 and Q8 in your own words using these numbers — why post-filtering breaks at low selectivity (cite the actual underfill numbers) and what happens when the selectivity estimate is wrong (cite the actual fallback rate).

- [ ] **Step 2: Commit and push**

```bash
git add docs/superpowers/milestones/05-pre-post-filter.md results/sweep_uncorrelated.csv results/sweep_correlated.csv results/figures/selectivity_estimation_uncorrelated.png results/figures/selectivity_estimation_correlated.png
git commit -m "docs: Phase 5 milestone report"
git push origin main
```

---

## Phase 6 — Predicate-aware traversal (Strategy C)

The hardest phase conceptually. `_search_layer_filtered` lives on `HNSWIndex` (it needs direct graph/store access, same as `_search_layer`); `FilteredHNSWStrategy` in `vecdb/index/strategies.py` wraps it, mirroring how `PostFilterStrategy` wraps plain `_search_layer`/`search`.

### Task 29: Two-tier admission and seeded entry points

**Files:**
- Modify: `vecdb/index/hnsw.py` (add `._search_layer_filtered`)
- Modify: `vecdb/index/strategies.py` (add `FilteredHNSWStrategy`)
- Test: `tests/test_filtered_hnsw.py`

**Interfaces:**
- Produces: `HNSWIndex._search_layer_filtered(self, q, entry_points, ef, layer, mask, budget_remaining=None) -> tuple[list[tuple[float,int]], int]` — traverses through every node reachable in the graph, but admits a node into the results heap only if `mask[node]` is true; returns `(results, ops_used)`. `class FilteredHNSWStrategy(Index)` with `__init__(self, hnsw_index, fallback, ef_base=64, n_seed_matches=8, seed=0)`, `._ef_eff(self, sel_hat: float) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_filtered_hnsw.py
import numpy as np
from vecdb.index.hnsw import HNSWIndex
from vecdb.index.flat import FlatIndex
from vecdb.index.strategies import PreFilterStrategy, FilteredHNSWStrategy
from vecdb.store.vectors import VectorStore

def _chain_index(n=20, d=2):
    """A chain graph 0-1-2-...-(n-1) on layer 0, so traversal must cross several
    non-matching nodes to reach a distant match."""
    points = np.array([[float(i), 0.0] for i in range(n)], dtype=np.float32)
    idx = HNSWIndex(dim=d, seed=0)
    idx.store = VectorStore(points)
    idx._visited_stamp = np.zeros(n, dtype=np.uint32)
    idx._visit_generation = 0
    idx._ensure_layers(0)
    idx.levels = [0] * n
    for i in range(n):
        idx.graph[0][i] = [j for j in (i - 1, i + 1) if 0 <= j < n]
    idx.entry_point = 0
    idx.max_level = 0
    return idx, points

def test_two_tier_admission_only_returns_matching_nodes():
    idx, points = _chain_index()
    mask = np.zeros(20, dtype=bool)
    mask[[15, 16, 17]] = True  # far from entry point 0; must be crossed-through to reach
    results, ops_used = idx._search_layer_filtered(
        np.array([16.0, 0.0], dtype=np.float32), entry_points=[0], ef=3, layer=0, mask=mask,
    )
    returned_ids = {node for _, node in results}
    assert returned_ids.issubset({15, 16, 17})
    assert ops_used > 3  # had to traverse through non-matching nodes to get there

def test_search_finds_matches_via_seeded_entry_point_even_when_far_from_global_entry():
    idx, points = _chain_index(n=50)
    flat = FlatIndex()
    flat.add(points, np.arange(50))
    strategy = FilteredHNSWStrategy(idx, fallback=PreFilterStrategy(flat), n_seed_matches=4, seed=1)
    mask = np.zeros(50, dtype=bool)
    mask[45:50] = True  # far from entry_point=0
    result = strategy.search(np.array([47.0, 0.0], dtype=np.float32), k=3, mask=mask,
                               params={"selectivity_hat": 0.1})
    assert all(mask[i] for i in result.ids)
    assert result.n_returned == 3

def test_ef_eff_widens_as_selectivity_drops():
    idx, points = _chain_index()
    flat = FlatIndex(); flat.add(points, np.arange(20))
    strategy = FilteredHNSWStrategy(idx, fallback=PreFilterStrategy(flat))
    assert strategy._ef_eff(sel_hat=0.5) < strategy._ef_eff(sel_hat=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_filtered_hnsw.py -v`
Expected: FAIL — `AttributeError`/`ImportError`

- [ ] **Step 3: Implement — add to `vecdb/index/hnsw.py`**

```python
    def _search_layer_filtered(self, q: np.ndarray, entry_points: list[int], ef: int, layer: int,
                                 mask: np.ndarray, budget_remaining: int | None = None) -> tuple[list[tuple[float, int]], int]:
        """Predicate-aware variant of _search_layer. Traverses through every node
        reachable in the graph (visitable) but admits a node into the results heap
        only if mask[node] is True (admissible). Refusing to cross non-matching nodes
        would strand greedy search whenever the induced subgraph is disconnected
        (source plan §2.2) — traversing through preserves connectivity at the cost of
        wasted distance computations on nodes that will never be returned. Returns
        (results, ops_used) so the caller can track a distance-op budget."""
        if budget_remaining is None:
            budget_remaining = len(self.store)
        self._visit_generation += 1
        gen = self._visit_generation
        stamp = self._visited_stamp
        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []
        ops_used = 0

        if entry_points:
            for ep in entry_points:
                stamp[ep] = gen
            dists = self.store.distances(q, np.array(entry_points, dtype=np.int64))
            ops_used += len(entry_points)
            for ep, d in zip(entry_points, dists):
                d = float(d)
                heapq.heappush(candidates, (d, ep))
                if mask[ep]:
                    heapq.heappush(results, (-d, ep))
                    if len(results) > ef:
                        heapq.heappop(results)

        while candidates and ops_used < budget_remaining:
            d_c, c = heapq.heappop(candidates)
            worst_d = -results[0][0] if results else float("inf")
            if d_c > worst_d and len(results) >= ef:
                break
            neighbours = [n for n in self.graph[layer].get(c, []) if stamp[n] != gen]
            if not neighbours:
                continue
            for n in neighbours:
                stamp[n] = gen
            dists = self.store.distances(q, np.array(neighbours, dtype=np.int64))
            ops_used += len(neighbours)
            for n, d in zip(neighbours, dists):
                d = float(d)
                heapq.heappush(candidates, (d, n))  # traverse through regardless of mask
                if mask[n]:
                    if len(results) < ef:
                        heapq.heappush(results, (-d, n))
                    elif d < -results[0][0]:
                        heapq.heappush(results, (-d, n))
                        heapq.heappop(results)

        return sorted((-nd, node) for nd, node in results), ops_used
```

- [ ] **Step 4: Implement — add to `vecdb/index/strategies.py`**

```python
class FilteredHNSWStrategy(Index):
    """Strategy C: predicate-aware graph traversal. See HNSWIndex._search_layer_filtered
    for the two-tier admission rule. Seeded with a handful of randomly sampled matching
    nodes alongside the normal hierarchical entry point — cheap insurance against
    starting stranded in a match-free region of the graph."""

    def __init__(self, hnsw_index: HNSWIndex, fallback: Index, ef_base: int = 64,
                 n_seed_matches: int = 8, seed: int = 0):
        self.hnsw = hnsw_index
        self.fallback = fallback
        self.ef_base = ef_base
        self.n_seed_matches = n_seed_matches
        self._rng = np.random.default_rng(seed)

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        raise NotImplementedError("FilteredHNSWStrategy wraps an already-built HNSWIndex")

    def _ef_eff(self, sel_hat: float) -> int:
        """The beam widens as matches get scarcer: ef_eff = ef_base * min(4, 1/max(s, 0.05))."""
        sel_hat = max(sel_hat, 0.05)
        return int(self.ef_base * min(4.0, 1.0 / sel_hat))

    def search(self, q: np.ndarray, k: int, mask: np.ndarray | None = None,
                params: dict | None = None) -> SearchResult:
        assert mask is not None, "FilteredHNSWStrategy requires a mask"
        params = params or {}
        sel_hat = params.get("selectivity_hat", 1.0)
        ef_eff = max(self._ef_eff(sel_hat), k)

        t0 = time.perf_counter()
        ops_before = self.hnsw.store.n_distance_ops

        ep = [self.hnsw.entry_point]
        for layer in range(self.hnsw.max_level, 0, -1):
            nearest = self.hnsw._search_layer(q, ep, ef=1, layer=layer)
            if nearest:
                ep = [nearest[0][1]]

        matches = np.nonzero(mask)[0]
        if matches.size > 0:
            n_seed = min(self.n_seed_matches, matches.size)
            seeds = self._rng.choice(matches, size=n_seed, replace=False).tolist()
            ep = sorted(set(ep) | set(seeds))

        results, _ = self.hnsw._search_layer_filtered(q, ep, ef=ef_eff, layer=0, mask=mask)
        results = results[:k]

        latency_ms = (time.perf_counter() - t0) * 1000
        n_ops = self.hnsw.store.n_distance_ops - ops_before
        ids = np.array([n for _, n in results], dtype=np.int64)
        dists = np.array([d for d, _ in results], dtype=np.float32)
        return SearchResult(ids=ids, distances=dists, n_distance_ops=n_ops,
                             strategy="predicate_aware", latency_ms=latency_ms, n_returned=int(ids.size))
```

Add the import at the top of `vecdb/index/strategies.py`: it already imports `HNSWIndex`; no new imports needed.

- [ ] **Step 5: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_filtered_hnsw.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add vecdb/index/hnsw.py vecdb/index/strategies.py tests/test_filtered_hnsw.py
git commit -m "feat(strategies): FilteredHNSWStrategy - two-tier admission + seeded matching entry points"
```

---

### Task 30: Two-hop expansion

**Files:**
- Modify: `vecdb/index/hnsw.py` (`_search_layer_filtered`)
- Modify: `vecdb/index/strategies.py` (`FilteredHNSWStrategy`)
- Test: `tests/test_filtered_hnsw.py` (extend)

**Interfaces:**
- Modifies: `_search_layer_filtered` gains a `two_hop_threshold: float = 0.1` parameter (default preserves prior behaviour only when explicitly passed `0.0`; the new default `0.1` activates the feature). `FilteredHNSWStrategy.__init__` gains `two_hop_threshold: float = 0.1` and passes it through.

- [ ] **Step 1: Write the failing test (append to `tests/test_filtered_hnsw.py`)**

```python
def test_two_hop_expansion_reaches_matches_two_hops_from_the_nearest_neighbour():
    # star graph: hub 0 connects to 1..5; 1 connects onward to 6 (a match). A search
    # that only expands one hop from the hub never reaches node 6 through node 1
    # unless two-hop expansion looks past 1 to 1's own neighbour, 6.
    idx = HNSWIndex(dim=2, seed=0)
    points = np.zeros((7, 2), dtype=np.float32)
    points[6] = [0.01, 0.0]  # node 6 nearly coincides with the query so it's clearly best
    from vecdb.store.vectors import VectorStore
    idx.store = VectorStore(points)
    idx._visited_stamp = np.zeros(7, dtype=np.uint32)
    idx._visit_generation = 0
    idx._ensure_layers(0)
    idx.graph[0] = {0: [1, 2, 3, 4, 5], 1: [0, 6], 2: [0], 3: [0], 4: [0], 5: [0], 6: [1]}
    idx.entry_point = 0
    idx.max_level = 0

    mask = np.zeros(7, dtype=bool)
    mask[6] = True  # only node 6 matches, and it is two hops from the entry point
    q = np.array([0.0, 0.0], dtype=np.float32)
    results, ops_used = idx._search_layer_filtered(q, entry_points=[0], ef=1, layer=0,
                                                      mask=mask, two_hop_threshold=0.5)
    assert results and results[0][1] == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_filtered_hnsw.py -v`
Expected: FAIL — `TypeError: _search_layer_filtered() got an unexpected keyword argument 'two_hop_threshold'`

- [ ] **Step 3: Implement — replace `_search_layer_filtered` in `vecdb/index/hnsw.py`**

Same as Task 29's version, but add the parameter and, right after the existing neighbour-expansion block inside the `while` loop, insert the two-hop block:

```python
    def _search_layer_filtered(self, q: np.ndarray, entry_points: list[int], ef: int, layer: int,
                                 mask: np.ndarray, budget_remaining: int | None = None,
                                 two_hop_threshold: float = 0.1) -> tuple[list[tuple[float, int]], int]:
        if budget_remaining is None:
            budget_remaining = len(self.store)
        self._visit_generation += 1
        gen = self._visit_generation
        stamp = self._visited_stamp
        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []
        ops_used = 0

        def _admit(node: int, d: float) -> None:
            if mask[node]:
                if len(results) < ef:
                    heapq.heappush(results, (-d, node))
                elif d < -results[0][0]:
                    heapq.heappush(results, (-d, node))
                    heapq.heappop(results)

        if entry_points:
            for ep in entry_points:
                stamp[ep] = gen
            dists = self.store.distances(q, np.array(entry_points, dtype=np.int64))
            ops_used += len(entry_points)
            for ep, d in zip(entry_points, dists):
                d = float(d)
                heapq.heappush(candidates, (d, ep))
                _admit(ep, d)

        while candidates and ops_used < budget_remaining:
            d_c, c = heapq.heappop(candidates)
            worst_d = -results[0][0] if results else float("inf")
            if d_c > worst_d and len(results) >= ef:
                break
            neighbours = [n for n in self.graph[layer].get(c, []) if stamp[n] != gen]
            if not neighbours:
                continue
            for n in neighbours:
                stamp[n] = gen
            dists = self.store.distances(q, np.array(neighbours, dtype=np.int64))
            ops_used += len(neighbours)
            for n, d in zip(neighbours, dists):
                d = float(d)
                heapq.heappush(candidates, (d, n))
                _admit(n, d)

            match_rate = sum(1 for n in neighbours if mask[n]) / len(neighbours)
            if match_rate < two_hop_threshold and ops_used < budget_remaining:
                two_hop = []
                seen = set()
                for n in neighbours:
                    for nn in self.graph[layer].get(n, []):
                        if stamp[nn] != gen and nn not in seen:
                            seen.add(nn)
                            two_hop.append(nn)
                if two_hop:
                    for nn in two_hop:
                        stamp[nn] = gen
                    d2 = self.store.distances(q, np.array(two_hop, dtype=np.int64))
                    ops_used += len(two_hop)
                    for n2, d in zip(two_hop, d2):
                        d = float(d)
                        heapq.heappush(candidates, (d, n2))
                        _admit(n2, d)

        return sorted((-nd, node) for nd, node in results), ops_used
```

- [ ] **Step 4: Implement — update `FilteredHNSWStrategy` in `vecdb/index/strategies.py`**

Add `two_hop_threshold: float = 0.1` to `__init__`, store it as `self.two_hop_threshold`, and pass `two_hop_threshold=self.two_hop_threshold` into the `_search_layer_filtered` call in `search()`.

- [ ] **Step 5: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_filtered_hnsw.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add vecdb/index/hnsw.py vecdb/index/strategies.py tests/test_filtered_hnsw.py
git commit -m "feat(strategies): two-hop neighbour expansion when local match rate is low (ACORN-style)"
```

---

### Task 31: Budget cap and honest bail-out

**Files:**
- Modify: `vecdb/index/strategies.py` (`FilteredHNSWStrategy`)
- Test: `tests/test_filtered_hnsw.py` (extend)

**Interfaces:**
- Modifies: `FilteredHNSWStrategy.__init__` gains `budget_fraction: float = 0.3`. `.search()` now hard-caps `_search_layer_filtered`'s `budget_remaining` at `budget_fraction * len(store)` and, if fewer than `k` results come back, bails out to `self.fallback.search(...)` with `strategy` relabeled `"predicate_aware_fallback"` and `n_distance_ops` including the wasted traversal cost. Adds `.bail_count: int`, `.query_count: int`, `.bail_rate: float` (property).

- [ ] **Step 1: Write the failing test (append to `tests/test_filtered_hnsw.py`)**

```python
def test_bails_out_to_fallback_when_budget_exhausted_without_k_results():
    idx, points = _chain_index(n=30)
    flat = FlatIndex(); flat.add(points, np.arange(30))
    # a tiny budget_fraction guarantees the traversal budget is exhausted before
    # it can reach any match, forcing a bail-out
    strategy = FilteredHNSWStrategy(idx, fallback=PreFilterStrategy(flat), budget_fraction=0.01)
    mask = np.zeros(30, dtype=bool)
    mask[29] = True  # the one match is at the far end of the chain from entry_point=0
    result = strategy.search(np.array([29.0, 0.0], dtype=np.float32), k=1, mask=mask,
                               params={"selectivity_hat": 0.03})
    assert result.strategy == "predicate_aware_fallback"
    assert result.n_returned == 1
    assert result.ids[0] == 29
    assert strategy.bail_count == 1
    assert strategy.query_count == 1
    assert strategy.bail_rate == 1.0

def test_does_not_bail_when_budget_is_generous():
    idx, points = _chain_index(n=30)
    flat = FlatIndex(); flat.add(points, np.arange(30))
    strategy = FilteredHNSWStrategy(idx, fallback=PreFilterStrategy(flat), budget_fraction=1.0)
    mask = np.zeros(30, dtype=bool)
    mask[29] = True
    result = strategy.search(np.array([29.0, 0.0], dtype=np.float32), k=1, mask=mask,
                               params={"selectivity_hat": 0.03})
    assert result.strategy == "predicate_aware"
    assert strategy.bail_count == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_filtered_hnsw.py -v`
Expected: FAIL — the first test's `result.strategy` will be `"predicate_aware"` with an under-filled (0-length) result instead of a fallback, because nothing bails out yet.

- [ ] **Step 3: Implement — modify `FilteredHNSWStrategy` in `vecdb/index/strategies.py`**

```python
    def __init__(self, hnsw_index: HNSWIndex, fallback: Index, ef_base: int = 64,
                 n_seed_matches: int = 8, two_hop_threshold: float = 0.1,
                 budget_fraction: float = 0.3, seed: int = 0):
        self.hnsw = hnsw_index
        self.fallback = fallback
        self.ef_base = ef_base
        self.n_seed_matches = n_seed_matches
        self.two_hop_threshold = two_hop_threshold
        self.budget_fraction = budget_fraction
        self._rng = np.random.default_rng(seed)
        self.bail_count = 0
        self.query_count = 0

    @property
    def bail_rate(self) -> float:
        return self.bail_count / self.query_count if self.query_count else 0.0
```

And in `search()`, after computing `ep` (seeded entry points) and before returning, replace the tail with:

```python
        self.query_count += 1
        budget = int(self.budget_fraction * len(self.hnsw.store))
        results, _ = self.hnsw._search_layer_filtered(
            q, ep, ef=ef_eff, layer=0, mask=mask,
            budget_remaining=budget, two_hop_threshold=self.two_hop_threshold,
        )
        results = results[:k]

        latency_ms = (time.perf_counter() - t0) * 1000
        n_ops = self.hnsw.store.n_distance_ops - ops_before

        if len(results) < k:
            self.bail_count += 1
            fb = self.fallback.search(q, k, mask=mask)
            fb.strategy = "predicate_aware_fallback"
            fb.n_distance_ops += n_ops
            return fb

        ids = np.array([n for _, n in results], dtype=np.int64)
        dists = np.array([d for d, _ in results], dtype=np.float32)
        return SearchResult(ids=ids, distances=dists, n_distance_ops=n_ops,
                             strategy="predicate_aware", latency_ms=latency_ms, n_returned=int(ids.size))
```

(`self.query_count += 1` moves from wherever it was to the top of the real search work; remove any duplicate increment left over from Task 29.)

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_filtered_hnsw.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add vecdb/index/strategies.py tests/test_filtered_hnsw.py
git commit -m "feat(strategies): distance-op budget cap with honest bail-out to fallback, tracked bail rate"
```

---

### Task 32: Selectivity-grid sweep for Strategy C

**Files:**
- Modify: `scripts/run_sweep_ab.py` → generalize into `scripts/run_sweep_all.py` (supersedes it; Phase 7 also builds on this generalized version)
- Test: none new (this reuses Task 27's tested `predicate_for_selectivity`/`SELECTIVITY_GRID` and Task 12's tested `run_benchmark`)

**Interfaces:**
- Consumes: `FilteredHNSWStrategy`, everything from Task 27's `scripts/run_sweep_ab.py`.
- Produces: `results/sweep_uncorrelated.csv` and `results/sweep_correlated.csv` now include `strategy in {"pre_filter", "post_filter", "post_filter_fallback", "predicate_aware", "predicate_aware_fallback"}` rows for every selectivity.

- [ ] **Step 1: Rewrite `scripts/run_sweep_ab.py` as `scripts/run_sweep_all.py`**

```python
# scripts/run_sweep_all.py
"""Phase 6 gate: full selectivity grid x both metadata variants for Strategies A, B,
and C. Supersedes scripts/run_sweep_ab.py (delete that file)."""
from pathlib import Path
import numpy as np
import pandas as pd
from vecdb.io.dataset import load
from vecdb.io.metadata_gen import load_metadata
from vecdb.store.metadata import MetaStore
from vecdb.predicate.compile import compile as compile_pred
from vecdb.predicate.selectivity import estimate_selectivity
from vecdb.index.flat import FlatIndex
from vecdb.index.hnsw import HNSWIndex
from vecdb.index.strategies import PreFilterStrategy, PostFilterStrategy, FilteredHNSWStrategy
from vecdb.bench.sweep import SELECTIVITY_GRID, predicate_for_selectivity
from vecdb.bench.groundtruth import compute_filtered_groundtruth, cache_groundtruth, load_groundtruth
from vecdb.bench.harness import run_benchmark

N_QUERIES = 200
WARMUP = 20

def run_variant(variant: str, bundle, hnsw: HNSWIndex, flat: FlatIndex) -> pd.DataFrame:
    cols = load_metadata(Path(f"data/sift1m_100k_meta_{variant}.npz"))
    meta = MetaStore(cols)
    queries = bundle.queries[:N_QUERIES]
    rows = []
    for target_s in SELECTIVITY_GRID:
        pred = predicate_for_selectivity(meta, target_s)
        mask = compile_pred(pred, meta)
        true_s = mask.sum() / meta.n
        sel_hat = estimate_selectivity(pred, meta)
        masks = [mask] * len(queries)

        gt_path = Path(f"results/gt_cache/{variant}_{target_s}.npy")
        if gt_path.exists():
            gt = load_groundtruth(gt_path)
        else:
            gt = compute_filtered_groundtruth(flat, queries, masks, k=10)
            cache_groundtruth(gt_path, gt)

        strategies = {
            "pre_filter": PreFilterStrategy(flat),
            "post_filter": PostFilterStrategy(hnsw, fallback=PreFilterStrategy(flat)),
            "predicate_aware": FilteredHNSWStrategy(hnsw, fallback=PreFilterStrategy(flat)),
        }
        for name, strat in strategies.items():
            df = run_benchmark(strat, queries, gt, k=10, masks=masks,
                                 params={"selectivity_hat": sel_hat}, warmup=WARMUP)
            df["target_selectivity"] = target_s
            df["true_selectivity"] = true_s
            df["sel_hat"] = sel_hat
            df["metadata_variant"] = variant
            rows.append(df)
            print(f"[{variant}] s={target_s} {name}: recall={df['recall'].mean():.3f}"
                  f" p95={df['latency_ms'].quantile(0.95):.2f}ms dist_ops_mean={df['dist_ops'].mean():.0f}")
    return pd.concat(rows, ignore_index=True)

def main() -> None:
    bundle = load("sift1m_100k", cache_dir=Path("data"))
    flat = FlatIndex()
    flat.add(bundle.base, np.arange(len(bundle.base)))
    hnsw = HNSWIndex.load(Path("data/hnsw_100k"))

    for variant, out_name in [("uncorrelated", "sweep_uncorrelated.csv"), ("correlated", "sweep_correlated.csv")]:
        df = run_variant(variant, bundle, hnsw, flat)
        df.to_csv(f"results/{out_name}", index=False)
        print(f"wrote results/{out_name}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Delete the superseded script and run the new one**

```bash
rm scripts/run_sweep_ab.py
```

Run: `".venv/Scripts/python" scripts/run_sweep_all.py`
Expected: per-selectivity, per-strategy print lines for both metadata variants, `results/sweep_uncorrelated.csv` / `results/sweep_correlated.csv` rewritten with all three strategies. **This is the "Done when" gate for Phase 6** (source plan §5 Day 5): on uncorrelated metadata, predicate-aware's latency should be below both pre- and post-filter for at least two selectivity values in the middle of the range, at recall@10 ≥ 0.90.

**If Strategy C never wins anywhere:** do not tune parameters until it does and do not fake the numbers. Work through source plan §5 Day 5's causes first (`ef_eff` too small, non-matches leaking into the admission logic, graph too sparse). If it's still genuinely losing everywhere after honest debugging, that is the documented, acceptable outcome from spec §8's risk register — write it up as a measured negative result with the dist_ops accounting explaining why, in the Phase 6 milestone report and later in the README limitations section.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_sweep_all.py
git rm scripts/run_sweep_ab.py
git commit -m "feat: generalize the sweep driver to all three strategies (A/B/C), Phase 6 gate run"
```

---

### Task 33: Phase 6 milestone report

**Files:**
- Create: `docs/superpowers/milestones/06-predicate-aware-traversal.md`

- [ ] **Step 1: Write the report**

Create `docs/superpowers/milestones/06-predicate-aware-traversal.md` with: **What got built** (two-tier admission, seeded entry points, two-hop expansion, dynamic ef, budget cap + bail-out). **Numbers**: from the Task 32 sweep — where (if anywhere) predicate-aware beats both fixed strategies on uncorrelated metadata, its recall there, and how it behaves on correlated metadata (expected: worse, possibly catastrophically so per spec §5 — report the actual bail rate). **Gate status**: honest statement of whether the Phase 6 "Done when" gate was met, and if not, the negative-result writeup per source plan §5. **Interview note**: answer source plan §7 Q11 ("what breaks if attributes correlate with vector position?") using your own bail-rate and latency numbers from both metadata variants side by side.

- [ ] **Step 2: Commit and push**

```bash
git add docs/superpowers/milestones/06-predicate-aware-traversal.md results/sweep_uncorrelated.csv results/sweep_correlated.csv
git commit -m "docs: Phase 6 milestone report"
git push origin main
```

---

## Phase 7 — Cost model, planner, full sweep, headline figures

Protect this phase — its output is the deliverable. `vecdb/planner/cost_model.py` and `vecdb/planner/planner.py` are hand-written ★ files.

### Task 34: Cost model formulas

**Files:**
- Create: `vecdb/planner/cost_model.py`
- Test: `tests/test_cost_model.py`

**Interfaces:**
- Produces: `@dataclass CostModelParams(c_scan, c_hop, alpha, beta, ef_min=16, ef_base=64, M=16, N=100_000, gamma_cap=50.0)` with `.save(path)`/`@classmethod .load(path)` (JSON), `cost_pre(params, k, sel_hat) -> float`, `ef_required(params, k, sel_hat) -> float`, `cost_post(params, k, sel_hat) -> float`, `gamma(params, sel_hat) -> float`, `cost_pred(params, k, sel_hat) -> float`. Formulas are exactly spec §3's: `C_pre = c_scan·N·ŝ`, `ef_required = clamp(alpha·k/ŝ, ef_min, N)`, `C_post = c_hop·M·ef_required`, `γ(ŝ) = 1 + β·(1/ŝ - 1)` capped at `gamma_cap`, `C_pred = c_hop·M·ef_base·γ(ŝ)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cost_model.py
from vecdb.planner.cost_model import CostModelParams, cost_pre, cost_post, cost_pred, ef_required, gamma

def make_params():
    return CostModelParams(c_scan=1.0, c_hop=1.0, alpha=4.0, beta=2.0, ef_min=16, ef_base=64, M=16, N=100_000)

def test_cost_pre_scales_linearly_with_selectivity():
    p = make_params()
    assert cost_pre(p, k=10, sel_hat=0.1) == p.c_scan * p.N * 0.1
    assert cost_pre(p, k=10, sel_hat=0.2) == 2 * cost_pre(p, k=10, sel_hat=0.1)

def test_ef_required_clamped_to_ef_min_and_n():
    p = make_params()
    assert ef_required(p, k=10, sel_hat=1.0) == p.ef_min
    assert ef_required(p, k=10, sel_hat=1e-9) == p.N

def test_cost_post_increases_as_selectivity_drops():
    p = make_params()
    assert cost_post(p, k=10, sel_hat=0.5) < cost_post(p, k=10, sel_hat=0.01)

def test_gamma_is_capped():
    p = make_params()
    assert gamma(p, sel_hat=1e-9) == p.gamma_cap

def test_cost_pred_increases_as_selectivity_drops():
    p = make_params()
    assert cost_pred(p, k=10, sel_hat=0.5) < cost_pred(p, k=10, sel_hat=0.01)

def test_params_save_and_load_roundtrip(tmp_path):
    p = make_params()
    path = tmp_path / "calibration.json"
    p.save(path)
    loaded = CostModelParams.load(path)
    assert loaded == p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_cost_model.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# vecdb/planner/cost_model.py
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np


@dataclass
class CostModelParams:
    c_scan: float
    c_hop: float
    alpha: float
    beta: float
    ef_min: int = 16
    ef_base: int = 64
    M: int = 16
    N: int = 100_000
    gamma_cap: float = 50.0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CostModelParams":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def cost_pre(params: CostModelParams, k: int, sel_hat: float) -> float:
    return params.c_scan * params.N * sel_hat


def ef_required(params: CostModelParams, k: int, sel_hat: float) -> float:
    sel_hat = max(sel_hat, 1e-6)
    return float(np.clip(params.alpha * k / sel_hat, params.ef_min, params.N))


def cost_post(params: CostModelParams, k: int, sel_hat: float) -> float:
    return params.c_hop * params.M * ef_required(params, k, sel_hat)


def gamma(params: CostModelParams, sel_hat: float) -> float:
    sel_hat = max(sel_hat, 1e-6)
    return min(1.0 + params.beta * (1.0 / sel_hat - 1.0), params.gamma_cap)


def cost_pred(params: CostModelParams, k: int, sel_hat: float) -> float:
    return params.c_hop * params.M * params.ef_base * gamma(params, sel_hat)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_cost_model.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add vecdb/planner/cost_model.py tests/test_cost_model.py
git commit -m "feat(planner): cost model formulas for all three strategies"
```

---

### Task 35: Calibration script

**Files:**
- Create: `scripts/calibrate.py`

**Interfaces:**
- Consumes: `results/sweep_uncorrelated.csv` (Task 32), `vecdb.planner.cost_model.CostModelParams`.
- Produces: `results/calibration.json`.

`alpha` (post-filter's ef-widening constant), `ef_min`, `ef_base`, and `M` are the design constants already baked into `PostFilterStrategy`/`FilteredHNSWStrategy` (Phases 5-6) — they are not fit, they are read off the code that's already running. `c_scan`, `c_hop`, and `beta` **are** fit by least squares against measured `dist_ops`, which is what makes this a calibrated cost model rather than three guessed constants (spec §3.2).

- [ ] **Step 1: Write and run `scripts/calibrate.py`**

```python
# scripts/calibrate.py
"""Phase 7: fit c_scan, c_hop, beta by least squares against Task 32's measured
dist_ops on the uncorrelated sweep. alpha/ef_min/ef_base/M are read from the
strategy defaults already in use, not fit."""
from pathlib import Path
import numpy as np
import pandas as pd
from vecdb.planner.cost_model import CostModelParams

ALPHA = 4.0     # PostFilterStrategy default
EF_MIN = 16     # PostFilterStrategy default
EF_BASE = 64    # FilteredHNSWStrategy default
M = 16          # HNSW build parameter (Phase 3/4)
K = 10

def fit_c_scan(df: pd.DataFrame, N: int) -> float:
    pre = df[df["strategy"] == "pre_filter"]
    x = pre["true_selectivity"].to_numpy() * N
    y = pre["dist_ops"].to_numpy()
    return float(np.sum(x * y) / np.sum(x * x))

def fit_c_hop(df: pd.DataFrame, N: int) -> float:
    post = df[df["strategy"] == "post_filter"]
    s = post["true_selectivity"].to_numpy()
    ef_req = np.clip(ALPHA * K / np.maximum(s, 1e-6), EF_MIN, N)
    x = M * ef_req
    y = post["dist_ops"].to_numpy()
    return float(np.sum(x * y) / np.sum(x * x))

def fit_beta(df: pd.DataFrame, c_hop: float) -> float:
    pred = df[df["strategy"] == "predicate_aware"]
    s = np.maximum(pred["true_selectivity"].to_numpy(), 1e-6)
    x = (1.0 / s) - 1.0
    y = pred["dist_ops"].to_numpy() / max(c_hop * M * EF_BASE, 1e-9)
    design = np.column_stack([np.ones_like(x), x])
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(b / a) if a != 0 else 0.0

def main() -> None:
    N = 100_000
    df = pd.read_csv("results/sweep_uncorrelated.csv")
    c_scan = fit_c_scan(df, N)
    c_hop = fit_c_hop(df, N)
    beta = fit_beta(df, c_hop)
    params = CostModelParams(c_scan=c_scan, c_hop=c_hop, alpha=ALPHA, beta=beta,
                               ef_min=EF_MIN, ef_base=EF_BASE, M=M, N=N)
    params.save(Path("results/calibration.json"))
    print(params)

if __name__ == "__main__":
    main()
```

Run: `".venv/Scripts/python" scripts/calibrate.py`
Expected: prints the fitted `CostModelParams` and writes `results/calibration.json`. Sanity-check `c_scan` — since `pre_filter`'s `dist_ops` is *exactly* `N·s` by construction (Task 11), the fit should land very close to `c_scan ≈ 1.0`; if it's wildly off, something upstream (likely `true_selectivity` computation) is broken and needs fixing before trusting the planner built on top of it.

- [ ] **Step 2: Commit**

```bash
git add scripts/calibrate.py results/calibration.json
git commit -m "feat: least-squares calibration of cost model constants from measured dist_ops"
```

---

### Task 36: Planner

**Files:**
- Create: `vecdb/planner/planner.py`
- Test: `tests/test_planner.py`

**Interfaces:**
- Consumes: `vecdb.planner.cost_model.CostModelParams`, `cost_pre`, `cost_post`, `cost_pred`.
- Produces: `@dataclass ExecutionPlan(strategy: str, reason: str, sel_hat: float, costs: dict[str, float])`, `class Planner` with `__init__(self, params: CostModelParams)`, `.plan(self, k: int, sel_hat: float) -> ExecutionPlan`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_planner.py
from vecdb.planner.cost_model import CostModelParams
from vecdb.planner.planner import Planner

def make_planner():
    params = CostModelParams(c_scan=1.0, c_hop=0.01, alpha=4.0, beta=5.0,
                               ef_min=16, ef_base=64, M=16, N=100_000)
    return Planner(params)

def test_pre_filter_wins_at_very_low_selectivity():
    plan = make_planner().plan(k=10, sel_hat=0.0005)
    assert plan.strategy == "pre_filter"

def test_plan_reason_names_the_chosen_strategy_and_its_cost():
    plan = make_planner().plan(k=10, sel_hat=0.0005)
    assert plan.strategy in plan.reason
    assert "ŝ=" in plan.reason

def test_plan_reports_all_three_costs():
    plan = make_planner().plan(k=10, sel_hat=0.1)
    assert set(plan.costs) == {"pre_filter", "post_filter", "predicate_aware"}
    assert plan.costs[plan.strategy] == min(plan.costs.values())

def test_plan_records_the_selectivity_it_was_given():
    plan = make_planner().plan(k=10, sel_hat=0.02)
    assert plan.sel_hat == 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_planner.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# vecdb/planner/planner.py
from __future__ import annotations
from dataclasses import dataclass
from vecdb.planner.cost_model import CostModelParams, cost_pre, cost_post, cost_pred


@dataclass
class ExecutionPlan:
    strategy: str
    reason: str
    sel_hat: float
    costs: dict[str, float]


class Planner:
    """Computes all three strategy costs from an estimated selectivity and picks the
    argmin. The reason string is returned to the API caller (Phase 8) — an explainable
    planner is far more compelling in a demo than a black box (spec §3.2)."""

    def __init__(self, params: CostModelParams):
        self.params = params

    def plan(self, k: int, sel_hat: float) -> ExecutionPlan:
        costs = {
            "pre_filter": cost_pre(self.params, k, sel_hat),
            "post_filter": cost_post(self.params, k, sel_hat),
            "predicate_aware": cost_pred(self.params, k, sel_hat),
        }
        best = min(costs, key=costs.get)
        others = ", ".join(f"{name}={cost:.0f}" for name, cost in costs.items() if name != best)
        reason = f"{best}: ŝ={sel_hat:.4f} -> cost={costs[best]:.0f} < {others}"
        return ExecutionPlan(strategy=best, reason=reason, sel_hat=sel_hat, costs=costs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_planner.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add vecdb/planner/planner.py tests/test_planner.py
git commit -m "feat(planner): cost-based strategy selection with human-readable reason string"
```

---

### Task 37: Regret measurement

**Files:**
- Create: `vecdb/bench/regret.py`
- Test: `tests/test_regret.py`

**Interfaces:**
- Consumes: a `pandas.DataFrame` with one row per query carrying each fixed strategy's latency plus the planner's realized latency.
- Produces: `compute_regret(df, fixed_strategies: list[str], chosen_col: str) -> pd.Series`, `regret_summary(regret) -> dict[str, float]` (`mean_regret_ms`, `p95_regret_ms`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_regret.py
import pandas as pd
import pytest
from vecdb.bench.regret import compute_regret, regret_summary

def test_regret_is_zero_when_planner_always_picks_the_best():
    df = pd.DataFrame({
        "pre_filter_latency_ms": [10.0, 5.0, 8.0],
        "post_filter_latency_ms": [20.0, 3.0, 9.0],
        "predicate_aware_latency_ms": [15.0, 4.0, 7.0],
        "planner_latency_ms": [10.0, 3.0, 7.0],  # always the min of the three
    })
    regret = compute_regret(df, ["pre_filter", "post_filter", "predicate_aware"], "planner_latency_ms")
    assert (regret == 0).all()

def test_regret_is_positive_when_planner_picks_worse_option():
    df = pd.DataFrame({
        "pre_filter_latency_ms": [10.0],
        "post_filter_latency_ms": [20.0],
        "predicate_aware_latency_ms": [15.0],
        "planner_latency_ms": [20.0],  # picked the worst of the three
    })
    regret = compute_regret(df, ["pre_filter", "post_filter", "predicate_aware"], "planner_latency_ms")
    assert regret.iloc[0] == pytest.approx(10.0)  # 20 - min(10,20,15)

def test_regret_summary_reports_mean_and_p95():
    df = pd.DataFrame({
        "pre_filter_latency_ms": [10.0] * 100,
        "post_filter_latency_ms": [10.0] * 100,
        "predicate_aware_latency_ms": [10.0] * 100,
        "planner_latency_ms": [10.0] * 99 + [50.0],  # one bad outlier
    })
    regret = compute_regret(df, ["pre_filter", "post_filter", "predicate_aware"], "planner_latency_ms")
    summary = regret_summary(regret)
    assert summary["mean_regret_ms"] > 0
    assert summary["p95_regret_ms"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_regret.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# vecdb/bench/regret.py
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_regret(df: pd.DataFrame, fixed_strategies: list[str], chosen_col: str) -> pd.Series:
    """regret_ms = chosen strategy's latency - best fixed strategy's latency, per
    query, in hindsight. df needs one f'{strategy}_latency_ms' column per fixed
    strategy plus chosen_col holding the planner's realized latency."""
    best = df[[f"{s}_latency_ms" for s in fixed_strategies]].min(axis=1)
    return df[chosen_col] - best


def regret_summary(regret: pd.Series) -> dict[str, float]:
    return {
        "mean_regret_ms": float(np.mean(regret)),
        "p95_regret_ms": float(np.quantile(regret, 0.95)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_regret.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add vecdb/bench/regret.py tests/test_regret.py
git commit -m "feat: regret computation (chosen vs hindsight-best latency)"
```

---

### Task 38: Full sweep with the planner, and regret report

**Files:**
- Create: `scripts/run_full_sweep.py` (supersedes `scripts/run_sweep_all.py` from Task 32 — delete it)
- Test: none new (composes already-tested pieces)

**Interfaces:**
- Consumes: `Planner`, `CostModelParams.load`, everything from Task 32's sweep, `compute_regret`/`regret_summary`.
- Produces: final `results/sweep_uncorrelated.csv` / `results/sweep_correlated.csv`, each including `planner` rows (the chosen strategy's own measured latency for every query in a cell, relabeled) and a `planner_reason` column; prints the mean/p95 regret gate from spec §1.2 (`planner mean latency < min(mean latency of any single fixed strategy)` across the sweep).

- [ ] **Step 1: Write `scripts/run_full_sweep.py`**

```python
# scripts/run_full_sweep.py
"""Phase 7 gate: the full 8 selectivities x 2 metadata variants x 4 executors
(3 fixed strategies + planner) x 200 queries sweep, plus the regret report."""
from pathlib import Path
import numpy as np
import pandas as pd
from vecdb.io.dataset import load
from vecdb.io.metadata_gen import load_metadata
from vecdb.store.metadata import MetaStore
from vecdb.predicate.compile import compile as compile_pred
from vecdb.predicate.selectivity import estimate_selectivity
from vecdb.index.flat import FlatIndex
from vecdb.index.hnsw import HNSWIndex
from vecdb.index.strategies import PreFilterStrategy, PostFilterStrategy, FilteredHNSWStrategy
from vecdb.bench.sweep import SELECTIVITY_GRID, predicate_for_selectivity
from vecdb.bench.groundtruth import compute_filtered_groundtruth, cache_groundtruth, load_groundtruth
from vecdb.bench.harness import run_benchmark
from vecdb.bench.regret import compute_regret, regret_summary
from vecdb.planner.cost_model import CostModelParams
from vecdb.planner.planner import Planner

N_QUERIES = 200
WARMUP = 20

def run_variant(variant: str, bundle, hnsw: HNSWIndex, flat: FlatIndex, planner: Planner) -> pd.DataFrame:
    cols = load_metadata(Path(f"data/sift1m_100k_meta_{variant}.npz"))
    meta = MetaStore(cols)
    queries = bundle.queries[:N_QUERIES]
    rows = []
    for target_s in SELECTIVITY_GRID:
        pred = predicate_for_selectivity(meta, target_s)
        mask = compile_pred(pred, meta)
        true_s = mask.sum() / meta.n
        sel_hat = estimate_selectivity(pred, meta)
        masks = [mask] * len(queries)

        gt_path = Path(f"results/gt_cache/{variant}_{target_s}.npy")
        gt = load_groundtruth(gt_path) if gt_path.exists() else compute_filtered_groundtruth(flat, queries, masks, k=10)
        if not gt_path.exists():
            cache_groundtruth(gt_path, gt)

        plan = planner.plan(k=10, sel_hat=sel_hat)
        strategies = {
            "pre_filter": PreFilterStrategy(flat),
            "post_filter": PostFilterStrategy(hnsw, fallback=PreFilterStrategy(flat)),
            "predicate_aware": FilteredHNSWStrategy(hnsw, fallback=PreFilterStrategy(flat)),
        }
        per_strategy = {}
        for name, strat in strategies.items():
            df = run_benchmark(strat, queries, gt, k=10, masks=masks,
                                 params={"selectivity_hat": sel_hat}, warmup=WARMUP)
            df["target_selectivity"] = target_s
            df["true_selectivity"] = true_s
            df["sel_hat"] = sel_hat
            df["metadata_variant"] = variant
            per_strategy[name] = df
            rows.append(df)

        planner_df = per_strategy[plan.strategy].copy()
        planner_df["strategy"] = "planner"
        planner_df["planner_reason"] = plan.reason
        rows.append(planner_df)
        print(f"[{variant}] s={target_s} planner picked {plan.strategy} ({plan.reason})")

    return pd.concat(rows, ignore_index=True)

def main() -> None:
    bundle = load("sift1m_100k", cache_dir=Path("data"))
    flat = FlatIndex()
    flat.add(bundle.base, np.arange(len(bundle.base)))
    hnsw = HNSWIndex.load(Path("data/hnsw_100k"))
    params = CostModelParams.load(Path("results/calibration.json"))
    planner = Planner(params)

    for variant, out_name in [("uncorrelated", "sweep_uncorrelated.csv"), ("correlated", "sweep_correlated.csv")]:
        df = run_variant(variant, bundle, hnsw, flat, planner)
        df.to_csv(f"results/{out_name}", index=False)
        print(f"wrote results/{out_name}")

    # Regret, computed on the uncorrelated sweep (the planner's calibration target)
    df = pd.read_csv("results/sweep_uncorrelated.csv")
    wide = df.groupby(["query_idx", "target_selectivity", "strategy"])["latency_ms"].mean().unstack("strategy")
    wide = wide.rename(columns={s: f"{s}_latency_ms" for s in ["pre_filter", "post_filter", "predicate_aware"]})
    wide = wide.rename(columns={"planner": "planner_latency_ms"})
    regret = compute_regret(wide, ["pre_filter", "post_filter", "predicate_aware"], "planner_latency_ms")
    print("regret:", regret_summary(regret))

    mean_fixed = {s: df[df["strategy"] == s]["latency_ms"].mean() for s in ["pre_filter", "post_filter", "predicate_aware"]}
    mean_planner = df[df["strategy"] == "planner"]["latency_ms"].mean()
    print("mean latency — fixed strategies:", mean_fixed, " planner:", mean_planner)
    print("Phase 7 planner-beats-fixed gate:", mean_planner < min(mean_fixed.values()))

if __name__ == "__main__":
    main()
```

Run: `".venv/Scripts/python" scripts/run_full_sweep.py`
Expected: full sweep completes (source plan §5 Day 6 estimates 30-60 minutes — run it, do not shortcut the grid), prints the planner's choice and reason per cell, the regret summary, and whether the planner-beats-every-fixed-strategy gate (spec §1.2 objective 5) held. Record the actual numbers in the Phase 7 milestone report — if the gate did not hold, that is itself a reportable finding, not something to massage.

**If the calibrated cost model clearly doesn't fit** (e.g. `fit_c_scan`/`fit_c_hop` come out negative or wildly unstable, or the planner's choices look nonsensical against the measured crossover), this is spec §8's documented "cost model doesn't fit" risk — the fallback is a decile lookup-table planner instead of the formula-based one: bucket `sel_hat` into deciles from `results/sweep_uncorrelated.csv`, record the empirically-best strategy per decile, and have `Planner.plan()` look up that table instead of computing `cost_pre`/`cost_post`/`cost_pred`. Only take this path if the calibrated formula genuinely doesn't work — implement it as `Planner.from_lookup_table(...)` alongside (not replacing) the existing constructor, and say plainly in the milestone report and README that it's a lookup table, not a cost model.

- [ ] **Step 2: Delete the superseded script, commit, do not push yet (Task 39 adds figures to the same commit set)**

```bash
git rm scripts/run_sweep_all.py
git add scripts/run_full_sweep.py results/sweep_uncorrelated.csv results/sweep_correlated.csv
git commit -m "feat: full sweep with planner as 4th executor, regret report"
```

---

### Task 39: Headline figures and Phase 7 milestone report

**Files:**
- Create: `scripts/generate_figures.py`
- Create: `docs/superpowers/milestones/07-planner-sweep-figures.md`

**Interfaces:**
- Consumes: `results/sweep_uncorrelated.csv`, `results/sweep_correlated.csv`, `vecdb.bench.plots.plot_lines`.
- Produces: `results/figures/crossover.png`, `crossover_correlated.png`, `recall_vs_selectivity.png`, `underfill.png`, `dist_ops.png` (the two `selectivity_estimation_*.png` and `pareto_unfiltered.png` figures already exist from Phases 4-5).

- [ ] **Step 1: Write `scripts/generate_figures.py`**

```python
# scripts/generate_figures.py
"""Phase 7 headline figures. crossover.png is THE deliverable — it goes at the top
of the README, above the fold."""
from pathlib import Path
import pandas as pd
from vecdb.bench.plots import plot_lines

STRATEGY_LABELS = {
    "pre_filter": "pre-filter", "post_filter": "post-filter",
    "predicate_aware": "predicate-aware", "planner": "planner (chosen)",
}

def _series_by_strategy(df: pd.DataFrame, value_col: str, agg: str) -> dict[str, list[tuple[float, float]]]:
    grouped = df.groupby(["strategy", "target_selectivity"])[value_col]
    agg_df = grouped.quantile(0.95) if agg == "p95" else grouped.mean()
    series: dict[str, list[tuple[float, float]]] = {}
    for (strategy, s), value in agg_df.items():
        if strategy not in STRATEGY_LABELS:
            continue
        series.setdefault(STRATEGY_LABELS[strategy], []).append((s, value))
    return series

def crossover(df: pd.DataFrame, out_path: Path, title: str) -> None:
    series = _series_by_strategy(df, "latency_ms", "p95")
    plot_lines(series, out_path, xlabel="selectivity (log)", ylabel="p95 latency (ms, log)",
               title=title, xscale="log", yscale="log")

def recall_vs_selectivity(df: pd.DataFrame, out_path: Path) -> None:
    series = _series_by_strategy(df, "recall", "mean")
    plot_lines(series, out_path, xlabel="selectivity (log)", ylabel="mean recall@10",
               title="Recall vs selectivity", xscale="log")

def underfill(df: pd.DataFrame, out_path: Path) -> None:
    post = df[df["strategy"] == "post_filter"]
    grouped = post.groupby("target_selectivity")["underfill"].mean()
    series = {"post-filter underfill rate": list(grouped.items())}
    plot_lines(series, out_path, xlabel="selectivity (log)", ylabel="underfill rate",
               title="Post-filter under-fill rate (the correctness bug, visualised)", xscale="log")

def dist_ops(df: pd.DataFrame, out_path: Path) -> None:
    series = _series_by_strategy(df, "dist_ops", "mean")
    plot_lines(series, out_path, xlabel="selectivity (log)", ylabel="mean dist_ops (log)",
               title="Distance computations vs selectivity (hardware-independent)",
               xscale="log", yscale="log")

def main() -> None:
    uncorr = pd.read_csv("results/sweep_uncorrelated.csv")
    corr = pd.read_csv("results/sweep_correlated.csv")

    crossover(uncorr, Path("results/figures/crossover.png"),
              "Crossover: winning strategy by selectivity (uncorrelated metadata)")
    crossover(corr, Path("results/figures/crossover_correlated.png"),
              "Crossover: winning strategy by selectivity (correlated metadata)")
    recall_vs_selectivity(uncorr, Path("results/figures/recall_vs_selectivity.png"))
    underfill(uncorr, Path("results/figures/underfill.png"))
    dist_ops(uncorr, Path("results/figures/dist_ops.png"))
    print("wrote crossover.png, crossover_correlated.png, recall_vs_selectivity.png, underfill.png, dist_ops.png")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `".venv/Scripts/python" scripts/generate_figures.py`
Expected: 5 new PNGs in `results/figures/`. Open `crossover.png` and confirm by eye that the winning strategy changes at least twice across the selectivity axis — this is the Phase 7 "Done when" gate (spec §6). If it doesn't, that's a finding for the milestone report, not something to force by cherry-picking the plot range.

- [ ] **Step 3: Write the Phase 7 milestone report**

Create `docs/superpowers/milestones/07-planner-sweep-figures.md` with: **What got built** (calibration, cost model, planner with reason strings, full sweep, 5 figures). **Numbers**: the calibrated `c_scan`/`c_hop`/`beta` from Task 35, the mean/p95 regret from Task 38, whether the planner-beats-fixed-strategies gate held (with the actual mean latencies), and where the crossover chart's strategy changes actually land on the selectivity axis (compare to source plan §5 Day 6's expected shape — note agreement or divergence honestly). **Gate status**. **Interview note**: answer source plan §7 Q7-Q9 (selectivity estimation, bounded harm when the estimate is wrong, how the constants were fit) in your own words with your own numbers.

- [ ] **Step 4: Commit and push**

```bash
git add scripts/generate_figures.py docs/superpowers/milestones/07-planner-sweep-figures.md results/figures/crossover.png results/figures/crossover_correlated.png results/figures/recall_vs_selectivity.png results/figures/underfill.png results/figures/dist_ops.png
git commit -m "feat: headline figures (crossover chart), Phase 7 milestone report"
git push origin main
```

---

## Phase 8 — FastAPI service, tests, README, polish

### Task 40: FastAPI service

**Files:**
- Modify: `pyproject.toml` (add `httpx` to the `dev` extra, needed by `fastapi.testclient.TestClient`)
- Create: `vecdb/service/schemas.py`
- Create: `vecdb/service/app.py`
- Test: `tests/test_service.py`

**Interfaces:**
- Consumes: `FlatIndex`, `HNSWIndex`, `MetaStore`, `IdMap`, `Planner`, `PreFilterStrategy`/`PostFilterStrategy`/`FilteredHNSWStrategy`, `compile`, `estimate_selectivity`.
- Produces: `class VecDBService` with `__init__(self, flat, hnsw, meta, idmap, planner)` (test-friendly — components injected directly) and `@classmethod .from_disk(cls, data_dir=Path("data")) -> VecDBService` (production startup, loads the persisted 100K index). Methods `.insert(req: InsertRequest) -> None`, `.search(req: SearchRequest) -> SearchResponse`, `.stats() -> StatsResponse`, `.persist(path: str) -> PersistResponse`. FastAPI `app` with routes `POST /insert`, `POST /search`, `GET /stats`, `POST /persist`, all depending on `get_service` via `Depends` (overridable in tests).

Newly inserted vectors live in a small in-memory "staged" overlay that `/search` always brute-force-scans and merges with the main index's results — this avoids mutating the hand-written HNSW graph live, and is called out plainly in the README as the real-time-insert design choice (analogous to near-real-time segment overlays in production search systems).

- [ ] **Step 1: Add `httpx` to `pyproject.toml`'s dev extra and reinstall**

Change `[project.optional-dependencies]` to:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27"]
```

Run: `".venv/Scripts/python" -m pip install -e ".[dev]"`

- [ ] **Step 2: Write the failing test**

```python
# tests/test_service.py
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `".venv/Scripts/python" -m pytest tests/test_service.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement `vecdb/service/schemas.py`**

```python
# vecdb/service/schemas.py
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
```

- [ ] **Step 5: Implement `vecdb/service/app.py`**

```python
# vecdb/service/app.py
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
def persist(path: str = "data/hnsw_100k", service: VecDBService = Depends(get_service)):
    return service.persist(path)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `".venv/Scripts/python" -m pytest tests/test_service.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Manual smoke test against the real 100K index (optional but recommended)**

Run (background): `".venv/Scripts/python" -m uvicorn vecdb.service.app:app --port 8000`
Then in another shell: `curl -s -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" -d "{\"vector\": [0.0], \"k\": 5}"` will 400 (wrong dim) — use a real 128-d vector from `bundle.queries[0]` instead if you want a genuine smoke test; otherwise `curl http://127.0.0.1:8000/stats` alone confirms the service boots against the real persisted index.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml vecdb/service/schemas.py vecdb/service/app.py tests/test_service.py
git commit -m "feat: FastAPI service (insert/search/stats/persist) with staged-insert overlay"
```

---

### Task 41: `test_strategies_agree.py`

**Files:**
- Create: `tests/test_strategies_agree.py`

**Interfaces:**
- Consumes: `FlatIndex`, `HNSWIndex`, `PreFilterStrategy`, `PostFilterStrategy`, `FilteredHNSWStrategy`.

- [ ] **Step 1: Write the test**

```python
# tests/test_strategies_agree.py
import numpy as np
from vecdb.index.flat import FlatIndex
from vecdb.index.hnsw import HNSWIndex
from vecdb.index.strategies import PreFilterStrategy, PostFilterStrategy, FilteredHNSWStrategy

def _build(n=400, d=16, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.random((n, d)).astype(np.float32)
    flat = FlatIndex(); flat.add(data, np.arange(n))
    hnsw = HNSWIndex(dim=d, M=16, ef_construction=200, seed=seed)
    hnsw.add(data, np.arange(n))
    return data, flat, hnsw

def test_all_three_strategies_agree_with_flat_on_tiny_data_with_generous_budgets():
    """With ef/budget large enough relative to N, every strategy should recover the
    exact top-k that FlatIndex does — this is the correctness contract every strategy
    must satisfy regardless of how it gets there."""
    data, flat, hnsw = _build(n=300)
    rng = np.random.default_rng(1)
    mask = rng.random(300) < 0.4  # generous selectivity so none of the strategies underfill

    pre = PreFilterStrategy(flat)
    post = PostFilterStrategy(hnsw, fallback=pre, alpha=8.0, ef_min=64)
    pred_aware = FilteredHNSWStrategy(hnsw, fallback=pre, ef_base=128, budget_fraction=0.9)

    for trial in range(5):
        q = rng.random(16).astype(np.float32)
        true_ids = set(flat.search(q, k=5, mask=mask).ids.tolist())

        pre_ids = set(pre.search(q, k=5, mask=mask).ids.tolist())
        post_ids = set(post.search(q, k=5, mask=mask, params={"selectivity_hat": 0.4}).ids.tolist())
        pred_ids = set(pred_aware.search(q, k=5, mask=mask, params={"selectivity_hat": 0.4}).ids.tolist())

        assert pre_ids == true_ids, f"trial {trial}: pre-filter disagreed with Flat"
        # HNSW-backed strategies are approximate even with generous budgets; require
        # substantial overlap rather than exact equality
        assert len(post_ids & true_ids) >= 4, f"trial {trial}: post-filter recall too low"
        assert len(pred_ids & true_ids) >= 4, f"trial {trial}: predicate-aware recall too low"
```

- [ ] **Step 2: Run it**

Run: `".venv/Scripts/python" -m pytest tests/test_strategies_agree.py -v`
Expected: PASS. If post/predicate-aware assertions fail, it means the generous parameters chosen here aren't generous enough for this data/seed — widen `ef_min`/`ef_base`/`budget_fraction` further rather than weakening the assertion below 4/5.

- [ ] **Step 3: Commit**

```bash
git add tests/test_strategies_agree.py
git commit -m "test: all three strategies agree with FlatIndex under generous budgets"
```

---

### Task 42: `test_planner_regret.py`

**Files:**
- Create: `tests/test_planner_regret.py`

**Interfaces:**
- Consumes: `results/sweep_uncorrelated.csv` (Task 38's cached full-sweep output), `vecdb.bench.regret.compute_regret`/`regret_summary`.

- [ ] **Step 1: Write the test**

```python
# tests/test_planner_regret.py
from pathlib import Path
import pandas as pd
import pytest
from vecdb.bench.regret import compute_regret, regret_summary

SWEEP_PATH = Path("results/sweep_uncorrelated.csv")

@pytest.mark.skipif(not SWEEP_PATH.exists(), reason="requires scripts/run_full_sweep.py to have been run")
def test_mean_regret_is_bounded_relative_to_oracle():
    df = pd.read_csv(SWEEP_PATH)
    wide = (
        df[df["strategy"].isin(["pre_filter", "post_filter", "predicate_aware", "planner"])]
        .groupby(["query_idx", "target_selectivity", "strategy"])["latency_ms"]
        .mean()
        .unstack("strategy")
        .rename(columns={
            "pre_filter": "pre_filter_latency_ms",
            "post_filter": "post_filter_latency_ms",
            "predicate_aware": "predicate_aware_latency_ms",
            "planner": "planner_latency_ms",
        })
        .dropna()
    )
    regret = compute_regret(wide, ["pre_filter", "post_filter", "predicate_aware"], "planner_latency_ms")
    summary = regret_summary(regret)
    oracle_mean = wide[["pre_filter_latency_ms", "post_filter_latency_ms", "predicate_aware_latency_ms"]].min(axis=1).mean()
    # spec §5 target: mean regret < 15% of the oracle's mean latency
    assert summary["mean_regret_ms"] < 0.15 * oracle_mean, (
        f"mean regret {summary['mean_regret_ms']:.3f}ms exceeds 15% of oracle mean {oracle_mean:.3f}ms — "
        "this is a real finding, report it in the milestone/README rather than loosening the threshold"
    )
```

- [ ] **Step 2: Run it**

Run: `".venv/Scripts/python" -m pytest tests/test_planner_regret.py -v`
Expected: PASS if Task 38's regret gate held; if it fails, this is the same honest-negative-result situation as elsewhere — report the actual mean regret and oracle mean in the README limitations section rather than adjusting the 15% threshold to make it pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_planner_regret.py
git commit -m "test: planner mean regret is bounded relative to the hindsight oracle"
```

---

### Task 43: Final README assembly

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: every milestone report in `docs/superpowers/milestones/`, every figure in `results/figures/`, `results/sweep_*.csv`, `results/calibration.json`.

- [ ] **Step 1: Replace each placeholder section of `README.md` with real content**

Follow source plan §7 Day 7's structure exactly — this ordering matters because most readers stop after section 3:

1. **Above the fold**: the one-sentence delta (already in the skeleton) immediately followed by `results/figures/crossover.png` embedded as an image.
2. **Results table**: recall@10, p95 latency, dist_ops for all three strategies at three representative selectivities (e.g. 0.001, 0.05, 0.5) — pull the real numbers from `results/sweep_uncorrelated.csv`, do not estimate them.
3. Restate the one-sentence delta as its own short paragraph.
4. **Architecture**: the layer diagram from spec §4, plus 3-4 sentences per major design decision (why hand-written HNSW, why three strategies, why a calibrated cost model) each linking to the milestone report that has the real numbers.
5. **How to run**: `pip install -e .`, `python scripts/download_data.py`, `python scripts/build_100k.py`, `python scripts/calibrate.py`, `python scripts/run_full_sweep.py`, `python scripts/generate_figures.py`, `uvicorn vecdb.service.app:app`. This must actually work from a clean clone — verified in Task 44.
6. **What I got wrong / limitations** — do not skip this. At minimum, port forward from spec §8's risk register plus whatever your own milestone reports actually found (the real FAISS latency gap number, the real HNSW-fallback-or-not outcome, the real Strategy C win/loss outcome, the real regret number, plus: metadata is synthetic, no deletes, no durability, single-threaded, the AND independence assumption's measured error).
7. **What I'd do next**: deletes (tombstone + rebuild-at-20%-tombstoned), a WAL, DiskANN/Vamana at larger scale, multi-threaded search.
8. **References**: already in the README skeleton.

- [ ] **Step 2: Verify the README's embedded images render**

Run: `".venv/Scripts/python" -c "from pathlib import Path; assert Path('results/figures/crossover.png').exists(); print('ok')"`

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: final README — results, architecture, how-to-run, honest limitations"
```

---

### Task 44: Clean-clone verification and `v1.0` tag

**Files:** none (verification + tagging only)

- [ ] **Step 1: Clone into a scratch directory and verify the documented flow works end to end**

```bash
cd /tmp || cd C:/Users/ASUS/AppData/Local/Temp
rm -rf vecdb-clean-clone
git clone D:/vecdb vecdb-clean-clone
cd vecdb-clean-clone
python -m venv .venv
".venv/Scripts/python" -m pip install --upgrade pip
".venv/Scripts/python" -m pip install -e ".[dev]"
".venv/Scripts/python" -m pytest
```

Expected: install succeeds, and every test that doesn't require the full downloaded dataset/persisted 100K index/`results/calibration.json` passes. Tests that `@pytest.mark.skipif` on missing large artifacts (Task 42) are expected to skip in a truly clean clone that hasn't run the data/build scripts yet — that's correct behaviour, not a failure.

- [ ] **Step 2: Confirm the spec §9 Definition of Done checklist item-by-item against the real repo** (not from memory — open each file):

- [ ] `results/figures/crossover.png` exists and is referenced at the top of `README.md`
- [ ] The results table in `README.md` has recall@10, p95 latency, and dist_ops for all three strategies at three selectivities
- [ ] `pip install -e . && python scripts/run_full_sweep.py` (after the data/build scripts) runs end to end
- [ ] `pytest` is green (or skips only the artifact-gated test) on a clean clone
- [ ] The limitations section names at least four real limitations
- [ ] The one-sentence delta is stated in the first paragraph
- [ ] HNSW recall is within 0.02 of FAISS's at matched parameters, or Phase 3's milestone report documents the fallback honestly

- [ ] **Step 3: Tag `v1.0` in the real repo (not the scratch clone) and push**

```bash
cd D:/vecdb
git tag -a v1.0 -m "v1.0: filtered vector DB with cost-based planner and measured crossover"
git push origin v1.0
```

---

### Task 45: Final milestone report

**Files:**
- Create: `docs/superpowers/milestones/08-service-tests-readme.md`

- [ ] **Step 1: Write the report**

Create `docs/superpowers/milestones/08-service-tests-readme.md` with: **What got built** (FastAPI service with staged-insert overlay, `test_strategies_agree`, `test_planner_regret`, final README, clean-clone verification, `v1.0` tag). **Numbers**: full `pytest` pass/skip counts on the clean clone from Task 44. **Gate status**: the spec §9 Definition of Done checklist, copied in with each item's actual pass/fail state. **Interview note**: a 90-second spoken summary — literally write out what you'd say out loud, per source plan §7's instruction to rehearse it — covering the one-sentence delta, the crossover chart, and your single strongest and single most honest-limitation numbers.

- [ ] **Step 2: Commit and push**

```bash
git add docs/superpowers/milestones/08-service-tests-readme.md
git commit -m "docs: final milestone report, project complete"
git push origin main
```

---

## Post-plan note

Every phase's milestone report accumulates in `docs/superpowers/milestones/` — by Phase 8 that folder itself is a second, more granular narrative of the whole build, useful on its own as interview prep material alongside the README.
