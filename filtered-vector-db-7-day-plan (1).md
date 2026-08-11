# Filtered Vector Database — 7-Day Implementation Plan

**Project codename:** `vecdb` (repo: `filtered-vecdb`)
**Language:** Python 3.11 + NumPy (+ optional Numba)
**Duration:** 7 days, ~7 hours/day (≈50 hours)
**Status:** Plan v1.0

---

## 0. The One-Sentence Delta

> **A cost-based query planner for approximate nearest-neighbour search that chooses between pre-filtering, post-filtering, and predicate-aware graph traversal using an estimated predicate selectivity — with a measured crossover curve showing exactly where each strategy wins and why.**

If you can say that sentence in an interview and then defend every noun in it with a number you measured, the project has done its job. Everything below exists to make that sentence true.

**What this is NOT:** it is not "I implemented HNSW." Thousands of people have implemented HNSW. HNSW is the *substrate*. The project is the planner and the crossover chart on top of it.

---

## 1. Objective

### 1.1 Primary objective

Build a working vector database that answers this query correctly and fast:

```
SELECT id FROM vectors
WHERE category = 'hull' AND year > 2019
ORDER BY l2_distance(embedding, :q)
LIMIT 10
```

...and, crucially, that **decides at runtime how to execute it** rather than hardcoding one strategy.

### 1.2 Secondary objectives (in priority order)

| # | Objective | Success criterion |
|---|---|---|
| 1 | Correct exact baseline | Flat index recall@10 = 1.000 by construction; used as ground truth |
| 2 | Hand-written HNSW | Unfiltered recall@10 ≥ 0.95 vs flat at some `efSearch` |
| 3 | Three filter strategies implemented | All three return valid results at every selectivity |
| 4 | The crossover measured | A plot where the winning strategy changes at least twice across selectivity |
| 5 | Planner beats every fixed strategy | Planner mean latency < min(mean latency of any single fixed strategy) across the sweep |
| 6 | Honest FAISS comparison | Recall parity within 0.02; latency gap explained, not hidden |
| 7 | Serviceable | FastAPI endpoints, index persists to disk and reloads |

### 1.3 Explicit non-objectives

State these in the README. Scoping honestly is a signal, not a weakness.

- **Not beating FAISS on wall-clock latency.** FAISS is C++ with SIMD. You are NumPy. You will lose by 2–5x on identical algorithms. This is expected and you will explain it.
- **Not distributed.** Single node, single process.
- **Not durable in the ACID sense.** Snapshot persistence only; no WAL, no crash recovery.
- **No deletes in v1.** Tombstone design is sketched but not built.
- **Not billion-scale.** 100K vectors in the main benchmark, 1M as an optional overnight run.

---

## 2. The Problem It Solves

### 2.1 Formal statement

Given:
- A set of vectors $V = \{v_1, \dots, v_N\}$, $v_i \in \mathbb{R}^d$
- Metadata attributes $A(v_i)$ for each vector
- A query vector $q \in \mathbb{R}^d$, integer $k$, and a boolean predicate $P$ over attributes

Return the $k$ vectors minimising $\|q - v\|_2$ **subject to** $P(A(v)) = \text{true}$.

Define **selectivity** $s = \frac{|\{v : P(A(v))\}|}{N}$ — the fraction of the dataset that passes the filter. Low $s$ = highly selective filter (few survivors). $s = 1$ = no filter.

### 2.2 Why this is genuinely hard

An HNSW index is a proximity graph built over the **entire** dataset. Its navigability — the property that greedy descent reaches the true nearest neighbour — is a property of the *whole* graph. The moment you constrain results to a subset, that property no longer holds for the subgraph you care about, and you are in trouble. There are three obvious things to do and all three break somewhere:

#### Strategy A — Pre-filter (filter, then brute force)

Materialise the set $S = \{v : P(v)\}$, then linear-scan it.

- **Correctness:** exact. Recall = 1.0 always.
- **Cost:** $O(N \cdot s \cdot d)$ distance computations.
- **Breaks when:** $s$ is large. At $s = 0.5$ on 100K vectors you are scanning 50,000 vectors per query. That's slower than an unfiltered ANN search by an order of magnitude, and you've thrown away the index entirely.

#### Strategy B — Post-filter (search, then discard)

Run normal HNSW with some `efSearch`, take the top $\text{ef}$ results, discard non-matching, return the top $k$ survivors.

- **Cost:** normal HNSW cost, if `ef` is normal.
- **Breaks when:** $s$ is small. Expected survivors from a top-$\text{ef}$ list is $\text{ef} \cdot s$. To expect $k$ survivors you need $\text{ef} \gtrsim k/s$. At $k=10$, $s=0.001$, that's $\text{ef} \geq 10{,}000$ — at which point HNSW's search cost has exploded past a brute-force scan of the entire dataset. Worse, it's *probabilistic*: sometimes you return fewer than $k$ results, or none. **Returning 3 results when the user asked for 10 is a correctness bug, not a performance issue.** This is the failure mode most student projects ship without noticing.

#### Strategy C — Predicate-aware traversal

Run the greedy graph search, but only admit matching nodes into the result heap while still *traversing through* non-matching ones.

- **Why traverse through non-matching nodes?** Because the subgraph induced by the filter is often **disconnected**. If you refuse to visit non-matching nodes, greedy search gets stranded in a region of the graph with no survivors and terminates with garbage. Traversing through preserves connectivity at the cost of wasted distance computations.
- **Breaks when:** $s$ is very small. Almost every node you touch is a non-match; you burn the entire distance budget on nodes you'll never return. Also, if matching nodes are clustered far from your entry point, you may never reach them.

#### The actual insight

**No strategy dominates.** There is a crossover region — empirically somewhere around $s \in [0.5\%, 30\%]$ — where the right answer changes. The project is to (a) implement all three properly, (b) measure the crossover, (c) build a cost model that *predicts* it, and (d) show the planner routing correctly.

### 2.3 Why this is the right project for you specifically

Both of your other projects need this component:

- **GD trainer** — you already embed utterances with `all-MiniLM-L6-v2` for topic relevance. Filter by session, by speaker, by time window. That is a filtered vector query.
- **NK/BV compliance checker** — retrieval over rule documents filtered by *class society* and *rule version*. That is **literally** the query in §1.1. Searching BV rules and getting an NK rule back is a correctness failure, not a ranking nuisance.

Building this turns three unrelated apps into one narrative: *"I built the retrieval layer, then built two systems on top of it."* That is a materially stronger portfolio shape.

---

## 3. Architecture

### 3.1 Layer diagram

```
┌──────────────────────────────────────────────────────────────┐
│  SERVICE LAYER            FastAPI  (app.py, schemas.py)      │
│  POST /insert  POST /search  GET /stats  POST /persist       │
└───────────────────────────┬──────────────────────────────────┘
                            │  SearchRequest{q, k, filter, ef?}
┌───────────────────────────▼──────────────────────────────────┐
│  QUERY PLANNER            planner/                           │
│   1. compile predicate  → bitmask + estimated selectivity ŝ  │
│   2. cost model         → C_pre(ŝ), C_post(ŝ), C_pred(ŝ)     │
│   3. choose argmin, emit ExecutionPlan                       │
│   4. record actual cost → feed calibration log               │
└───────┬──────────────────┬──────────────────┬────────────────┘
        │                  │                  │
┌───────▼──────┐  ┌────────▼───────┐  ┌───────▼──────────────┐
│ PRE-FILTER   │  │ POST-FILTER    │  │ PREDICATE-AWARE      │
│ exact scan   │  │ HNSW + discard │  │ filtered HNSW walk   │
│ over mask    │  │ adaptive ef    │  │ 2-hop expansion      │
└───────┬──────┘  └────────┬───────┘  └───────┬──────────────┘
        │                  │                  │
┌───────▼──────────────────▼──────────────────▼────────────────┐
│  INDEX LAYER              index/                             │
│   FlatIndex (exact, ground truth)                            │
│   HNSWIndex (hand-written: layers, M, efC, heuristic prune)  │
│   shared: DistanceCounter instrumentation                    │
└───────┬──────────────────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────┐
│  PREDICATE LAYER          predicate/                         │
│   DSL parse → AST → NumPy bool mask                          │
│   Selectivity estimator (histograms + value counts)          │
└───────┬──────────────────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────┐
│  STORAGE LAYER            store/                             │
│   VectorStore   float32 (N,d) np.memmap + norms cache        │
│   MetaStore     columnar dict[str, np.ndarray] + stats       │
│   IdMap         external id ↔ internal row index             │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Component contracts

#### `store/vectors.py` — VectorStore

```python
class VectorStore:
    """Row-major float32 vectors, memmapped so 1M x 128 doesn't need to fit in RAM."""
    data: np.memmap          # (N, d) float32
    sq_norms: np.ndarray     # (N,) float32, precomputed ||v||^2 for fast L2
    n_distance_ops: int      # instrumentation counter

    def distances(self, q: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Batched L2^2 from q to self.data[rows]. THE hot path.
        Uses ||q-v||^2 = ||q||^2 - 2 q.v + ||v||^2 so it's one gemv, not a loop.
        Increments n_distance_ops by len(rows)."""
```

> **Performance note that will save your week:** never compute distances one at a time in a Python loop. In `search_layer`, gather *all* unvisited neighbours of the current node into an array and compute their distances in a single vectorised call. This is worth 20–50x and is the difference between a 20-minute build and a 12-hour build.

#### `store/metadata.py` — MetaStore

```python
class MetaStore:
    """Columnar attributes. One np.ndarray per column, aligned to vector row index."""
    columns: dict[str, np.ndarray]     # 'category' -> int32 codes, 'year' -> int16, ...
    categoricals: dict[str, dict]      # column -> {label: code}
    stats: dict[str, ColumnStats]      # value counts (categorical) / 64-bin histogram (numeric)
```

#### `predicate/` — DSL, compiler, estimator

Keep the DSL deliberately small. It is not the interesting part; do not gold-plate it.

```python
# Wire format (JSON, from the API):
{"op": "and", "clauses": [
    {"col": "category", "op": "eq", "val": "hull"},
    {"col": "year",     "op": "gt", "val": 2019}
]}

# Supported: eq, ne, lt, lte, gt, gte, in, and, or, not
```

```python
def compile(pred: Predicate, meta: MetaStore) -> np.ndarray:  # (N,) bool mask
def estimate_selectivity(pred: Predicate, meta: MetaStore) -> float:
    """From stats only — must NOT touch the mask. That's the whole point:
    the planner has to decide BEFORE materialising anything expensive.
    - eq on categorical  -> value_count / N
    - range on numeric   -> histogram bin interpolation
    - AND                -> product of children (independence assumption — a lie,
                            and you will measure how big a lie)
    - OR                 -> s1 + s2 - s1*s2
    - NOT                -> 1 - s
    """
```

> **Interview gold:** the independence assumption for AND is exactly the assumption Postgres makes, and exactly the reason Postgres has extended statistics. Measure your estimation error (predicted ŝ vs true $s$) on correlated columns and put that scatter plot in the README. That single plot moves you from "student built ANN" to "student understands query optimisation."

#### `index/base.py` — the common interface

```python
class Index(ABC):
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, q: np.ndarray, k: int,
               mask: np.ndarray | None = None,
               params: dict | None = None) -> SearchResult: ...

@dataclass
class SearchResult:
    ids: np.ndarray
    distances: np.ndarray
    n_distance_ops: int      # for hardware-independent comparison
    strategy: str
    latency_ms: float
    n_returned: int          # < k means an under-fill — track this!
```

`n_distance_ops` is the metric you lead with when comparing against FAISS. It is implementation-independent, language-independent, and hardware-independent. "FAISS is 3x faster in wall-clock but performs 1.15x more distance computations" is a far more sophisticated claim than a raw timing table.

#### `planner/cost_model.py`

```
C_pre(ŝ)  = c_scan · N · ŝ
C_post(ŝ) = c_hop · M · ef_required(ŝ),   ef_required = clamp(α·k/ŝ, ef_min, N)
C_pred(ŝ) = c_hop · M · ef_base · γ(ŝ)

where γ(ŝ) is the traversal-inflation factor: how many extra nodes you visit
because most neighbours fail the predicate. Fit empirically on Day 5;
expect roughly γ ≈ 1 + β·(1/ŝ - 1) capped at some ceiling.
```

`c_scan`, `c_hop`, `α`, `β` are **calibrated from your own Day 4–5 measurements**, written to `results/calibration.json`, and loaded at planner init. Calibration-from-measurement is what makes it a cost model rather than three magic numbers.

Planner quality metric: **regret** = (chosen strategy's latency) − (best strategy's latency in hindsight), averaged over the query set. Report mean and p95 regret. Aim for mean regret < 15% of oracle.

### 3.3 Repository layout

```
filtered-vecdb/
├── README.md                  ← the deliverable everyone actually reads
├── pyproject.toml
├── vecdb/
│   ├── io/
│   │   ├── fvecs.py           # SIFT .fvecs/.ivecs readers
│   │   └── dataset.py         # download, cache, subset, synth metadata
│   ├── store/
│   │   ├── vectors.py
│   │   ├── metadata.py
│   │   └── idmap.py
│   ├── predicate/
│   │   ├── dsl.py
│   │   ├── compile.py
│   │   └── selectivity.py
│   ├── index/
│   │   ├── base.py
│   │   ├── flat.py
│   │   ├── hnsw.py            ★ hand-written, ~400 lines
│   │   └── strategies.py      ★ hand-written, ~250 lines
│   ├── planner/
│   │   ├── cost_model.py      ★ hand-written
│   │   └── planner.py
│   ├── service/
│   │   ├── app.py
│   │   └── schemas.py
│   └── bench/
│       ├── groundtruth.py
│       ├── harness.py
│       ├── sweep.py
│       └── plots.py
├── tests/
│   ├── test_predicate.py
│   ├── test_hnsw_correctness.py
│   ├── test_strategies_agree.py
│   └── test_planner_regret.py
├── scripts/
│   ├── download_data.sh
│   ├── build_index.py
│   ├── run_sweep.py
│   └── calibrate.py
└── results/
    ├── calibration.json
    ├── sweep_uncorrelated.csv
    ├── sweep_correlated.csv
    └── figures/*.png
```

★ = **hand-write these**. Everything else, vibe-code freely — I/O, plotting, FastAPI boilerplate, dataset download, Pydantic schemas. That's your usual split and it's the right one here: the starred files are the ~25% you will be asked to defend line by line.

---

## 4. Data & Experimental Design

### 4.1 Dataset

**SIFT** from the TEXMEX corpus (`irisa.fr/texmex/columns/ann`). Two sizes:

| Set | N | d | queries | Use |
|---|---|---|---|---|
| `siftsmall` | 10,000 | 128 | 100 | Days 1–2 development. Fast iteration. |
| `sift1m` (100K subset) | 100,000 | 128 | 1,000 | Days 3–7 headline benchmark. |
| `sift1m` (full) | 1,000,000 | 128 | 10,000 | Optional overnight run for the README number. |

Ships with exact ground truth for the *unfiltered* case, which is a free correctness check on Day 1.

**Why not embeddings from your own GD app?** Because SIFT is a standard benchmark and FAISS numbers on it are published, so your numbers are comparable to something. Add a 10K-vector `all-MiniLM-L6-v2` run at the end as a "does it work on real embeddings" appendix — good, but not the main table.

### 4.2 Synthetic metadata — and why you generate two versions

SIFT has no attributes, so you generate them. Generate **two** metadata sets over the *same* vectors:

**(a) Uncorrelated.** `category` uniform over 100 values, `year` uniform over 2000–2025, `score` ~ U(0,1). The filter is a random subsample of the graph. This is the easy, well-studied case.

**(b) Correlated.** Run k-means with 100 clusters over the vectors; assign `category` = cluster ID with 85% probability, random otherwise. Now the filter selects a *spatially contiguous region* of the vector space.

Running both is cheap and it is the single highest-value-per-hour addition in this plan, because the two cases behave **oppositely**:

- Uncorrelated: predicate-aware traversal works well — matching nodes are scattered evenly, so you hit them constantly.
- Correlated: predicate-aware traversal can be *catastrophic* — greedy descent enters a region of the graph containing zero matches and gets stranded. Meanwhile pre-filtering gets *better*, because survivors are clustered and mutually near.

"My predicate-aware strategy wins by 4x on uncorrelated attributes and *loses by 2x* on correlated ones, and here's the mechanism" is a far better interview answer than any single clean win.

### 4.3 Selectivity grid

Sweep $s \in \{0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0\}$ — log-ish spacing because the interesting action is at the low end. Construct predicates that hit each target selectivity exactly (choose `score < t` with $t$ = the empirical quantile).

Per cell: 200 queries, discard first 20 as warmup, report p50/p95.

### 4.4 Metrics

| Metric | Definition | Why |
|---|---|---|
| `recall@k` | $\|\text{returned} \cap \text{true}\| / k$ | The core quality number |
| `underfill_rate` | fraction of queries returning $< k$ results | The post-filter correctness bug |
| `latency_p50/p95` | ms per query, single-threaded | What users feel |
| `dist_ops` | distance computations per query | Hardware-independent comparison |
| `qps` | throughput, batch mode | Ops-facing number |
| `build_time_s` | index construction wall clock | The cost side of the tradeoff |
| `index_bytes` | resident index size excl. raw vectors | Memory tradeoff |
| `sel_error` | $\|\hat{s} - s\| / s$ | Planner input quality |
| `regret_ms` | chosen − oracle-best latency | Planner output quality |

**Ground truth for filtered queries must be computed by exact brute force over the mask.** There is no shortcut and no published ground truth for your synthetic predicates. Cache it — computing it for the full grid takes minutes and you'll rerun the sweep a dozen times.

---

## 5. Day-by-Day Plan

Each day has: goal, tasks, a **Done when** gate, and what to vibe-code vs hand-write. If you miss a Done-when gate, take the cut listed in §6 rather than sliding the schedule — the Day 6 chart is the deliverable and it cannot be the thing that gets dropped.

---

### Day 0 — Setup *(evening before, ~2.5h)*

**Goal:** never touch environment plumbing again after tonight.

- `uv init` / venv. Deps: `numpy`, `faiss-cpu`, `fastapi`, `uvicorn`, `pydantic`, `matplotlib`, `pandas`, `pytest`, `tqdm`, `scikit-learn` (k-means only).
- Download `siftsmall` + `sift1m`. Write `.fvecs`/`.ivecs` readers, verify shapes.
- Write the metadata generator (both correlated and uncorrelated variants), persist as `.npz`.
- `git init`, first commit, push. Write the README skeleton with empty results sections **now** — it forces you to know what the outputs are.

**Done when:** `python -c "from vecdb.io.dataset import load; X,Q,GT,meta = load('siftsmall'); print(X.shape, Q.shape, GT.shape)"` prints `(10000,128) (100,128) (100,100)`.

**Vibe-code:** all of it.

---

### Day 1 — Exact baseline + benchmark harness

**Goal:** you cannot measure improvement without a ruler. Build the ruler first.

1. `VectorStore` with memmap, cached squared norms, batched `distances()`, `n_distance_ops` counter.
2. `MetaStore` + column stats (value counts, 64-bin histograms).
3. Predicate DSL → AST → boolean mask. Selectivity estimator from stats.
4. `FlatIndex.search(q, k, mask)` — exact filtered search. **This is your ground-truth oracle.**
5. Benchmark harness: takes an index + query set + predicate set, returns the metrics table in §4.4.
6. FAISS baselines wired in: `IndexFlatL2` (sanity) and `IndexHNSWFlat` (the real comparison).
7. Verify against the shipped SIFT ground truth: your `FlatIndex` unfiltered recall@100 must be exactly 1.000.

**Done when:** `pytest tests/test_predicate.py` green, and you can print a metrics table for Flat and FAISS-Flat on siftsmall with recall = 1.0 for both.

**Hand-write:** the recall computation (get the tie-handling right — with duplicate distances, naive set-intersection recall can exceed 1.0), the mask semantics, the selectivity estimator.
**Vibe-code:** table formatting, plotting utilities, FAISS wrappers, memmap plumbing.

---

### Day 2 — HNSW construction

**Goal:** the substrate. This is the longest hand-written day. Budget the full 7 hours and don't get clever.

Read Malkov & Yashunin (2016), *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs* — Algorithms 1–5. Implement them directly.

**Structures:**
```python
class HNSWIndex(Index):
    M: int                          # max neighbours per node, layers > 0    (default 16)
    M0: int                         # max neighbours at layer 0              (default 2*M)
    ef_construction: int            # build-time beam width                  (default 200)
    mL: float                       # = 1 / ln(M)
    entry_point: int
    max_level: int
    levels: np.ndarray              # (N,) int8
    graph: list[list[list[int]]]    # graph[layer][node] -> neighbour ids
```

**Algorithms to implement, in order:**

1. **Level assignment** — `l = floor(-log(uniform(0,1)) * mL)`. Geometric decay; expected fraction of nodes at level ≥ ℓ is $M^{-\ell}$. Sanity check: with M=16 and N=100K you should see roughly 100K / 6.25K / 390 / 25 / 1 nodes per level.

2. **`search_layer(q, entry_points, ef, layer, mask=None)`** — the core routine. Three collections: a min-heap of *candidates* by distance, a max-heap of *results* (so you can cheaply evict the worst), and a *visited* set.
   ```
   while candidates non-empty:
       c = pop nearest candidate
       if dist(c) > worst in results and len(results) == ef: break   # the stopping rule
       neighbours = graph[layer][c] not in visited
       mark all visited
       D = store.distances(q, neighbours)          ← ONE batched call, not a loop
       for each (n, d): if d < worst-in-results or len(results) < ef:
           push to candidates and results; evict worst if over ef
   ```
   The early-break is what makes HNSW fast. Get it wrong and you'll silently degrade into a full traversal — one of the two most common bugs.

3. **`select_neighbours_heuristic(q, candidates, M)`** — Algorithm 4. **Do not skip this and just keep the M closest.** Iterate candidates in ascending distance from `q`; keep candidate `c` only if `d(c, q) < d(c, r)` for every already-kept `r`. This enforces a relative-neighbourhood-graph property: it keeps *diverse* neighbours in different directions rather than M clustered ones. It is the single biggest quality lever in HNSW and the second-most-common thing students get wrong. Expect a 0.05–0.15 recall improvement over naive top-M.

4. **`insert(v)`** — assign level ℓ; greedy-descend from entry point to layer ℓ+1 with ef=1; from ℓ down to 0 run `search_layer` with `ef_construction`, select M neighbours by heuristic, link bidirectionally, and **re-prune** any neighbour that now exceeds its degree cap (using the same heuristic). Update entry point if ℓ > max_level.

5. Persistence: `save()`/`load()` via `np.savez` + pickled adjacency.

**Done when:** build on siftsmall (10K) completes in under 2 minutes and unfiltered recall@10 vs `FlatIndex` is ≥ 0.95 at `efSearch=100`.

**Hand-write:** every line of `hnsw.py`. This is the part you defend in interviews.

> **Debugging checklist if recall is bad:**
> - Recall stuck near 0.3–0.6 → you skipped the neighbour heuristic, or you're not re-pruning neighbours after bidirectional linking.
> - Recall high but search is slow → the early-break condition in `search_layer` is wrong.
> - Recall ~1.0 but build is glacial → you're not using the layer hierarchy; check level assignment actually produces multiple layers.
> - Nondeterministic recall across runs → seed your RNG; level assignment is random.

---

### Day 3 — Search tuning + scale to 100K

**Goal:** turn a working HNSW into a *characterised* HNSW, and get it to benchmark scale.

1. **`efSearch` sweep:** ef ∈ {10, 20, 40, 80, 160, 320}. Plot recall@10 vs latency → **Pareto curve**. Overlay FAISS `IndexHNSWFlat` with the same M and efConstruction on the same axes. This is your first real figure.
2. **Build-parameter sweep:** M ∈ {8, 16, 32} × efConstruction ∈ {100, 200, 400} on siftsmall. Table of build time / index memory / recall@10 at fixed ef. Pick your operating point and *justify it in one sentence*.
3. **Optimise the hot path.** Profile with `cProfile`. Expected wins: batch neighbour distances (if you haven't already), replace `set` visited with a `np.uint8` array + a generation counter (avoids reallocating per query), store adjacency in flat `np.int32` arrays rather than lists of lists.
4. **Scale to 100K.** Build, time it, persist it. Expect 15–40 minutes with vectorised distances.
5. *(Optional)* `@numba.njit` on `search_layer`'s inner loop. Only if you're ahead — it's a 5–10x win but a real time sink to debug.

**Done when:** you have `figures/pareto_unfiltered.png` with your curve and FAISS's, and a persisted 100K index on disk.

**Reality check on the gap:** you'll likely sit at 2–5x FAISS's latency at equal recall. That is the correct, expected result. Note the `dist_ops` comparison alongside — if your distance-op count is within ~20% of FAISS's, your *algorithm* is right and the gap is purely SIMD and memory layout. Say exactly that.

---

### Day 4 — Pre-filter and post-filter + selectivity estimation

**Goal:** two of three strategies, plus the planner's input signal.

1. **Strategy A — `PreFilterStrategy`:** `mask.nonzero()` → gather rows → one batched distance call → `argpartition` for top-k. Exact. ~30 lines. Instrument `dist_ops = |S|`.
2. **Strategy B — `PostFilterStrategy`:** run HNSW with `ef = clamp(α · k / ŝ, ef_min, N)`, filter results, take top-k.
   - Track `underfill_rate` explicitly. Do not silently return short lists.
   - Add the honest fallback: if underfilled, retry once with 4x ef, and if still short, fall back to pre-filter. **Log every fallback** — the fallback rate vs selectivity is itself a good plot.
3. **Selectivity estimator validation:** for 500 random predicates (single-clause, AND of 2, AND of 3, OR), scatter-plot $\hat{s}$ vs true $s$ on log-log axes for both the uncorrelated and correlated metadata. The independence assumption will hold on (a) and fail visibly on (b). **This is a headline figure, not a footnote.**
4. Run the full selectivity grid for A and B. Persist to `results/sweep_*.csv`.

**Done when:** `sweep_uncorrelated.csv` has complete rows for pre and post across all 8 selectivities, and you can state the ŝ-error at p95 for correlated ANDs.

**Hand-write:** both strategies, the adaptive-ef formula, the estimator.
**Vibe-code:** the sweep driver, CSV writing, plots.

---

### Day 5 — Predicate-aware traversal

**Goal:** the strategy that makes this a project rather than an exercise. Also the hardest day conceptually.

`FilteredHNSWStrategy` — a modified `search_layer` with these changes:

1. **Two-tier admission.** A node is *visitable* if it's in the graph; a node is *admissible to results* only if `mask[node]`. Traverse through everything, collect only matches. This alone fixes most of the connectivity problem.

2. **Seeded entry points.** Instead of always starting from the global entry point, seed the layer-0 search with $r$ (say 8) randomly sampled *matching* nodes alongside the hierarchical entry point. Cheap insurance against starting stranded in a match-free region. Sample from `mask.nonzero()` — you already have the mask.

3. **Two-hop expansion.** Track the match rate among the neighbours you've expanded. If it drops below a threshold (say 0.1·ŝ... tune it), expand *neighbours-of-neighbours* for the current node instead of just neighbours. This is the ACORN-style trick: it effectively densifies the induced subgraph on the fly, at the cost of a wider fan-out. Guard it with a budget cap so a pathological query can't blow up.

4. **Dynamic ef.** Widen the beam as selectivity falls: `ef_eff = ef_base · min(4, 1/max(ŝ, 0.05))`. The beam is holding matches, and matches are scarce, so it needs to be wider.

5. **Budget cap and honest termination.** Hard-cap `dist_ops` per query (e.g. $0.3N$). If you hit the cap without $k$ results, **bail out to pre-filter**. Track the bail rate. A strategy that knows when it's losing and hands off is exactly what a query executor should do, and saying so is a strong interview moment.

Then: run the full grid for Strategy C on both metadata variants.

**Done when:** on uncorrelated metadata, Strategy C's latency is below both A and B for at least two selectivity values in the middle of the range, at recall@10 ≥ 0.90.

**If C never wins anywhere:** don't panic and don't fake it. Common causes: `ef_eff` too small (matches evicted from the beam by nearer non-matches — check your admission logic isn't accidentally letting non-matches into the result heap), or the graph is too sparse (M too small). If after honest debugging it still loses everywhere, **that is a publishable finding** — write up "predicate-aware traversal did not beat the pre-filter/post-filter envelope at N=100K on this workload, and here is the distance-op accounting showing why." An honest negative result defended with numbers beats a fabricated win, and interviewers can smell the difference.

---

### Day 6 — Query planner + the full sweep + the chart

**Goal:** the deliverable. Protect this day.

1. **Calibrate** (`scripts/calibrate.py`): fit `c_scan`, `c_hop`, `α`, `β` by least squares against Day 4–5 measurements. Write `results/calibration.json`.
2. **Cost model + planner:** compute all three costs from ŝ, pick the argmin, emit an `ExecutionPlan` with a human-readable reason string (`"pre-filter: ŝ=0.004 → 400 scans < 2100 est. hops"`). Return the reason in the API response — an explainable planner is much more compelling in a demo than a black box.
3. **Regret measurement:** run every query under all three strategies plus the planner. Compute per-query regret vs the hindsight-best. Report mean and p95.
4. **THE FULL SWEEP:** 8 selectivities × 2 metadata variants × 4 executors (3 fixed + planner) × 200 queries. Should take 30–60 minutes; start it and write the README while it runs.
5. **The figures** (this is what people actually look at):
   - **`crossover.png`** — x: selectivity (log), y: p95 latency (log), one line per strategy + planner as a dashed line, shaded regions showing which strategy the planner chose. *This is your headline image. Put it at the top of the README.*
   - `crossover_correlated.png` — same axes, correlated metadata. The shape should differ visibly.
   - `recall_vs_selectivity.png` — shows pre-filter flat at 1.0, the others degrading.
   - `underfill.png` — post-filter's underfill rate climbing as selectivity drops. The correctness bug, visualised.
   - `dist_ops.png` — the hardware-independent version of the crossover.
   - `selectivity_estimation.png` — from Day 4.
   - `pareto_unfiltered.png` — from Day 3.
6. **FastAPI service:** `/insert`, `/search` (returns results + chosen strategy + reason + timing breakdown), `/stats`, `/persist`. Half an hour, vibe-coded.

**Done when:** `crossover.png` exists, shows the winning strategy changing at least twice, and the planner line tracks the lower envelope.

**Expected shape** (so you can sanity-check — don't tune toward these, just be suspicious if you're wildly off):
- $s < \sim1\%$: pre-filter wins. Scanning 1,000 of 100,000 vectors is just cheap.
- $\sim1\% < s < \sim30\%$: predicate-aware wins on uncorrelated data.
- $s > \sim30\%$: post-filter (≈ plain HNSW) wins; the filter barely constrains anything.
- Correlated data: the middle region shrinks or vanishes; pre-filter's region extends further right.

---

### Day 7 — Writeup, tests, interview prep

**Goal:** make six days of work legible in ninety seconds.

1. **README structure** (this ordering matters — most readers stop after §3):
   1. One-paragraph problem statement + `crossover.png` **above the fold**
   2. The results table: recall@10, p95 latency, dist_ops per strategy at three representative selectivities
   3. The one-sentence delta
   4. Architecture diagram + design decisions with justifications
   5. How to run (must actually work from a clean clone — test this)
   6. **What I got wrong / limitations** ← do not skip
   7. What I'd do next
   8. References (Malkov & Yashunin; Gorilla if you mention it; ACORN / Filtered-DiskANN for the traversal ideas)
2. **Tests:** `test_strategies_agree.py` (all three return the same top-k as FlatIndex on tiny data with ef large enough), `test_hnsw_correctness.py` (recall floor), `test_predicate.py` (mask semantics, including empty and full masks), `test_planner_regret.py` (regret below threshold on cached results).
3. **The limitations section**, written honestly:
   - 2–5x slower than FAISS at equal recall; SIMD and memory layout, not algorithm — and here's the dist_ops table proving the algorithm is right.
   - Metadata is synthetic. Real workloads have correlations I only approximated with the k-means variant.
   - No deletes. Design sketch: tombstone bitmap + exclude at admission + periodic rebuild when tombstone fraction > 20%.
   - No durability. Snapshot only; a WAL is the obvious next step.
   - Single-threaded. Search parallelises trivially across queries; build doesn't, easily.
   - Independence assumption in the AND estimator; measured error is in `selectivity_estimation.png`.
4. **Interview prep:** write out answers to §7 below and say them aloud once. Genuinely — aloud. The gap between "I know this" and "I can say this in 45 seconds under pressure" is large and only closes by rehearsal.
5. Tag `v1.0`. Clean commit history if it's a mess (`git rebase -i`).

---

## 6. Risk Register and Cut List

Compressing six weeks into seven days means something will slip. Decide *now* what gets cut, so that at 11pm on Day 5 you're executing a decision instead of making one.

| Risk | Likelihood | Mitigation / cut |
|---|---|---|
| HNSW recall stuck low on Day 2 | **High** | The debugging checklist in Day 2 covers ~90% of cases. Hard stop: if unfixed by end of Day 3, wrap `faiss.IndexHNSWFlat` behind your `Index` interface, ship the filtering work on top of it, and say plainly in the README that HNSW is FAISS's. **The filtering research is the project; the HNSW is the substrate.** This trade is fine. |
| Build too slow at 100K | Medium | Drop to 50K. Nobody's evaluation of this project turns on 100K vs 50K. |
| Predicate-aware never wins | Medium | Ship it as a measured negative result with the dist_ops accounting. Genuinely fine — see Day 5. |
| Days 4–5 overrun | Medium | Cut the correlated-metadata sweep to a single selectivity value (0.01) as an appendix rather than a full second sweep. |
| Cost model doesn't fit | Low | Fall back to a lookup-table planner: bucket ŝ into deciles, pick the empirically-best strategy per bucket. Less elegant, still a planner, still routes correctly. Say it's a lookup table. |
| You get sick / a lab deadline lands | Always | Days 1–5 alone with a partial sweep is still a good project. Day 6's chart is the only truly non-negotiable artefact — if you have three days, do Days 1, 4, 6 with FAISS as the index. |

**Ranked cut order** (cut from the bottom of the value stack first):
1. Numba optimisation *(cut freely)*
2. Full 1M run *(cut freely)*
3. Correlated metadata full sweep → reduce to one point
4. Two-hop expansion → seeded entry points alone are enough for a working Strategy C
5. FastAPI service → a CLI demo will do
6. — everything below this line is load-bearing —
7. The crossover chart, the planner, the three strategies, the honest README

---

## 7. Interview Questions You Must Be Able to Answer

Write your own answers to all of these on Day 7. Sketches given; the numbers must be yours.

**On the problem**
1. *Why not just filter after searching?* → Expected survivors is ef·s; you need ef ≥ k/s, which explodes as s→0, and it's probabilistic so you sometimes return fewer than k. Cite your underfill curve.
2. *Why not just filter first and scan?* → O(N·s·d). Fine at s=0.001, terrible at s=0.5. Cite your crossover point.
3. *So where's the crossover, and what determines it?* → Your measured number. Determined by the ratio of scan throughput to graph-hop cost, and by whether attributes correlate with vector position.

**On HNSW**
4. *What does the layer hierarchy buy you?* → Long-range links at sparse upper layers give logarithmic descent to the right region; layer 0 gives fine-grained local search. Without it you're doing NSW with linear-ish hop counts.
5. *Why the neighbour-selection heuristic instead of keeping the M closest?* → Diversity. Keeping the M closest gives you M neighbours in the same direction, so the graph has poor global connectivity and greedy search gets trapped in local minima. Quote your measured recall delta.
6. *What's your recall@10 and why isn't it 1.0?* → It's approximate: greedy descent with a bounded beam can miss the true nearest if it's reachable only through a locally-worse node. Widening ef trades latency for recall — point at the Pareto curve.

**On the planner**
7. *How do you estimate selectivity without touching the data?* → Precomputed value counts and histograms; independence assumption for AND. Show the estimation-error scatter and name the assumption as an assumption.
8. *What happens when the estimate is wrong?* → Bounded harm: the strategies all return correct results, just at suboptimal cost. Post-filter has a retry-and-fallback path. Quote your p95 regret.
9. *How did you fit the cost model constants?* → Least squares on measured latencies from the Day 4–5 sweep, persisted to calibration.json, reloaded at init. It's calibrated, not guessed.

**On rigour and limits**
10. *Why is your implementation slower than FAISS?* → SIMD-vectorised distance kernels and cache-friendly memory layout. Then the good part: my distance-op count is within X% of theirs, so the algorithm is equivalent — the gap is entirely in the constant factor per distance computation.
11. *What breaks if attributes correlate with vector position?* → Predicate-aware traversal degrades badly: greedy descent enters match-free regions and strands. Pre-filter improves. Show both charts side by side.
12. *How would you handle deletes?* → Tombstone bitmap, excluded at admission, graph left intact. Degrades over time as tombstoned nodes still cost traversal; rebuild when the tombstone fraction crosses ~20%. The hard part is that deleting a hub node damages connectivity for its neighbours.
13. *What would you do at 100M vectors?* → Doesn't fit in RAM, so HNSW's random access pattern is fatal. DiskANN/Vamana with a flatter graph and SSD-resident adjacency, plus product quantisation for an in-memory rerank set. That's the next project, not this one.
14. *Why Python?* → I'm benchmarking index algorithms, not language speed; NumPy's distance kernels are BLAS. FAISS is C++ behind a Python API too. If the goal were raw throughput I'd write it in Rust or C++ — and my dist_ops metric is exactly the language-independent way to show the algorithm is sound regardless.

---

## 8. Reading (only what you'll actually use)

| Paper | What you need from it | When |
|---|---|---|
| Malkov & Yashunin, *HNSW* (2016) | Algorithms 1–5, verbatim | Day 2 — read before you code |
| Gollapudi et al., *Filtered-DiskANN* (2023) | The filtered-graph problem framing; label-aware edges | Skim Day 4 evening |
| Patel et al., *ACORN* (2024) | Predicate-agnostic traversal; the neighbour-expansion trick | Skim Day 4 evening |
| Selinger et al., *Access Path Selection in a RDBMS* (1979) | Where cost-based planning came from. Short, readable, and citing it in an interview is a genuine flex | Day 6, 20 minutes |

Skim, don't study. You need the mechanism, not the proofs.

---

## 9. Definition of Done

The project is complete when a stranger can clone the repo and, in five minutes, reach all of:

- [ ] `crossover.png` at the top of the README, showing at least two strategy changes
- [ ] A results table with recall@10, p95 latency, and dist_ops for all three strategies at three selectivities
- [ ] `pip install -e . && python scripts/run_sweep.py --dataset siftsmall` runs end to end on a clean clone
- [ ] `pytest` green
- [ ] A limitations section that names at least four real limitations
- [ ] The one-sentence delta, stated in the first paragraph
- [ ] Your HNSW recall within 0.02 of FAISS's at matched parameters — or a documented, honest explanation of why not

And when *you* can say the §0 sentence out loud, cold, and defend all fourteen questions in §7 with your own numbers.
