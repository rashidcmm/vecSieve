# Phase 2 Milestone: Storage, Predicate DSL, Flat Baseline

## What got built

Task 6–13 delivered the core storage and indexing layer. We built `vecdb.store.vectors.VectorStore` (row-major float32 vector storage that computes batched squared-L2 distance as a single matrix-vector product via the `||q-v||^2 = ||q||^2 - 2 q.v + ||v||^2` identity, and tracks a running `n_distance_ops` counter), `vecdb.store.metadata.MetaStore` (columnar attribute storage, one `np.ndarray` per column, that precomputes per-column statistics — value counts for categorical `int32` columns, a 64-bin histogram for numeric columns — and never re-touches the raw columns at query time), and `vecdb.store.idmap.IdMap` (external ID ↔ internal dense row index, assigned sequentially on `add()`, with roundtrip tests).

The predicate DSL in `vecdb.predicate.dsl` is dict-based, not a class hierarchy: predicates are nested, JSON-compatible dicts with a lowercase string `"op"` key. Leaf ops (`"eq"`, `"ne"`, `"lt"`, `"lte"`, `"gt"`, `"gte"`, `"in"`) carry `"col"` and `"val"`; combinator ops (`"and"`, `"or"`) carry a non-empty `"clauses"` list; `"not"` carries a single `"clause"`. `validate_predicate()` recursively checks this shape and raises `ValueError` on malformed input. `vecdb.predicate.compile.compile()` recursively walks the same dict structure and evaluates it against a `MetaStore`, combining boolean masks with `&`, `|`, and `~` for `and`/`or`/`not`, and calling straight through to numpy comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`, `np.isin`) for the leaf ops, producing a single boolean mask over rows.

Selectivity estimation in `vecdb.predicate.selectivity` walks the same predicate dicts but reads *only* `MetaStore` statistics (value counts / histogram edges and counts), never touching raw columns or masks — categorical `eq`/`ne`/`in` divide value counts by `n`, numeric `lt`/`lte`/`gt`/`gte` interpolate within the histogram bin containing the threshold, `and` multiplies child selectivities under an independence assumption, `or` uses inclusion-exclusion, and `not` is `1 - child`. The `vecdb.index.flat.FlatIndex` is a reference exact-search implementation using brute-force L2 distance over all vectors, returning results sorted by distance. The `vecdb.bench.harness` runs end-to-end benchmarks on arbitrary indexes against ground truth, computing recall@k from comparison sets. Finally, FAISS baselines (`vecdb.index.faiss_baseline.FaissFlatIndex` and `FaissHNSWIndex`) wrap FAISS's own exact and approximate search for sanity checking and performance comparison across all later phases.

## Numbers

**Test pass counts by file:**
- Tasks 6–13 combined: 47 tests passing (all green, no failures)
- Test breakdown: `tests/test_fvecs.py` (2), `tests/test_dataset.py` (3), `tests/test_metadata_gen.py` (3), `tests/test_vectors.py` (4), `tests/test_metastore.py` (3), `tests/test_idmap.py` (4), `tests/test_predicate.py` (10), `tests/test_selectivity.py` (7), `tests/test_flat_index.py` (3), `tests/test_harness.py` (6), `tests/test_faiss_baseline.py` (2) — sums to 47

**Baseline recall verification (Phase 2 gate):**
- FlatIndex recall@100 (siftsmall, 100 queries): **1.0**
- FaissFlatIndex recall@100 (siftsmall, 100 queries): **1.0**

Both indexes are exact-search implementations and each achieves perfect recall@100 against the SIFT ground truth. `scripts/verify_baseline.py` verifies this via `recall@100 == 1.0` for both, which is agnostic to tie-order among equidistant neighbors — it does not assert that the two indexes return bitwise-identical result arrays.

## Gate status

✓ PASS. `scripts/verify_baseline.py` confirms that both FlatIndex (our pure-Python brute-force) and FaissFlatIndex (FAISS's L2 reference) achieve perfect recall@100 on siftsmall queries. This gate ensures Phase 3–7 can trust the baseline for all downstream Pareto curves and dist_ops comparisons. No regressions; all prior task tests remain green.

## Interview note

Selectivity estimation reads *only* precomputed MetaStore statistics (histograms and value counts) and deliberately never touches raw attribute columns or the query-time mask. This design enforces separation of concerns: cost modeling happens at plan time using aggregate statistics, while actual filtering is deferred to search time. Touching the mask during estimation would couple the cost model to runtime data, making plans depend on query specifics rather than data distribution properties, which defeats cardinality-based optimization. For recall tie-handling in the benchmark harness, we use `min(k, len(true_ids))` as the denominator rather than naively using `len(true_ids)`. When the ground truth has fewer than k results (e.g., a rare category returns only 50 true neighbors when k=100), using the full k would incorrectly penalize our index for returning all 50 correct results; the corrected formula scores recall as 50/50 = 1.0, properly crediting exact queries.
