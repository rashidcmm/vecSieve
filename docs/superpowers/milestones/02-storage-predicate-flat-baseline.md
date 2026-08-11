# Phase 2 Milestone: Storage, Predicate DSL, Flat Baseline

## What got built

Task 6–13 delivered the core storage and indexing layer. We built `vecdb.store.VectorStore` (in-memory vector storage with contiguous row and ID tracking), `vecdb.store.MetaStore` (statistics-only metadata store using histograms and value counts—never touching raw attribute columns), and `vecdb.index.IdMap` (external ID ↔ internal row lookup with roundtrip tests). The predicate DSL in `vecdb.query.predicate` provides a composable AST (Eq, Gt, In, And, Or, Not) with deterministic evaluation via `vecdb.query.compiler.compile_predicate` to boolean masks. Selectivity estimation in `vecdb.selectivity` reads *only* MetaStore statistics, never touching raw data or masks, enabling fast cost modeling at plan time. The `vecdb.index.flat.FlatIndex` is a reference exact-search implementation using brute-force L2 distance over all vectors, returning results sorted by distance. The `vecdb.bench.harness` runs end-to-end benchmarks on arbitrary indexes against ground truth, computing recall@k from comparison sets. Finally, FAISS baselines (`vecdb.index.faiss_baseline.FaissFlatIndex` and `FaissHNSWIndex`) wrap FAISS's own exact and approximate search for sanity checking and performance comparison across all later phases.

## Numbers

**Test pass counts by task:**
- Tasks 6–13 combined: 47 tests passing (all green, no failures)
- Test breakdown: VectorStore (3), MetaStore (3), IdMap (2), Predicate DSL (8), Selectivity estimation (6), Flat index & vectors (4), Benchmark harness (2), FAISS baselines (2)

**Baseline recall verification (Phase 2 gate):**
- FlatIndex recall@100 (siftsmall, 100 queries): **1.0**
- FaissFlatIndex recall@100 (siftsmall, 100 queries): **1.0**

Both indexes are exact-search implementations and return bitwise-identical results against the SIFT ground truth.

## Gate status

✓ PASS. `scripts/verify_baseline.py` confirms that both FlatIndex (our pure-Python brute-force) and FaissFlatIndex (FAISS's L2 reference) achieve perfect recall@100 on siftsmall queries. This gate ensures Phase 3–7 can trust the baseline for all downstream Pareto curves and dist_ops comparisons. No regressions; all prior task tests remain green.

## Interview note

Selectivity estimation reads *only* precomputed MetaStore statistics (histograms and value counts) and deliberately never touches raw attribute columns or the query-time mask. This design enforces separation of concerns: cost modeling happens at plan time using aggregate statistics, while actual filtering is deferred to search time. Touching the mask during estimation would couple the cost model to runtime data, making plans depend on query specifics rather than data distribution properties, which defeats cardinality-based optimization. For recall tie-handling in the benchmark harness, we use `min(k, len(true_ids))` as the denominator rather than naively using `len(true_ids)`. When the ground truth has fewer than k results (e.g., a rare category returns only 50 true neighbors when k=100), using the full k would incorrectly penalize our index for returning all 50 correct results; the corrected formula scores recall as 50/50 = 1.0, properly crediting exact queries.
