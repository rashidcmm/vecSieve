# Filtered Vector Database — Design Spec

**Status:** Approved for implementation
**Repo:** https://github.com/rashidcmm/filtered-vecdb- (private)
**Source plan:** `filtered-vector-db-7-day-plan (1).md` (repo root) — the authoritative technical reference for every algorithm below. This spec adopts its architecture, locks the scope decisions, and defines how the project gets executed autonomously.
**Purpose:** A portfolio/resume project. The user (Rashid) will not be hand-writing this code himself in real time — it is built autonomously — but the finished repo must be something he can defend line-by-line in interviews around December 2026, after studying the milestone explainers produced during the build.

---

## 1. The One-Sentence Delta

> A cost-based query planner for approximate nearest-neighbour search that chooses between pre-filtering, post-filtering, and predicate-aware graph traversal using estimated predicate selectivity — with a measured crossover curve showing exactly where each strategy wins and why.

This is not "an HNSW implementation." HNSW is the substrate; the planner and the measured crossover are the deliverable.

## 2. Objective

**Primary:** correctly and efficiently answer queries of the form:
```
SELECT id FROM vectors
WHERE category = 'hull' AND year > 2019
ORDER BY l2_distance(embedding, :q)
LIMIT 10
```
...deciding **at runtime** which execution strategy to use, rather than hardcoding one.

**Secondary objectives** (in priority order — see source plan §1.2 for full success criteria): exact flat baseline → hand-written HNSW (recall@10 ≥ 0.95 unfiltered) → three working filter strategies → a measured crossover where the winning strategy changes at least twice → a planner that beats every fixed strategy on mean latency → an honest FAISS comparison → a servable FastAPI layer with persistence.

**Explicit non-objectives:** beating FAISS on wall-clock (NumPy vs. C++/SIMD — expected 2–5x gap, explained not hidden), distributed operation, ACID durability, deletes in v1, billion-scale data. These are stated up front in the README, not discovered as excuses later.

## 3. Scope Decisions Locked for This Build

These are the deltas from the source plan, decided during brainstorming:

| Decision | Value |
|---|---|
| Benchmark scale | **100K SIFT subset only.** The 1M full run is explicitly cut — not attempted, not left "optional" in a way that creates ambiguity. |
| Metadata variants | Both uncorrelated and correlated (k-means-based), per source plan §4.2 — this is the highest-value/hour addition and stays in. |
| HNSW implementation | **Hand-written**, per source plan Day 2, algorithms 1–5 of Malkov & Yashunin. Fallback to `faiss.IndexHNSWFlat` wrapped behind the same `Index` interface is permitted **only** if recall@10 ≥ 0.95 is not achieved after genuine debugging against the plan's own debugging checklist (source plan §5, Day 2) — and if that fallback is used, the README states it plainly. This is the plan's own documented contingency, not a new option. |
| Service layer | FastAPI included (`/insert`, `/search`, `/stats`, `/persist`), per source plan §3.3 and Day 6. |
| Numba optimization | Cut (source plan's own lowest-value cut-list item). |
| Repo hosting | Private GitHub repo `rashidcmm/filtered-vecdb-`, pushed incrementally as milestones complete. |
| Environment | Windows 11, Python 3.12, local machine. No GPU/cloud dependency required anywhere in the plan. |

Everything else — architecture, component contracts, predicate DSL, cost model, dataset, metrics, day-by-day algorithm details — is adopted unmodified from the source plan. This spec does not restate all of it; implementation should treat the source plan as the algorithmic reference and this spec as the execution contract on top of it.

## 4. Architecture (summary)

```
SERVICE LAYER (FastAPI)  →  QUERY PLANNER (selectivity → cost model → strategy choice)
                                   │
              ┌────────────────────┼────────────────────┐
        PRE-FILTER          POST-FILTER          PREDICATE-AWARE
        (exact scan)      (HNSW + discard)     (filtered HNSW walk)
                                   │
                             INDEX LAYER (FlatIndex, hand-written HNSWIndex)
                                   │
                          PREDICATE LAYER (DSL → mask, selectivity estimator)
                                   │
                     STORAGE LAYER (VectorStore memmap, MetaStore columnar, IdMap)
```

Full component contracts (`VectorStore`, `MetaStore`, predicate DSL grammar, `Index` ABC, `SearchResult` fields, cost model formulas, repository file layout) are specified in source plan §3 and carry over verbatim. The `★`-marked hand-written files in source plan §3.3 (`hnsw.py`, `strategies.py`, `cost_model.py`) remain the files that must be written line-by-line, not scaffolded generically — they're what gets defended in interviews.

## 5. Data & Experimental Design

Adopted from source plan §4, unmodified except for the 100K-only scale lock:

- **Dataset:** SIFT (`siftsmall` 10K for dev iteration, 100K subset of `sift1m` for the headline benchmark). Downloaded from the TEXMEX corpus — reachable over plain HTTP from this environment (HTTPS cert on that host is broken; HTTP works).
- **Synthetic metadata:** two variants over the same vectors — uncorrelated (random category/year/score) and correlated (k-means-cluster-based category). Both are required; this is what produces the "predicate-aware traversal wins on uncorrelated, loses on correlated" finding that makes the write-up interesting.
- **Selectivity grid:** s ∈ {0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0}, 200 queries per cell, first 20 discarded as warmup.
- **Metrics:** recall@k, underfill_rate, latency p50/p95, dist_ops, qps, build_time_s, index_bytes, sel_error, regret_ms — full definitions in source plan §4.4. Ground truth for filtered queries computed by brute force over the mask, cached to avoid recomputation across sweep reruns.

## 6. Execution Model

This is the part that's genuinely new relative to the source plan, which was written for a human learning HNSW over 7 calendar days. Since this is executed autonomously:

- **Milestones replace days.** The 8-day structure (Day 0–7) becomes a sequence of build milestones with the same content and the same "Done when" gates, but no artificial per-day time-boxing — compute-bound steps run as background jobs and I check in when they complete rather than blocking a day around them:
  1. Setup — repo scaffold, dataset download, dependency install, metadata generation
  2. Storage + predicate layer + Flat baseline + benchmark harness (the "ruler")
  3. Hand-written HNSW construction and correctness validation
  4. Search tuning (efSearch/build-param sweeps) + scale-up to 100K
  5. Pre-filter and post-filter strategies + selectivity estimator validation
  6. Predicate-aware traversal strategy
  7. Cost-model calibration + planner + full sweep + headline figures
  8. FastAPI service + tests + README + final polish
- **Milestone reports.** After each milestone, a short written report: what was built, the actual measured numbers it produced, whether its "Done when" gate was met, and a plain-language note on how to explain that piece in an interview. These reports are the study material for December.
- **No plan/reality gap.** Every "Done when" gate is a hard checkpoint, not a suggestion. If a gate fails (e.g. HNSW recall stuck below target), the response is genuine debugging against the plan's checklist first — silently lowering the bar or writing README claims the code doesn't support is not an acceptable outcome. If a gate still can't be met after honest effort, that gets reported as its own finding (the source plan explicitly treats a well-measured negative result, e.g. "Strategy C never wins," as an acceptable, even valuable, outcome) — never hidden or glossed over.
- **Git workflow.** One commit per milestone (or logical sub-step within a large milestone), pushed to the private GitHub repo as the build progresses, so the commit history itself is legible and interview-usable.
- **Definition of Done** for the whole project is source plan §9, unmodified: crossover chart above the fold in the README showing ≥2 strategy changes, results table (recall/p95/dist_ops for all 3 strategies at 3 selectivities), clean-clone `pip install -e . && python scripts/run_sweep.py` working end to end, green test suite, an honest limitations section (≥4 real limitations), the one-sentence delta stated up front, and HNSW recall within 0.02 of FAISS's at matched parameters (or a documented, honest explanation of why not).

## 7. Testing Strategy

Adopted from source plan Day 7 / §3.3 `tests/`:
- `test_predicate.py` — DSL/mask semantics, including empty and full masks
- `test_hnsw_correctness.py` — recall floor vs. `FlatIndex` ground truth
- `test_strategies_agree.py` — all three strategies return the same top-k as `FlatIndex` on tiny data with `ef` large enough
- `test_planner_regret.py` — regret below threshold on cached sweep results

## 8. Risk Register (carried forward, adjusted for locked decisions)

| Risk | Mitigation |
|---|---|
| Hand-written HNSW recall stuck low | Debug against source plan's Day 2 checklist first; fall back to FAISS-substrate wrapped behind `Index`, documented plainly, only if unresolved |
| Predicate-aware traversal never wins anywhere | Ship as a measured, explained negative result with dist_ops accounting — not fabricated |
| Cost model doesn't fit well | Fall back to a decile lookup-table planner, explicitly labeled as such |
| 1M run temptation later | Out of scope for this build; not pursued unless separately requested |

## 9. References

Malkov & Yashunin (HNSW, 2016) · Gollapudi et al. (Filtered-DiskANN, 2023) · Patel et al. (ACORN, 2024) · Selinger et al. (Access Path Selection in a RDBMS, 1979) — full reading notes in source plan §8.
