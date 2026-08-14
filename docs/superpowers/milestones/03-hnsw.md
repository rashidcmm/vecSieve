# Phase 3 Milestone: Hand-written HNSW

## What got built

Tasks 14–19 delivered a complete, hand-written Hierarchical Navigable Small World index (`vecdb.index.hnsw.HNSWIndex`), implemented directly from the Malkov & Yashunin (2016) algorithm rather than wrapping FAISS.

- **Class skeleton and level assignment** (Task 14): `HNSWIndex.__init__` sets up per-layer graph storage (`graph: list[dict[int, list[int]]]`, one adjacency dict per layer), `M`/`M0` degree caps, and `ef_construction`. `_assign_level()` draws a level for each inserted node from the geometric distribution `floor(-ln(U) * mL)` with `mL = 1/ln(M)`, which is what gives HNSW its logarithmically-shrinking layer population and O(log N) navigation.
- **`_search_layer`** (Task 15): a greedy beam search over a single layer, using a min-heap of candidates to expand and a bounded max-heap (via negated distances) of the best `ef` results seen so far, with a visited set to avoid re-expanding nodes. It early-breaks once the nearest unexplored candidate is already worse than the current worst kept result.
- **`_select_neighbors_heuristic`** (Task 16): Algorithm 4 from the paper. Given a candidate list sorted by distance to the query, it greedily accepts a candidate only if it is closer to the query than it is to every neighbour already selected, capping at `M` accepted neighbours. This is what keeps the graph's edges pointing in diverse directions instead of all clustering toward one dense region.
- **`_insert` / `add` / `search`** (Task 17): full insertion — descend from the entry point through upper layers with `ef=1` greedy search to find a good entry into the target node's top level, then at each layer from `min(level, max_level)` down to 0, run `_search_layer` with `ef_construction`, select diverse neighbours via Algorithm 4, link bidirectionally, and re-prune any neighbour whose degree cap (`M` or `M0` at layer 0) is exceeded. `search()` does the same descent with `ef=1` on upper layers, then a wider `_search_layer` pass at layer 0 with the caller's `ef`.
- **Persistence** (Task 18): `.save()`/`.load()` round-trip the raw vectors (`vectors.npy`), the per-layer adjacency (`graph.pkl`), and metadata (`meta.pkl`: dim, M, M0, ef_construction, entry_point, max_level, levels) to a directory, verified by `test_hnsw_persistence.py` to produce identical search results after reload.
- **Phase 3 gate** (Task 19): `scripts/build_index.py` builds the index on the full siftsmall dataset (10,000 base vectors, dim 128), benchmarks recall@10 against `FlatIndex`/ground-truth at `ef=100`, asserts the numeric gates, and persists the built index to `data/hnsw_siftsmall/`. `tests/test_hnsw_correctness.py` is the permanent regression test: it loads that persisted artifact (does not rebuild) and re-asserts the recall floor on every `pytest` run.

No existing modules were modified — HNSWIndex, the harness, and the dataset loader were used as-is per the brief.

## Numbers

**Phase 3 gate (`scripts/build_index.py` on siftsmall, M=16, ef_construction=200, seed=42):**
- Build time: **31.2s** (gate requires < 120s)
- Recall@10 at efSearch=100: **0.9980** (gate requires >= 0.95)
- Persisted index: `data/hnsw_siftsmall/` (vectors.npy, graph.pkl, meta.pkl)

**Test results:**
- Tasks 14–18 combined (`test_hnsw_levels.py`, `test_hnsw_search_layer.py`, `test_hnsw_heuristic.py`, `test_hnsw_insert.py`, `test_hnsw_persistence.py`): 16 tests, all passing
- Task 19 (`test_hnsw_correctness.py`): 1 test, passing (not skipped — the artifact from `scripts/build_index.py` is present)
- Full suite (`pytest -v` from repo root): **64 passed**, 0 failed, 0 skipped, in ~4.7s

## Gate status

PASS. `scripts/build_index.py` ran cleanly on the first attempt — no debugging or fallback was needed. Build time (31.2s) is well under the 120s ceiling and recall@10 (0.9980) comfortably clears the 0.95 floor. The hand-written implementation was used as specified; the documented FAISS-wrapping fallback was not invoked.

## Interview note

**What does Algorithm 4's diversity heuristic (`_select_neighbors_heuristic`) buy over naively keeping the top-M nearest candidates?**

If you just keep the M nearest candidates to a newly-inserted point, in a dense cluster those M nearest points are often all clustered in the same direction from the query — think of a point surrounded by other points on one side and open space on the other; naive top-M would fill all M edge slots with near-duplicates from the crowded side, leaving the node with no edges reaching toward the sparser regions of the space. That starves the graph of long-range/cross-cluster connectivity, which is exactly what makes greedy search able to jump across the space quickly instead of getting stuck hill-climbing through one dense pocket.

Algorithm 4 fixes this by testing each candidate, in increasing distance-to-query order, against the neighbours already accepted: a candidate `c` is kept only if it is closer to the query than it is to every neighbour already selected. Concretely in the code, for each accepted neighbour `r` we compute `d(c, r)` and reject `c` if `d(c, r) < d(query, c)` for any `r`. This is a proxy for "does c bring a genuinely new direction, or is it redundant with something already picked" — if `c` is nearer to an already-selected point `r` than to the query itself, `r` already "covers" that direction about as well, so `c` is skipped in favour of a later, more diverse candidate. The net effect is a relative-neighbourhood-graph-like structure: fewer redundant clustered edges, more edges reaching into different directions, which is what gives HNSW its long-range navigability and is the main reason greedy beam search on top of it achieves near-exact recall while only touching a small fraction of the graph.

**What does the early-break stopping condition in `_search_layer` buy over a full graph traversal?**

`_search_layer` maintains a min-heap of candidates still to expand and a bounded max-heap of the best `ef` results found so far. Each iteration pops the nearest unexpanded candidate `c` and compares its distance to the query against the current worst (furthest) distance among the kept `ef` results. If `c` is already farther from the query than our current worst kept result, *and* we already have `ef` results, the loop breaks immediately instead of continuing to pop and expand the rest of the candidate heap.

The reasoning is monotonicity of the min-heap pop order: everything still sitting in the candidate heap is at least as far as `c` (that's what makes it a min-heap), so if `c` itself can't possibly improve on our current worst kept result, nothing after it can either — expanding those candidates' neighbours could only ever produce distances greater than or equal to what we've already ruled out as unhelpful, given the graph has reasonably well-behaved edges. Once that condition is true, further work is provably wasted.

Without this check, `_search_layer` would keep expanding every node it manages to add to the visited set until the whole reachable component of that layer's graph was exhausted — degenerating into a full traversal that costs O(N) distance computations at query time instead of the roughly logarithmic number HNSW is supposed to need. The early break is what actually delivers HNSW's sublinear search cost in practice, not just on paper; it's also why, per the code's own comment, if a search were to end up slow despite decent recall, that stopping condition (and whether `ef` and the beam are sized sensibly) is the first place to look.
