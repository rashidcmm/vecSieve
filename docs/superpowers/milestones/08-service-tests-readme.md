# Phase 8 milestone: FastAPI service, agreement/regret tests, final README, v1.0

## What got built

**FastAPI service layer** (Task 40). `vecdb/service/schemas.py` (6 Pydantic models:
`InsertRequest`, `SearchRequest`, `SearchResultItem`, `SearchResponse`, `StatsResponse`,
`PersistResponse`) and `vecdb/service/app.py` (`VecDBService` class + FastAPI routes). Four
endpoints: `POST /insert`, `POST /search`, `GET /stats`, `POST /persist`. The service is built on
a **staged-insert overlay**: new vectors from `POST /insert` land in in-memory lists
(`staged_vectors`, `staged_ids`, `staged_meta`) rather than mutating the hand-written HNSW graph
live; `POST /search` brute-force-scans the staged overlay and merges those results with the main
index's results, mirroring how production systems layer a mutable segment on top of an immutable
one. Search is planner-driven: the request's filter is compiled to a bitmask, `Planner` picks a
strategy, and the response reports both the chosen strategy name and a human-readable reason
string. `VecDBService` supports both constructor injection (for tests) and a `from_disk()`
production path that loads the persisted 100K SIFT bundle and `results/calibration.json`.

A high-severity **path-traversal vulnerability** was found and fixed in `POST /persist`
post-review: the original endpoint accepted an arbitrary `path: str`, allowing writes outside the
intended data directory (e.g. `../../../../whatever`). The fix changed the parameter to a bare
`name: str` (filename only), added a `DATA_ROOT = Path("data").resolve()` constraint, and
rejected any name containing a path separator, any name starting with `.`, or any resolved path
that doesn't stay under `DATA_ROOT`. Two tests were added to cover it:
`test_persist_endpoint_accepts_valid_name` and `test_persist_endpoint_rejects_path_traversal`.
Total service test coverage: 6 tests in `tests/test_service.py`, all passing.

**`tests/test_strategies_agree.py`** (Task 41). One test,
`test_all_three_strategies_agree_with_flat_on_tiny_data_with_generous_budgets`, builds a tiny
index (n=300, d=16) with a generous 40% selectivity and checks all three strategies against the
`FlatIndex` ground truth over 5 random queries at k=5: `pre_filter` must match exactly;
`post_filter` and `predicate_aware` must hit ≥4/5 recall under generous budgets. Passed on the
first run, no parameter widening needed.

**`tests/test_planner_regret.py`** (Task 42). Loads the Task 38 full-sweep CSV
(`results/sweep_uncorrelated.csv`), reshapes to per-(query, selectivity) mean latencies for
`pre_filter`, `post_filter`, `predicate_aware`, and `planner`, computes the hindsight oracle (the
per-row minimum across the three fixed strategies) and the planner's regret against it via
`compute_regret()`/`regret_summary()`, then asserts mean regret is under 15% of the oracle's mean
latency (spec §5 target). This test **genuinely fails**: mean regret 29.782ms vs. a 0.135ms
threshold (15% of a 0.899ms oracle mean) — the planner it exercises is the **cost-model**
`Planner`, and this is a documented negative result about that planner, not a bug in the test or
the harness (see "Numbers" and "Gate status" below).

**Final README assembly** (Task 43). `README.md` was rewritten end-to-end: the one-sentence delta
up top followed by the embedded `results/figures/crossover.png` with an honest caption; a 9-row
results table (3 strategies × 3 selectivities) pulled live from `results/sweep_uncorrelated.csv`
via pandas, not estimated or copied from an older milestone doc; the §3.1 architecture diagram
reproduced verbatim with four design-decision paragraphs; the exact 7-command "How to run"
sequence plus an explicit statement that `pytest` is not fully green; and a "What I got wrong"
section leading with the two mandatory negative results (Strategy C never winning; the two-part
planner story kept in two separate, clearly-labeled sub-bullets) before the single-crossover
finding, the FAISS gap, the AND-independence tail error, and the remaining limitations from source
plan §8. A post-review fix corrected a false claim that `predicate_aware` was "the slowest
strategy at every selectivity shown" — the table itself contradicts that (`post_filter` is far
slower at s=0.001; `pre_filter` is slower at s=0.5) — replaced with two verified claims: it's
never the *fastest* at any of the 8 full-sweep selectivities, and it is not uniformly the slowest
in the 3-row table either.

**Clean-clone verification and `v1.0` tag** (Task 44). A fresh `git clone D:/vecdb` into a scratch
directory, a fresh venv, `pip install -e ".[dev]"`, and a full `pytest -v` run confirmed the
documented flow works end to end outside the working directory's local state. The spec's
Definition of Done checklist (7 items, reproduced below) was checked item-by-item against real
files, not from memory. The annotated tag `v1.0` was created on commit `8aaf588` (the README
fix-round commit) and pushed to `origin`; confirmed on the remote via `git ls-remote --tags
origin` (tag object `8dcbe159...` → commit `8aaf5886...`).

## Numbers

**Clean-clone `pytest` result (Task 44, the authoritative count for this report):**

```
1 failed, 106 passed, 1 skipped, 1 warning in 9.95s
```

- **Failed**: `tests/test_planner_regret.py::test_mean_regret_is_bounded_relative_to_oracle` —
  mean regret 29.782ms exceeds 15% of oracle mean 0.899ms (threshold 0.135ms). Genuinely fails,
  not skipped: `results/sweep_uncorrelated.csv` is committed to the repo, so the `skipif` guard
  does not fire in a clean clone.
- **Skipped**: `tests/test_hnsw_correctness.py::test_recall_at_10_floor_against_flat_ground_truth`
  — `skipif`s on the absence of `data/hnsw_siftsmall/`, the persisted Phase 3 build artifact,
  which is not committed to the repo. Correct, expected behavior in a clean clone that hasn't run
  the data/build scripts.

This differs from the count quoted in `README.md`'s own "How to run" section — **107 passed, 1
failed** — because the local working copy at `D:\vecdb` happens to have `data/hnsw_siftsmall/`
present on disk (a build artifact from earlier phases that was never committed), so
`test_recall_at_10_floor_against_flat_ground_truth` runs and passes locally instead of skipping.
Both counts are consistent: **107 non-failing tests either way** (106 passed + 1 skipped, or 107
passed), and the same single test — `test_planner_regret.py` — fails in both environments for the
same documented reason. This report quotes the **clean-clone** figures (1 failed, 106 passed, 1
skipped) as the authoritative numbers, per this task's brief, because they reflect what anyone
who actually clones the repo fresh will see, not an artifact of this particular machine's build
history.

**The two planners, kept separate (do not merge these into one verdict):**

- The **cost-model planner** (`Planner(params)` — what `crossover.png` actually plots as "planner
  (chosen)") **failed** its Phase 7 gate: mean latency 30.68ms vs. `pre_filter`'s 3.66ms, roughly
  **8x worse**, on the full uncorrelated sweep. This is the same planner `test_planner_regret.py`
  exercises and the same failure the clean-clone `pytest` run reproduces (29.782ms regret vs. a
  0.135ms threshold).
- `Planner.from_lookup_table(...)` — the source plan's own documented risk-register mitigation —
  buckets estimated selectivity and picks the empirically fastest strategy per bucket, calibrated
  on half the sweep (`query_idx < 110`) and evaluated on the disjoint held-out half
  (`query_idx ≥ 110`). On that held-out data it **passed**: mean latency 0.9046ms vs.
  `pre_filter`'s 3.6410ms, roughly **4x faster**. It was evaluated as a calibration/analysis pass
  over already-collected sweep data (`scripts/evaluate_lookup_planner.py`), not a new 100K-scale
  sweep, and it is **not plotted** in `crossover.png` or any other figure.

Quoting either fact alone misrepresents the project: the planner that is actually swept and
plotted lost by ~8x; a documented fallback for exactly that failure mode won by ~4x on held-out
data, via a different mechanism, on a different data slice.

## Gate status — spec §9 Definition of Done, verified against real files

The checklist below reproduces the source technical plan's own §9 wording
(`filtered-vector-db-7-day-plan (1).md:642-648`, read directly in this session — not the
`task-44-brief.md` paraphrase this report originally relied on, which silently dropped a clause
from item 1; see the fix note in `task-45-report.md` for what changed and why), with one
deliberate substitution: item 3 names `scripts/run_full_sweep.py`, the script that actually
exists and runs the sweep today, in place of the source's original `scripts/run_sweep.py
--dataset siftsmall`, an earlier name the codebase has since superseded. This report re-confirms
each verdict against current file state rather than copying any prior summary uncritically.

| # | Item | Verdict |
|---|---|---|
| 1 | `crossover.png` at the top of the README, **showing at least two strategy changes** | **PARTIAL — fails the full criterion.** `results/figures/crossover.png` exists (89,051 bytes) and is referenced at the top of `README.md` (line 8, immediately after the opening delta paragraph) — that half holds. But the chart shows **exactly one** strategy change (crossover), not the ≥2 this item requires: `predicate_aware` never wins any of the 8 measured selectivities, so the expected three-region shape (pre_filter → predicate_aware → post_filter, two crossovers) collapses to two regions (pre_filter → post_filter, one crossover). This is the project's own headline honest finding — see "What got built" and the interview note above — so it is graded here as a genuine miss, not rounded up to a pass. |
| 2 | Results table has recall@10, p95 latency, dist_ops for all three strategies at three selectivities | **PASS** — `README.md`'s 9-row table (3 strategies × selectivities 0.001/0.05/0.5), pulled live from `results/sweep_uncorrelated.csv` |
| 3 | `pip install -e . && python scripts/run_full_sweep.py` runs end to end | **PASS** — confirmed by artifact evidence (`results/sweep_uncorrelated.csv`, `results/sweep_correlated.csv` both present and non-empty) plus Task 38's narrative record; not re-run in this verification pass (a fresh 100K run takes 30–60 min) |
| 4 | `pytest` is green (or skips only the artifact-gated test) on a clean clone | **CONDITIONAL — not fully green.** One genuine, documented failure: `tests/test_planner_regret.py::test_mean_regret_is_bounded_relative_to_oracle` (cost-model planner regret, 29.782ms vs. 0.135ms threshold). This is a real, honestly-red assertion failure, not a skip and not a bug — the underlying finding (the cost-model planner is ~8x worse than the best fixed strategy) is real and documented, not a test-harness defect. One test correctly skips (`test_hnsw_correctness.py`, artifact-gated). Do not read this row as "pytest ✅." |
| 5 | Limitations section names at least four real limitations | **PASS** — `README.md`'s "What I got wrong / limitations" names 11 distinct items: Strategy C's universal loss, the single-crossover finding, the cost-model planner's gate failure, the FAISS latency gap, the AND-independence tail error, synthetic metadata, no deletes, no durability, single-threaded, `pytest` not fully green, and the never-triggered post-filter fallback |
| 6 | The one-sentence delta is stated in the first paragraph | **PASS** — `README.md`'s opening blockquote states it verbatim |
| 7 | HNSW recall is within 0.02 of FAISS's at matched parameters, or Phase 3's milestone report documents the fallback honestly | **PASS on the primary clause** — max measured recall gap is 0.011 (hand-written HNSW actually *higher* at ef=10/20/40) across all six matched `ef` values in `docs/superpowers/milestones/04-tuning-scale.md`; the FAISS-wrapping fallback was never invoked. (Note: this item is about *recall*, not *latency* — the ~13–15x latency gap is real but is a separate, honestly-documented metric, not a violation of this checklist item.) |

**Bottom line: 5 of 7 items are unconditional PASS. Two items do not hold unconditionally, both
for documented, understood, non-bug reasons: item 1 (`crossover.png` showing ≥2 strategy changes)
shows only 1, because `predicate_aware` never wins; item 4 (`pytest` green) has one genuine,
documented failure.** The project does not claim a fully green test suite, nor a two-crossover
chart, anywhere in this report or in `README.md` — both of these are the same class of honestly-
reported negative result as everything else in "What got built" above, not gaps this table papers
over.

## Interview note (spoken, ~90 seconds, per source plan §7)

"The one-sentence version: this is a cost-based query planner for approximate nearest-neighbor
search that picks between three strategies — pre-filter, post-filter, and predicate-aware graph
traversal — based on estimated selectivity, and I measured a real crossover curve showing exactly
where each one wins. On the crossover chart, the honest shape is one crossover, not two. I
expected pre-filter to win at low selectivity, predicate-aware to own a middle band, and
post-filter to take over at high selectivity — three regimes. What actually happened is
predicate-aware never won anywhere, not once across eight selectivity points, so the chart shows
pre-filter winning the whole low-to-mid range and post-filter taking over above about ten percent
selectivity. One crossover, because the middle strategy lost its own middle band. My strongest
number is on the fallback side: I built a lookup-table planner as a documented backup, calibrated
on half the queries and evaluated on the held-out other half, and it beat pre-filter by roughly
four times — point-nine milliseconds versus three-point-six. My most honest limitation number is
that the cost-model planner — the one actually plotted in the chart — lost its own gate by about
eight times, thirty milliseconds of mean regret against a hindsight oracle instead of near-zero,
because it was calibrated against distance-operation counts, not wall-clock time, and those don't
rank the same way once one strategy's operations cost three orders of magnitude more per op than
another's. That test is still red in the suite today, on purpose — I left it failing rather than
loosening the threshold to hide it."

(Word count: 260 words, roughly 90 seconds at natural speaking pace.)

## Files referenced

- `vecdb/service/app.py`, `vecdb/service/schemas.py` — FastAPI service
- `tests/test_service.py` — 6 service tests
- `tests/test_strategies_agree.py` — strategy agreement test
- `tests/test_planner_regret.py` — planner regret test (currently red)
- `README.md` — final assembled README
- `results/figures/crossover.png`, `results/sweep_uncorrelated.csv`, `results/sweep_correlated.csv`
- `.superpowers/sdd/2026-08-11-filtered-vector-db/task-40-report.md` (+ addendum),
  `task-41-report.md`, `task-42-report.md`, `task-43-report.md` (+ addendum), `task-44-report.md`,
  `task-38-followup-report.md`
- `docs/superpowers/milestones/07-planner-sweep-figures.md`,
  `docs/superpowers/milestones/06-predicate-aware-traversal.md`,
  `docs/superpowers/milestones/04-tuning-scale.md`
- `filtered-vector-db-7-day-plan (1).md` (repo root) — source technical plan, §9 Definition of
  Done (lines 638-650), read directly for the Gate status table above
