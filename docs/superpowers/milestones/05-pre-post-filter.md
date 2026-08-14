# Phase 5 Milestone: Pre-filter and post-filter strategies

## What got built

Tasks 24–27 built the first two of the three filtered-search strategies (Strategy A and Strategy B from the source plan), validated the selectivity estimator that Strategy B and the eventual planner both depend on, and ran the full selectivity-grid A/B sweep at 100K scale that is the headline artifact for this phase.

- **`PreFilterStrategy`** (Task 24, `vecdb/index/strategies.py`): wraps an already-built `FlatIndex` and delegates masked search to it unchanged — an exact masked brute-force scan, relabeled `"pre_filter"` in the returned `SearchResult`. No approximation: this is the ground-truth-correct, O(N) strategy that the other two are measured against.
- **`PostFilterStrategy`** (Task 25, appended to `vecdb/index/strategies.py`): wraps the hand-written `HNSWIndex`. `_ef_for(k, sel_hat)` computes an adaptive beam width `ef = clamp(alpha * k / sel_hat, ef_min, N)` with `alpha=4.0` by default, runs HNSW at that `ef`, then filters results down to the masked subset. If the masked result count is below `k`, it retries with a widened `ef` (up to `max_retries` times, `retry_multiplier` per retry); if it still can't fill `k` results after retries, it falls back to `PreFilterStrategy`'s exact scan and reports the combined distance-op cost of the failed HNSW attempts plus the fallback scan (verified honest in Task 25's own accounting trace). `fallback_count`/`query_count`/`fallback_rate` are tracked as running counters across calls.
- **Selectivity-estimator validation** (Task 26, `scripts/run_selectivity_validation.py`): 500 random predicates evaluated against both metadata variants, producing a true-vs-estimated selectivity log-log scatter for each and printing median/p95 relative error — the headline "measure your estimation error and put the scatter plot in the README" evidence the source plan calls out (§3.2). Figures: `results/figures/selectivity_estimation_uncorrelated.png`, `results/figures/selectivity_estimation_correlated.png`.
- **Selectivity-grid A/B sweep** (Task 27, `vecdb/bench/sweep.py` + `scripts/run_sweep_ab.py`): ran both strategies across the full 8-point selectivity grid (`{0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0}`) on both metadata variants (uncorrelated, correlated) at 100K scale, 200 queries per cell with the first 20 discarded as warmup (180 measured queries/cell), against a brute-force ground-truth cache (`results/gt_cache/`, 16 `.npy` files). Output: `results/sweep_uncorrelated.csv`, `results/sweep_correlated.csv` — 2,880 rows each (8 selectivities x 2 strategies x 180 queries).

No existing modules outside `vecdb/index/strategies.py` and the new `vecdb/bench/sweep.py` were touched; 76/76 tests passing at the end of Phase 5.

## Numbers

All numbers below are computed directly from `results/sweep_uncorrelated.csv` and `results/sweep_correlated.csv` via pandas (`groupby(['strategy','target_selectivity'])`), not copied from prior task reports.

### Uncorrelated metadata

| strategy | selectivity | recall (mean) | latency p50 (ms) | latency p95 (ms) | underfill rate | dist_ops (mean) |
|---|---|---|---|---|---|---|
| post_filter | 0.001 | 1.0000 | 645.28 | 812.14 | 0.00 | 61,774 |
| post_filter | 0.005 | 1.0000 | 131.97 | 222.80 | 0.00 | 27,056 |
| post_filter | 0.010 | 1.0000 | 66.41  | 124.54 | 0.00 | 18,017 |
| post_filter | 0.050 | 1.0000 | 22.90  | 29.43  | 0.00 | 6,242 |
| post_filter | 0.100 | 0.9994 | 12.45  | 14.44  | 0.00 | 3,866 |
| post_filter | 0.250 | 0.9972 | 2.56   | 6.23   | 0.00 | 1,997 |
| post_filter | 0.500 | 0.9922 | 3.17   | 3.52   | 0.00 | 1,178 |
| post_filter | 1.000 | 0.9733 | 1.74   | 2.13   | 0.00 | 712 |
| pre_filter  | 0.001 | 1.0000 | 0.023  | 0.024  | 0.00 | 100 |
| pre_filter  | 0.005 | 1.0000 | 0.044  | 0.048  | 0.00 | 500 |
| pre_filter  | 0.010 | 1.0000 | 0.078  | 0.087  | 0.00 | 1,000 |
| pre_filter  | 0.050 | 1.0000 | 1.18   | 1.94   | 0.00 | 5,000 |
| pre_filter  | 0.100 | 1.0000 | 2.18   | 2.75   | 0.00 | 10,000 |
| pre_filter  | 0.250 | 1.0000 | 5.22   | 5.85   | 0.00 | 25,000 |
| pre_filter  | 0.500 | 1.0000 | 9.85   | 10.44  | 0.00 | 50,000 |
| pre_filter  | 1.000 | 1.0000 | 16.10  | 18.61  | 0.00 | 100,000 |

### Correlated metadata

| strategy | selectivity | recall (mean) | latency p50 (ms) | latency p95 (ms) | underfill rate | dist_ops (mean) |
|---|---|---|---|---|---|---|
| post_filter | 0.001 | 1.0000 | 603.58 | 740.16 | 0.00 | 57,194 |
| post_filter | 0.005 | 1.0000 | 163.08 | 213.16 | 0.00 | 25,845 |
| post_filter | 0.010 | 1.0000 | 58.76  | 115.27 | 0.00 | 17,507 |
| post_filter | 0.050 | 1.0000 | 11.38  | 27.43  | 0.00 | 6,275 |
| post_filter | 0.100 | 1.0000 | 5.82   | 14.11  | 0.00 | 3,873 |
| post_filter | 0.250 | 0.9978 | 2.52   | 4.61   | 0.00 | 1,997 |
| post_filter | 0.500 | 0.9933 | 1.36   | 3.29   | 0.00 | 1,178 |
| post_filter | 1.000 | 0.9733 | 1.86   | 2.04   | 0.00 | 712 |
| pre_filter  | 0.001 | 1.0000 | 0.026  | 0.027  | 0.00 | 100 |
| pre_filter  | 0.005 | 1.0000 | 0.045  | 0.048  | 0.00 | 500 |
| pre_filter  | 0.010 | 1.0000 | 0.077  | 0.093  | 0.00 | 1,000 |
| pre_filter  | 0.050 | 1.0000 | 1.08   | 1.32   | 0.00 | 5,000 |
| pre_filter  | 0.100 | 1.0000 | 2.10   | 2.62   | 0.00 | 10,000 |
| pre_filter  | 0.250 | 1.0000 | 4.71   | 5.84   | 0.00 | 25,000 |
| pre_filter  | 0.500 | 1.0000 | 9.90   | 10.72  | 0.00 | 50,000 |
| pre_filter  | 1.000 | 1.0000 | 16.32  | 19.54  | 0.00 | 100,000 |

`underfill` is boolean per query (`n_returned < k` before any fallback resolves it); the mean over 180 queries per cell is the underfill rate. Both variants show **0.00 underfill rate at all 8 selectivities**, and correspondingly **0.00 fallback_rate at all 8 selectivities, both variants** (`fallback_rate` column, post-filter rows only) — see the Interview note (Q8) for why, and note the two are the same finding: fallback only fires after underfill survives all retries, so a flat 0.00 underfill curve necessarily produces a flat 0.00 fallback curve.

**Latency crossover.** Pre-filter's p50 scales linearly with selectivity (O(N·s)) — 0.023ms at s=0.001 up to ~16ms at s=1.0. Post-filter's p50 is dominated by the widened `ef` at low selectivity and only becomes cheap once selectivity is high enough that `ef` stays near its floor. In both metadata variants, the crossover where post-filter's p50 drops below pre-filter's p50 falls **between s=0.10 and s=0.25** (uncorrelated: pre=2.18ms/post=12.45ms at s=0.10, pre=5.22ms/post=2.56ms at s=0.25; correlated: pre=2.10ms/post=5.82ms at s=0.10, pre=4.71ms/post=2.52ms at s=0.25). This is consistent between the two metadata variants — the crossover location does not depend visibly on whether `category` correlates with vector-space clusters, only on selectivity.

### Selectivity-estimation error (Task 26, `scripts/run_selectivity_validation.py`, 500 random predicates/variant)

| metadata variant | median relative error | p95 relative error |
|---|---|---|
| uncorrelated | 0.008 | 0.467 |
| correlated | 0.005 | 0.466 |

The correlated variant does **not** show visibly larger estimation error than the uncorrelated one — both median and p95 are within noise of each other. This is expected once you look at how the correlated dataset is actually generated: its correlation is between `category` and vector-space clusters (via k-means), not between `category`/`year`/`score` themselves. The AND-independence assumption the estimator relies on (`P(A and B) ≈ P(A)·P(B)`) is a statement about correlation *between predicate columns*, and those columns remain independent of each other in both variants — only their relationship to the embedding geometry differs. That geometric correlation is what Phase 6's predicate-aware traversal cares about; it is orthogonal to what the selectivity estimator's independence assumption is exposed to.

## Gate status

**PASS.** Verified directly against the CSVs with pandas (`groupby(['strategy','target_selectivity']).size()`), not assumed:

- `results/sweep_uncorrelated.csv`: 2,880 rows total. Every one of the 8 target selectivities has exactly 180 rows for `pre_filter` and 180 rows for `post_filter` (200 queries/cell minus 20 warmup, as the plan's §4.3 selectivity grid specifies).
- `results/sweep_correlated.csv`: same shape, same complete 8x2x180 coverage.
- `results/figures/selectivity_estimation_uncorrelated.png` and `results/figures/selectivity_estimation_correlated.png` exist on disk (71,813 and 71,766 bytes respectively) — Task 26's headline scatter figures, referenced above, committed as part of this milestone.

Both CSVs are complete; no cell is missing, truncated, or padded.

## Interview note

**Q1 (source plan §7): Why not just filter after searching?**

The mechanism the plan predicts is: expected survivors after post-filtering is roughly `ef * s`, so to keep `k` results you need `ef >= k/s`, which explodes as `s -> 0`, and because it's probabilistic you sometimes return fewer than `k` (underfill). `PostFilterStrategy._ef_for` implements exactly this (`ef = clamp(alpha * k / sel_hat, ef_min, N)`, `alpha=4.0`), with a bounded retry loop and an exact-scan fallback as the safety net if retries don't fill `k`.

What the real 100K sweep shows is that the safety net worked *completely* — underfill rate is 0.00 at every one of the 8 selectivities, both metadata variants (table above). So post-filtering did not "break" in the sense of returning wrong-sized or wrong result sets anywhere in this sweep; recall stayed >=0.97 everywhere and hit 1.0 at every selectivity below 0.10. But it broke in the sense the plan actually cares about — cost. At `s=0.001`, `ef` widens toward its cap and post-filter's p50 latency is 645ms (uncorrelated) / 604ms (correlated) versus pre-filter's 0.02-0.03ms — roughly **27,000x** slower — and its dist_ops count (61,774 / 57,194) is over **600x** pre-filter's (100). That is the `ef >= k/s` explosion made concrete: instead of failing quietly (returning too few results), `alpha=4.0`'s aggressive default pushed `ef` up hard enough to *always* find enough survivors, at a distance-computation cost that approaches doing almost a full unfiltered graph search per query. The underfill curve is flat at zero precisely because the cost curve is doing the exploding instead — same mechanism, different observable.

**Q8 (source plan §7): What happens when the selectivity estimate is wrong?**

The bounded-harm claim from the plan is: all three strategies keep returning correct results even when `sel_hat` misses `true_selectivity`, just at higher cost, and post-filter specifically has a retry-and-fallback path for the case where the misestimate causes real underfill. Two separate pieces of evidence back this up, and they say different (compatible) things:

1. **The mechanism is unit-tested and works.** Task 25's `test_fallback_rate_is_tracked_across_calls` deliberately constructed an adversarial mask small enough that HNSW's widened `ef` still can't surface `k` matches, forcing `fallback_count`/`query_count` to produce `fallback_rate=1.0`. That test is the proof the fallback path is reachable and functions correctly when the estimate is badly enough wrong (or the true selectivity genuinely is that extreme) — it is not vaporware.

2. **In this 100K sweep, it was never needed.** `fallback_rate` is 0.00 at all 8 selectivities, both metadata variants — identical to the underfill rate, because fallback only triggers once underfill survives every retry, and underfill never survived even the first attempt here. The reason is algebraic, not a sweep artifact: with `alpha=4.0` and `k=10`, expected survivors under `_ef_for`'s unclamped formula is `alpha*k = 40` regardless of `sel_hat`, roughly 4.8 standard deviations above the `k=10` cutoff for a hypergeometric-ish survival process — an underfill event is a ~1e-6-probability tail event per query at these default settings, not something 180 queries/cell should be expected to hit even once. Selectivity estimates being off (Task 26's own p95 relative error of ~0.467 on both variants shows the estimator genuinely does have a heavy tail) does perturb `ef`, but `alpha=4.0`'s safety margin absorbs that perturbation without ever tripping the retry/fallback machinery in this sweep.

Put together: "bounded harm" holds, but the honest picture is that in this benchmark the bound was never actually tested by real data — the harm bound is proven by construction (the unit test) and by the survival-probability argument, not by an observed rising fallback curve. A sweep at a much smaller `alpha` (or one deliberately using a badly biased estimator) would be needed to see the fallback path fire under real sweep conditions; that is a natural follow-up, not something this report should paper over as already demonstrated end-to-end.
