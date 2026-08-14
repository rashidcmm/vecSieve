# Phase 6 Milestone: Predicate-aware traversal (Strategy C)

## What got built

Tasks 29–32 built `FilteredHNSWStrategy` (Strategy C from the source plan) — the graph-native
alternative to brute-force pre-filtering (Strategy A) and post-hoc masked HNSW (Strategy B) —
ran it across the full selectivity grid on both metadata variants, and (Task 32's follow-up)
root-caused why it does not win the Phase 6 gate.

- **Two-tier admission and seeded entry points** (Task 29, `HNSWIndex._search_layer_filtered`,
  `vecdb/index/strategies.py`): a node is *visitable* if it's anywhere in the graph but only
  *admissible to results* if `mask[node]` is true — the search traverses through non-matching
  nodes to preserve connectivity but never returns them. Layer-0 search is seeded with
  `n_seed_matches=8` randomly sampled matching nodes (from `mask.nonzero()`) alongside the
  normal hierarchical entry point, as cheap insurance against starting stranded in a
  match-free region. `_ef_eff(sel_hat)` widens the beam as selectivity drops.
- **Two-hop expansion** (Task 30): ACORN-style densification — when the match rate among a
  node's just-expanded neighbours drops below `two_hop_threshold=0.1`, the search additionally
  expands neighbours-of-neighbours for that node, injecting more candidates into the beam.
  Guarded by the same budget check as ordinary expansion.
- **Dynamic ef** (Task 29): `_ef_eff = ef_base * min(4.0, 1 / max(sel_hat, 0.05))`, i.e. the
  admission-heap capacity widens up to 4x `ef_base` (`ef_base=64` by default, so up to 256) as
  estimated selectivity shrinks — the same "beam must be wider because matches are scarce"
  idea as `PostFilterStrategy`'s `alpha*k/sel_hat`, but applied to a filtered admission quota
  rather than to a post-hoc-discard beam.
- **Budget cap and honest bail-out** (Task 31): distance-ops are hard-capped at
  `budget_fraction * len(store)` (default 0.3, i.e. 30% of the dataset) per query. If the
  budget is exhausted before `k` matches are admitted, the strategy bails out honestly to its
  `fallback` (`PreFilterStrategy`), reports the combined cost of the wasted traversal plus the
  fallback scan, and relabels the result `"predicate_aware_fallback"`. `bail_count` /
  `query_count` / `bail_rate` are tracked as running counters, mirroring `PostFilterStrategy`'s
  own fallback accounting from Phase 5.
- **Selectivity-grid sweep, all three strategies** (Task 32, `scripts/run_sweep_all.py`,
  generalizing Task 27's A/B-only driver): ran `pre_filter`, `post_filter`, and
  `predicate_aware` across the full 8-point selectivity grid on both metadata variants at 100K
  scale, 200 queries/cell with 20 discarded as warmup (180 measured/cell). Output:
  `results/sweep_uncorrelated.csv`, `results/sweep_correlated.csv` — 4,320 rows each
  (8 selectivities x 3 strategies x 180 queries).
- **Follow-up debugging investigation** (Task 32's debug pass, not a code change): instrumented
  re-run of `_search_layer_filtered` at `s=0.01` on 8 real queries to determine *why*
  `predicate_aware` runs to (near) its budget cap at low/mid selectivity. Result: no
  implementation bug — see Gate status below.

No existing modules outside `vecdb/index/hnsw.py`, `vecdb/index/strategies.py`,
`scripts/run_sweep_all.py`, and their tests were touched. 82/82 tests passing at the end of
Phase 6 (re-verified for this report).

## Numbers

All numbers below are taken directly from `results/sweep_uncorrelated.csv` and
`results/sweep_correlated.csv` (independently recomputed via pandas `groupby` for this report,
not copied from the prior task reports without re-checking).

### Gate check — uncorrelated metadata

Gate (source plan §5 Day 5): `predicate_aware`'s p95 latency should be below **both**
`pre_filter` and `post_filter` for **at least two** selectivity values in the middle of the
range, at recall@10 >= 0.90.

| s | pre_filter p95 (ms) | post_filter p95 (ms) | predicate_aware p95 (ms) | predicate_aware recall | beats both? |
|---|---|---|---|---|---|
| 0.001 | 0.031 | 396.592 | 80.497 | 0.986 | No (loses to pre) |
| 0.005 | 0.047 | 93.850 | 78.807 | 1.000 | No (loses to pre) |
| 0.01  | 0.119 | 56.571 | 78.832 | 1.000 | No (loses to both) |
| 0.05  | 1.095 | 10.513 | 67.118 | 1.000 | No (loses to both) |
| 0.1   | 1.944 | 6.906  | 50.172 | 1.000 | No (loses to both) |
| 0.25  | 4.284 | 2.524  | 23.771 | 1.000 | No (loses to both) |
| 0.5   | 7.913 | 1.356  | 5.365  | 1.000 | No (loses to post; beats pre) |
| 1.0   | 14.654| 0.781  | 1.561  | 0.992 | No (loses to post; beats pre) |

**`predicate_aware` never beats both fixed strategies at any of the 8 selectivity points on
uncorrelated metadata.** There is no selectivity in this grid where it wins — not "just barely
misses the two-point minimum," but zero wins anywhere. Recall is never the limiting factor:
`predicate_aware`'s recall@10 is >= 0.986 everywhere and 1.000 at 6 of 8 points, comfortably
above the 0.90 floor the gate requires.

### Correlated metadata

| s | pre_filter p95 (ms) | post_filter p95 (ms) | predicate_aware p95 (ms) | predicate_aware recall | beats both? |
|---|---|---|---|---|---|
| 0.001 | 0.022 | 381.944 | 79.786 | 0.996 | No (loses to pre) |
| 0.005 | 0.047 | 88.803  | 78.032 | 1.000 | No (loses to pre) |
| 0.01  | 0.075 | 47.684  | 77.043 | 1.000 | No (loses to both) |
| 0.05  | 1.124 | 10.458  | 63.976 | 1.000 | No (loses to both) |
| 0.1   | 1.939 | 5.818   | 47.811 | 1.000 | No (loses to both) |
| 0.25  | 4.243 | 2.478   | 24.568 | 1.000 | No (loses to both) |
| 0.5   | 8.277 | 1.454   | 6.042  | 0.999 | No (loses to post; beats pre) |
| 1.0   | 14.227| 0.863   | 1.520  | 0.992 | No (loses to post; beats pre) |

Same story: zero wins on correlated metadata either. Comparing the two variants directly,
`predicate_aware`'s p95 latency is close between the two at 7 of 8 selectivities, but not
uniformly in one direction: correlated is faster at 6 of 8 points (s=0.001, 0.005, 0.01, 0.05,
0.1, 1.0), by 1-5%. At the other two points correlated is *slower*: s=0.25 (24.57ms correlated
vs 23.77ms uncorrelated, ~3.4% slower) and, most notably, **s=0.5 — the single largest gap
anywhere in the grid — where correlated p95 is 6.042ms vs uncorrelated's 5.365ms, ~12.6%
slower**. Recall is close between the two variants everywhere (within 0.01 at every point).
**Correlated metadata is not uniformly worse for `predicate_aware` in this run** — it is faster
at most selectivities, but the largest single directional swing (s=0.5) runs the other way, with
correlated metadata costing more latency there, not less (see Interview note for the full
side-by-side and the bail-rate picture).

### Bail rate

`strategy` in both CSVs takes on exactly three values — `pre_filter`, `post_filter`,
`predicate_aware` — with no `predicate_aware_fallback` rows in either file (checked directly:
`df['strategy'].unique()` on both CSVs). `predicate_aware`'s `underfill` column is `False` for
every one of the 1,440 rows in each file. **Bail rate is 0.0 at every selectivity, on both
metadata variants, in this sweep** — the budget cap (30% of the 100K store = 30,000 dist_ops)
was never exceeded without first admitting `k=10` matches, even at the sparsest selectivity
(s=0.001, ~100 matching rows). This is a fact of this run, not a claim that the bail-out
mechanism is untested or unreachable: Task 31's own unit tests (`test_bails_out_to_fallback_...`)
construct an adversarial small-budget scenario specifically to prove the bail-out path fires
and behaves honestly when it does. It just never had to, here.

## Gate status

**FAIL.** `predicate_aware` never simultaneously beats both `pre_filter` and `post_filter` on
p95 latency at any of the 8 selectivity points, on either metadata variant — not the "two
points in the middle" the gate asks for, but zero points anywhere. Recall was never the
limiting factor (>= 0.986 everywhere, mostly 1.000, versus the 0.90 floor). This is a clean,
unambiguous negative result on latency alone.

### Why (the debugging investigation's finding, not a hand-wave)

Following the source plan's own Day-5 instruction ("if C never wins anywhere... after honest
debugging it still loses everywhere, that is a publishable finding"), Task 32's follow-up
instrumented `_search_layer_filtered` at `s=0.01` on 8 real queries against the actual 100K
index, before concluding anything. The result: **no implementation bug.** Specifically checked
and confirmed correct:

- The break condition (`d_c > worst_d and len(results) >= ef`) is structurally identical to the
  already-correct, already-reviewed `_search_layer`. No off-by-one, no stale-visited-set reuse.
- The two-hop trigger (`match_rate < two_hop_threshold`) fires exactly when documented, scoped
  correctly to the just-processed node's immediate neighbours.
- The min-heap ordering is never violated — two-hop candidates never "jump the queue" ahead of
  closer ones.

The real, mechanistic cause is a **design-level double compensation for selectivity**:
`_ef_eff`'s beam-widening formula (`ef_base * min(4, 1/max(s, 0.05))`) inflates the admission
*quota* to 256 (its ceiling, saturating for every `s <= 0.25`) at low/mid selectivity. But the
two-tier admission mechanism already compensates for scarcity *during* traversal, by walking
through non-matches and via two-hop densification — so requiring the results heap to
additionally fill to a 1/selectivity-scaled quota before the natural stopping bound is even
evaluated pays for the same scarcity twice. At `s=0.01`, reaching that 256-match quota (25.6%
of all 1,000 matching rows in the entire 100K index) takes 22,000-26,000 of the ~30,000-op
budget across the 8 sampled queries — by which point there is essentially no room left for the
natural break to fire before the budget cap trips instead. Two-hop expansion is not
misbehaving; it correctly triggers on the vast majority of iterations (1,200-2,400 times per
query, ~88% of all distance ops) because at ~1% global match rate a node's local match rate is
almost always below the 10% two-hop threshold — that is the intended, working densification
mechanism, and it is also the dominant cost of reaching the inflated quota.

**Counterfactual verification, not just theory**: re-running the identical 8 queries with
`ef=k=10` instead of `ef=ef_eff=256` terminated naturally every time (12,996-22,906 ops, well
under budget) and returned the **exact same top-10 set** as the wide, budget-capped run in
**8/8 queries**. The extra 246 admitted slots (ranks 11-256) never once displaced anything in
the final top-10 for these queries. In plain terms: at this selectivity, the wide beam buys
zero additional recall while dominating the cost — the quota, not the two-hop mechanism's
correctness, is the binding cost driver.

This mirrors the CSV numbers directly: `predicate_aware`'s mean dist_ops (~13.8k-30.1k across
the mid/low range, s=0.01-0.25; it never exceeds ~30,200 anywhere in the full 8-point grid, on
either variant) are actually competitive with or better than `post_filter`'s at the sparse end, but each
dist-op is a Python-level distance call versus `pre_filter`'s single vectorized numpy matmul
over the masked rows — three orders of magnitude cheaper per unit of work at low selectivity —
so being dist-op-competitive with `post_filter` is not enough to win against `pre_filter`'s
vectorized path. There is no selectivity band in this grid where `predicate_aware`
simultaneously beats the vectorized brute-force baseline and the unfiltered-graph baseline.

Per the debugging report's scope, no fix was attempted or applied — `_ef_eff`'s formula,
`two_hop_threshold`, and `budget_fraction` were explicitly out of scope for tuning in both Task
32 and its follow-up. `vecdb/index/hnsw.py` and `vecdb/index/strategies.py` are unchanged from
what Task 31 committed.

### This is the plan's own documented, acceptable outcome

Source plan §6 risk register: *"Predicate-aware never wins | Medium | Ship it as a measured
negative result with the dist_ops accounting. Genuinely fine — see Day 5."* Day 5 itself: *"If
after honest debugging it still loses everywhere, that is a publishable finding... An honest
negative result defended with numbers beats a fabricated win, and interviewers can smell the
difference."* Both conditions are met here: the debugging pass happened first and found no
bug, and this report gives the dist-ops accounting the plan asks for. This is not spun as a
partial win — it is a clean gate failure with a mechanistic, verified explanation.

## Interview note

**Q11 (source plan §7): What breaks if attributes correlate with vector position?**

The plan's a-priori prediction is specific: *"Predicate-aware traversal degrades badly: greedy
descent enters match-free regions and strands. Pre-filter improves. Show both charts side by
side."* That prediction describes a **catastrophic stranding failure mode** — the search
walking into a region of the graph with no matches and having no way out except the budget cap
firing and a fallback (or an outright failure to reach `k` results).

**That failure mode did not occur in this run.** Side by side, uncorrelated vs. correlated:

| metric | uncorrelated | correlated |
|---|---|---|
| bail rate (all 8 selectivities) | 0.0 | 0.0 |
| predicate_aware p95, s=0.001 | 80.497ms | 79.786ms |
| predicate_aware p95, s=0.01  | 78.832ms | 77.043ms |
| predicate_aware p95, s=0.1   | 50.172ms | 47.811ms |
| predicate_aware p95, s=0.5   | 5.365ms  | 6.042ms  |
| predicate_aware recall, s=0.001 | 0.986 | 0.996 |
| predicate_aware recall, s=1.0   | 0.992 | 0.992 |

Bail rate is identically 0.0 on both variants at every selectivity — the budget cap was never
exceeded without first admitting `k` matches, on either metadata variant. Across the full 8-point
grid, correlated is faster than uncorrelated at 6 of 8 points (s=0.001, 0.005, 0.01, 0.05, 0.1,
1.0; e.g. 47.81ms vs 50.17ms at s=0.1, 77.04ms vs 78.83ms at s=0.01), each by roughly 1-5%. It is
slower at the other two: s=0.25 (24.57ms vs 23.77ms, ~3.4% slower) and, the largest single gap
anywhere in the grid, **s=0.5, where correlated is ~12.6% slower (6.042ms vs 5.365ms)** — shown
in the table above. So the direction is not uniform, though the effect size stays modest (at most
~13%) relative to the multi-order-of-magnitude latency gap between strategies driving the gate
failure itself. Recall is within 0.01 of each other everywhere, if anything slightly higher for
correlated at the sparsest point (0.996 vs 0.986 at s=0.001). Pre-filter does not "improve" relative to
predicate-aware under correlation in any way this data shows beyond what it already does under
uncorrelated metadata — its own numbers barely move between variants either (e.g. s=1.0:
14.654ms uncorrelated vs 14.227ms correlated).

**Why the theoretical failure mode didn't show up here**: two-tier admission (Task 29) plus
seeded entry points plus two-hop expansion (Task 30) are exactly the mitigations the plan
itself prescribes against greedy-descent stranding, and this correlated dataset's correlation
(`category` tied to vector-space k-means clusters, per Task 26's construction) evidently isn't
extreme enough, combined with those mitigations, to produce a region the search can't escape
within budget — seeded matching entry points in particular mean the search is never solely
dependent on greedy descent from a single, possibly mismatched, global entry point. The honest
answer to Q11 for *this implementation, on this dataset, at this budget*: nothing catastrophic
breaks — bail rate stays at 0.0, and while p95 latency moves by up to ~13% between variants at
individual selectivities (in both directions, not consistently favoring one variant), that is
far from the order-of-magnitude degradation ("degrades badly... strands") the plan's prediction
describes. The mechanism the plan warns about (stranding) is a real risk the
mitigations were built to address, and on this workload they succeed at preventing it; the
gate still fails, but for the unrelated reason documented above (the admission-quota /
dist-ops-cost mismatch), not for stranding. Answering this question honestly means reporting
that the specific theoretical failure mode in the plan's prediction did not manifest in the
measured data — not inventing a stranding narrative to match the plan's expectation when the
numbers don't show one.
