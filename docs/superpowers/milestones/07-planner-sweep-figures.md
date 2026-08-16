# Phase 7 Milestone: Query planner, full sweep, and the headline figures

## What got built

Tasks 34-39 built the cost-model query planner, calibrated it against measured data, ran
the full 4-executor sweep across the whole selectivity grid on both metadata variants, and
produced the five headline figures plus (as an unplanned but plan-sanctioned follow-up) a
lookup-table fallback planner that rescues the gate the cost-model planner failed.

- **Cost model** (`vecdb/planner/cost_model.py`, Task 34): `CostModelParams` dataclass plus
  three cost functions — `cost_pre = c_scan * N * sel_hat` (pre-filter's scanned-element
  count), `cost_post = c_hop * M * ef_required(alpha, k, sel_hat)` (post-filter's
  hop-proxy, `ef_required` clipped to `[ef_min, N]`), `cost_pred = c_hop * M * ef_base *
  gamma(sel_hat, beta, gamma_cap)` (predicate-aware's hop-proxy with a selectivity-scaled
  inflation term, capped at `gamma_cap`).
- **Calibration** (`scripts/calibrate.py`, Task 35): least-squares fit of `c_scan`, `c_hop`,
  `beta` against measured `dist_ops` from `results/sweep_uncorrelated.csv` (Phase 6's
  sweep). `alpha`, `ef_min`, `ef_base`, `M`, `gamma_cap` are fixed design constants, not
  fit. Written to `results/calibration.json`.
- **Planner with reason strings** (`vecdb/planner/planner.py`, Task 36): `Planner(params)`
  computes all three costs from an estimated selectivity `ŝ`, picks the argmin, and returns
  an `ExecutionPlan` carrying a human-readable `reason` string naming the chosen strategy
  and the losing costs (e.g. `"pre_filter: ŝ=0.0008 -> cost=83 < post_filter=79327,
  predicate_aware=219"`).
- **Regret measurement** (`vecdb/bench/regret.py`, Task 37): `compute_regret` /
  `regret_summary` compare the planner's realized per-query latency against the
  hindsight-best of the three fixed strategies for that same query, reporting mean and p95
  regret in milliseconds.
- **The full sweep** (`scripts/run_full_sweep.py`, Task 38): 8 selectivities x 2 metadata
  variants x 4 executors (3 fixed + planner) x 200 queries (180 measured after a 20-query
  warmup). Regenerated `results/sweep_uncorrelated.csv` and `results/sweep_correlated.csv`
  with a populated `planner_reason` column on the `planner` rows.
- **Lookup-table fallback planner** (`vecdb/planner/planner.py`'s `Planner.from_lookup_table`
  and `scripts/evaluate_lookup_planner.py`, Task 38 follow-up, commit `fa994c7`): the plan's
  own risk-register mitigation for "cost model doesn't fit" (source plan §6). Buckets `ŝ`
  and picks the empirically-fastest strategy per bucket from an 8-entry table fit on a
  calibration half of `results/sweep_uncorrelated.csv` (`query_idx < 110`), evaluated on the
  disjoint held-out half (`query_idx >= 110`). Purely additive to `Planner` — the original
  `Planner(params)` cost-model constructor and `.plan()` branch are byte-identical to
  before.
- **The five figures** (`scripts/generate_figures.py`, Task 39, this task):
  `results/figures/crossover.png`, `crossover_correlated.png`, `recall_vs_selectivity.png`,
  `underfill.png`, `dist_ops.png`. (`selectivity_estimation_{correlated,uncorrelated}.png`
  and `pareto_unfiltered.png` already existed from Phases 4-5.)

## Numbers

### Calibrated cost-model constants (Task 35, `results/calibration.json`)

| constant | value | fit or fixed |
|---|---|---|
| `c_scan` | 1.0 | fit (least squares, `pre_filter` rows) |
| `c_hop` | 0.10286764330640595 | fit (least squares, `post_filter` rows) |
| `alpha` | 4.0 | fixed design constant |
| `beta` | 0.0008928984488808203 | fit (linear regression, `predicate_aware` rows) |
| `ef_min` | 16 | fixed design constant |
| `ef_base` | 64 | fixed design constant |
| `M` | 16 | fixed design constant |
| `N` | 100000 | fixed (dataset size) |
| `gamma_cap` | 50.0 | fixed design constant |

`c_scan` recovered exactly `1.0`, which Task 35 read as a strong sanity check: `pre_filter`'s
`dist_ops` is `N * true_selectivity` by construction, so a least-squares fit against exactly
that quantity should — and did — recover a unit coefficient with no scaling bug.

### Regret (Task 37's metric, computed by Task 38 on the full sweep)

Two distinct planners were evaluated against this metric, on two different slices of data,
and they diverge completely. Both are reported below — see "Gate status" for why neither
number alone tells the full story.

**Cost-model planner** (`Planner(params)`, the actual `planner` rows in
`results/sweep_uncorrelated.csv` / `sweep_correlated.csv`, i.e. what `crossover.png` plots
as "planner (chosen)"), full uncorrelated sweep (all 180 measured queries x 8 selectivities):

- `mean_regret_ms`: **29.78**
- `p95_regret_ms`: **71.75**

**Lookup-table fallback planner** (`Planner.from_lookup_table(...)`, evaluated by
`scripts/evaluate_lookup_planner.py` on the held-out evaluation half of the same
`sweep_uncorrelated.csv`, `query_idx >= 110`, 90 of the 180 queries — not a new sweep run,
pure analysis over already-collected data, and not plotted in any figure):

- `mean_regret_ms`: **1.8333e-05** (~0)
- `p95_regret_ms`: **0.0**

### Mean latencies and the gate check — cost-model planner (full uncorrelated sweep)

| strategy | mean latency (ms) |
|---|---|
| pre_filter | 3.660249028052931 |
| predicate_aware | 37.987278680621884 |
| planner (cost-model, "planner (chosen)" in crossover.png) | 30.68058743063173 |
| post_filter | 66.67915305553025 |

Gate (spec §1.2 objective 5): `mean_planner < min(mean_fixed.values())`.
`30.68 < 3.66` is **False**. The cost-model planner beats two of the three fixed strategies
(`post_filter`, `predicate_aware`) but loses to `pre_filter` by roughly 8x, because it picks
`predicate_aware` for 7 of the 8 selectivity points (see the "where the crossover lands"
section below) — a strategy that, per the numbers, never actually wins.

### Mean latencies and the gate check — lookup-table fallback planner (held-out evaluation half)

| strategy | mean latency (ms), `query_idx >= 110` |
|---|---|
| pre_filter | 3.6410416665603407 |
| predicate_aware | 37.62661263879434 |
| lookup-table planner | 0.9046354165346015 |
| post_filter | 66.17494999987281 |

Gate: `0.9046354165346015 < 3.6410416665603407` is **True** — the lookup-table planner is
roughly 4x *faster* than the best fixed strategy on data it was never fit on (calibration
was on `query_idx < 110`; this is the disjoint `query_idx >= 110` half). A secondary,
exploratory check applying the same uncorrelated-fitted table to
`results/sweep_correlated.csv`'s evaluation half also holds: 0.895855ms vs. 3.573200ms.

**This lookup-table result has no line of its own in any figure and no rows in the sweep
CSVs** — it was never run as a new 100K-scale sweep, only evaluated by calibration/held-out
analysis over the sweep data already collected in Task 38. `crossover.png`'s "planner
(chosen)" line is the cost-model planner only.

### Where the crossover chart's strategy changes actually land, vs. the plan's expected shape

Reading `results/figures/crossover.png` and `crossover_correlated.png` (p95 latency, log-log,
one line per fixed strategy plus the cost-model planner) directly off the underlying p95
numbers:

| target_selectivity | uncorrelated winner (p95) | correlated winner (p95) |
|---|---|---|
| 0.001 | pre_filter | pre_filter |
| 0.005 | pre_filter | pre_filter |
| 0.01 | pre_filter | pre_filter |
| 0.05 | pre_filter | pre_filter |
| 0.1 | pre_filter | pre_filter |
| 0.25 | post_filter | post_filter |
| 0.5 | post_filter | post_filter |
| 1.0 | post_filter | post_filter |

Source plan §5 Day 6's expected shape:
- `s < ~1%`: pre-filter wins.
- `~1% < s < ~30%`: predicate-aware wins on uncorrelated data.
- `s > ~30%`: post-filter wins.
- Correlated data: the middle (predicate-aware) region shrinks or vanishes; pre-filter's
  region extends further right.

**Measured reality diverges from this on the middle band.** `pre_filter` wins the entire
low-and-mid range (s=0.001 through s=0.1, i.e. up to 10%, further right than the expected
~1%), and `post_filter` wins the entire high range (s=0.25 through s=1.0). There is exactly
**one** crossover — between s=0.1 and s=0.25 — not two. `predicate_aware` never wins at any
of the 8 grid points on either metadata variant, so the expected middle "predicate-aware
wins" band does not appear at all: this is consistent with, and further confirms, Phase 6's
own gate-failure finding (`docs/superpowers/milestones/06-predicate-aware-traversal.md`) that
`predicate_aware` never wins anywhere in this implementation. The correlated chart is nearly
identical in shape to the uncorrelated one (same single crossover, same location) rather than
showing a visibly shrunk/vanished middle band or an extended pre-filter region — because
there was no predicate-aware-winning middle band in the uncorrelated chart to begin with, it
can't visibly shrink further under correlation.

I also checked the `dist_ops.png` (hardware-independent op-count) figure for the same
question, since Task 38's cost-model diagnosis attributed the planner's error to comparing
op-counts across strategies with very different real cost-per-op. Even in pure op-count
terms, `predicate_aware` never has the lowest mean `dist_ops` at any of the 8 points either —
the crossover in `dist_ops.png` is also a single pre_filter -> post_filter switch, in the
same s=0.05-to-0.1 to s=0.25 neighborhood. So the absence of a predicate-aware-winning region
is not just a per-op-cost-normalization artifact of wall-clock latency; it holds by the
hardware-independent metric too, in this run.

## Gate status

Two gates are in play here, evaluated on two different planners over two different slices of
data. Reporting both plainly, without blending them into one verdict:

**Phase 7 gate, cost-model planner (spec §1.2 objective 5, as literally swept and plotted in
`crossover.png`'s "planner (chosen)" line): FAILED.**
`mean_planner (30.68 ms)` is not `< min(mean_fixed) (3.66 ms, pre_filter)` on the full
uncorrelated sweep. Root cause (diagnosed in Task 38's report, reconfirmed above via the
`dist_ops` check): the calibrated cost formula ranks `predicate_aware` as cheaper than
`pre_filter` for 7 of 8 selectivity points, but `predicate_aware` is never actually the
fastest strategy anywhere in this implementation — its Python-level per-hop traversal cost
does not scale the way the linear `c_hop * hops` term assumes relative to `pre_filter`'s
single vectorized NumPy scan.

**Phase 7 gate, lookup-table fallback planner (`Planner.from_lookup_table`, spec §6's
documented "cost model doesn't fit" mitigation, evaluated by
`scripts/evaluate_lookup_planner.py` on the held-out half of already-collected sweep data,
not plotted in any figure): PASSED.**
`mean_planner_eval (0.9046354165346015 ms)` is `< min(mean_fixed_eval) (3.6410416665603407
ms, pre_filter)` on 90 held-out queries never used to fit the table — roughly a 4x win, and a
complete reversal of the cost-model planner's ~8x loss. Regret on that held-out half is
essentially zero (1.83e-5 ms mean, 0.0 p95).

**Net honest verdict**: the planner *as literally swept, plotted, and shipped in
`crossover.png`* — the `Planner(params)` cost-model path — does not clear the Phase 7 gate.
The plan's own documented fallback for exactly this failure mode does clear it, on held-out
data, as an unplanned but plan-sanctioned follow-up analysis rather than a new sweep run. Both
facts are true and neither should be quoted without the other.

**Day 6's separate "Done when" gate** (`crossover.png` exists, shows the winning strategy
changing at least twice, and the planner line tracks the lower envelope) also does **not**
fully hold, checked by eye against `results/figures/crossover.png` and against the table
above: `crossover.png` exists and the cost-model planner line does track the lower envelope
at the two extremes (s=0.001 and s=1.0, where it matches the winning fixed strategy) but
diverges sharply from it in the middle (s=0.005 through s=0.5, where it tracks
`predicate_aware`'s losing line instead). The winning-strategy-among-fixed-strategies changes
**once**, not "at least twice" — pre_filter to post_filter, between s=0.1 and s=0.25. This is
reported honestly rather than cherry-picking the plotted range to manufacture a second
crossing that isn't in the data.

## Interview note

**Q7 (source plan §7): How do you estimate selectivity without touching the data?**

Precomputed value counts and histograms per attribute, built once at index time
(`vecdb/predicate/estimator.py`, Phase 4-5) — no per-query scan of the actual vector or
metadata store. For a leaf predicate (`eq`, `in`, range comparisons) the estimator looks up
the attribute's histogram and returns the fraction of rows matching. For a conjunction
(`and`), the estimator multiplies per-clause selectivities together, i.e. assumes
independence between attributes. I measured how wrong that assumption is in
`results/figures/selectivity_estimation_uncorrelated.png` and
`selectivity_estimation_correlated.png` — an estimated-vs-true selectivity scatter across the
grid — and I name the independence assumption explicitly here rather than let it hide inside
the formula: it is an approximation, not a guarantee, and its error is largest exactly when
attributes are correlated with each other (which is why the correlated-metadata variant
exists as its own experiment, not merely a robustness check).

**Q8 (source plan §7): What happens when the estimate is wrong?**

Bounded harm by construction: `ŝ` only ever selects *which* of three strategies runs, not
whether the result is correct. All three fixed strategies return valid top-`k` results
regardless of the true selectivity — the recall floor tracks the strategy actually chosen and
its own parameters (e.g. `ef_eff`), not the accuracy of `ŝ`. Concretely, `results/figures/recall_vs_selectivity.png`
shows every strategy at or near recall@10 = 1.0 across almost the whole grid; the one place
harm shows up is `post_filter`'s underfill path (fewer than `k` results returned when too few
survivors pass the mask), which `results/figures/underfill.png` tracks directly — and in this
run that underfill rate measured exactly `0.0` at every one of the 8 selectivity points, so
even that particular harm channel did not fire on this data. Where a wrong `ŝ` *does* show up
is in cost: my own p95 regret numbers quantify this precisely. The cost-model planner's
`p95_regret_ms = 71.75` (mean `29.78`) on the full uncorrelated sweep is the actual measured
price of picking a suboptimal-but-still-correct strategy, i.e. of the estimate (or, in this
case, more precisely the cost model's ranking of strategies given the estimate) being "wrong"
in the cost-optimization sense rather than in the correctness sense. The lookup-table
fallback's regret on held-out data (`p95_regret_ms = 0.0`, mean `~1.8e-5`) shows the same
bounded-harm property holding much more tightly once the ranking mechanism itself is fixed to
match measured reality.

**Q9 (source plan §7): How did you fit the cost model constants?**

Least squares (`scripts/calibrate.py`, Task 35) against measured `dist_ops` from
`results/sweep_uncorrelated.csv`, the Phase 6 sweep — not guessed, and not fit against
latency (fit against the hardware-independent op count, deliberately, so the constants
describe algorithmic cost rather than absorbing machine-specific noise). `c_scan` is fit by
regressing `pre_filter` rows' `dist_ops` against `N * true_selectivity`; it recovered exactly
`1.0`, which is both the expected value (`pre_filter`'s `dist_ops` is defined as `N *
true_selectivity`) and a sanity check that the CSV's `true_selectivity` and `dist_ops`
columns have no unit or scaling bug upstream. `c_hop` is fit the same way against
`post_filter` rows' `dist_ops` versus an `M * ef_required(...)` proxy, recovering
`~0.1029`. `beta` is fit by linear regression of normalized `predicate_aware` `dist_ops`
against `(1/s - 1)`, recovering `~0.000893`. `alpha`, `ef_min`, `ef_base`, `M`, `gamma_cap`
are fixed design constants, not fit — they're architectural choices (HNSW's own `M`, the
beam-widening ceiling) rather than quantities with a clean single-parameter regression
target. All nine values are persisted to `results/calibration.json` and reloaded by
`Planner(CostModelParams(**json.load(...)))` at init, so the planner never re-derives them at
runtime. **The caveat this milestone's own gate result adds to Q9's textbook answer**: fitting
constants by least squares against `dist_ops` guarantees the model matches the *op-count*
data it was fit against; it does not guarantee the resulting cost *ranking* between
strategies matches *wall-clock* reality when the strategies' cost-per-op differs by orders of
magnitude (a vectorized NumPy scan vs. Python-level graph hops) — which is exactly what went
wrong here and is why the lookup-table fallback exists as the plan's documented mitigation
for this specific case.
