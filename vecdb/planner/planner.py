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
    planner is far more compelling in a demo than a black box (spec §3.2).

    Two construction modes:
    - `Planner(params)` (Task 36): the calibrated cost-model path. `.plan()` computes
      cost_pre/cost_post/cost_pred and picks the argmin.
    - `Planner.from_lookup_table(lookup_table)` (Task 38 follow-up): a decile lookup-table
      fallback, built empirically from measured sweep data instead of a cost formula — see
      spec §8's "cost model doesn't fit" risk. `.plan()` in this mode does not touch
      CostModelParams or the cost_* functions at all; it just looks up the nearest
      calibration selectivity in `lookup_table` and returns its recorded strategy.
    """

    def __init__(self, params: CostModelParams):
        self.params = params
        self._lookup_table: dict[float, str] | None = None

    @classmethod
    def from_lookup_table(cls, lookup_table: dict[float, str]) -> "Planner":
        """Build a Planner that dispatches by nearest-selectivity lookup instead of a cost
        model. `lookup_table` maps a calibration selectivity (float) to the strategy name
        empirically fastest there. Nearest match is by absolute difference |sel_hat - key|,
        ties broken by Python's min() (first key encountered in iteration order)."""
        self = cls.__new__(cls)
        self.params = None
        self._lookup_table = dict(lookup_table)
        return self

    def plan(self, k: int, sel_hat: float) -> ExecutionPlan:
        if self._lookup_table is not None:
            return self._plan_lookup(sel_hat)
        costs = {
            "pre_filter": cost_pre(self.params, k, sel_hat),
            "post_filter": cost_post(self.params, k, sel_hat),
            "predicate_aware": cost_pred(self.params, k, sel_hat),
        }
        best = min(costs, key=costs.get)
        others = ", ".join(f"{name}={cost:.0f}" for name, cost in costs.items() if name != best)
        reason = f"{best}: ŝ={sel_hat:.4f} -> cost={costs[best]:.0f} < {others}"
        return ExecutionPlan(strategy=best, reason=reason, sel_hat=sel_hat, costs=costs)

    def _plan_lookup(self, sel_hat: float) -> ExecutionPlan:
        nearest = min(self._lookup_table, key=lambda point: abs(point - sel_hat))
        strategy = self._lookup_table[nearest]
        reason = (
            f"lookup_table: ŝ={sel_hat:.4f} -> nearest calibration point "
            f"{nearest:.4f} -> {strategy} (empirically fastest there)"
        )
        return ExecutionPlan(strategy=strategy, reason=reason, sel_hat=sel_hat, costs={})
