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
    planner is far more compelling in a demo than a black box (spec §3.2)."""

    def __init__(self, params: CostModelParams):
        self.params = params

    def plan(self, k: int, sel_hat: float) -> ExecutionPlan:
        costs = {
            "pre_filter": cost_pre(self.params, k, sel_hat),
            "post_filter": cost_post(self.params, k, sel_hat),
            "predicate_aware": cost_pred(self.params, k, sel_hat),
        }
        best = min(costs, key=costs.get)
        others = ", ".join(f"{name}={cost:.0f}" for name, cost in costs.items() if name != best)
        reason = f"{best}: ŝ={sel_hat:.4f} -> cost={costs[best]:.0f} < {others}"
        return ExecutionPlan(strategy=best, reason=reason, sel_hat=sel_hat, costs=costs)
