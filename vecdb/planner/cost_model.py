from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np


@dataclass
class CostModelParams:
    c_scan: float
    c_hop: float
    alpha: float
    beta: float
    ef_min: int = 16
    ef_base: int = 64
    M: int = 16
    N: int = 100_000
    gamma_cap: float = 50.0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CostModelParams":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def cost_pre(params: CostModelParams, k: int, sel_hat: float) -> float:
    return params.c_scan * params.N * sel_hat


def ef_required(params: CostModelParams, k: int, sel_hat: float) -> float:
    sel_hat = max(sel_hat, 1e-6)
    return float(np.clip(k / sel_hat, params.ef_min, params.N))


def cost_post(params: CostModelParams, k: int, sel_hat: float) -> float:
    return params.c_hop * params.M * ef_required(params, k, sel_hat)


def gamma(params: CostModelParams, sel_hat: float) -> float:
    sel_hat = max(sel_hat, 1e-6)
    return min(1.0 + params.beta * (1.0 / sel_hat - 1.0), params.gamma_cap)


def cost_pred(params: CostModelParams, k: int, sel_hat: float) -> float:
    return params.c_hop * params.M * params.ef_base * gamma(params, sel_hat)
