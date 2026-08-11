from __future__ import annotations
import numpy as np
from vecdb.store.metadata import MetaStore


def compile(pred: dict, meta: MetaStore) -> np.ndarray:
    op = pred["op"]
    if op == "and":
        result = compile(pred["clauses"][0], meta).copy()
        for clause in pred["clauses"][1:]:
            result &= compile(clause, meta)
        return result
    if op == "or":
        result = compile(pred["clauses"][0], meta).copy()
        for clause in pred["clauses"][1:]:
            result |= compile(clause, meta)
        return result
    if op == "not":
        return ~compile(pred["clause"], meta)

    col = meta.columns[pred["col"]]
    val = pred["val"]
    if op == "eq":
        return col == val
    if op == "ne":
        return col != val
    if op == "lt":
        return col < val
    if op == "lte":
        return col <= val
    if op == "gt":
        return col > val
    if op == "gte":
        return col >= val
    if op == "in":
        return np.isin(col, val)
    raise ValueError(f"unsupported op: {op!r}")
