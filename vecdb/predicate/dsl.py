from __future__ import annotations

LEAF_OPS = {"eq", "ne", "lt", "lte", "gt", "gte", "in"}
COMBINATOR_OPS = {"and", "or"}


def validate_predicate(pred: dict) -> None:
    if not isinstance(pred, dict) or "op" not in pred:
        raise ValueError(f"predicate must be a dict with an 'op' key, got {pred!r}")
    op = pred["op"]
    if op in LEAF_OPS:
        if "col" not in pred or "val" not in pred:
            raise ValueError(f"leaf predicate {pred!r} must have 'col' and 'val'")
        return
    if op in COMBINATOR_OPS:
        if "clauses" not in pred or not isinstance(pred["clauses"], list) or not pred["clauses"]:
            raise ValueError(f"'{op}' predicate must have a non-empty 'clauses' list")
        for clause in pred["clauses"]:
            validate_predicate(clause)
        return
    if op == "not":
        if "clause" not in pred:
            raise ValueError("'not' predicate must have a 'clause'")
        validate_predicate(pred["clause"])
        return
    raise ValueError(f"unsupported predicate op: {op!r}")
