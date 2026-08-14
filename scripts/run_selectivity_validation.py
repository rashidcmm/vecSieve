# scripts/run_selectivity_validation.py
"""Phase 5 headline figure: for 500 random predicates (single-clause, AND-of-2,
AND-of-3, OR), scatter estimated selectivity vs true selectivity on log-log axes,
for both uncorrelated and correlated metadata. The independence assumption for AND
should hold on uncorrelated columns and visibly fail on correlated ones."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vecdb.io.metadata_gen import load_metadata
from vecdb.store.metadata import MetaStore
from vecdb.predicate.compile import compile as compile_pred
from vecdb.predicate.selectivity import estimate_selectivity

def random_predicate(rng: np.random.Generator, meta: MetaStore) -> dict:
    shape = rng.choice(["single", "and2", "and3", "or2"])
    def leaf():
        col = rng.choice(["category", "year", "score"]) if "score" in meta.columns else rng.choice(["category", "year"])
        if meta.stats[col].kind == "categorical":
            val = int(rng.choice(list(meta.stats[col].value_counts.keys())))
            return {"op": "eq", "col": col, "val": val}
        lo = float(meta.stats[col].hist_edges[0])
        hi = float(meta.stats[col].hist_edges[-1])
        return {"op": rng.choice(["lt", "gt"]), "col": col, "val": float(rng.uniform(lo, hi))}
    if shape == "single":
        return leaf()
    if shape == "and2":
        return {"op": "and", "clauses": [leaf(), leaf()]}
    if shape == "and3":
        return {"op": "and", "clauses": [leaf(), leaf(), leaf()]}
    return {"op": "or", "clauses": [leaf(), leaf()]}

def collect(meta: MetaStore, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_s, est_s = [], []
    for _ in range(n):
        pred = random_predicate(rng, meta)
        mask = compile_pred(pred, meta)
        true = mask.sum() / meta.n
        est = estimate_selectivity(pred, meta)
        if true > 0:  # log-log plot can't show zero
            true_s.append(true)
            est_s.append(est if est > 0 else 1e-6)
    return np.array(true_s), np.array(est_s)

def plot_scatter(true_s, est_s, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_s, est_s, alpha=0.4, s=15)
    ax.plot([1e-4, 1], [1e-4, 1], "k--", linewidth=1, label="perfect estimate")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("true selectivity"); ax.set_ylabel("estimated selectivity (ŝ)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main() -> None:
    for variant in ["uncorrelated", "correlated"]:
        cols = load_metadata(Path(f"data/sift1m_100k_meta_{variant}.npz"))
        meta = MetaStore(cols)
        true_s, est_s = collect(meta, n=500, seed=0)
        err = np.abs(est_s - true_s) / true_s
        print(f"{variant}: median rel. error={np.median(err):.3f}  p95={np.quantile(err, 0.95):.3f}")
        plot_scatter(true_s, est_s, f"Selectivity estimation error ({variant})",
                     f"results/figures/selectivity_estimation_{variant}.png")

if __name__ == "__main__":
    main()
