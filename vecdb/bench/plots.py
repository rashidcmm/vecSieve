from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """points: (latency_ms, recall). Returns the subset not dominated by any other
    point (no other point has both lower-or-equal latency and higher-or-equal recall,
    strictly better in at least one)."""
    frontier = []
    for p in points:
        dominated = any(
            o != p and o[0] <= p[0] and o[1] >= p[1] and (o[0] < p[0] or o[1] > p[1])
            for o in points
        )
        if not dominated:
            frontier.append(p)
    return sorted(frontier)


def plot_lines(series: dict[str, list[tuple[float, float]]], out_path: Path, xlabel: str, ylabel: str,
                title: str, xscale: str = "linear", yscale: str = "linear") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, points in series.items():
        points = sorted(points)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
