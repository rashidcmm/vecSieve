from pathlib import Path
from vecdb.bench.plots import pareto_frontier, plot_lines

def test_pareto_frontier_drops_dominated_points():
    # (latency, recall): (10, 0.9) dominates (20, 0.85) — lower latency, higher recall
    points = [(10.0, 0.9), (20.0, 0.85), (5.0, 0.5), (30.0, 0.99)]
    frontier = pareto_frontier(points)
    assert (20.0, 0.85) not in frontier
    assert (10.0, 0.9) in frontier
    assert (5.0, 0.5) in frontier   # cheapest latency, nothing beats it on latency
    assert (30.0, 0.99) in frontier  # highest recall, nothing beats it on recall

def test_plot_lines_writes_a_png_file(tmp_path):
    out = tmp_path / "figures" / "test.png"
    plot_lines(
        series={"a": [(1.0, 0.5), (2.0, 0.8)], "b": [(1.0, 0.4), (2.0, 0.9)]},
        out_path=out, xlabel="x", ylabel="y", title="test",
    )
    assert out.exists()
    assert out.stat().st_size > 0
