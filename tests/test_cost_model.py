from vecdb.planner.cost_model import CostModelParams, cost_pre, cost_post, cost_pred, ef_required, gamma

def make_params():
    return CostModelParams(c_scan=1.0, c_hop=1.0, alpha=4.0, beta=2.0, ef_min=16, ef_base=64, M=16, N=100_000)

def test_cost_pre_scales_linearly_with_selectivity():
    p = make_params()
    assert cost_pre(p, k=10, sel_hat=0.1) == p.c_scan * p.N * 0.1
    assert cost_pre(p, k=10, sel_hat=0.2) == 2 * cost_pre(p, k=10, sel_hat=0.1)

def test_ef_required_clamped_to_ef_min_and_n():
    p = make_params()
    assert ef_required(p, k=1, sel_hat=1.0) == p.ef_min
    assert ef_required(p, k=10, sel_hat=1e-9) == p.N

def test_cost_post_increases_as_selectivity_drops():
    p = make_params()
    assert cost_post(p, k=10, sel_hat=0.5) < cost_post(p, k=10, sel_hat=0.01)

def test_gamma_is_capped():
    p = make_params()
    assert gamma(p, sel_hat=1e-9) == p.gamma_cap

def test_cost_pred_increases_as_selectivity_drops():
    p = make_params()
    assert cost_pred(p, k=10, sel_hat=0.5) < cost_pred(p, k=10, sel_hat=0.01)

def test_params_save_and_load_roundtrip(tmp_path):
    p = make_params()
    path = tmp_path / "calibration.json"
    p.save(path)
    loaded = CostModelParams.load(path)
    assert loaded == p
