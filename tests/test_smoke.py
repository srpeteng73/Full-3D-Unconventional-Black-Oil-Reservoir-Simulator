# tests/test_smoke.py
from core.full3d import simulate

def test_fast_proxy_runs_quickly():
    """
    Exercise the router and the fast proxy (no heavy solver).
    This should always be quick and stable in CI.
    """
    inputs = {
        "engine_type": "Proxy",           # forces the fast proxy path
        "L_ft": 10000.0,
        "xf_ft": 300.0,
        "hf_ft": 180.0,
        "pad_interf": 0.2,
        "n_laterals": 1,
        "Rs_pb_scf_stb": 650.0,
        "pb_psi": 5200.0,
        "fluid_model": "unconventional",
        "grid": {"nx": 1, "ny": 1, "nz": 1},
        "init": {"p_init_psi": 3000.0},
    }

    out = simulate(inputs)
    # basic sanity checks
    assert "qg" in out and out["qg"].size == 360
    assert "qo" in out and out["qo"].size == 360

