# core/full3d.py
import numpy as np
import time
from engines.fast import fallback_fast_solver

try:
    from engines.implicit import simulate_3phase_implicit
    _HAS_IMPLICIT = True
except Exception:
    simulate_3phase_implicit = None
    _HAS_IMPLICIT = False


def simulate(inputs):
    """
    Router for the simulation engine.
    - If engine_type contains "Implicit" and engines/implicit.py is available:
        -> use simulate_3phase_implicit
    - Else:
        -> use fast analytical proxy (fallback_fast_solver)
    If the implicit engine raises, we catch it and return the fast proxy with an error string.
    """
    engine_type = str(inputs.get("engine_type", ""))
    use_implicit = ("Implicit" in engine_type) and _HAS_IMPLICIT

    if use_implicit:
        try:
            return simulate_3phase_implicit(inputs)
        except Exception as e:
            # annotate and fall back so the UI keeps working
            err = f"Implicit engine failed in engines/implicit.py: {e.__class__.__name__}: {e}"
            print(err)
            rng = np.random.default_rng(1234)
            out = fallback_fast_solver(inputs, rng)
            out["engine_error"] = err
            return out

    rng = np.random.default_rng(1234)
    return fallback_fast_solver(inputs, rng)


def simulate_3D_implicit(inputs):
    """
    Legacy Phase-1b proxy (kept as a safe fallback only).
    """
    print("--- Running Phase 1b: 3D Implicit Engine Blueprint (proxy fallback) ---")
    start_time = time.time()

    rng = np.random.default_rng(1234)
    proxy_results = fallback_fast_solver(inputs, rng)

    grid = inputs.get("grid", {})
    nz, ny, nx = grid.get("nz"), grid.get("ny"), grid.get("nx")
    p_init = inputs.get("init", {}).get("p_init_psi")

    proxy_results["p3d_psi"] = np.full((nz, ny, nx), p_init)
    proxy_results["p_init_3d"] = np.full((nz, ny, nx), p_init)
    proxy_results["ooip_3d"] = np.zeros((nz, ny, nx))
    proxy_results["pm_mid_psi"] = [np.full((ny, nx), p) for p in np.linspace(p_init, 2500, 360)]
    proxy_results["runtime_s"] = time.time() - start_time
    return proxy_results
