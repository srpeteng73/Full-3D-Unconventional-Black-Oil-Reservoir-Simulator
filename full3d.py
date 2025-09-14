# core/full3d.py
import numpy as np
import time
from engines.fast import fallback_fast_solver

# Try to import the Phase-1 implicit driver (optional; safe if missing)
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
        -> use simulate_3phase_implicit (Phase 1 driver)
    - Else:
        -> use fast analytical proxy (fallback_fast_solver)
    """
    engine_type = inputs.get("engine_type", "")
    if "Implicit" in engine_type and _HAS_IMPLICIT:
        return simulate_3phase_implicit(inputs)
    elif "Implicit" in engine_type and not _HAS_IMPLICIT:
        # If implicit driver isn't present yet, use local proxy blueprint
        return simulate_3D_implicit(inputs)
    else:
        rng = np.random.default_rng(1234)
        return fallback_fast_solver(inputs, rng)


def simulate_3D_implicit(inputs):
    """
    Phase 1b blueprint: returns proxy results.
    Kept here as a safe fallback if engines/implicit.py is not yet added.
    """
    print("--- Running Phase 1b: 3D Implicit Engine Blueprint (proxy fallback) ---")
    start_time = time.time()

    # Proxy production using the fast solver
    rng = np.random.default_rng(1234)
    proxy_results = fallback_fast_solver(inputs, rng)

    # Minimal 3D arrays the UI expects
    grid = inputs.get("grid", {})
    nz, ny, nx = grid.get("nz"), grid.get("ny"), grid.get("nx")
    p_init = inputs.get("init", {}).get("p_init_psi")

    proxy_results["p3d_psi"] = np.full((nz, ny, nx), p_init)
    proxy_results["p_init_3d"] = np.full((nz, ny, nx), p_init)
    proxy_results["ooip_3d"] = np.zeros((nz, ny, nx))
    proxy_results["pm_mid_psi"] = [np.full((ny, nx), p) for p in np.linspace(p_init, 2500, 360)]
    proxy_results["runtime_s"] = time.time() - start_time

    return proxy_results
