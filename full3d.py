# full3d.py  (ROOT FILE)

from __future__ import annotations
import numpy as np
import time

from engines.fast import fallback_fast_solver

# Try implicit driver (optional)
try:
    from engines.implicit import simulate_3phase_implicit
    _HAS_IMPLICIT = True
except Exception:
    simulate_3phase_implicit = None
    _HAS_IMPLICIT = False

# IMPORTANT: bring in our PVT compatibility layer
from core.blackoil_pvt1 import BlackOilPVT


def _coerce_pvt(inputs: dict) -> dict:
    """Ensure inputs['pvt'] is a BlackOilPVT instance (convert legacy Fluid/dict)."""
    pvt_in = inputs.get("pvt", {})
    if not isinstance(pvt_in, BlackOilPVT):
        try:
            inputs["pvt"] = BlackOilPVT.from_inputs(pvt_in)
        except Exception as e:
            # Fallback to defaults if anything odd comes through
            print("[full3d] PVT coercion failed:", e)
            inputs["pvt"] = BlackOilPVT.from_inputs({})
    return inputs


def simulate(inputs: dict) -> dict:
    """
    Router for the simulation engine.

    - If engine_type contains "Implicit" and engines/implicit.py is available:
        -> use simulate_3phase_implicit (Phase 1 driver)
    - Else:
        -> use fast analytical proxy (fallback_fast_solver)
    """
    # Make sure we always have a proper BlackOilPVT object
    inputs = dict(inputs)  # shallow copy to avoid side-effects upstream
    inputs = _coerce_pvt(inputs)

    # --- DEBUG GUARD: prove we have a proper PVT object
    pvt = inputs["pvt"]
    print("[full3d] PVT type:", type(pvt))
    assert hasattr(pvt, "Rs"), f"PVT missing Rs; got {type(pvt)}"

    engine_type = inputs.get("engine_type", "")
    if "Implicit" in engine_type and _HAS_IMPLICIT:
        return simulate_3phase_implicit(inputs)
    elif "Implicit" in engine_type and not _HAS_IMPLICIT:
        # Soft fallback if implicit driver isn't available
        return simulate_3D_implicit(inputs)
    else:
        rng = np.random.default_rng(1234)
        return fallback_fast_solver(inputs, rng)


def simulate_3D_implicit(inputs: dict) -> dict:
    """
    Phase 1b blueprint: returns proxy results. Kept as a safe fallback
    if engines/implicit.py is not yet added or fails to import.
    """
    print("--- Running Phase 1b: 3D Implicit Engine Blueprint (proxy fallback) ---")
    start_time = time.time()

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
