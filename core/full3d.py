# core/full3d.py
from __future__ import annotations
import numpy as np
import time
from collections.abc import Mapping

from engines.fast import fallback_fast_solver

# Try implicit driver (optional)
try:
    from engines.implicit import simulate_3phase_implicit
    _HAS_IMPLICIT = True
except Exception:
    simulate_3phase_implicit = None
    _HAS_IMPLICIT = False

# Bring in the BlackOil PVT class (with dict/legacy compatibility)
from core.blackoil_pvt1 import BlackOilPVT


def _coerce_pvt_inplace(inputs: dict) -> None:
    """
    Ensure inputs['pvt'] is a BlackOilPVT instance.
    Accepts:
      - BlackOilPVT (returned as-is)
      - dict / Mapping (passed to BlackOilPVT.from_inputs)
      - legacy object (e.g., 'Fluid') with attributes like pb_psi, Rs_pb_scf_stb, etc.
    Falls back to sensible defaults if anything is missing.
    """
    pvt_in = inputs.get("pvt", {})

    # Already correct type
    if isinstance(pvt_in, BlackOilPVT):
        print("[core/full3d] PVT is already BlackOilPVT")
        return

    try:
        # Mapping (dict-like)
        if isinstance(pvt_in, Mapping):
            print("[core/full3d] Coercing PVT from Mapping")
            inputs["pvt"] = BlackOilPVT.from_inputs(pvt_in)
            return

        # Legacy object (e.g., 'Fluid') – pull attributes safely
        print("[core/full3d] Coercing PVT from legacy object:", type(pvt_in))

        def take(*names, default=None):
            for n in names:
                if hasattr(pvt_in, n):
                    return getattr(pvt_in, n)
            return default

        coerced = {
            "pb_psi":           take("pb_psi", "pb", default=5200.0),
            "Rs_pb_scf_stb":    take("Rs_pb_scf_stb", "Rs_pb", "rs_pb", default=650.0),
            "Bo_pb_rb_stb":     take("Bo_pb_rb_stb", "Bo_pb", "bo_pb", default=1.35),
            "muo_pb_cp":        take("muo_pb_cp", "mu_oil_pb_cp", "mu_o", default=1.2),
            "mug_pb_cp":        take("mug_pb_cp", "mu_gas_pb_cp", "mu_g", default=0.020),
            "ct_o_1psi":        take("ct_o_1psi", "cto", default=8e-6),
            "ct_g_1psi":        take("ct_g_1psi", "ctg", default=3e-6),
            "ct_w_1psi":        take("ct_w_1psi", "ctw", default=3e-6),
            "Bw_ref":           take("Bw_ref", "bw_ref", default=1.01),
        }

        inputs["pvt"] = BlackOilPVT.from_inputs(coerced)
        return

    except Exception as e:
        print("[core/full3d] PVT coercion failed; falling back to defaults:", e)
        inputs["pvt"] = BlackOilPVT.from_inputs({})
        return


def simulate(inputs: dict) -> dict:
    """
    Router for the simulation engine.

    - If engine_type contains "Implicit" and engines/implicit.py is available:
        -> use simulate_3phase_implicit (Phase 1 driver)
    - Else:
        -> use fast analytical proxy (fallback_fast_solver)
    """
    print(f"[core/full3d] simulate() from: {__file__}")

    # Make sure we always have a proper BlackOilPVT object
    inputs = dict(inputs)  # shallow copy to avoid side-effects upstream
    _coerce_pvt_inplace(inputs)

    # DEBUG GUARD: prove we have a proper PVT object
    pvt = inputs["pvt"]
    print("[core/full3d] PVT type:", type(pvt))
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
    nz = grid.get("nz", 1)
    ny = grid.get("ny", 1)
    nx = grid.get("nx", 1)
    p_init = inputs.get("init", {}).get("p_init_psi", 3000.0)

    proxy_results["p3d_psi"]   = np.full((nz, ny, nx), p_init)
    proxy_results["p_init_3d"] = np.full((nz, ny, nx), p_init)
    proxy_results["ooip_3d"]   = np.zeros((nz, ny, nx))
    proxy_results["pm_mid_psi"] = [np.full((ny, nx), p) for p in np.linspace(p_init, 2500, 360)]
    proxy_results["runtime_s"] = time.time() - start_time

    return proxy_results
