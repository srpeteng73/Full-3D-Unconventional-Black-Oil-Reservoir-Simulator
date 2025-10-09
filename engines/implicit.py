# engines/implicit.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import time

# Always use the new PVT implementation
from core.blackoil_pvt1 import BlackOilPVT
from core.relperm1      import CoreyRelPerm
from core.grid1         import Grid
from core.wells1        import WellSet
from core.assembler     import Assembler

# Linear solver
try:
    from core.linear1 import solve_linear
except Exception:
    from core.linear import solve_linear


@dataclass
class EngineOptions:
    newton_tol: float = 1e-6
    max_newton: int = 12
    max_lin:   int = 200
    # line search
    ls_max_backtracks: int = 6
    armijo_c1: float = 1e-4
    # time stepping
    dt_init_days: float = (30.0 * 365.0) / 360.0
    dt_min_days:  float = 0.25
    dt_max_days:  float = 90.0
    dt_grow:      float = 1.25
    dt_shrink:    float = 0.5
    # fail safety
    max_step_retries: int = 3


def _enforce_pvt_iface(pvt: object) -> None:
    required = ("Bo", "Bw", "Bg", "Rs", "dBo_dP", "dBw_dP", "dBg_dP", "dRs_dP", "mu_oil", "mu_gas", "mu_water")
    missing = [name for name in required if not hasattr(pvt, name)]
    if missing:
        raise TypeError(f"PVT must be BlackOilPVT; got {type(pvt).__name__} missing {missing}.")


def _res_norm(R: np.ndarray) -> float:
    return float(np.linalg.norm(R, ord=np.inf))


def _newton_with_linesearch(
    asm: Assembler, x_start: np.ndarray, x_prev: np.ndarray,
    dt_days: float, opts: EngineOptions
) -> Tuple[bool, np.ndarray, int, int, float]:
    x_n = x_start.copy()
    for it in range(opts.max_newton):
        R, J = asm.residual_and_jacobian(x_n, x_prev, dt_days)
        norm0 = _res_norm(R)
        if norm0 < opts.newton_tol:
            return True, x_n, it, 0, norm0

        dx = solve_linear(J, -R, max_iter=opts.max_lin)
        alpha = 1.0
        accepted = False
        for bt in range(opts.ls_max_backtracks + 1):
            x_try = x_n + alpha * dx
            asm.clamp_state_inplace(x_try)
            R_try, _ = asm.residual_and_jacobian(x_try, x_prev, dt_days)
            norm_try = _res_norm(R_try)
            if norm_try <= (1.0 - opts.armijo_c1 * alpha) * norm0:
                x_n = x_try
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            return False, x_n, it + 1, opts.ls_max_backtracks, norm0

    R_end, _ = asm.residual_and_jacobian(x_n, x_prev, dt_days)
    return False, x_n, opts.max_newton, 0, _res_norm(R_end)


def simulate_3phase_implicit(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Three-phase implicit driver with line-search + adaptive dt."""
    t_start = time.time()
    
    # --- Build model objects
    grid = Grid.from_inputs(inputs)
    pvt  = BlackOilPVT.from_inputs(inputs)
    kr   = CoreyRelPerm.from_inputs(inputs)
    wells = WellSet.from_inputs(inputs, grid)
    _enforce_pvt_iface(pvt)

    opts = EngineOptions(
        newton_tol=float(inputs.get("newton_tol", 1e-6)),
        max_newton=int(inputs.get("max_newton", 12)),
        max_lin=int(inputs.get("max_lin", 200)),
    )

    # --- Initial State
    ncell = grid.num_cells
    P0  = np.full(ncell, float(inputs["p_init_psi"]))
    Sw0 = np.full(ncell, float(inputs["Swc"]))
    Sg0 = np.zeros(ncell)
    x_cells = np.stack([P0, Sw0, Sg0], axis=1).reshape(-1)
    asm = Assembler(grid=grid, pvt=pvt, kr=kr, wells=wells, opts=vars(opts))
    x = x_cells.copy()

    # --- Adaptive time integration
    total_days = float(inputs.get("t_end_days", 10 * 365.25))
    t, dt = 0.0, opts.dt_init_days
    t_days, qg_list, qo_list, qw_list = [0.0], [0.0], [0.0], [0.0]
    x_prev = x.copy()

    step_index = 0
    while t < total_days - 1e-9:
        dt = min(dt, total_days - t)
        retries = 0
        while True:
            ok, x_next, iters, bts, norm_end = _newton_with_linesearch(asm, x, x_prev, dt, opts)
            if ok:
                x_prev, x = x_next.copy(), x_next
                t += dt
                step_index += 1
                qo, qg, qw = wells.surface_rates(x, grid, pvt, kr)
                t_days.append(t); qg_list.append(qg); qo_list.append(qo); qw_list.append(qw)
                dt = min(dt * opts.dt_grow, opts.dt_max_days)
                if step_index % 10 == 0:
                    print(f"[implicit] t={t:8.2f} d, dt={dt:6.2f}, iters={iters}, norm={norm_end:.3e}")
                break
            
            retries += 1
            dt = max(dt * opts.dt_shrink, opts.dt_min_days)
            print(f"[implicit] step reject: dt-> {dt:.3f} d (retry {retries}/{opts.max_step_retries}), iters={iters}, norm={norm_end:.3e}")
            if retries >= opts.max_step_retries:
                print("[implicit] giving up after max retries.")
                t = total_days # Force exit
                break
    
    # --- CORRECTED: Pack final 3D results ---
    t_arr  = np.asarray(t_days, float)
    qg_arr = np.asarray(qg_list, float)
    qo_arr = np.asarray(qo_list, float)
    qw_arr = np.asarray(qw_list, float)

    # Unpack final state into 3D arrays
    final_state_grid = x[:ncell*3].reshape((ncell, 3))
    P_final = final_state_grid[:, 0]
    Sw_final = final_state_grid[:, 1]
    Sg_final = final_state_grid[:, 2]
    So_final = 1.0 - Sw_final - Sg_final
    
    press_matrix = P_final.reshape((grid.nz, grid.ny, grid.nx))
    
    # Calculate OOIP
    Vp = grid.dx * grid.dy * grid.dz * grid.phi.flatten() # Pore volume
    Soi = 1.0 - Sw0 # Initial oil saturation
    Boi = pvt.Bo(P0) # Initial oil formation volume factor
    ooip_per_cell = (Vp * Soi) / Boi
    ooip_3d = ooip_per_cell.reshape((grid.nz, grid.ny, grid.nx))

    return dict(
        t=t_arr, qg=qg_arr, qo=qo_arr, qw=qw_arr,
        p_init_3d=P0.reshape((grid.nz, grid.ny, grid.nx)),
        press_matrix=press_matrix,
        pm_mid_psi=np.mean(press_matrix, axis=(0,1,2)),
        ooip_3d=ooip_3d,
        runtime_s=time.time() - t_start
    )
