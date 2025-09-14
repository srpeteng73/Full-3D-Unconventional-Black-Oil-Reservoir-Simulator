# engines/implicit.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from core.blackoil_pvt1 import BlackOilPVT
from core.relperm1      import CoreyRelPerm
from core.grid1         import Grid
from core.wells1        import WellSet
from core.assembler     import Assembler

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
    dt_init_days: float = (30.0 * 365.0) / 360.0  # ~1 month / 12 ≈ 30-day months
    dt_min_days:  float = 0.25                    # 6 hours
    dt_max_days:  float = 90.0                    # ~quarterly
    dt_grow:      float = 1.25
    dt_shrink:    float = 0.5
    # fail safety
    max_step_retries: int = 3


def _enforce_pvt_iface(pvt: BlackOilPVT) -> None:
    required = ("Bo", "Bw", "Bg", "Rs",
                "dBo_dP", "dBw_dP", "dBg_dP", "dRs_dP",
                "mu_oil", "mu_water", "mu_gas")
    missing = [m for m in required if not hasattr(pvt, m)]
    if missing:
        raise TypeError(
            f"BlackOilPVT is missing {missing}. "
            "Did an old placeholder Fluid class slip in?"
        )


def _res_norm(R: np.ndarray) -> float:
    # infinity norm is robust for accept/reject decisions
    return float(np.linalg.norm(R, ord=np.inf))


def _newton_with_linesearch(
    asm: Assembler, x_start: np.ndarray, x_prev: np.ndarray,
    dt_days: float, opts: EngineOptions
) -> Tuple[bool, np.ndarray, int, int, float]:
    """
    Try to take one implicit step using Newton + Armijo backtracking.
    Returns: (success, x_next, newton_iters, backtracks_used, final_norm)
    """
    x_n = x_start.copy()
    for it in range(opts.max_newton):
        R, J = asm.residual_and_jacobian(x_n, x_prev, dt_days)
        norm0 = _res_norm(R)
        if norm0 < opts.newton_tol:
            return True, x_n, it, 0, norm0

        # Solve J dx = -R
        dx = solve_linear(J, -R, max_iter=opts.max_lin)

        # Armijo backtracking line search
        alpha = 1.0
        accepted = False
        for bt in range(opts.ls_max_backtracks + 1):
            x_try = x_n + alpha * dx
            asm.clamp_state_inplace(x_try)
            R_try, _ = asm.residual_and_jacobian(x_try, x_prev, dt_days)
            norm_try = _res_norm(R_try)
            # sufficient decrease: f(x+αdx) <= (1 - c1*α) f(x)
            if norm_try <= (1.0 - opts.armijo_c1 * alpha) * norm0:
                x_n = x_try
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            # Line search failed at this Newton iteration
            return False, x_n, it + 1, opts.ls_max_backtracks, norm0

    # Max Newton iterations hit
    R_end, _ = asm.residual_and_jacobian(x_n, x_prev, dt_days)
    return False, x_n, opts.max_newton, 0, _res_norm(R_end)


def simulate_3phase_implicit(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Three-phase implicit driver with line-search + adaptive dt."""
    # --- Build model objects
    grid = Grid.from_inputs(inputs["grid"], inputs.get("rock", {}))
    pvt  = BlackOilPVT.from_inputs(inputs["pvt"])
    kr   = CoreyRelPerm.from_inputs(inputs["relperm"])
    wells = WellSet.from_inputs(inputs.get("msw", {}), inputs.get("schedule", {}), grid, inputs)
    _enforce_pvt_iface(pvt)

    opts = EngineOptions(
        newton_tol=float(inputs.get("newton_tol", 1e-6)),
        max_newton=int(inputs.get("max_newton", 12)),
        max_lin=int(inputs.get("max_lin", 200)),
    )

    # --- State layout: [P, Sw, Sg] per cell (+ optional extra well DOFs for RATE)
    ncell = grid.num_cells
    P0  = np.full(ncell, float(inputs["init"]["p_init_psi"]))
    Sw0 = np.full(ncell, float(inputs["relperm"]["Swc"]))
    Sg0 = np.zeros(ncell)
    x_cells = np.stack([P0, Sw0, Sg0], axis=1).reshape(-1)

    asm = Assembler(grid=grid, pvt=pvt, kr=kr, wells=wells, opts=vars(opts))

    nextra = 0
    if hasattr(asm, "n_extra_dof") and callable(getattr(asm, "n_extra_dof")):
        try:
            nextra = int(asm.n_extra_dof())
        except Exception:
            nextra = 0

    if nextra > 0:
        pwf_init = float(inputs.get("pad_bhp_psi", P0[0] - 500.0))
        x = np.concatenate([x_cells, np.full(nextra, pwf_init)])
    else:
        x = x_cells.copy()

    # --- Adaptive time integration
    total_days = 30.0 * 365.0
    t, dt = 0.0, opts.dt_init_days
    t_days, qg_list, qo_list = [], [], []
    x_prev = x.copy()

    step_index = 0
    while t < total_days - 1e-9:
        dt = min(dt, total_days - t)  # don't overshoot final time
        retries = 0

        while True:
            ok, x_next, iters, bts, norm_end = _newton_with_linesearch(asm, x, x_prev, dt, opts)
            if ok:
                # accept step
                x_prev = x_next.copy()
                x      = x_next
                t     += dt
                step_index += 1

                # report rates (optionally with pwf overrides for RATE wells)
                pwf_overrides = None
                if nextra > 0 and hasattr(asm, "rate_well_indices") and hasattr(asm, "ixWell"):
                    try:
                        pwf_overrides = {}
                        for extra_i, wi in enumerate(asm.rate_well_indices):
                            pwf_overrides[wi] = x[asm.ixWell(extra_i)]
                    except Exception:
                        pwf_overrides = None

                try:
                    qo, qg, _ = wells.surface_rates(x, grid, pvt, kr, pwf_overrides=pwf_overrides)
                except TypeError:
                    qo, qg, _ = wells.surface_rates(x, grid, pvt, kr)

                t_days.append(t); qg_list.append(qg); qo_list.append(qo)

                # successful -> grow dt a bit (within bounds)
                dt = min(dt * opts.dt_grow, opts.dt_max_days)

                if step_index % 10 == 0:
                    print(f"[implicit] t={t:8.2f} d, dt={dt:6.2f}, iters={iters}, norm={norm_end:.3e}")

                break  # proceed to next global step

            # not ok -> shrink dt and retry
            retries += 1
            dt = max(dt * opts.dt_shrink, opts.dt_min_days)

            print(f"[implicit] step reject: dt-> {dt:.3f} d (retry {retries}/{opts.max_step_retries}), "
                  f"iters={iters}, norm={norm_end:.3e}")

            if retries >= opts.max_step_retries:
                # give up on implicit for this path — return what we have so far
                print("[implicit] giving up this step after retries; returning partial time series.")
                t_arr  = np.asarray(t_days, float)
                qg_arr = np.asarray(qg_list, float)
                qo_arr = np.asarray(qo_list, float)
                EUR_g_BCF  = np.trapz(qg_arr, t_arr) / 1e6 if len(t_arr) else 0.0
                EUR_o_MMBO = np.trapz(qo_arr, t_arr) / 1e6 if len(t_arr) else 0.0
                return dict(
                    t=t_arr, qg=qg_arr, qo=qo_arr,
                    EUR_g_BCF=EUR_g_BCF, EUR_o_MMBO=EUR_o_MMBO,
                    p_init_3d=np.full((grid.nz, grid.ny, grid.nx), P0[0]),
                    press_matrix=None, pm_mid_psi=None, ooip_3d=None, runtime_s=0.0
                )

    # --- pack results
    t_arr  = np.asarray(t_days, float)
    qg_arr = np.asarray(qg_list, float)  # Mscf/d
    qo_arr = np.asarray(qo_list, float)  # STB/d
    EUR_g_BCF  = np.trapz(qg_arr, t_arr) / 1e6 if len(t_arr) else 0.0
    EUR_o_MMBO = np.trapz(qo_arr, t_arr) / 1e6 if len(t_arr) else 0.0

    return dict(
        t=t_arr, qg=qg_arr, qo=qo_arr,
        EUR_g_BCF=EUR_g_BCF, EUR_o_MMBO=EUR_o_MMBO,
        p_init_3d=np.full((grid.nz, grid.ny, grid.nx), P0[0]),
        press_matrix=None, pm_mid_psi=None, ooip_3d=None, runtime_s=0.0
    )
