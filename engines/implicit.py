# engines/implicit.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

# --- Option B: import from the *1.py modules ---
from core.blackoil_pvt1 import BlackOilPVT
from core.relperm1 import CoreyRelPerm
from core.grid1 import Grid
from core.wells1 import WellSet
from core.timestepping1 import TimeStepper
from core.linear1 import solve_linear

# Assembler stays in core/assembler.py (no suffix)
from core.assembler import Assembler


@dataclass
class EngineOptions:
    newton_tol: float = 1e-6
    max_newton: int = 12
    max_lin: int = 200
    gravity: float = 0.0   # ft/s^2, reserved for future gz terms
    clamp_eps: float = 1e-9


def simulate_3phase_implicit(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 1 implicit driver (skeleton).
    Uses accumulation + simple well sinks for now, with Newton + linear solve.
    """
    # ---- Build objects from inputs
    grid = Grid.from_inputs(inputs["grid"])
    pvt = BlackOilPVT.from_inputs(inputs["pvt"])
    kr = CoreyRelPerm.from_inputs(inputs["relperm"])

    opts = EngineOptions(
        newton_tol=float(inputs.get("newton_tol", 1e-6)),
        max_newton=int(inputs.get("max_newton", 12)),
        max_lin=int(inputs.get("max_lin", 200)),
    )

    # ---- State vector per cell: [P, Sw, Sg]
    ncell = grid.num_cells
    p_init = float(inputs["init"]["p_init_psi"])
    swc = float(inputs["relperm"]["Swc"])

    P0 = np.full(ncell, p_init)
    Sw0 = np.full(ncell, swc)
    Sg0 = np.zeros(ncell)
    x_prev = np.stack([P0, Sw0, Sg0], axis=1).reshape(-1)

    # ---- Wells
    wells = WellSet.from_inputs(inputs.get("msw", {}), inputs.get("schedule", {}), grid, inputs)

    # ---- Simple time schedule (placeholder)
    ts = TimeStepper(total_days=30.0 * 365.0, nsteps=360)

    assembler = Assembler(grid=grid, pvt=pvt, kr=kr, wells=wells)

    # ---- March in time with Newton iterations
    t_days = []
    qg_list, qo_list = [], []
    x_n = x_prev.copy()

    for step_idx, (dt_days, t_accum) in enumerate(ts):
        for it in range(opts.max_newton):
            R, J = assembler.residual_and_jacobian(x_n, x_prev, dt_days)
            if np.linalg.norm(R, ord=np.inf) < opts.newton_tol:
                break
            dx = solve_linear(J, -R, max_iter=opts.max_lin)
            x_n += dx
            assembler.clamp_state_inplace(x_n)

        # accept step
        x_prev = x_n.copy()

        # Report surface rates at this step
        qo_s, qg_s, _ = wells.surface_rates(x_prev, grid, pvt, kr)
        t_days.append(t_accum)
        qg_list.append(qg_s)
        qo_list.append(qo_s)

    # ---- Package results for the UI
    t = np.asarray(t_days, float)
    qg = np.asarray(qg_list, float)   # Mscf/d
    qo = np.asarray(qo_list, float)   # STB/d

    EUR_g_BCF = np.trapz(qg, t) / 1e6
    EUR_o_MMBO = np.trapz(qo, t) / 1e6

    return dict(
        t=t,
        qg=qg,
        qo=qo,
        EUR_g_BCF=EUR_g_BCF,
        EUR_o_MMBO=EUR_o_MMBO,
        # placeholders expected by some tabs (filled when full fields exist)
        p_init_3d=np.full((grid.nz, grid.ny, grid.nx), p_init),
        press_matrix=None,
        pm_mid_psi=None,
        ooip_3d=None,
        runtime_s=0.0,
    )
