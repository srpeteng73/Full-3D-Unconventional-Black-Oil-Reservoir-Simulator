# engines/implicit.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

from core.blackoil_pvt1 import BlackOilPVT
from core.relperm1 import CoreyRelPerm
from core.grid1 import Grid
from core.wells1 import WellSet
from core.timestepping1 import TimeStepper
from core.linear1 import solve_linear
from core.assembler import Assembler

@dataclass
class EngineOptions:
    newton_tol: float = 1e-6
    max_newton: int = 12
    max_lin: int = 200
    gravity: float = 0.0
    clamp_eps: float = 1e-9

def simulate_3phase_implicit(inputs: Dict[str, Any]) -> Dict[str, Any]:
    # engines/implicit.py  (only this line changes)
grid = Grid.from_inputs(inputs["grid"], inputs.get("rock", {}))

    pvt  = BlackOilPVT.from_inputs(inputs["pvt"])
    kr   = CoreyRelPerm.from_inputs(inputs["relperm"])

    opts = EngineOptions(
        newton_tol=float(inputs.get("newton_tol", 1e-6)),
        max_newton=int(inputs.get("max_newton", 12)),
        max_lin=int(inputs.get("max_lin", 200)),
    )

    # initial cell state [P, Sw, Sg] per cell
    ncell = grid.num_cells
    p_init = float(inputs["init"]["p_init_psi"])
    swc    = float(inputs["relperm"]["Swc"])

    P0  = np.full(ncell, p_init)
    Sw0 = np.full(ncell, swc)
    Sg0 = np.zeros(ncell)

    wells = WellSet.from_inputs(inputs.get("msw", {}), inputs.get("schedule", {}), grid, inputs)
    asm = Assembler(grid=grid, pvt=pvt, kr=kr, wells=wells)

    ncell_dof = asm.n_cell_dof()
    nextra    = asm.n_extra_dof()

    # Build full state vector: [cells..., pwf_rate_well_0, pwf_rate_well_1, ...]
    x_prev = np.zeros(ncell_dof + nextra)
    x_prev[0::3][:] = P0
    x_prev[1::3][:] = Sw0
    x_prev[2::3][:] = Sg0

    # initialize pwf for rate wells slightly below reservoir pressure
    for extra_i, wi in enumerate(asm.rate_well_indices):
        row = asm.ixWell(extra_i)
        x_prev[row] = p_init - 500.0

    # time stepping
    ts = TimeStepper(total_days=30.0 * 365.0, nsteps=360)

    t_days, qg_list, qo_list = [], [], []
    x_n = x_prev.copy()

    for _, (dt_days, t_accum) in enumerate(ts):
        # Newton loop
        for _ in range(opts.max_newton):
            R, J = asm.residual_and_jacobian(x_n, x_prev, dt_days)
            if np.linalg.norm(R, ord=np.inf) < opts.newton_tol:
                break
            dx = solve_linear(J, -R, max_iter=opts.max_lin)
            x_n += dx
            asm.clamp_state_inplace(x_n)

        # accept step
        x_prev = x_n.copy()

        # Report rates — pass pwf overrides for rate wells
        pwf_overrides = {extra_i: x_prev[asm.ixWell(extra_i)] for extra_i, _ in enumerate(asm.rate_well_indices)} if nextra > 0 else None
        qo_s, qg_s, _ = wells.surface_rates(x_prev, grid, pvt, kr, pwf_overrides=pwf_overrides)

        t_days.append(t_accum)
        qg_list.append(qg_s)
        qo_list.append(qo_s)

    t  = np.asarray(t_days, float)
    qg = np.asarray(qg_list, float)
    qo = np.asarray(qo_list, float)

    EUR_g_BCF  = np.trapz(qg, t) / 1e6
    EUR_o_MMBO = np.trapz(qo, t) / 1e6

    return dict(
        t=t, qg=qg, qo=qo,
        EUR_g_BCF=EUR_g_BCF, EUR_o_MMBO=EUR_o_MMBO,
        p_init_3d=np.full((grid.nz, grid.ny, grid.nx), p_init),
        press_matrix=None, pm_mid_psi=None, ooip_3d=None,
        runtime_s=0.0,
    )
