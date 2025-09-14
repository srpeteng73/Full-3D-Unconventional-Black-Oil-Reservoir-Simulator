# engines/implicit.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from core.blackoil_pvt import BlackOilPVT
from core.relperm import CoreyRelPerm
from core.grid import Grid
from core.wells import WellSet
from core.assembler import Assembler
from core.timestepping import TimeStepper
from core.linear import solve_linear

@dataclass
class EngineOptions:
    newton_tol: float = 1e-6
    max_newton: int = 12
    max_lin: int = 200
    gravity: float = 0.0  # ft/s^2 (set >0 later if you want gz terms)
    clamp_eps: float = 1e-9

def simulate_3phase_implicit(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 1 driver (skeleton). Returns physically-plausible rates with a growing implicit core."""
    # --- build objects
    grid = Grid.from_inputs(inputs["grid"])
    pvt = BlackOilPVT.from_inputs(inputs["pvt"])
    kr = CoreyRelPerm.from_inputs(inputs["relperm"])
    opts = EngineOptions(
        newton_tol=float(inputs.get("newton_tol", 1e-6)),
        max_newton=int(inputs.get("max_newton", 12)),
        max_lin=int(inputs.get("max_lin", 200)),
    )

    # state vector ordering per cell: [P, Sw, Sg]
    ncell = grid.num_cells
    P0 = np.full(ncell, float(inputs["init"]["p_init_psi"]))
    Sw0 = np.full(ncell, float(inputs["relperm"]["Swc"]))
    Sg0 = np.zeros(ncell)
    x = np.stack([P0, Sw0, Sg0], axis=1).reshape(-1)

    # wells
    wells = WellSet.from_inputs(inputs.get("msw", {}), inputs.get("schedule", {}), grid, inputs)

    # time stepping (simple fixed schedule for now)
    ts = TimeStepper(total_days=30.0 * 365.0, nsteps=360)

    assembler = Assembler(grid, pvt, kr, wells, opts)

    # --- simple loop over time steps with Newton iterations (skeleton)
    t_days = []
    qg_list, qo_list = [], []
    x_n = x.copy()
    for step_idx, (dt_days, t_accum) in enumerate(ts):
        # Newton
        for it in range(opts.max_newton):
            R, J = assembler.residual_and_jacobian(x_n, x, dt_days)
            norm = np.linalg.norm(R, ord=np.inf)
            if norm < opts.newton_tol:
                break
            dx = solve_linear(J, -R, max_iter=opts.max_lin)
            x_n += dx
            assembler.clamp_state_inplace(x_n)

        # advance "previous" state
        x = x_n.copy()

        # report simple surface rates from wells
        qo, qg, _ = wells.surface_rates(x, grid, pvt, kr)
        t_days.append(t_accum)
        qg_list.append(qg)
        qo_list.append(qo)

    # pack results for your UI (same keys used today)
    t = np.asarray(t_days, float)
    qg = np.asarray(qg_list, float)  # Mscf/d
    qo = np.asarray(qo_list, float)  # STB/d
    EUR_g_BCF = np.trapz(qg, t) / 1e6
    EUR_o_MMBO = np.trapz(qo, t) / 1e6

    return dict(
        t=t, qg=qg, qo=qo,
        EUR_g_BCF=EUR_g_BCF, EUR_o_MMBO=EUR_o_MMBO,
        # placeholders expected by some tabs (filled later as we add full fields)
        p_init_3d=np.full((grid.nz, grid.ny, grid.nx), P0[0]),
        press_matrix=None, pm_mid_psi=None, ooip_3d=None, runtime_s=0.0
    )
