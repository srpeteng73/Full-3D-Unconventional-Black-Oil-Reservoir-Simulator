# engines/implicit.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

# --- Core imports (Phase 1.2: use *1 modules) ---
from core.blackoil_pvt1 import BlackOilPVT
from core.relperm1 import CoreyRelPerm
from core.grid1 import Grid
from core.wells1 import WellSet
from core.assembler import Assembler

# Try both "*1" and base names for linear solver & time stepping
try:
    from core.linear1 import solve_linear
except Exception:
    from core.linear import solve_linear  # if you have a base version

try:
    from core.timestepping1 import TimeStepping as _TS
except Exception:
    try:
        from core.timestepping import TimeStepper as _TS
    except Exception:
        _TS = None  # we'll fall back to a tiny local iterator

@dataclass
class EngineOptions:
    newton_tol: float = 1e-6
    max_newton: int = 12
    max_lin: int = 200

def _ts_iter(total_days: float, nsteps: int):
    """Local fallback if no timestepping module is present."""
    dt = total_days / float(nsteps)
    t_accum = 0.0
    for _ in range(nsteps):
        t_accum += dt
        yield dt, t_accum

def simulate_3phase_implicit(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Three-phase implicit driver (Phase 1.2 skeleton)."""
    # --- build objects
    grid = Grid.from_inputs(inputs["grid"], inputs.get("rock", {}))
    pvt  = BlackOilPVT.from_inputs(inputs["pvt"])
    kr   = CoreyRelPerm.from_inputs(inputs["relperm"])

    # wells (BHP or RATE). If RATE, assembler adds extra DOFs (one per rate well)
    wells = WellSet.from_inputs(inputs.get("msw", {}), inputs.get("schedule", {}), grid, inputs)

    opts = EngineOptions(
        newton_tol=float(inputs.get("newton_tol", 1e-6)),
        max_newton=int(inputs.get("max_newton", 12)),
        max_lin=int(inputs.get("max_lin", 200)),
    )

    # --- initial state: [P, Sw, Sg] per cell
    ncell = grid.num_cells
    P0  = np.full(ncell, float(inputs["init"]["p_init_psi"]))
    Sw0 = np.full(ncell, float(inputs["relperm"]["Swc"]))
    Sg0 = np.zeros(ncell)
    x_cell = np.stack([P0, Sw0, Sg0], axis=1).reshape(-1)

    asm = Assembler(grid=grid, pvt=pvt, kr=kr, wells=wells, opts=vars(opts))

    # If there are RATE wells, append one extra unknown per rate well (the Pwf)
    nextra = asm.n_extra_dof()
    if nextra > 0:
        # initialize each Pwf near BHP target or a bit below reservoir P
        pwf_init = float(inputs.get("pad_bhp_psi", P0[0] - 500.0))
        x = np.concatenate([x_cell, np.full(nextra, pwf_init)])
    else:
        x = x_cell.copy()

    # --- time stepping
    total_days = 30.0 * 365.0
    nsteps = 360
    if _TS is not None:
        try:
            ts = _TS(total_days=total_days, nsteps=nsteps)
            iterator = iter(ts)
        except TypeError:
            # Some versions may not be iterable directly
            iterator = _ts_iter(total_days, nsteps)
    else:
        iterator = _ts_iter(total_days, nsteps)

    # --- loop
    t_days, qg_list, qo_list = [], [], []
    x_prev = x.copy()
    for dt_days, t_accum in iterator:
        x_n = x.copy()
        # Newton
        for _ in range(opts.max_newton):
            R, J = asm.residual_and_jacobian(x_n, x_prev, dt_days)
            if np.linalg.norm(R, ord=np.inf) < opts.newton_tol:
                break
            dx = solve_linear(J, -R, max_iter=opts.max_lin)
            x_n += dx
            asm.clamp_state_inplace(x_n)

        # accept step
        x_prev = x_n.copy()
        x = x_n

        # Report surface rates (with Pwf overrides from extra DOFs if present)
        pwf_overrides = None
        if nextra > 0:
            pwf_overrides = {}
            for extra_i, wi in enumerate(asm.rate_well_indices):
                pwf_overrides[wi] = x[asm.ixWell(extra_i)]

        qo, qg, _ = wells.surface_rates(x, grid, pvt, kr, pwf_overrides=pwf_overrides)
        t_days.append(t_accum); qg_list.append(qg); qo_list.append(qo)

    # --- pack results (keys your UI expects)
    t  = np.asarray(t_days, float)
    qg = np.asarray(qg_list, float)  # Mscf/d
    qo = np.asarray(qo_list, float)  # STB/d
    EUR_g_BCF  = np.trapz(qg, t) / 1e6
    EUR_o_MMBO = np.trapz(qo, t) / 1e6

    return dict(
        t=t, qg=qg, qo=qo,
        EUR_g_BCF=EUR_g_BCF, EUR_o_MMBO=EUR_o_MMBO,
        # placeholders for UI tabs that expect 3D fields (we’ll fill these in Phase 1.3)
        p_init_3d=np.full((grid.nz, grid.ny, grid.nx), P0[0]),
        press_matrix=None, pm_mid_psi=None, ooip_3d=None, runtime_s=0.0
    )
