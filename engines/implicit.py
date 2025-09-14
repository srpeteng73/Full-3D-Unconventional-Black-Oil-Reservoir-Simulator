# engines/implicit.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

# ---- Core (use the *1 modules) ----
from core.blackoil_pvt1 import BlackOilPVT
from core.relperm1      import CoreyRelPerm
from core.grid1         import Grid
from core.wells1        import WellSet
from core.assembler     import Assembler

# Linear solver / time stepping (prefer *1, fall back to base if needed)
try:
    from core.linear1 import solve_linear
except Exception:
    from core.linear import solve_linear

try:
    from core.timestepping1 import TimeStepping as _TS
except Exception:
    try:
        from core.timestepping import TimeStepper as _TS  # older name
    except Exception:
        _TS = None  # use a tiny local iterator


@dataclass
class EngineOptions:
    newton_tol: float = 1e-6
    max_newton: int = 12
    max_lin:   int = 200


def _ts_iter(total_days: float, nsteps: int):
    """Local fallback if no timestepping class is available."""
    dt = total_days / float(nsteps)
    t_accum = 0.0
    for _ in range(nsteps):
        t_accum += dt
        yield dt, t_accum


def simulate_3phase_implicit(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Three-phase implicit driver (Phase 1.x skeleton, safe with/without rate DOFs)."""
    # ---- Build model objects
    grid = Grid.from_inputs(inputs["grid"], inputs.get("rock", {}))
    pvt  = BlackOilPVT.from_inputs(inputs["pvt"])

    # Explicit sanity check so legacy Fluid can't slip in
    for req in ("Bo", "Bw", "Bg", "Rs"):
        if not hasattr(pvt, req):
            raise TypeError(
                f"BlackOilPVT is missing required method '{req}'. "
                "Did an old 'Fluid' object slip in?"
            )

    kr    = CoreyRelPerm.from_inputs(inputs["relperm"])
    wells = WellSet.from_inputs(inputs.get("msw", {}), inputs.get("schedule", {}), grid, inputs)

    opts = EngineOptions(
        newton_tol=float(inputs.get("newton_tol", 1e-6)),
        max_newton=int(inputs.get("max_newton", 12)),
        max_lin=int(inputs.get("max_lin", 200)),
    )

    # ---- Initial state: [P, Sw, Sg] per cell
    ncell = grid.num_cells
    P0  = np.full(ncell, float(inputs["init"]["p_init_psi"]))
    Sw0 = np.full(ncell, float(inputs["relperm"]["Swc"]))
    Sg0 = np.zeros(ncell)
    x_cell = np.stack([P0, Sw0, Sg0], axis=1).reshape(-1)

    # Assembler
    asm = Assembler(grid=grid, pvt=pvt, kr=kr, wells=wells, opts=vars(opts))

    # Optional RATE-control extras (only if your Assembler exposes them)
    nextra = 0
    if hasattr(asm, "n_extra_dof") and callable(getattr(asm, "n_extra_dof")):
        try:
            nextra = int(asm.n_extra_dof())
        except Exception:
            nextra = 0

    if nextra > 0:
        pwf_init = float(inputs.get("pad_bhp_psi", P0[0] - 500.0))
        x = np.concatenate([x_cell, np.full(nextra, pwf_init)])
    else:
        x = x_cell.copy()

    # ---- Time stepping
    total_days = 30.0 * 365.0
    nsteps = 360
    if _TS is not None:
        try:
            iterator = iter(_TS(total_days=total_days, nsteps=nsteps))
        except TypeError:
            iterator = _ts_iter(total_days, nsteps)
    else:
        iterator = _ts_iter(total_days, nsteps)

    # ---- Loop
    t_days, qg_list, qo_list = [], [], []
    x_prev = x.copy()
    for dt_days, t_accum in iterator:
        x_n = x.copy()

        # Newton iterations
        for _ in range(opts.max_newton):
            R, J = asm.residual_and_jacobian(x_n, x_prev, dt_days)
            if np.linalg.norm(R, ord=np.inf) < opts.newton_tol:
                break
            dx = solve_linear(J, -R, max_iter=opts.max_lin)
            x_n += dx
            asm.clamp_state_inplace(x_n)

        # Accept the step
        x_prev = x_n.copy()
        x = x_n

        # Surface rates; pass RATE overrides only if your API supports them
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
            # older signature without pwf_overrides
            qo, qg, _ = wells.surface_rates(x, grid, pvt, kr)

        t_days.append(t_accum)
        qg_list.append(qg)
        qo_list.append(qo)

    # ---- Pack results for the UI
    t  = np.asarray(t_days, float)
    qg = np.asarray(qg_list, float)  # Mscf/d
    qo = np.asarray(qo_list, float)  # STB/d
    EUR_g_BCF  = np.trapz(qg, t) / 1e6
    EUR_o_MMBO = np.trapz(qo, t) / 1e6

    return dict(
        t=t, qg=qg, qo=qo,
        EUR_g_BCF=EUR_g_BCF, EUR_o_MMBO=EUR_o_MMBO,
        # placeholders for 3D fields (we’ll fill later)
        p_init_3d=np.full((grid.nz, grid.ny, grid.nx), P0[0]),
        press_matrix=None, pm_mid_psi=None, ooip_3d=None, runtime_s=0.0,
    )
