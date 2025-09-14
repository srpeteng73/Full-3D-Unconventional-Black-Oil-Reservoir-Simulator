# core/assembler.py
from __future__ import annotations
import numpy as np
from scipy.sparse import lil_matrix

from dataclasses import dataclass

# Use the *1.py modules (Option B)
from core.blackoil_pvt1 import BlackOilPVT
from core.relperm1 import CoreyRelPerm
from core.grid1 import Grid
from core.wells1 import WellSet
# (linear1 / timestepping1 are used by the engine driver, not here)

@dataclass
class Assembler:
    grid: Grid
    pvt: BlackOilPVT
    kr: CoreyRelPerm
    wells: WellSet
    opts: dict | None = None

    # Clamp to physical bounds so Newton doesn’t wander
    def clamp_state_inplace(self, x: np.ndarray) -> None:
        # x is [P0, Sw0, Sg0, P1, Sw1, Sg1, ...]
        P = x[0::3]; Sw = x[1::3]; Sg = x[2::3]
        Sw[:] = np.clip(Sw, self.kr.Swc + 1e-6, 1.0 - self.kr.Sor - 1e-6)
        Sg[:] = np.clip(Sg, self.kr.Sgc, 1.0 - self.kr.Swc - 1e-6)
        So = 1.0 - Sw - Sg
        neg = So < 0
        if np.any(neg):
            # If oil would go negative, shave it from gas
            Sg[neg] = Sg[neg] + So[neg]
        x[0::3], x[1::3], x[2::3] = P, Sw, Sg

    def residual_and_jacobian(self, x_n: np.ndarray, x_prev: np.ndarray, dt_days: float):
        """
        Skeleton black-oil residuals with accumulation + simple well sinks.
        (Next pass: add full face fluxes and 3x3 Jacobian blocks.)
        """
        n = self.grid.num_cells
        ndof = 3 * n
        R = np.zeros(ndof)
        J = lil_matrix((ndof, ndof))

        # unpack
        Pn = x_n[0::3]; Swn = x_n[1::3]; Sgn = x_n[2::3]
        Pm = x_prev[0::3]; Swm = x_prev[1::3]; Sgm = x_prev[2::3]

        So_n = 1.0 - Swn - Sgn
        So_m = 1.0 - Swm - Sgm

        # PVT at n and m
        Bo_n, Bg_n, Bw_n = self.pvt.Bo(Pn), self.pvt.Bg(Pn), self.pvt.Bw(Pn)
        Bo_m, Bg_m, Bw_m = self.pvt.Bo(Pm), self.pvt.Bg(Pm), self.pvt.Bw(Pm)
        Rs_n, Rs_m = self.pvt.Rs(Pn), self.pvt.Rs(Pm)

        # accumulation (rock compressibility ignored for now)
        phi = 0.12  # placeholder porosity (will be field/array later)
        Vb = (self.grid.dx * self.grid.dy * self.grid.dz) / 5.615  # ft^3 → bbl
        dt = max(dt_days, 1e-9)

        def ixP(c): return 3 * c
        def ixSw(c): return 3 * c + 1
        def ixSg(c): return 3 * c + 2

        for c in range(n):
            # Oil accumulation
            Ao_n = phi * So_n[c] / max(Bo_n[c], 1e-12)
            Ao_m = phi * So_m[c] / max(Bo_m[c], 1e-12)
            R[ixP(c)] += (Ao_n - Ao_m) * Vb / dt

            # Water accumulation
            Aw_n = phi * Swn[c] / max(Bw_n[c], 1e-12)
            Aw_m = phi * Swm[c] / max(Bw_m[c], 1e-12)
            R[ixSw(c)] += (Aw_n - Aw_m) * Vb / dt

            # Gas accumulation: free gas minus dissolved from oil
            Ag_n = phi * Sgn[c] / max(Bg_n[c], 1e-12) - phi * So_n[c] * Rs_n[c] / max(Bg_n[c], 1e-12)
            Ag_m = phi * Sgm[c] / max(Bg_m[c], 1e-12) - phi * So_m[c] * Rs_m[c] / max(Bg_m[c], 1e-12)
            R[ixSg(c)] += (Ag_n - Ag_m) * Vb / dt

        # Simple well sinks (BHP control). Next pass: add Jacobian & rate control unknowns.
        qo_s, qg_s, qw_s = self.wells.surface_rates(x_n, self.grid, self.pvt, self.kr)
        total_perfs = sum(len(w.perfs) for w in self.wells.wells)
        if total_perfs > 0:
            qo_cell = qo_s / total_perfs
            qw_cell = qw_s / total_perfs
            qg_cell = qg_s / total_perfs
            for w in self.wells.wells:
                for p in w.perfs:
                    R[ixP(p.cell)]  -= qo_cell
                    R[ixSw(p.cell)] -= qw_cell
                    R[ixSg(p.cell)] -= qg_cell

        # Diagonal stabilization so Newton doesn’t explode before full Jacobian
        for c in range(n):
            J[ixP(c), ixP(c)]   = 1.0
            J[ixSw(c), ixSw(c)] = 1.0
            J[ixSg(c), ixSg(c)] = 1.0

        return R, J.tocsc()
