# core/assembler.py
from __future__ import annotations
import numpy as np
from scipy.sparse import lil_matrix
from dataclasses import dataclass

from core.grid import Grid
from core.blackoil_pvt import BlackOilPVT
from core.relperm import CoreyRelPerm
from core.wells import WellSet

@dataclass
class Assembler:
    grid: Grid
    pvt: BlackOilPVT
    kr: CoreyRelPerm
    wells: WellSet
    opts: any

    def clamp_state_inplace(self, x):
        # x: [P, Sw, Sg] per cell
        P = x[0::3]; Sw = x[1::3]; Sg = x[2::3]
        Sw[:] = np.clip(Sw, self.kr.Swc + 1e-6, 1.0 - self.kr.Sor - 1e-6)
        Sg[:] = np.clip(Sg, self.kr.Sgc, 1.0 - self.kr.Swc - 1e-6)
        So = 1.0 - Sw - Sg
        neg = So < 0
        if np.any(neg):
            # push Sg down a bit if So went negative
            Sg[neg] = Sg[neg] + So[neg]
        x[0::3], x[1::3], x[2::3] = P, Sw, Sg

    def residual_and_jacobian(self, x_n, x_prev, dt_days):
        """Skeletal 3eq per cell (oil, water, gas) with accumulation only + simple well sinks.
           Next pass we’ll add full flux terms and all partials.
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

        # accumulation terms (black-oil canonical forms, simplified rock compressibility)
        phi = 0.12  # placeholder porosity
        Vb = (self.grid.dx * self.grid.dy * self.grid.dz) / 5.615  # bbl per cell (ft^3 -> bbl)
        dt = max(dt_days, 1e-9)

        # Oil equation index map helpers
        def ixP(c): return 3*c
        def ixSw(c): return 3*c + 1
        def ixSg(c): return 3*c + 2

        for c in range(n):
            # oil accumulation (with Rs coupling in gas eq later)
            Ao_n = phi * So_n[c] / max(Bo_n[c], 1e-12)
            Ao_m = phi * So_m[c] / max(Bo_m[c], 1e-12)
            R[ixP(c)] += (Ao_n - Ao_m) * Vb / dt

            # water accumulation
            Aw_n = phi * Swn[c] / max(Bw_n[c], 1e-12)
            Aw_m = phi * Swm[c] / max(Bw_m[c], 1e-12)
            R[ixSw(c)] += (Aw_n - Aw_m) * Vb / dt

            # gas accumulation (free gas minus dissolved from oil)
            Ag_n = phi * Sgn[c] / max(Bg_n[c], 1e-12) - phi * So_n[c] * Rs_n[c] / max(Bg_n[c], 1e-12)
            Ag_m = phi * Sgm[c] / max(Bg_m[c], 1e-12) - phi * So_m[c] * Rs_m[c] / max(Bg_m[c], 1e-12)
            R[ixSg(c)] += (Ag_n - Ag_m) * Vb / dt

        # simple well sink terms (BHP control) — add to residual
        # (In next pass we’ll add Jacobian entries and full RATE control unknowns)
        qo_s, qg_s, qw_s = self.wells.surface_rates(x_n, self.grid, self.pvt, self.kr)
        # Distribute equally over perfs (placeholder)
        total_perfs = sum(len(w.perfs) for w in self.wells.wells)
        if total_perfs > 0:
            qo_cell = qo_s / total_perfs
            qw_cell = qw_s / total_perfs
            qg_cell = qg_s / total_perfs
            for w in self.wells.wells:
                for p in w.perfs:
                    R[ixP(p.cell)] -= qo_cell
                    R[ixSw(p.cell)] -= qw_cell
                    R[ixSg(p.cell)] -= qg_cell

        # Jacobian: start with diagonal stabilization so Newton doesn’t blow up before we add full partials
        for c in range(n):
            J[ixP(c), ixP(c)] = 1.0
            J[ixSw(c), ixSw(c)] = 1.0
            J[ixSg(c), ixSg(c)] = 1.0

        return R, J.tocsc()
