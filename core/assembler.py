# core/assembler.py
from __future__ import annotations
import numpy as np
from scipy.sparse import lil_matrix
from dataclasses import dataclass, field

from core.blackoil_pvt1 import BlackOilPVT
from core.relperm1 import CoreyRelPerm
from core.grid1 import Grid
from core.wells1 import WellSet

@dataclass
class Assembler:
    grid: Grid
    pvt: BlackOilPVT
    kr: CoreyRelPerm
    wells: WellSet
    opts: dict | None = None

    # neighbor topology and transmissibilities
    neighbors: list[tuple[int, int, float]] = field(default_factory=list)
    rate_well_indices: list[int] = field(default_factory=list)  # indices into wells.wells

    def __post_init__(self):
        self._build_neighbors()
        self.rate_well_indices = [wi for wi, w in enumerate(self.wells.wells) if w.control == "RATE"]

    # --- DOF layout helpers
    def n_cell_dof(self) -> int:
        return 3 * self.grid.num_cells

    def n_extra_dof(self) -> int:
        return len(self.rate_well_indices)  # one pwf unknown per rate-controlled well

    def total_dof(self) -> int:
        return self.n_cell_dof() + self.n_extra_dof()

    def ixP(self, c: int) -> int: return 3 * c
    def ixSw(self, c: int) -> int: return 3 * c + 1
    def ixSg(self, c: int) -> int: return 3 * c + 2
    def ixWell(self, iw_extra: int) -> int: return self.n_cell_dof() + iw_extra  # row/col for well pwf or constraint

    def clamp_state_inplace(self, x: np.ndarray) -> None:
        P = x[0::3]; Sw = x[1::3]; Sg = x[2::3]
        Sw[:] = np.clip(Sw, self.kr.Swc + 1e-6, 1.0 - self.kr.Sor - 1e-6)
        Sg[:] = np.clip(Sg, self.kr.Sgc, 1.0 - self.kr.Swc - 1e-6)
        So = 1.0 - Sw - Sg
        neg = So < 0
        if np.any(neg):
            Sg[neg] = Sg[neg] + So[neg]
        x[0::3], x[1::3], x[2::3] = P, Sw, Sg

        # No clamp on well unknowns (pwf) here; leave Newton to solve it

    def _build_neighbors(self):
        """Simple 6-connectivity with a constant effective permeability (skeleton).
        T = 0.001127 * k_eff * Area / Distance  (units folded into k_eff)"""
        g = self.grid
        k_eff_md = 0.1  # TODO: wire real Kx,Ky,Kz fields
        conv = 0.001127

        # x-faces
        for k in range(g.nz):
            for j in range(g.ny):
                for i in range(g.nx - 1):
                    c = g.get_idx(i, j, k)
                    n = g.get_idx(i + 1, j, k)
                    area = g.dy * g.dz
                    dist = g.dx
                    T = conv * k_eff_md * area / dist
                    self.neighbors.append((c, n, T))

        # y-faces
        for k in range(g.nz):
            for j in range(g.ny - 1):
                for i in range(g.nx):
                    c = g.get_idx(i, j, k)
                    n = g.get_idx(i, j + 1, k)
                    area = g.dx * g.dz
                    dist = g.dy
                    T = conv * k_eff_md * area / dist
                    self.neighbors.append((c, n, T))

        # z-faces
        for k in range(g.nz - 1):
            for j in range(g.ny):
                for i in range(g.nx):
                    c = g.get_idx(i, j, k)
                    n = g.get_idx(i, j, k + 1)
                    area = g.dx * g.dy
                    dist = g.dz
                    T = conv * k_eff_md * area / dist
                    self.neighbors.append((c, n, T))

    def _pwf_overrides_from_x(self, x: np.ndarray) -> dict[int, float]:
        """Map well index -> pwf for RATE wells, from tail of x."""
        overrides = {}
        for extra_i, wi in enumerate(self.rate_well_indices):
            overrides[wi] = x[self.ixWell(extra_i)]
        return overrides

    def residual_and_jacobian(self, x_n: np.ndarray, x_prev: np.ndarray, dt_days: float):
        n = self.grid.num_cells
        ndof = self.total_dof()
        R = np.zeros(ndof)
        J = lil_matrix((ndof, ndof))
        dt = max(dt_days, 1e-9)

        # unpack states (cells)
        Pn = x_n[0::3]; Swn = x_n[1::3]; Sgn = x_n[2::3]
        Pm = x_prev[0::3]; Swm = x_prev[1::3]; Sgm = x_prev[2::3]
        So_n = 1.0 - Swn - Sgn
        So_m = 1.0 - Swm - Sgm

        # fluid & mobility at n
        Bo_n, Bg_n, Bw_n = self.pvt.Bo(Pn), self.pvt.Bg(Pn), self.pvt.Bw(Pn)
        Bo_m, Bg_m, Bw_m = self.pvt.Bo(Pm), self.pvt.Bg(Pm), self.pvt.Bw(Pm)
        Rs_n, Rs_m = self.pvt.Rs(Pn), self.pvt.Rs(Pm)
        muo, mug, muw = self.pvt.mu_oil(Pn), self.pvt.mu_gas(Pn), self.pvt.mu_water(Pn)
        kro, krw, krg = self.kr.kr(Swn, Sgn)

        lam_o = kro / (muo * Bo_n + 1e-12)
        lam_w = krw / (muw * Bw_n + 1e-12)
        lam_g = krg / (mug * Bg_n + 1e-12)

        # accumulation
        phi = 0.12  # placeholder porosity
        Vb = (self.grid.dx * self.grid.dy * self.grid.dz) / 5.615  # bbl per cell

        for c in range(n):
            # Oil accumulation
            Ao_n = phi * So_n[c] / max(Bo_n[c], 1e-12)
            Ao_m = phi * So_m[c] / max(Bo_m[c], 1e-12)
            R[self.ixP(c)] += (Ao_n - Ao_m) * Vb / dt

            # Water accumulation
            Aw_n = phi * Swn[c] / max(Bw_n[c], 1e-12)
            Aw_m = phi * Swm[c] / max(Bw_m[c], 1e-12)
            R[self.ixSw(c)] += (Aw_n - Aw_m) * Vb / dt

            # Gas accumulation (free - dissolved)
            Ag_n = phi * Sgn[c] / max(Bg_n[c], 1e-12) - phi * So_n[c] * Rs_n[c] / max(Bg_n[c], 1e-12)
            Ag_m = phi * Sgm[c] / max(Bg_m[c], 1e-12) - phi * So_m[c] * Rs_m[c] / max(Bg_m[c], 1e-12)
            R[self.ixSg(c)] += (Ag_n - Ag_m) * Vb / dt

        # flux terms (TPFA-like) — symmetric add/subtract, simple averaging of mobility
        for c, nbh, T in self.neighbors:
            # oil
            lam_face_o = 0.5 * (lam_o[c] + lam_o[nbh])
            R[self.ixP(c)]   += T * lam_face_o * (Pn[c] - Pn[nbh])
            R[self.ixP(nbh)] += T * lam_face_o * (Pn[nbh] - Pn[c])
            # Jacobian wrt pressures (ignore dλ/dP for now)
            J[self.ixP(c),   self.ixP(c)]   += T * lam_face_o
            J[self.ixP(c),   self.ixP(nbh)] += -T * lam_face_o
            J[self.ixP(nbh), self.ixP(nbh)] += T * lam_face_o
            J[self.ixP(nbh), self.ixP(c)]   += -T * lam_face_o

            # water
            lam_face_w = 0.5 * (lam_w[c] + lam_w[nbh])
            R[self.ixSw(c)]   += T * lam_face_w * (Pn[c] - Pn[nbh])
            R[self.ixSw(nbh)] += T * lam_face_w * (Pn[nbh] - Pn[c])
            J[self.ixSw(c),   self.ixP(c)]   += T * lam_face_w
            J[self.ixSw(c),   self.ixP(nbh)] += -T * lam_face_w
            J[self.ixSw(nbh), self.ixP(nbh)] += T * lam_face_w
            J[self.ixSw(nbh), self.ixP(c)]   += -T * lam_face_w

            # gas
            lam_face_g = 0.5 * (lam_g[c] + lam_g[nbh])
            R[self.ixSg(c)]   += T * lam_face_g * (Pn[c] - Pn[nbh])
            R[self.ixSg(nbh)] += T * lam_face_g * (Pn[nbh] - Pn[c])
            J[self.ixSg(c),   self.ixP(c)]   += T * lam_face_g
            J[self.ixSg(c),   self.ixP(nbh)] += -T * lam_face_g
            J[self.ixSg(nbh), self.ixP(nbh)] += T * lam_face_g
            J[self.ixSg(nbh), self.ixP(c)]   += -T * lam_face_g

        # wells — sinks into cell residuals
        pwf_over = self._pwf_overrides_from_x(x_n) if self.n_extra_dof() > 0 else {}
        # distribute sinks and add Jacobian entries wrt P (and pwf for rate wells)
        for wi, w in enumerate(self.wells.wells):
            # choose pwf
            if w.control == "BHP":
                pwf = w.target
                extra_idx = None
            else:
                # RATE well: pwf is an unknown at ixWell(extra_i)
                extra_i = self.rate_well_indices.index(wi)
                extra_idx = self.ixWell(extra_i)
                pwf = x_n[extra_idx]

            for perf in w.perfs:
                c = perf.cell
                dp = Pn[c] - pwf
                if dp <= 0.0:
                    continue

                # phase sinks
                qo = perf.WI * lam_o[c] * dp
                qw = perf.WI * lam_w[c] * dp
                qg = perf.WI * lam_g[c] * dp

                # subtract into residuals
                R[self.ixP(c)]  -= qo
                R[self.ixSw(c)] -= qw
                R[self.ixSg(c)] -= qg

                # derivatives w.r.t P(c)
                J[self.ixP(c),  self.ixP(c)]  += -perf.WI * lam_o[c]
                J[self.ixSw(c), self.ixP(c)]  += -perf.WI * lam_w[c]
                J[self.ixSg(c), self.ixP(c)]  += -perf.WI * lam_g[c]

                # derivatives w.r.t pwf (if RATE well)
                if extra_idx is not None:
                    J[self.ixP(c),  extra_idx] +=  perf.WI * lam_o[c]
                    J[self.ixSw(c), extra_idx] +=  perf.WI * lam_w[c]
                    J[self.ixSg(c), extra_idx] +=  perf.WI * lam_g[c]

        # rate-control constraints (gas rate in Mscf/d)
        for extra_i, wi in enumerate(self.rate_well_indices):
            w = self.wells.wells[wi]
            row = self.ixWell(extra_i)  # constraint row also placed at this index
            pwf = x_n[row]
            F = 0.0
            for perf in w.perfs:
                c = perf.cell
                dp = Pn[c] - pwf
                if dp <= 0.0:
                    continue
                lamg = lam_g[c]
                qg = perf.WI * lamg * dp
                F += qg

                # dF/dP(c)
                J[row, self.ixP(c)] += perf.WI * lamg
                # dF/dpwf
                J[row, row] += -perf.WI * lamg

            R[row] = F - w.target  # enforce gas rate = target

        return R, J.tocsc()
