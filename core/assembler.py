# core/assembler.py
from __future__ import annotations
import numpy as np
from scipy.sparse import lil_matrix
from dataclasses import dataclass

# use the *1 modules everywhere
from core.blackoil_pvt1 import BlackOilPVT
from core.relperm1 import CoreyRelPerm
from core.grid1 import Grid
from core.wells1 import WellSet


_EPS = 1e-12

@dataclass
class Assembler:
    grid: Grid
    pvt: BlackOilPVT
    kr: CoreyRelPerm
    wells: WellSet
    opts: dict | None = None

    # neighbor list of (c, n, T_o, T_w, T_g) with directional harmonic averages
    neighbors: list[tuple[int, int, float, float, float]] = field(default_factory=list)
    rate_well_indices: list[int] = field(default_factory=list)

    def __post_init__(self):
        self._build_neighbors()
        self.rate_well_indices = [wi for wi, w in enumerate(self.wells.wells) if w.control == "RATE"]

    # DOF layout
    def n_cell_dof(self) -> int: return 3 * self.grid.num_cells
    def n_extra_dof(self) -> int: return len(self.rate_well_indices)
    def total_dof(self) -> int:   return self.n_cell_dof() + self.n_extra_dof()

    def ixP(self, c: int) -> int:  return 3 * c
    def ixSw(self, c: int) -> int: return 3 * c + 1
    def ixSg(self, c: int) -> int: return 3 * c + 2
    def ixWell(self, iw_extra: int) -> int: return self.n_cell_dof() + iw_extra

    def clamp_state_inplace(self, x: np.ndarray) -> None:
        P = x[0::3]; Sw = x[1::3]; Sg = x[2::3]
        Sw[:] = np.clip(Sw, self.kr.Swc + 1e-6, 1.0 - self.kr.Sor - 1e-6)
        Sg[:] = np.clip(Sg, self.kr.Sgc, 1.0 - self.kr.Swc - 1e-6)
        So = 1.0 - Sw - Sg
        neg = So < 0
        if np.any(neg):
            Sg[neg] = Sg[neg] + So[neg]
        x[0::3], x[1::3], x[2::3] = P, Sw, Sg

    @staticmethod
    def _harm(a, b):
        return 2.0 * a * b / (a + b + _EPS)

    def _build_neighbors(self):
        g = self.grid
        conv = 0.001127  # unit conversion folded into T
        nbrs = []

        # x-faces
        for k in range(g.nz):
            for j in range(g.ny):
                for i in range(g.nx - 1):
                    c = g.get_idx(i, j, k); n = g.get_idx(i + 1, j, k)
                    area = g.dy * g.dz; dist = g.dx
                    kxo = self._harm(g.kx[c], g.kx[n])
                    kyo = self._harm(g.ky[c], g.ky[n])
                    kzo = self._harm(g.kz[c], g.kz[n])
                    T = conv * area / dist
                    # directional k per phase (same permeability; phase mobility goes elsewhere)
                    nbrs.append((c, n, T * kxo, T * kyo, T * kzo))

        # y-faces
        for k in range(g.nz):
            for j in range(g.ny - 1):
                for i in range(g.nx):
                    c = g.get_idx(i, j, k); n = g.get_idx(i, j + 1, k)
                    area = g.dx * g.dz; dist = g.dy
                    kxo = self._harm(g.kx[c], g.kx[n])
                    kyo = self._harm(g.ky[c], g.ky[n])
                    kzo = self._harm(g.kz[c], g.kz[n])
                    T = conv * area / dist
                    nbrs.append((c, n, T * kxo, T * kyo, T * kzo))

        # z-faces
        for k in range(g.nz - 1):
            for j in range(g.ny):
                for i in range(g.nx):
                    c = g.get_idx(i, j, k); n = g.get_idx(i, j, k + 1)
                    area = g.dx * g.dy; dist = g.dz
                    kxo = self._harm(g.kx[c], g.kx[n])
                    kyo = self._harm(g.ky[c], g.ky[n])
                    kzo = self._harm(g.kz[c], g.kz[n])
                    T = conv * area / dist
                    nbrs.append((c, n, T * kxo, T * kyo, T * kzo))

        self.neighbors = nbrs

    def _pwf_overrides_from_x(self, x: np.ndarray) -> dict[int, float]:
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

        # unpack states
        Pn = x_n[0::3]; Swn = x_n[1::3]; Sgn = x_n[2::3]
        Pm = x_prev[0::3]; Swm = x_prev[1::3]; Sgm = x_prev[2::3]
        So_n = 1.0 - Swn - Sgn
        So_m = 1.0 - Swm - Sgm

        # PVT and relperm + derivatives
        Bo_n, Bg_n, Bw_n = self.pvt.Bo(Pn), self.pvt.Bg(Pn), self.pvt.Bw(Pn)
        Bo_m, Bg_m, Bw_m = self.pvt.Bo(Pm), self.pvt.Bg(Pm), self.pvt.Bw(Pm)
        dBo_dP, dBg_dP, dBw_dP = self.pvt.dBo_dP(Pn), self.pvt.dBg_dP(Pn), self.pvt.dBw_dP(Pn)
        Rs_n, Rs_m = self.pvt.Rs(Pn), self.pvt.Rs(Pm)
        dRs_dP = self.pvt.dRs_dP(Pn)

        muo, mug, muw = self.pvt.mu_oil(Pn), self.pvt.mu_gas(Pn), self.pvt.mu_water(Pn)
        (kro, krw, krg,
         dkro_dSw, dkro_dSg,
         dkrw_dSw, dkrw_dSg,
         dkrg_dSw, dkrg_dSg) = self.kr.kr_and_derivs(Swn, Sgn)

        inv_o = 1.0 / (muo * Bo_n + _EPS)
        inv_w = 1.0 / (muw * Bw_n + _EPS)
        inv_g = 1.0 / (mug * Bg_n + _EPS)

        lam_o = kro * inv_o
        lam_w = krw * inv_w
        lam_g = krg * inv_g

        dlamo_dSw = dkro_dSw * inv_o
        dlamo_dSg = dkro_dSg * inv_o
        dlamw_dSw = dkrw_dSw * inv_w
        dlamw_dSg = dkrw_dSg * inv_w
        dlamg_dSw = dkrg_dSw * inv_g
        dlamg_dSg = dkrg_dSg * inv_g

        phi = self.grid.phi
        Vb = (self.grid.dx * self.grid.dy * self.grid.dz) / 5.615

        # Accumulation with Jacobian pieces
        for c in range(n):
            # oil: Ao = phi*So/Bo
            Ao_n = phi[c] * So_n[c] / max(Bo_n[c], _EPS)
            Ao_m = phi[c] * So_m[c] / max(Bo_m[c], _EPS)
            R[self.ixP(c)] += (Ao_n - Ao_m) * Vb / dt

            # ∂R_o/∂P via dBo/dP
            dAo_dP = phi[c] * So_n[c] * (-1.0) * dBo_dP[c] / (Bo_n[c]**2 + _EPS)
            J[self.ixP(c), self.ixP(c)] += dAo_dP * Vb / dt
            # ∂R_o/∂Sw, ∂R_o/∂Sg via So = 1 - Sw - Sg
            J[self.ixP(c), self.ixSw(c)] += (-phi[c] / (Bo_n[c] + _EPS)) * Vb / dt
            J[self.ixP(c), self.ixSg(c)] += (-phi[c] / (Bo_n[c] + _EPS)) * Vb / dt

            # water: Aw = phi*Sw/Bw
            Aw_n = phi[c] * Swn[c] / max(Bw_n[c], _EPS)
            Aw_m = phi[c] * Swm[c] / max(Bw_m[c], _EPS)
            R[self.ixSw(c)] += (Aw_n - Aw_m) * Vb / dt
            # ∂R_w/∂P via dBw/dP
            dAw_dP = phi[c] * Swn[c] * (-1.0) * dBw_dP[c] / (Bw_n[c]**2 + _EPS)
            J[self.ixSw(c), self.ixP(c)] += dAw_dP * Vb / dt
            # ∂R_w/∂Sw
            J[self.ixSw(c), self.ixSw(c)] += (phi[c] / (Bw_n[c] + _EPS)) * Vb / dt

            # gas: Ag = phi*(Sg/Bg - So*Rs/Bg)
            Ag_n = phi[c] * (Sgn[c] / max(Bg_n[c], _EPS) - So_n[c] * Rs_n[c] / max(Bg_n[c], _EPS))
            Ag_m = phi[c] * (Sgm[c] / max(Bg_m[c], _EPS) - So_m[c] * Rs_m[c] / max(Bg_m[c], _EPS))
            R[self.ixSg(c)] += (Ag_n - Ag_m) * Vb / dt
            # ∂R_g/∂P via dBg/dP and dRs/dP
            term = (So_n[c] * Rs_n[c] - Sgn[c]) / (Bg_n[c]**2 + _EPS)
            dAg_dP = phi[c] * (term * dBg_dP[c] - So_n[c] * dRs_dP[c] / (Bg_n[c] + _EPS))
            J[self.ixSg(c), self.ixP(c)] += dAg_dP * Vb / dt
            # ∂R_g/∂Sw
            J[self.ixSg(c), self.ixSw(c)] += (phi[c] * (Rs_n[c] / (Bg_n[c] + _EPS))) * Vb / dt
            # ∂R_g/∂Sg
            J[self.ixSg(c), self.ixSg(c)] += (phi[c] * (1.0 / (Bg_n[c] + _EPS))) * Vb / dt
            J[self.ixSg(c), self.ixSg(c)] += (phi[c] * (Rs_n[c] / (Bg_n[c] + _EPS)) * 0.0)  # placeholder for ∂(So)/∂Sg already in accumulation above

        # Flux terms with λ dependence on Sw,Sg (TPFA-like)
        for c, nbh, Tx, Ty, Tz in self.neighbors:
            T = Tx  # single scalar T since directional already folded (using kx/ky/kz uniformly)
            dP = Pn[c] - Pn[nbh]

            # Oil
            lam_face = 0.5 * (lam_o[c] + lam_o[nbh])
            R[self.ixP(c)]   += T * lam_face * dP
            R[self.ixP(nbh)] += T * lam_face * (-dP)

            J[self.ixP(c),   self.ixP(c)]   += T * lam_face
            J[self.ixP(c),   self.ixP(nbh)] += -T * lam_face
            J[self.ixP(nbh), self.ixP(nbh)] += T * lam_face
            J[self.ixP(nbh), self.ixP(c)]   += -T * lam_face

            # λ derivatives wrt Sw/Sg
            dlam_face_dSw_c   = 0.5 * dlamo_dSw[c]
            dlam_face_dSw_nbh = 0.5 * dlamo_dSw[nbh]
            dlam_face_dSg_c   = 0.5 * dlamo_dSg[c]
            dlam_face_dSg_nbh = 0.5 * dlamo_dSg[nbh]

            J[self.ixP(c),   self.ixSw(c)]   += T * dlam_face_dSw_c   * dP
            J[self.ixP(c),   self.ixSw(nbh)] += T * dlam_face_dSw_nbh * dP
            J[self.ixP(c),   self.ixSg(c)]   += T * dlam_face_dSg_c   * dP
            J[self.ixP(c),   self.ixSg(nbh)] += T * dlam_face_dSg_nbh * dP

            J[self.ixP(nbh), self.ixSw(c)]   += -T * dlam_face_dSw_c   * dP
            J[self.ixP(nbh), self.ixSw(nbh)] += -T * dlam_face_dSw_nbh * dP
            J[self.ixP(nbh), self.ixSg(c)]   += -T * dlam_face_dSg_c   * dP
            J[self.ixP(nbh), self.ixSg(nbh)] += -T * dlam_face_dSg_nbh * dP

            # Water
            lam_face = 0.5 * (lam_w[c] + lam_w[nbh])
            R[self.ixSw(c)]   += T * lam_face * dP
            R[self.ixSw(nbh)] += T * lam_face * (-dP)

            J[self.ixSw(c),   self.ixP(c)]   += T * lam_face
            J[self.ixSw(c),   self.ixP(nbh)] += -T * lam_face
            J[self.ixSw(nbh), self.ixP(nbh)] += T * lam_face
            J[self.ixSw(nbh), self.ixP(c)]   += -T * lam_face

            dlam_face_dSw_c   = 0.5 * dlamw_dSw[c]
            dlam_face_dSw_nbh = 0.5 * dlamw_dSw[nbh]
            dlam_face_dSg_c   = 0.5 * dlamw_dSg[c]
            dlam_face_dSg_nbh = 0.5 * dlamw_dSg[nbh]

            J[self.ixSw(c),   self.ixSw(c)]   += T * dlam_face_dSw_c   * dP
            J[self.ixSw(c),   self.ixSw(nbh)] += T * dlam_face_dSw_nbh * dP
            J[self.ixSw(c),   self.ixSg(c)]   += T * dlam_face_dSg_c   * dP
            J[self.ixSw(c),   self.ixSg(nbh)] += T * dlam_face_dSg_nbh * dP

            J[self.ixSw(nbh), self.ixSw(c)]   += -T * dlam_face_dSw_c   * dP
            J[self.ixSw(nbh), self.ixSw(nbh)] += -T * dlam_face_dSw_nbh * dP
            J[self.ixSw(nbh), self.ixSg(c)]   += -T * dlam_face_dSg_c   * dP
            J[self.ixSw(nbh), self.ixSg(nbh)] += -T * dlam_face_dSg_nbh * dP

            # Gas
            lam_face = 0.5 * (lam_g[c] + lam_g[nbh])
            R[self.ixSg(c)]   += T * lam_face * dP
            R[self.ixSg(nbh)] += T * lam_face * (-dP)

            J[self.ixSg(c),   self.ixP(c)]   += T * lam_face
            J[self.ixSg(c),   self.ixP(nbh)] += -T * lam_face
            J[self.ixSg(nbh), self.ixP(nbh)] += T * lam_face
            J[self.ixSg(nbh), self.ixP(c)]   += -T * lam_face

            dlam_face_dSw_c   = 0.5 * dlamg_dSw[c]
            dlam_face_dSw_nbh = 0.5 * dlamg_dSw[nbh]
            dlam_face_dSg_c   = 0.5 * dlamg_dSg[c]
            dlam_face_dSg_nbh = 0.5 * dlamg_dSg[nbh]

            J[self.ixSg(c),   self.ixSw(c)]   += T * dlam_face_dSw_c   * dP
            J[self.ixSg(c),   self.ixSw(nbh)] += T * dlam_face_dSw_nbh * dP
            J[self.ixSg(c),   self.ixSg(c)]   += T * dlam_face_dSg_c   * dP
            J[self.ixSg(c),   self.ixSg(nbh)] += T * dlam_face_dSg_nbh * dP

            J[self.ixSg(nbh), self.ixSw(c)]   += -T * dlam_face_dSw_c   * dP
            J[self.ixSg(nbh), self.ixSw(nbh)] += -T * dlam_face_dSw_nbh * dP
            J[self.ixSg(nbh), self.ixSg(c)]   += -T * dlam_face_dSg_c   * dP
            J[self.ixSg(nbh), self.ixSg(nbh)] += -T * dlam_face_dSg_nbh * dP

        # Wells — include λ(S) sensitivity + P and pwf couplings
        for wi, w in enumerate(self.wells.wells):
            if w.control == "BHP":
                pwf = w.target
                extra_idx = None
            else:
                extra_i = self.rate_well_indices.index(wi)
                extra_idx = self.ixWell(extra_i)
                pwf = x_n[extra_idx]

            for perf in w.perfs:
                c = perf.cell
                dp = Pn[c] - pwf
                if dp <= 0.0:
                    continue

                # sinks
                qo = perf.WI * lam_o[c] * dp
                qw = perf.WI * lam_w[c] * dp
                qg = perf.WI * lam_g[c] * dp

                R[self.ixP(c)]  -= qo
                R[self.ixSw(c)] -= qw
                R[self.ixSg(c)] -= qg

                # dR/dP(c)
                J[self.ixP(c),  self.ixP(c)]  += -perf.WI * lam_o[c]
                J[self.ixSw(c), self.ixP(c)]  += -perf.WI * lam_w[c]
                J[self.ixSg(c), self.ixP(c)]  += -perf.WI * lam_g[c]

                # dR/dpwf
                if extra_idx is not None:
                    J[self.ixP(c),  extra_idx] +=  perf.WI * lam_o[c]
                    J[self.ixSw(c), extra_idx] +=  perf.WI * lam_w[c]
                    J[self.ixSg(c), extra_idx] +=  perf.WI * lam_g[c]

                # dR/dSw and dR/dSg via λ
                J[self.ixP(c),  self.ixSw(c)]  += -perf.WI * dlamo_dSw[c] * dp
                J[self.ixP(c),  self.ixSg(c)]  += -perf.WI * dlamo_dSg[c] * dp

                J[self.ixSw(c), self.ixSw(c)] += -perf.WI * dlamw_dSw[c] * dp
                J[self.ixSw(c), self.ixSg(c)] += -perf.WI * dlamw_dSg[c] * dp

                J[self.ixSg(c), self.ixSw(c)] += -perf.WI * dlamg_dSw[c] * dp
                J[self.ixSg(c), self.ixSg(c)] += -perf.WI * dlamg_dSg[c] * dp

        # Rate-well constraints (gas rate)
        for extra_i, wi in enumerate(self.rate_well_indices):
            w = self.wells.wells[wi]
            row = self.ixWell(extra_i)
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
                J[row, self.ixP(c)] += perf.WI * lamg
                J[row, row]         += -perf.WI * lamg
                # add λ(S) sensitivity
                J[row, self.ixSw(c)] += perf.WI * dlamg_dSw[c] * dp
                J[row, self.ixSg(c)] += perf.WI * dlamg_dSg[c] * dp

            R[row] = F - w.target

        return R, J.tocsc()
