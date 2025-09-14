# core/wells.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
from core.grid import Grid
from core.blackoil_pvt import BlackOilPVT
from core.relperm import CoreyRelPerm

@dataclass
class Perf:
    cell: int
    WI: float

@dataclass
class Well:
    name: str
    control: str  # "BHP" or "RATE"
    target: float
    bhp_min: float = 0.0
    perfs: List[Perf] = field(default_factory=list)

@dataclass
class WellSet:
    wells: List[Well]

    @staticmethod
    def from_inputs(msw: dict, schedule: dict, grid: Grid, inputs: Dict[str, Any]) -> "WellSet":
        # simple single-well construction along mid row, mid layer (matches your UI idea)
        L_ft = float(msw.get("L_ft", inputs.get("L_ft", 10000.0)))
        num_seg = max(1, int(L_ft / grid.dx))
        j = grid.ny // 2; k = grid.nz // 2
        perfs = []
        for i in range(num_seg):
            idx = grid.get_idx(i, j, k)
            if idx != -1:
                WI = peaceman_WI(grid, idx, kx=0.1, ky=0.1, dz=grid.dz)  # placeholder k; we’ll wire actual k later
                perfs.append(Perf(cell=idx, WI=WI))

        ctrl = "BHP"
        target = float(schedule.get("bhp_psi", inputs.get("pad_bhp_psi", 2500.0)))
        if schedule.get("mode", "").upper() == "RATE":
            ctrl = "RATE"
            target = float(schedule.get("total_rate_stbpd", 1000.0))

        w = Well(name="W1", control=ctrl, target=target, perfs=perfs)
        return WellSet([w])

    # surface rates using simple fractional flow at perforations (skeleton)
    def surface_rates(self, x, grid: Grid, pvt: BlackOilPVT, kr: CoreyRelPerm):
        ncell = grid.num_cells
        P = x[0::3]; Sw = x[1::3]; Sg = x[2::3]
        So = np.clip(1.0 - Sw - Sg, 0.0, 1.0)

        Bo, Bg, Bw = pvt.Bo(P), pvt.Bg(P), pvt.Bw(P)
        muo, mug, muw = pvt.mu_oil(P), pvt.mu_gas(P), pvt.mu_water(P)
        kro, krw, krg = kr.kr(Sw, Sg)

        lam_o = kro / (muo * Bo + 1e-12)
        lam_w = krw / (muw * Bw + 1e-12)
        lam_g = krg / (mug * Bg + 1e-12)

        qo_s, qg_s, qw_s = 0.0, 0.0, 0.0
        for w in self.wells:
            if w.control == "BHP":
                pwf = w.target
            else:
                # For RATE control we’ll back-calculate pwf later (Phase 1b). For now assume target ~ pwf proxy.
                pwf = w.bhp_min or w.target

            for perf in w.perfs:
                dp = max(P[perf.cell] - pwf, 0.0)
                qo_s += perf.WI * lam_o[perf.cell] * dp
                qw_s += perf.WI * lam_w[perf.cell] * dp
                qg_s += perf.WI * lam_g[perf.cell] * dp + perf.WI * lam_o[perf.cell] * dp * pvt.Rs(P[perf.cell])

        # convert gas to Mscf/d (roughly already surface vol), oil STB/d okay
        return qo_s, qg_s/1000.0, qw_s

def peaceman_WI(grid: Grid, cell: int, kx: float, ky: float, dz: float, rw: float = 0.25, skin: float = 0.0):
    """Anisotropic Peaceman WI (2D horizontal well through a cell)."""
    # indices from cell id
    nx, ny = grid.nx, grid.ny
    k = cell // (nx * ny); j = (cell - k * nx * ny) // nx; i = cell % nx

    # effective radius (common anisotropic form)
    re = 0.28 * ((grid.dx**2) * (ky/kx)**0.5 + (grid.dy**2) * (kx/ky)**0.5) ** 0.5
    kh = np.sqrt(kx * ky)
    WI = 2.0 * np.pi * kh * dz / (np.log(re / rw + 1e-12) + skin + 1e-12)
    # unit conversion for field units is handled elsewhere; keep dimensionless-ish here
    return WI
