# core/wells1.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from core.grid1 import Grid
from core.blackoil_pvt1 import BlackOilPVT
from core.relperm1 import CoreyRelPerm

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
        # Build simple mid-row/layer lateral with Peaceman WI
        L_ft = float(msw.get("L_ft", inputs.get("L_ft", 10000.0)))
        num_seg = max(1, int(L_ft / grid.dx))
        j = grid.ny // 2; k = grid.nz // 2
        perfs: List[Perf] = []
        for i in range(num_seg):
            idx = grid.get_idx(i, j, k)
            if idx != -1:
                WI = peaceman_WI(grid, idx, kx=0.1, ky=0.1, dz=grid.dz)  # TODO: wire actual kx,ky
                perfs.append(Perf(cell=idx, WI=WI))

        # Decide control using inputs style from app.py
        pad_ctrl = (inputs.get("pad_ctrl") or "BHP").upper()
        if pad_ctrl == "RATE":
            ctrl = "RATE"
            target = float(inputs.get("pad_rate_mscfd", 100000.0))  # gas rate target in Mscf/d
        else:
            ctrl = "BHP"
            target = float(schedule.get("bhp_psi", inputs.get("pad_bhp_psi", 2500.0)))

        return WellSet([Well(name="W1", control=ctrl, target=target, perfs=perfs)])

    def surface_rates(
        self,
        x: np.ndarray,
        grid: Grid,
        pvt: BlackOilPVT,
        kr: CoreyRelPerm,
        pwf_overrides: Optional[Dict[int, float]] = None,
    ):
        """Return (qo_s, qg_s, qw_s) surface rates. If pwf_overrides is provided,
        it should map well index -> pwf for RATE-controlled wells."""
        P = x[0::3]; Sw = x[1::3]; Sg = x[2::3]
        So = np.clip(1.0 - Sw - Sg, 0.0, 1.0)

        Bo, Bg, Bw = pvt.Bo(P), pvt.Bg(P), pvt.Bw(P)
        muo, mug, muw = pvt.mu_oil(P), pvt.mu_gas(P), pvt.mu_water(P)
        kro, krw, krg = kr.kr(Sw, Sg)

        lam_o = kro / (muo * Bo + 1e-12)
        lam_w = krw / (muw * Bw + 1e-12)
        lam_g = krg / (mug * Bg + 1e-12)

        qo_s = qg_s = qw_s = 0.0
        for wi, w in enumerate(self.wells):
            if w.control == "BHP":
                pwf = w.target
            else:
                # RATE: use override if provided; else fall back to target as guess
                pwf = (pwf_overrides or {}).get(wi, w.target)

            for perf in w.perfs:
                dp = max(P[perf.cell] - pwf, 0.0)
                qo_s += perf.WI * lam_o[perf.cell] * dp
                qw_s += perf.WI * lam_w[perf.cell] * dp
                # free gas only for now (dissolved gas handled later)
                qg_s += perf.WI * lam_g[perf.cell] * dp

        return qo_s, qg_s, qw_s  # Gas = Mscf/d scale is approximate here

def peaceman_WI(grid: Grid, cell: int, kx: float, ky: float, dz: float, rw: float = 0.25, skin: float = 0.0):
    # Decode i,j,k from flat cell index
    nx, ny = grid.nx, grid.ny
    k = cell // (nx * ny); j = (cell - k * nx * ny) // nx; i = cell % nx
    # Anisotropic equivalent radius
    re = 0.28 * ((grid.dx**2) * (ky/kx)**0.5 + (grid.dy**2) * (kx/ky)**0.5) ** 0.5
    kh = np.sqrt(kx * ky)
    conv = 1.0  # fold units into WI for now
    return conv * 2.0 * np.pi * kh * dz / (np.log(re / rw + 1e-12) + skin + 1e-12)
