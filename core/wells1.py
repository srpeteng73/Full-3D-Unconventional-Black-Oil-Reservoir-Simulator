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
        L_ft = float(msw.get("L_ft", inputs.get("L_ft", 10000.0)))
        num_seg = max(1, int(L_ft / grid.dx))
        j = grid.ny // 2; k = grid.nz // 2

        perfs: List[Perf] = []
        for i in range(num_seg):
            idx = grid.get_idx(i, j, k)
            if idx != -1:
                kx = float(grid.kx[idx])
                ky = float(grid.ky[idx])
                WI = peaceman_WI(grid, idx, kx=kx, ky=ky, dz=grid.dz)
                perfs.append(Perf(cell=idx, WI=WI))

        pad_ctrl = (inputs.get("pad_ctrl") or "BHP").upper()
        if pad_ctrl == "RATE":
            ctrl = "RATE"
            target = float(inputs.get("pad_rate_mscfd", 100000.0))
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
        P = x[0::3]; Sw = x[1::3]; Sg = x[2::3]
        So = np.clip(1.0 - Sw - Sg, 0.0, 1.0)

        Bo = pvt.Bo(P); Bg = pvt.Bg(P); Bw = pvt.Bw(P)
        muo = pvt.mu_oil(P); mug = pvt.mu_gas(P); muw = pvt.mu_water(P)
        kro, krw, krg = kr.kr(Sw, Sg)

        lam_o = kro / (muo * Bo + 1e-12)
        lam_w = krw / (muw * Bw + 1e-12)
        lam_g = krg / (mug * Bg + 1e-12)

        qo_s = qg_s = qw_s = 0.0
        for wi, w in enumerate(self.wells):
            if w.control == "BHP":
                pwf = w.target
            else:
                pwf = (pwf_overrides or {}).get(wi, w.target)

            for perf in w.perfs:
                dp = max(P[perf.cell] - pwf, 0.0)
                qo_s += perf.WI * lam_o[perf.cell] * dp
                qw_s += perf.WI * lam_w[perf.cell] * dp
                qg_s += perf.WI * lam_g[perf.cell] * dp

        return qo_s, qg_s, qw_s

def peaceman_WI(grid: Grid, cell: int, kx: float, ky: float, dz: float, rw: float = 0.25, skin: float = 0.0):
    nx, ny = grid.nx, grid.ny
    k = cell // (nx * ny); j = (cell - k * nx * ny) // nx; i = cell % nx
    # anisotropic effective radius (Peaceman)
    re = 0.28 * np.sqrt((grid.dx**2) * np.sqrt(ky / max(kx, 1e-12)) +
                        (grid.dy**2) * np.sqrt(kx / max(ky, 1e-12)))
    kh = np.sqrt(kx * ky)
    conv = 1.0  # fold units in WI for now
    return conv * 2.0 * np.pi * max(kh, 1e-12) * dz / (np.log(max(re / rw, 1.0)) + skin + 1e-12)
