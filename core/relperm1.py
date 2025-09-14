# core/relperm.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class CoreyRelPerm:
    Swc: float
    Sor: float
    Sgc: float
    kro_end: float
    krw_end: float
    krg_end: float
    no: float
    nw: float
    ng: float = 2.0

    @staticmethod
    def from_inputs(rp: dict) -> "CoreyRelPerm":
        return CoreyRelPerm(
            Swc=float(rp["Swc"]), Sor=float(rp["Sor"]), Sgc=0.05,
            kro_end=float(rp["kro_end"]), krw_end=float(rp["krw_end"]),
            krg_end=0.9, no=float(rp["no"]), nw=float(rp["nw"])
        )

    def clamp(self, Sw, Sg):
        Sw = np.clip(Sw, self.Swc, 1.0 - self.Sor)
        Sg = np.clip(Sg, self.Sgc, 1.0 - self.Swc)
        So = np.clip(1.0 - Sw - Sg, 0.0, 1.0)
        return Sw, Sg, So

    def kr(self, Sw, Sg):
        Sw, Sg, So = self.clamp(Sw, Sg)
        denom_os = max(1e-12, (1.0 - self.Swc - self.Sor))
        denom_g = max(1e-12, (1.0 - self.Swc - self.Sgc))
        Swn = (Sw - self.Swc) / denom_os
        Son = (So - self.Sor) / denom_os
        Sgn = (Sg - self.Sgc) / denom_g
        krw = self.krw_end * np.power(np.clip(Swn, 0, 1), self.nw)
        kro = self.kro_end * np.power(np.clip(Son, 0, 1), self.no)
        krg = self.krg_end * np.power(np.clip(Sgn, 0, 1), self.ng)
        return kro, krw, krg
