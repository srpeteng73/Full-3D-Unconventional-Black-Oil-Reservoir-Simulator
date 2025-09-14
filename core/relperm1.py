# core/relperm1.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

_EPS = 1e-12

@dataclass
class CoreyRelPerm:
    Swc: float = 0.15
    Sor: float = 0.25
    Sgc: float = 0.05
    kro_end: float = 0.8
    krw_end: float = 0.6
    krg_end: float = 0.9
    no: float = 2.0
    nw: float = 2.0
    ng: float = 2.0

    @staticmethod
    def from_inputs(rp: dict) -> "CoreyRelPerm":
        return CoreyRelPerm(
            Swc=float(rp.get("Swc", 0.15)),
            Sor=float(rp.get("Sor", 0.25)),
            Sgc=float(rp.get("Sgc", 0.05)),
            kro_end=float(rp.get("kro_end", 0.8)),
            krw_end=float(rp.get("krw_end", 0.6)),
            krg_end=float(rp.get("krg_end", 0.9)),
            no=float(rp.get("no", 2.0)),
            nw=float(rp.get("nw", 2.0)),
            ng=float(rp.get("ng", 2.0)),
        )

    def _clip01(self, x):
        return np.clip(x, 0.0, 1.0)

    def kr(self, Sw, Sg):
        Sw = np.asarray(Sw, float); Sg = np.asarray(Sg, float)
        So = np.clip(1.0 - Sw - Sg, 0.0, 1.0)
        denom_o = (1.0 - self.Swc - self.Sor)
        denom_w = (1.0 - self.Swc - self.Sor)
        denom_g = (1.0 - self.Swc - self.Sgc)

        Son = self._clip01((So - self.Sor) / max(denom_o, _EPS))
        Swn = self._clip01((Sw - self.Swc) / max(denom_w, _EPS))
        Sgn = self._clip01((Sg - self.Sgc) / max(denom_g, _EPS))

        kro = self.kro_end * np.power(Son, self.no)
        krw = self.krw_end * np.power(Swn, self.nw)
        krg = self.krg_end * np.power(Sgn, self.ng)
        return kro, krw, krg

    def kr_and_derivs(self, Sw, Sg):
        Sw = np.asarray(Sw, float); Sg = np.asarray(Sg, float)
        So = np.clip(1.0 - Sw - Sg, 0.0, 1.0)
        dSo_dSw = -1.0; dSo_dSg = -1.0

        denom_o = (1.0 - self.Swc - self.Sor)
        denom_w = (1.0 - self.Swc - self.Sor)
        denom_g = (1.0 - self.Swc - self.Sgc)

        inv_o = 1.0 / max(denom_o, _EPS)
        inv_w = 1.0 / max(denom_w, _EPS)
        inv_g = 1.0 / max(denom_g, _EPS)

        Son = np.clip((So - self.Sor) * inv_o, 0.0, 1.0)
        Swn = np.clip((Sw - self.Swc) * inv_w, 0.0, 1.0)
        Sgn = np.clip((Sg - self.Sgc) * inv_g, 0.0, 1.0)

        kro = self.kro_end * np.power(Son, self.no)
        krw = self.krw_end * np.power(Swn, self.nw)
        krg = self.krg_end * np.power(Sgn, self.ng)

        # derivatives; zero outside (0,1) region due to clipping
        mask_Son = (Son > 0) & (Son < 1)
        mask_Swn = (Swn > 0) & (Swn < 1)
        mask_Sgn = (Sgn > 0) & (Sgn < 1)

        dkro_dSon = np.zeros_like(Son); dkro_dSon[mask_Son] = self.kro_end * self.no * np.power(Son[mask_Son], self.no - 1.0)
        dkrw_dSwn = np.zeros_like(Swn); dkrw_dSwn[mask_Swn] = self.krw_end * self.nw * np.power(Swn[mask_Swn], self.nw - 1.0)
        dkrg_dSgn = np.zeros_like(Sgn); dkrg_dSgn[mask_Sgn] = self.krg_end * self.ng * np.power(Sgn[mask_Sgn], self.ng - 1.0)

        dSon_dSw = dSo_dSw * inv_o
        dSon_dSg = dSo_dSg * inv_o
        dSwn_dSw = inv_w
        dSgn_dSg = inv_g

        dkro_dSw = dkro_dSon * dSon_dSw
        dkro_dSg = dkro_dSon * dSon_dSg
        dkrw_dSw = dkrw_dSwn * dSwn_dSw
        dkrw_dSg = np.zeros_like(Sg)
        dkrg_dSw = np.zeros_like(Sw)
        dkrg_dSg = dkrg_dSgn * dSgn_dSg

        return (kro, krw, krg,
                dkro_dSw, dkro_dSg,
                dkrw_dSw, dkrw_dSg,
                dkrg_dSw, dkrg_dSg)
