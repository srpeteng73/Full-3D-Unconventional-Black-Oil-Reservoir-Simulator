# core/blackoil_pvt1.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

_EPS = 1e-12

@dataclass
class BlackOilPVT:
    pb_psi: float
    Rs_pb_scf_stb: float
    Bo_pb_rb_stb: float
    muo_pb_cp: float
    mug_pb_cp: float
    ct_o_1psi: float = 8e-6
    ct_g_1psi: float = 3e-6
    ct_w_1psi: float = 3e-6
    Bw_ref: float = 1.01  # simple constant water FVF

    @staticmethod
    def from_inputs(pvt: dict) -> "BlackOilPVT":
        return BlackOilPVT(
            pb_psi=float(pvt.get("pb_psi", 5200.0)),
            Rs_pb_scf_stb=float(pvt.get("Rs_pb_scf_stb", 650.0)),
            Bo_pb_rb_stb=float(pvt.get("Bo_pb_rb_stb", 1.35)),
            muo_pb_cp=float(pvt.get("muo_pb_cp", 1.2)),
            mug_pb_cp=float(pvt.get("mug_pb_cp", 0.020)),
            ct_o_1psi=float(pvt.get("ct_o_1psi", 8e-6)),
            ct_g_1psi=float(pvt.get("ct_g_1psi", 3e-6)),
            ct_w_1psi=float(pvt.get("ct_w_1psi", 3e-6)),
        )

    # Oil FVF: exponential compressibility about pb
    def Bo(self, P):
        P = np.asarray(P, float)
        return self.Bo_pb_rb_stb * np.exp(-self.ct_o_1psi * (P - self.pb_psi))

    def dBo_dP(self, P):
        B = self.Bo(P)
        return -self.ct_o_1psi * B

    # Gas FVF: exponential w.r.t. pressure (proxy)
    def Bg(self, P):
        P = np.asarray(P, float)
        Bg_pb = 1.2e-5  # reference Bg at pb (proxy)
        return Bg_pb * np.exp(self.ct_g_1psi * (self.pb_psi - P))

    def dBg_dP(self, P):
        B = self.Bg(P)
        return -self.ct_g_1psi * B

    # Water FVF: near-constant with small compressibility
    def Bw(self, P):
        P = np.asarray(P, float)
        return self.Bw_ref * np.exp(self.ct_w_1psi * (P - self.pb_psi))

    def dBw_dP(self, P):
        B = self.Bw(P)
        return self.ct_w_1psi * B

    # Rs(P): simple piecewise proxy consistent with your earlier app
    def Rs(self, P):
        P = np.asarray(P, float)
        a = 1.2e-4
        out = np.where(P <= self.pb_psi, self.Rs_pb_scf_stb,
                       self.Rs_pb_scf_stb + a * np.power(P - self.pb_psi, 1.1))
        return out

    def dRs_dP(self, P):
        P = np.asarray(P, float)
        a = 1.2e-4
        d = np.where(P <= self.pb_psi, 0.0,
                     a * 1.1 * np.power(np.maximum(P - self.pb_psi, 0.0), 0.1))
        return d

    # viscosities (constants for now)
    def mu_oil(self, P): return np.full_like(np.asarray(P, float), self.muo_pb_cp)
    def mu_gas(self, P): return np.full_like(np.asarray(P, float), self.mug_pb_cp)
    def mu_water(self, P): return np.full_like(np.asarray(P, float), 0.5)
