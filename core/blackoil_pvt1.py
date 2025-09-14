# core/blackoil_pvt.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class BlackOilPVT:
    pb: float
    Rs_pb: float
    Bo_pb: float
    muo_pb: float
    mug_pb: float
    ct_oil: float

    @staticmethod
    def from_inputs(pvt: dict) -> "BlackOilPVT":
        return BlackOilPVT(
            pb=float(pvt["pb_psi"]),
            Rs_pb=float(pvt["Rs_pb_scf_stb"]),
            Bo_pb=float(pvt["Bo_pb_rb_stb"]),
            muo_pb=float(pvt["muo_pb_cp"]),
            mug_pb=float(pvt["mug_pb_cp"]),
            ct_oil=float(pvt.get("ct_o_1psi", 8e-6)),
        )

    # Simple, differentiable forms (we’ll refine later)
    def Bo(self, P):  # rb/STB
        return self.Bo_pb * (1.0 - self.ct_oil * (P - self.pb))
    def dBo_dP(self, P):
        return -self.Bo_pb * self.ct_oil * np.ones_like(P)

    def Rs(self, P):
        P = np.asarray(P, float)
        return np.where(P <= self.pb, self.Rs_pb, self.Rs_pb + 0.00012*(P - self.pb)**1.1)
    def dRs_dP(self, P):
        P = np.asarray(P, float)
        return np.where(P <= self.pb, 0.0, 0.00012*1.1*(P - self.pb)**0.1)

    def Bg(self, P):  # rb/scf (scaled)
        pmin, pmax = P.min(), P.max()
        return 1.2e-5 + (7.0e-6 - 1.2e-5) * (P - pmin) / (pmax - pmin + 1e-12)
    def dBg_dP(self, P):
        pmin, pmax = P.min(), P.max()
        return (7e-6 - 1.2e-5) / (pmax - pmin + 1e-12) * np.ones_like(P)

    def Bw(self, P):
        return np.full_like(P, 1.01)
    def dBw_dP(self, P):
        return np.zeros_like(P)

    def mu_oil(self, P): return np.full_like(P, self.muo_pb)
    def mu_gas(self, P): return np.full_like(P, self.mug_pb)
    def mu_water(self, P): return np.full_like(P, 0.5)
