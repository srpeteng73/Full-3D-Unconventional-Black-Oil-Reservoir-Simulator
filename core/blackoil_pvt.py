# core/blackoil_pvt.py
# Back-compat shim so ANY import of core.blackoil_pvt.{BlackOilPVT,Fluid,from_inputs}
# returns the new Phase-1.x implementation that HAS Rs, Bo/Bw/Bg, derivatives, etc.

from __future__ import annotations
from typing import Any
from .blackoil_pvt1 import BlackOilPVT as _NewBlackOilPVT

# Expose the new class under BOTH names
class BlackOilPVT(_NewBlackOilPVT):
    pass

# Legacy name used by older modules / UI code
class Fluid(_NewBlackOilPVT):
    pass

# Legacy constructor some code may still call
def from_inputs(pvt_inputs: dict[str, Any]) -> BlackOilPVT:
    return _NewBlackOilPVT.from_inputs(pvt_inputs)

__all__ = ["BlackOilPVT", "Fluid", "from_inputs"]
