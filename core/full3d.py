# core/full3d.py
from __future__ import annotations

import time
import numpy as np

from engines.fast import fallback_fast_solver

# Optional implicit driver
try:
    from engines.implicit import simulate_3phase_implicit
    _HAS_IMPLICIT = True
except Exception:
    simulate_3phase_implicit = None
    _HAS_IMPLICIT = False


def simulate(inputs: dict) -> dict:
    """
    Engine router.

    Expects `inputs['pvt']` to be an object that provides callables:
      .Rs(p), .Bo(p), .Bg(p), .mu_g(p), .mu_o(p)
    (This is exactly what the _PVTAdapter in app.py supplies.)
    """
    print(f"[core/full3d] simulate() - engine_type={inputs.get('engine_type')}")

    # Gentle PVT sanity (no coercion; we trust app.py's adapter)
    pvt = inputs.get("pvt", None)

    def _has_callable(name: str) -> bool:
        return (pvt is not None) and hasattr(pvt, name) and callable(getattr(pvt, name))

    missing = [n for n in ("Rs", "Bo", "Bg", "mu_g", "mu_o") if not _has_callable(n)]
    if missing:
        # Only warn; some paths (e.g., fast proxy) may not use all of these
        print(f"[core/full3d] WARNING: PVT is missing callables: {missing}")

    engine_type = str(inputs.get("engine_type") or "")

    # Try implicit if explicitly requested and available
    if "Implicit" in engine_type and _HAS_IMPLICIT:
        try:
            return simulate_3phase_implicit(inputs)
        except Exception as e:
            print(f"[core/full3d] Implicit engine failed: {e}. Falling back to fast proxy.")

    # Fast analytical proxy (default & fallback)
    rng = np.random.default_rng(1234)
    return fallback_fast_solver(inputs, rng)
