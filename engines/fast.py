# engines/fast.py
import numpy as np

__all__ = ["fallback_fast_solver"]

def _get(state, key, default=None):
    """
    Robust getter that works with both:
      - flat dicts (e.g., {"L_ft": ..., "xf_ft": ...})
      - nested dicts (e.g., {"msw": {...}, "pvt": {...}})
    """
    if not isinstance(state, dict):
        return default
    if key in state:
        return state[key]
    # Try common nested sections used by the app/engine
    for section in ("msw", "schedule", "pvt", "relperm", "init", "grid", "rock"):
        sub = state.get(section)
        if isinstance(sub, dict) and key in sub:
            return sub[key]
    return default


def fallback_fast_solver(state, rng=None):
    """
    Fast analytical proxy used for previews/sensitivities and as a fallback.
    This version is input-robust and avoids any dependency on app.py.
    Returns:
        dict(t, qg, qo, EUR_g_BCF, EUR_o_MMBO)
    """
    # Time base (days)
    t = np.linspace(1.0, 30.0 * 365.0, 360)

    # Pull inputs safely from flat or nested dicts
    L   = float(_get(state, "L_ft", 10000.0))
    xf  = float(_get(state, "xf_ft", 300.0))
    hf  = float(_get(state, "hf_ft", 180.0))
    nl  = int(_get(state, "n_laterals", _get(state, "laterals", 2)))
    pad = float(_get(state, "pad_interf", 0.20))

    # "Richness" proxy to nudge gas/oil split
    Rs_pb = float(_get(state, "Rs_pb_scf_stb", 650.0))
    pb    = float(_get(state, "pb_psi", 5200.0))
    richness = Rs_pb / max(1.0, pb)

    # Simple geometric scalers
    geo_g = (L / 10000.0) ** 0.85 * (xf / 300.0) ** 0.55 * (hf / 180.0) ** 0.20
    geo_o = (L / 10000.0) ** 0.85 * (xf / 300.0) ** 0.40 * (hf / 180.0) ** 0.30

    # Pad interference + lateral count penalty
    interf_mul = 1.0 / (1.00 + 1.25 * pad + 0.35 * max(0, nl - 1))

    fluid_model = _get(state, "fluid_model", "unconventional")
    if fluid_model == "black_oil":
        qi_g_base, qi_o_base = 8000.0, 1600.0
        rich_g = 1.0 + 0.05 * np.clip(richness, 0.0, 1.4)
        rich_o = 1.0 + 0.08 * np.clip(richness, 0.0, 1.4)
        Di_g_yr, b_g = 0.45, 0.80
        Di_o_yr, b_o = 0.42, 0.95
        qi_g_min, qi_g_max = 2000.0, 18000.0
        qi_o_min, qi_o_max = 700.0, 3500.0
    else:
        qi_g_base, qi_o_base = 12000.0, 1000.0
        rich_g = 1.0 + 0.30 * np.clip(richness, 0.0, 1.4)
        rich_o = 1.0 + 0.12 * np.clip(richness, 0.0, 1.4)
        Di_g_yr, b_g = 0.60, 0.85
        Di_o_yr, b_o = 0.50, 1.00
        qi_g_min, qi_g_max = 3000.0, 28000.0
        qi_o_min, qi_o_max = 400.0, 2500.0

    qi_g = np.clip(qi_g_base * geo_g * interf_mul * rich_g, qi_g_min, qi_g_max)
    qi_o = np.clip(qi_o_base * geo_o * interf_mul * rich_o, qi_o_min, qi_o_max)

    # Convert yearly decline to daily
    Di_g = Di_g_yr / 365.0
    Di_o = Di_o_yr / 365.0

    # Arps hyperbolic declines
    qg = qi_g / (1.0 + b_g * Di_g * t) ** (1.0 / b_g)   # Mscf/d
    qo = qi_o / (1.0 + b_o * Di_o * t) ** (1.0 / b_o)   # STB/d

    # EURs
    EUR_g_BCF  = np.trapz(qg, t) / 1e6
    EUR_o_MMBO = np.trapz(qo, t) / 1e6

    return dict(
        t=t, qg=qg, qo=qo,
        EUR_g_BCF=EUR_g_BCF,
        EUR_o_MMBO=EUR_o_MMBO,
    )
