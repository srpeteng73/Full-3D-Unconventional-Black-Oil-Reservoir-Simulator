import numpy as np
import streamlit as st # Import streamlit for logging

__all__ = ["fallback_fast_solver"]

def _get(state, key, default=None):
    """
    Robust getter that works with both flat and nested dicts.
    Returns the default if the key is not found or the value is None.
    """
    val = default
    if not isinstance(state, dict):
        return default
    if key in state and state[key] is not None:
        val = state[key]
    else:
        # Try common nested sections
        for section in ("msw", "schedule", "pvt", "relperm", "init", "grid", "rock"):
            sub = state.get(section)
            if isinstance(sub, dict) and key in sub and sub[key] is not None:
                val = sub[key]
                break # Found it, stop searching
    
    # If after all that, we still have the default, return it
    if val is default:
        return default
    
    # Try to convert to float/int, but fall back to default on failure
    try:
        if isinstance(default, float): return float(val)
        if isinstance(default, int): return int(val)
        return val
    except (ValueError, TypeError):
        return default


def fallback_fast_solver(state, rng=None):
    """
    Fast analytical proxy used for previews/sensitivities and as a fallback.
    This version is input-robust and avoids any dependency on app.py.
    Returns:
        dict(t, qg, qo, EUR_g_BCF, EUR_o_MMBO)
    """
    try:
        # --- DIAGNOSTIC LOGGING ---
        st.warning("ENTERING fallback_fast_solver in engines/fast.py")

        # Time base (days)
        t = np.linspace(1.0, 30.0 * 365.0, 360)

        # Pull inputs safely from flat or nested dicts
        L   = _get(state, "L_ft", 10000.0)
        xf  = _get(state, "xf_ft", 300.0)
        hf  = _get(state, "hf_ft", 180.0)
        nl  = _get(state, "n_laterals", 2)
        pad = _get(state, "pad_interf", 0.20)
        Rs_pb = _get(state, "Rs_pb_scf_stb", 650.0)
        pb    = _get(state, "pb_psi", 5200.0)

        # "Richness" proxy to nudge gas/oil split
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

        Di_g = Di_g_yr / 365.0
        Di_o = Di_o_yr / 365.0

        qg = qi_g / (1.0 + b_g * Di_g * t) ** (1.0 / b_g)
        qo = qi_o / (1.0 + b_o * Di_o * t) ** (1.0 / b_o)

        EUR_g_BCF  = np.trapz(qg, t) / 1e6
        EUR_o_MMBO = np.trapz(qo, t) / 1e6

        st.warning("EXITING fallback_fast_solver NORMALLY.")
        
        return dict(
            t=t, qg=qg, qo=qo,
            EUR_g_BCF=EUR_g_BCF,
            EUR_o_MMBO=EUR_o_MMBO,
        )

    except Exception as e:
        # If anything inside fails, log the error and return a safe, empty result
        st.error(f"FATAL ERROR inside fallback_fast_solver: {e}")
        st.exception(e)
        t = np.linspace(1.0, 30.0 * 365.0, 360)
        return dict(
            t=t, qg=np.zeros_like(t), qo=np.zeros_like(t),
            EUR_g_BCF=0.0, EUR_o_MMBO=0.0,
        )
