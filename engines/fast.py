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

    Inputs are pulled safely from a flat or nested dict via _get(...).
    If pad control is BHP, the chosen BHP will affect:
      - the overall drawdown scaling (qi multipliers)
      - the gas/oil balance (via bubble-point proximity)
    Returns:
        dict(t, qg, qo, EUR_g_BCF, EUR_o_MMBO)
    """
    try:
        # Time base (days)
        t = np.linspace(1.0, 30.0 * 365.0, 360)

        # --- Pull inputs (robustly) ---
        L        = _get(state, "L_ft", 10000.0)
        xf       = _get(state, "xf_ft", 300.0)
        hf       = _get(state, "hf_ft", 180.0)
        nl       = _get(state, "n_laterals", 2)
        pad      = _get(state, "pad_interf", 0.20)
        Rs_pb    = _get(state, "Rs_pb_scf_stb", 650.0)
        pb       = _get(state, "pb_psi", 5200.0)
        p_res    = _get(state, "p_init_psi", 6500.0)  # reservoir pressure
        ctrl     = str(_get(state, "pad_ctrl", "BHP"))
        bhp      = _get(state, "pad_bhp_psi", 2500.0)
        fluid_model = _get(state, "fluid_model", "unconventional")

        # --- Small DEBUG breadcrumb (remove later if you want) ---
        try:
            st.caption(f"DEBUG[fast]: ctrl={ctrl}  bhp={bhp:.0f} psi  p_res={p_res:.0f} psi  pb={pb:.0f} psi")
        except Exception:
            pass

        # --- "Richness" proxy to nudge gas/oil split ---
        # Same as before, but we’ll modulate it by bubble-point proximity if using BHP control.
        richness = Rs_pb / max(1.0, pb)

        # --- Geometry scalers (as before) ---
        geo_g = (L / 10000.0) ** 0.85 * (xf / 300.0) ** 0.55 * (hf / 180.0) ** 0.20
        geo_o = (L / 10000.0) ** 0.85 * (xf / 300.0) ** 0.40 * (hf / 180.0) ** 0.30

        # --- Pad interference + lateral count penalty (as before) ---
        interf_mul = 1.0 / (1.00 + 1.25 * pad + 0.35 * max(0, nl - 1))

        # --- Base rates & declines by "fluid model" (as before, b-factors clamped) ---
        if fluid_model == "black_oil":
            qi_g_base, qi_o_base = 8000.0, 1600.0
            rich_g_base = 1.0 + 0.05 * np.clip(richness, 0.0, 1.4)
            rich_o_base = 1.0 + 0.08 * np.clip(richness, 0.0, 1.4)
            Di_g_yr, Di_o_yr = 0.45, 0.42
            b_g = np.clip(0.80, 0.0, 0.99)
            b_o = np.clip(0.95, 0.0, 0.99)
            qi_g_min, qi_g_max = 2000.0, 18000.0
            qi_o_min, qi_o_max = 700.0, 3500.0
        else:  # "unconventional" default
            qi_g_base, qi_o_base = 12000.0, 1000.0
            rich_g_base = 1.0 + 0.30 * np.clip(richness, 0.0, 1.4)
            rich_o_base = 1.0 + 0.12 * np.clip(richness, 0.0, 1.4)
            Di_g_yr, Di_o_yr = 0.60, 0.50
            b_g = np.clip(0.85, 0.0, 0.99)
            b_o = np.clip(1.00, 0.0, 0.99)
            qi_g_min, qi_g_max = 3000.0, 28000.0
            qi_o_min, qi_o_max = 400.0, 2500.0

        # --- Drawdown/BHP effect (only if control == "BHP") ---
        # Drawdown = p_res - bhp; stronger drawdown → higher rates (within bounds).
        # Normalize w.r.t. (p_res - pb) so behavior is sensible around bubble point.
        drawdown = max(p_res - bhp, 50.0)
        ref_dd   = max(p_res - pb, 50.0)
        dd_scale = np.clip(drawdown / ref_dd, 0.40, 1.60)  # keep realistic
        # Bubble-point proximity factor: if bhp is near/above pb, less free gas liberation.
        # If bhp << pb, more gas liberation; if bhp >= pb, gas reduced slightly.
        if ctrl.upper() == "BHP":
            below_pb = (pb - bhp) / max(pb, 1.0)
            gas_bias = np.clip(1.0 + 0.50 * below_pb, 0.75, 1.35)  # boosts gas when bhp<<pb
            oil_bias = np.clip(1.0 - 0.25 * below_pb, 0.80, 1.20)  # modest counterweight
        else:
            gas_bias = 1.0
            oil_bias = 1.0

        # --- Combine influences for initial rates ---
        rich_g = rich_g_base * gas_bias
        rich_o = rich_o_base * oil_bias

        qi_g = np.clip(qi_g_base * geo_g * interf_mul * rich_g * dd_scale, qi_g_min, qi_g_max)
        qi_o = np.clip(qi_o_base * geo_o * interf_mul * rich_o * dd_scale, qi_o_min, qi_o_max)

        # --- Daily decline rates from annual ---
        Di_g = Di_g_yr / 365.0
        Di_o = Di_o_yr / 365.0

        # --- Arps hyperbolic (mathematically safe with clamped b) ---
        qg = qi_g / (1.0 + b_g * Di_g * t) ** (1.0 / b_g)
        qo = qi_o / (1.0 + b_o * Di_o * t) ** (1.0 / b_o)

        # --- Integrate to EURs (units: gas→BCF, oil→MMbbl) ---
        EUR_g_BCF  = np.trapz(qg, t) / 1e6
        EUR_o_MMBO = np.trapz(qo, t) / 1e6

        return dict(
            t=t, qg=qg, qo=qo,
            EUR_g_BCF=EUR_g_BCF,
            EUR_o_MMBO=EUR_o_MMBO,
        )

    except Exception as e:
        # If anything inside fails, log and return a safe empty result
        try:
            st.error(f"FATAL ERROR inside fallback_fast_solver: {e}")
            st.exception(e)
        except Exception:
            pass
        t = np.linspace(1.0, 30.0 * 365.0, 360)
        return dict(
            t=t, qg=np.zeros_like(t), qo=np.zeros_like(t),
            EUR_g_BCF=0.0, EUR_o_MMBO=0.0,
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
