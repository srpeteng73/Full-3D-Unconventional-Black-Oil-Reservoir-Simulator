# engines/fast.py
import numpy as np

def fallback_fast_solver(state, rng):
    t = np.linspace(1, 30 * 365, 360)

    L = float(state["L_ft"])
    xf = float(state["xf_ft"])
    hf = float(state["hf_ft"])
    pad_interf = float(state.get("pad_interf", 0.2))
    nlats = int(state.get("n_laterals", 1))

    # "Richness" proxy from Rs/pb to nudge oil/gas mix
    richness = float(state.get("Rs_pb_scf_stb", 650.0)) / max(1.0, float(state.get("pb_psi", 5200.0)))

    # Simple geometric scalers
    geo_g = (L / 10000.0) ** 0.85 * (xf / 300.0) ** 0.55 * (hf / 180.0) ** 0.20
    geo_o = (L / 10000.0) ** 0.85 * (xf / 300.0) ** 0.40 * (hf / 180.0) ** 0.30

    # Pad interference + lateral count penalty
    interf_mul = 1.0 / (1.00 + 1.25 * pad_interf + 0.35 * max(0, nlats - 1))

    fluid_model = state.get("fluid_model", "unconventional")
    if fluid_model == "unconventional":
        qi_g_base, qi_o_base = 12000.0, 1000.0
        rich_g = 1.0 + 0.30 * np.clip(richness, 0.0, 1.4)
        rich_o = 1.0 + 0.12 * np.clip(richness, 0.0, 1.4)
        Di_g_yr, b_g, Di_o_yr, b_o = 0.60, 0.85, 0.50, 1.00
    else:
        qi_g_base, qi_o_base = 8000.0, 1600.0
        rich_g = 1.0 + 0.05 * np.clip(richness, 0.0, 1.4)
        rich_o = 1.0 + 0.08 * np.clip(richness, 0.0, 1.4)
        Di_g_yr, b_g, Di_o_yr, b_o = 0.45, 0.80, 0.42, 0.95

    qi_g = np.clip(qi_g_base * geo_g * interf_mul * rich_g, 3000.0, 28000.0)
    qi_o = np.clip(qi_o_base * geo_o * interf_mul * rich_o, 400.0, 2500.0)

    Di_g = Di_g_yr / 365.0
    Di_o = Di_o_yr / 365.0

    qg = qi_g / (1.0 + b_g * Di_g * t) ** (1.0 / b_g)
    qo = qi_o / (1.0 + b_o * Di_o * t) ** (1.0 / b_o)

    # NOTE: use np.trapz (portable) instead of np.trapezoid
    EUR_g_BCF = np.trapz(qg, t) / 1e6
    EUR_o_MMBO = np.trapz(qo, t) / 1e6

    return dict(t=t, qg=qg, qo=qo, EUR_g_BCF=EUR_g_BCF, EUR_o_MMBO=EUR_o_MMBO)
