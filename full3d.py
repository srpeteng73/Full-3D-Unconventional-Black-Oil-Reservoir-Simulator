# core/full3d.py
import numpy as np

def simulate(inputs: dict):
    """
    Minimal stub so the app runs end-to-end on Streamlit Cloud.
    Replace with your real implicit engine when ready.
    """
    # time in days
    t = np.linspace(1.0, 3650.0, 240)  # ~10 years

    # super-simple declines (proxy)
    qi_g, di_g = 8000.0, 0.80   # Mcf/d, 1/yr
    qi_o, di_o = 1000.0, 0.70   # stb/d, 1/yr

    years = t / 365.25
    qg = qi_g * np.exp(-di_g * years)
    qo = qi_o * np.exp(-di_o * years)

    return {
        "t": t,
        "qg": qg,
        "qo": qo,
        # Optional fields used by downstream tabs:
        "press_matrix": None,
        "pm_mid_psi": None,
        "p_init_3d": None,
        "ooip_3d": None,
    }

