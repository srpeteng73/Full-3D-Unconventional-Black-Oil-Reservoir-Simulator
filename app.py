# Forcing a redeploy on Streamlit Cloud
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import streamlit as st
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy import stats
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
import numpy_financial as npf
from core.full3d import simulate
from engines.fast import fallback_fast_solver  # used in preview & fallbacks
import warnings  # trap analytical power warnings for Arps

# ========= EUR display policy & compact gauge (module-scope) =========
MIDLAND_BOUNDS = {
    "oil_mmbo": (0.4, 2.0),
    "gas_bcf":  (0.2, 3.5),
    "max_gor_scfstb": 2000.0,
}

def enforce_midland_bounds(eur_o_mmbo: float, eur_g_bcf: float):
    """
    Clamp DISPLAYED EURs to Permian–Midland oil-window bands and cap GOR.
    Returns (eur_o_disp, eur_g_disp, note).
    """
    lo_o, hi_o = MIDLAND_BOUNDS["oil_mmbo"]
    lo_g, hi_g = MIDLAND_BOUNDS["gas_bcf"]
    max_gor = MIDLAND_BOUNDS["max_gor_scfstb"]

    o = max(lo_o, min(float(eur_o_mmbo or 0.0), hi_o))
    g = max(lo_g, min(float(eur_g_bcf or 0.0), hi_g))

    # Keep implied GOR reasonable for display
    if o > 0:
        gor = (g * 1.0e9) / (o * 1.0e6)
        if gor > max_gor:
            g = (max_gor * (o * 1.0e6)) / 1.0e9

    note = (f"Display clamped to Midland oil-window: "
            f"Oil [{lo_o}, {hi_o}] MMBO; Gas [{lo_g}, {hi_g}] BCF; "
            f"GOR ≤ {max_gor:,.0f} scf/STB.")
    return o, g, note

def render_semi_gauge(title: str, value: float, unit: str,
                      vmin: float, vmax: float, bar_color: str):
    """Compact semicircle gauge that fits two-up on laptop screens."""
    import plotly.graph_objects as go
    v = 0.0 if value is None else float(value)
    vdisp = max(vmin, min(v, vmax))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=vdisp,
        number={"valueformat": ".2f", "font": {"size": 36}},
        title={"text": title, "font": {"size": 18}},
        gauge={
            "axis": {"range": [vmin, vmax], "tickwidth": 1, "tickcolor": "#9aa0a6"},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "white",
            "shape": "angular",
            "threshold": None,
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    # CORRECTED: Increased top margin from 40 to 60 to prevent title clipping
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=220)
    return fig
# =====================================================================


def _cum_trapz_days(t_days, y_per_day):
    if y_per_day is None:
        return None
    t = _np.asarray(t_days, float)
    y = _np.nan_to_num(_np.asarray(y_per_day, float), nan=0.0)
    return _ctr(y, t, initial=0.0)  # returns same length as t

def _apply_economic_cutoffs(t, y, *, cutoff_days=None, min_rate=0.0):
    if y is None:
        return _np.asarray(t, float), None
    t = _np.asarray(t, float)
    y = _np.asarray(y, float)
    mask = _np.ones_like(t, dtype=bool)
    if cutoff_days is not None and cutoff_days > 0:
        mask &= (t <= float(cutoff_days))
    if min_rate and _np.any(y < float(min_rate)):
        below = y < float(min_rate)
        first = _np.argmax(below) if _np.any(below) else None
        if first is not None and below[first]:
            mask[first:] = False
    return t[mask], y[mask]

def _compute_eurs_and_cums(t, qg=None, qo=None, qw=None):
    """
    Compute cumulative volumes and EURs from rate vectors.
    t  : days (1D)
    qg : gas rate, Mscf/d
    qo : oil rate, STB/d
    qw : water rate, STB/d
    Returns dict with cum arrays and EURs (gas in BCF, oil/water in MMbbl).
    """
    import numpy as np
    from scipy.integrate import cumulative_trapezoid
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    out = {}

    # Coerce & guard
    t = np.asarray(t, float)
    if t.size == 0:
        return out
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    t = np.maximum(t, 0.0)

    def _clean(y):
        if y is None:
            return None
        y = np.asarray(y, float)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.maximum(y, 0.0)
        return y

    qg = _clean(qg)
    qo = _clean(qo)
    qw = _clean(qw)

    # Integrate to cumulative (area under rate vs time)
    if qg is not None:
        cumg_mscf = cumulative_trapezoid(qg, t, initial=0.0)   # Mscf
        out["cum_g_BCF"] = cumg_mscf / 1.0e6                   # BCF
        out["eur_gas_BCF"] = float(out["cum_g_BCF"][-1])
    if qo is not None:
        cumo_stb = cumulative_trapezoid(qo, t, initial=0.0)    # STB
        out["cum_o_MMBO"] = cumo_stb / 1.0e6                   # MMbbl
        out["eur_oil_MMBO"] = float(out["cum_o_MMBO"][-1])
    if qw is not None:
        cumw_stb = cumulative_trapezoid(qw, t, initial=0.0)    # STB
        out["cum_w_MMBL"] = cumw_stb / 1.0e6                   # MMbbl
        out["eur_water_MMBL"] = float(out["cum_w_MMBL"][-1])

    # Implied EUR GOR (scf/STB) if both exist
    if ("eur_gas_BCF" in out) and ("eur_oil_MMBO" in out) and out["eur_oil_MMBO"] > 0:
        gas_scf = out["eur_gas_BCF"] * 1.0e9    # BCF -> scf
        oil_stb = out["eur_oil_MMBO"] * 1.0e6   # MMbbl -> STB
        out["eur_gor_scfstb"] = float(gas_scf / oil_stb)

    return out

def _apply_play_bounds_to_results(sim_like: dict, play_name: str, engine_name: str):
    """
    For Analytical engine ONLY: if EURs violate play bounds, scale the displayed cumulative
    curves to keep the UI realistic (debug-friendly). Adds 'eur_valid' + message.
    Full 3D engine remains enforced elsewhere (no soft clamping here).
    """
    import numpy as np

    bounds = _sanity_bounds_for_play(play_name)
    eur_valid = True
    msgs = []

    eur_g = sim_like.get("eur_gas_BCF")
    eur_o = sim_like.get("eur_oil_MMBO")

    if "analytical" in (engine_name or "").lower():
        # Gas bounds
        if eur_g is not None:
            lo, hi = bounds["gas_bcf"]
            if eur_g < lo or eur_g > hi:
                eur_valid = False
                clamp = min(max(eur_g, lo), hi)
                msgs.append(f"Gas EUR {eur_g:.2f} BCF clamped to [{lo:.1f}, {hi:.1f}] → {clamp:.2f} BCF.")
                if "cum_g_BCF" in sim_like and eur_g > 0:
                    scale = clamp / eur_g
                    sim_like["cum_g_BCF"] = np.asarray(sim_like["cum_g_BCF"], float) * scale
                sim_like["eur_gas_BCF"] = clamp

        # Oil bounds
        if eur_o is not None:
            lo, hi = bounds["oil_mmbo"]
            if eur_o < lo or eur_o > hi:
                eur_valid = False
                clamp = min(max(eur_o, lo), hi)
                msgs.append(f"Oil EUR {eur_o:.2f} MMBO clamped to [{lo:.1f}, {hi:.1f}] → {clamp:.2f} MMBO.")
                if "cum_o_MMBO" in sim_like and eur_o > 0:
                    scale = clamp / eur_o
                    sim_like["cum_o_MMBO"] = np.asarray(sim_like["cum_o_MMBO"], float) * scale
                sim_like["eur_oil_MMBO"] = clamp

        # Optional: enforce a max EUR GOR
        if ("eur_gor_scfstb" in sim_like) and ("eur_oil_MMBO" in sim_like):
            gor = float(sim_like["eur_gor_scfstb"])
            max_gor = bounds.get("max_eur_gor_scfstb", None)
            if max_gor and gor > max_gor and sim_like.get("eur_oil_MMBO", 0) > 0:
                eur_valid = False
                target_gas_scf = max_gor * (sim_like["eur_oil_MMBO"] * 1.0e6)
                target_gas_bcf = target_gas_scf / 1.0e9
                msgs.append(f"EUR GOR {gor:,.0f} > {max_gor:,.0f}; gas clamped to {target_gas_bcf:.2f} BCF.")
                if ("eur_gas_BCF" in sim_like) and ("cum_g_BCF" in sim_like) and sim_like["eur_gas_BCF"] > 0:
                    scale = target_gas_bcf / sim_like["eur_gas_BCF"]
                    sim_like["cum_g_BCF"] = np.asarray(sim_like["cum_g_BCF"], float) * scale
                    sim_like["eur_gas_BCF"] = target_gas_bcf
                sim_like["eur_gor_scfstb"] = max_gor

    sim_like["eur_valid"] = eur_valid
    sim_like["eur_validation_msg"] = "OK" if eur_valid else " | ".join(msgs)
    return sim_like

def validate_midland_eur(EUR_o_MMBO, EUR_g_BCF, *, pb_psi=None, Rs_pb=None):
    lo_o, hi_o = MIDLAND_BOUNDS["oil_mmbo"]
    lo_g, hi_g = MIDLAND_BOUNDS["gas_bcf"]
    msgs, ok = [], True

    if EUR_o_MMBO < lo_o or EUR_o_MMBO > hi_o:
        ok = False
        msgs.append(f"Oil EUR {EUR_o_MMBO:.2f} MMBO outside Midland sanity [{lo_o}, {hi_o}] MMBO.")
    if EUR_g_BCF < lo_g or EUR_g_BCF > hi_g:
        ok = False
        msgs.append(f"Gas EUR {EUR_g_BCF:.2f} BCF outside Midland sanity [{lo_g}, {hi_g}] BCF.")

    # Tolerance-aware PVT/GOR consistency (~3×Rs at pb)
    if EUR_o_MMBO > 0 and Rs_pb not in (None, 0):
        implied_GOR = (EUR_g_BCF * 1e9) / (EUR_o_MMBO * 1e6)  # scf/STB
        limit = 3.0 * float(Rs_pb)
        tol = 1e-6
        if implied_GOR > (limit + tol) and (pb_psi or 0) > 1.0:
            ok = False
            msgs.append(
                f"Implied EUR GOR {implied_GOR:,.0f} scf/STB inconsistent with Rs(pb)≈{Rs_pb:,.0f} "
                f"(>{limit:,.0f})."
            )

    return ok, " ".join(msgs) if msgs else "OK"

def gauge_max(value, typical_hi, floor=0.1, safety=0.15):
    if np.isnan(value) or value <= 0:
        return max(floor, typical_hi)
    # 95th-percentile-ish: typical_hi plus margin vs. observed value
    return max(floor, typical_hi*(1.0+safety), value*(1.25))

def fmt_qty(v, unit):
    if unit == "BCF":
        return f"{v:,.2f} BCF"
    if unit == "MMBO":
        return f"{v:,.2f} MMBO"
    return f"{v:,.2f} {unit}"
# ===============================================================================


# ---------------------- Plot Style Pack (Gas=RED, Oil=GREEN) ----------------------
COLOR_GAS = "#d62728"  # More vibrant red for gas
COLOR_OIL = "#2ca02c"  # More vibrant green for oil
COLOR_WATER = "#1f77b4" # Standard blue for water


# Clean global template
pio.templates.default = "plotly_white"


def _style_fig(fig, title, xlab, ylab_left, ylab_right=None):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0, xanchor="left"),
        font=dict(size=14),
        margin=dict(l=60, r=90, t=60, b=60),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
    )
    fig.update_xaxes(title=xlab, showline=True, linewidth=1, mirror=True)
    fig.update_yaxes(title=ylab_left, showline=True, linewidth=1, mirror=True, secondary_y=False)
    if ylab_right:
        fig.update_yaxes(title=ylab_right, secondary_y=True, showgrid=False)


def rate_chart(t, qg=None, qo=None, qw=None):
    """Dual-axis rate chart: Gas left (red), Liquids right (green/blue)."""
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    if qg is not None:
        fig.add_trace(
            go.Scatter(x=t, y=qg, name="Gas (Mscf/d)", line=dict(width=2, color=COLOR_GAS)),
            secondary_y=False,
        )
    if qo is not None:
        fig.add_trace(
            go.Scatter(x=t, y=qo, name="Oil (STB/d)", line=dict(width=2, color=COLOR_OIL)),
            secondary_y=True,
        )
    if qw is not None:
        fig.add_trace(
            go.Scatter(x=t, y=qw, name="Water (STB/d)", line=dict(width=2, color=COLOR_WATER)),
            secondary_y=True,
        )
    _style_fig(
        fig, "Production Rate vs. Time", "Time (days)", "Gas Rate (Mscf/d)", "Liquid Rate (STB/d)"
    )
    return fig


# High-resolution export button for all charts
PLOT_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "plot",
        "height": 720,
        "width": 1280,
        "scale": 3,
    },
}

# ----------------------------------------------------------------------------------
# ------------------------ Utils ------------------------
def _setdefault(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

def _on_play_change():
    # Clear prior results so the UI cannot show stale EURs
    st.session_state.sim = None

def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def _sim_signature_from_state():
    """
    Lightweight signature of knobs that materially change physics/EUR policy.
    Keep this at module scope so both the engine and Results tab can use it.
    """
    s = st.session_state
    play = s.get("play_sel", "")
    engine = s.get("engine_type", "")
    ctrl  = s.get("pad_ctrl", "BHP")
    bhp   = float(s.get("pad_bhp_psi", 0.0))
    r_m   = float(s.get("pad_rate_mscfd", 0.0))
    r_o   = float(s.get("pad_rate_stbd", 0.0))
    pb    = float(s.get("pb_psi", 0.0))
    rs    = float(s.get("Rs_pb_scf_stb", 0.0))
    bo    = float(s.get("Bo_pb_rb_stb", 1.0))
    pinit = float(s.get("p_init_psi", 0.0))
    key = (play, engine, ctrl, bhp, r_m, r_o, pb, rs, bo, pinit)
    return hash(key)
    
def is_heel_location_valid(x_heel_ft, y_heel_ft, state):
    """Simple feasibility check for well placement (stay inside model and avoid fault strip)."""
    x_max = state['nx'] * state['dx'] - state['L_ft']
    y_max = state['ny'] * state['dy']
    if not (0 <= x_heel_ft <= x_max and 0 <= y_heel_ft <= y_max):
        return False
    if state.get('use_fault'):
        plane = state.get('fault_plane', 'i-plane (vertical)')
        if 'i-plane' in plane:
            fault_x = state['fault_index'] * state['dx']
            return abs(x_heel_ft - fault_x) > 2 * state['dx']
        else:
            fault_y = state['fault_index'] * state['dy']
            return abs(y_heel_ft - fault_y) > 2 * state['dy']
    return True


st.set_page_config(page_title="3D Unconventional / Black-Oil Reservoir Simulator", layout="wide")

# ------------------------ Defaults ------------------------
_setdefault("apply_preset_payload", None)
_setdefault("sim", None)
_setdefault("rng_seed", 1234)

# --- Engine & Model type options ---
ENGINE_TYPES = [
    "Analytical Model (Fast Proxy)",
    "3D Three-Phase Implicit (Phase 1a)",
    "3D Three-Phase Implicit (Phase 1b)",
]
# Model Type options (must match the sidebar selectbox exactly)
VALID_MODEL_TYPES = ["Unconventional Reservoir", "Black Oil Reservoir"]
_setdefault("sim_mode", VALID_MODEL_TYPES[0])  # Default to the first allowed value
_setdefault("sim_mode", VALID_MODEL_TYPES[0])
_setdefault("dfn_segments", None)
_setdefault("use_dfn_sink", True)
_setdefault("use_auto_dfn", True)
_setdefault("vol_downsample", 2)
_setdefault("iso_value_rel", 0.5)

defaults = dict(
    nx=300,
    ny=60,
    nz=12,
    dx=40.0,
    dy=40.0,
    dz=15.0,
    k_stdev=0.02,
    phi_stdev=0.02,
    anis_kxky=1.0,
    facies_style="Continuous (Gaussian)",
    use_fault=False,
    fault_plane="i-plane (vertical)",
    fault_index=60,
    fault_tm=0.10,
    n_laterals=2,
    L_ft=10000.0,
    stage_spacing_ft=250.0,
    clusters_per_stage=3,
    dP_LE_psi=200.0,
    f_fric=0.02,
    wellbore_ID_ft=0.30,
    xf_ft=300.0,
    hf_ft=180.0,
    pad_interf=0.20,
    pad_ctrl="BHP",
    pad_bhp_psi=5200.0,
    pad_rate_mscfd=100000.0,
    outer_bc="Infinite-acting",
    p_outer_psi=7950.0,
    pb_psi=5200.0,
    Rs_pb_scf_stb=650.0,
    Bo_pb_rb_stb=1.35,
    muo_pb_cp=1.20,
    mug_pb_cp=0.020,
    a_g=0.15,
    z_g=0.90,
    p_init_psi=5800.0,
    p_min_bhp_psi=2500.0,
    ct_1_over_psi=0.000015,
    include_RsP=True,
    krw_end=0.6,
    kro_end=0.8,
    nw=2.0,
    no=2.0,
    Swc=0.15,
    Sor=0.25,
    pc_slope_psi=0.0,
    ct_o_1psi=8e-6,
    ct_g_1psi=3e-6,
    ct_w_1psi=3e-6,
    newton_tol=1e-6,
    trans_tol=1e-7,
    max_newton=12,
    max_lin=200,
    threads=0,
    use_omp=False,
    use_mkl=False,
    use_pyamg=False,
    use_cusparse=False,
    dfn_radius_ft=60.0,
    dfn_strength_psi=500.0,
    engine_type="Analytical Model (Fast Proxy)"  # Set stable engine as default
)
for k, v in defaults.items():
    _setdefault(k, v)

if st.session_state.apply_preset_payload is not None:
    for k, v in st.session_state.apply_preset_payload.items():
        st.session_state[k] = v
    st.session_state.apply_preset_payload = None
    _safe_rerun()

# ------------------------ PRESETS (US + Canada Shale Plays) ------------------------
# Typical, rounded values for quick-start modeling. Tune as needed per asset.
PLAY_PRESETS = {
    # --- UNITED STATES ---
    "Permian – Midland (Oil)": dict(
        L_ft=10000.0,
        stage_spacing_ft=250.0,
        xf_ft=300.0,
        hf_ft=180.0,
        Rs_pb_scf_stb=650.0,
        pb_psi=5200.0,
        pad_bhp_psi=5300.0,  # <-- ADDED: Sets BHP above bubble point by default
        Bo_pb_rb_stb=1.35,
        p_init_psi=5800.0,
    ),
    "Permian – Delaware (Oil/Gas)": dict(
        L_ft=10000.0,
        stage_spacing_ft=225.0,
        xf_ft=320.0,
        hf_ft=200.0,
        Rs_pb_scf_stb=700.0,
        pb_psi=5400.0,
        Bo_pb_rb_stb=1.36,
        p_init_psi=6000.0,
    ),
    "Eagle Ford (Oil Window)": dict(
        L_ft=9000.0,
        stage_spacing_ft=225.0,
        xf_ft=270.0,
        hf_ft=150.0,
        Rs_pb_scf_stb=700.0,
        pb_psi=5400.0,
        Bo_pb_rb_stb=1.34,
        p_init_psi=5600.0,
    ),
    "Eagle Ford (Condensate)": dict(
        L_ft=9000.0,
        stage_spacing_ft=225.0,
        xf_ft=300.0,
        hf_ft=160.0,
        Rs_pb_scf_stb=900.0,
        pb_psi=5600.0,
        Bo_pb_rb_stb=1.30,
        p_init_psi=5800.0,
    ),
    "Bakken / Three Forks (Oil)": dict(
        L_ft=10000.0,
        stage_spacing_ft=240.0,
        xf_ft=280.0,
        hf_ft=160.0,
        Rs_pb_scf_stb=350.0,
        pb_psi=4300.0,
        Bo_pb_rb_stb=1.20,
        p_init_psi=4700.0,
    ),
    "Haynesville (Dry Gas)": dict(
        L_ft=10000.0,
        stage_spacing_ft=200.0,
        xf_ft=350.0,
        hf_ft=180.0,
        Rs_pb_scf_stb=0.0,
        pb_psi=1.0,
        Bo_pb_rb_stb=1.00,
        p_init_psi=7000.0,
    ),
    "Marcellus (Dry Gas)": dict(
        L_ft=9000.0,
        stage_spacing_ft=225.0,
        xf_ft=300.0,
        hf_ft=150.0,
        Rs_pb_scf_stb=0.0,
        pb_psi=1.0,
        Bo_pb_rb_stb=1.00,
        p_init_psi=5200.0,
    ),
    "Utica (Liquids-Rich)": dict(
        L_ft=10000.0,
        stage_spacing_ft=225.0,
        xf_ft=320.0,
        hf_ft=180.0,
        Rs_pb_scf_stb=400.0,
        pb_psi=5000.0,
        Bo_pb_rb_stb=1.22,
        p_init_psi=5500.0,
    ),
    "Barnett (Gas)": dict(
        L_ft=6500.0,
        stage_spacing_ft=200.0,
        xf_ft=250.0,
        hf_ft=120.0,
        Rs_pb_scf_stb=0.0,
        pb_psi=1.0,
        Bo_pb_rb_stb=1.00,
        p_init_psi=4200.0,
    ),
    "Niobrara / DJ (Oil)": dict(
        L_ft=9000.0,
        stage_spacing_ft=225.0,
        xf_ft=280.0,
        hf_ft=140.0,
        Rs_pb_scf_stb=250.0,
        pb_psi=3800.0,
        Bo_pb_rb_stb=1.18,
        p_init_psi=4200.0,
    ),
    "Anadarko – Woodford": dict(
        L_ft=10000.0,
        stage_spacing_ft=225.0,
        xf_ft=300.0,
        hf_ft=160.0,
        Rs_pb_scf_stb=300.0,
        pb_psi=4600.0,
        Bo_pb_rb_stb=1.20,
        p_init_psi=5000.0,
    ),
    "Granite Wash": dict(
        L_ft=8000.0,
        stage_spacing_ft=225.0,
        xf_ft=280.0,
        hf_ft=150.0,
        Rs_pb_scf_stb=200.0,
        pb_psi=4200.0,
        Bo_pb_rb_stb=1.15,
        p_init_psi=4600.0,
    ),
    "Fayetteville (Gas)": dict(
        L_ft=6000.0,
        stage_spacing_ft=200.0,
        xf_ft=240.0,
        hf_ft=120.0,
        Rs_pb_scf_stb=0.0,
        pb_psi=1.0,
        Bo_pb_rb_stb=1.00,
        p_init_psi=3500.0,
    ),
    "Tuscaloosa Marine (Oil)": dict(
        L_ft=10000.0,
        stage_spacing_ft=250.0,
        xf_ft=300.0,
        hf_ft=160.0,
        Rs_pb_scf_stb=450.0,
        pb_psi=5000.0,
        Bo_pb_rb_stb=1.25,
        p_init_psi=5400.0,
    ),
    # --- CANADA ---
    "Montney (Condensate-Rich)": dict(
        L_ft=10000.0,
        stage_spacing_ft=225.0,
        xf_ft=330.0,
        hf_ft=180.0,
        Rs_pb_scf_stb=600.0,
        pb_psi=5200.0,
        Bo_pb_rb_stb=1.28,
        p_init_psi=5600.0,
    ),
    "Duvernay (Liquids)": dict(
        L_ft=9500.0,
        stage_spacing_ft=225.0,
        xf_ft=320.0,
        hf_ft=180.0,
        Rs_pb_scf_stb=700.0,
        pb_psi=5400.0,
        Bo_pb_rb_stb=1.32,
        p_init_psi=5800.0,
    ),
    "Horn River (Dry Gas)": dict(
        L_ft=9000.0,
        stage_spacing_ft=225.0,
        xf_ft=320.0,
        hf_ft=170.0,
        Rs_pb_scf_stb=0.0,
        pb_psi=1.0,
        Bo_pb_rb_stb=1.00,
        p_init_psi=6500.0,
    ),
}
PLAY_LIST = list(PLAY_PRESETS.keys())
# Create a single 'state' dictionary from session_state for cleaner access
# This makes the variable available globally for all tabs to use.
state = {k: st.session_state.get(k, v) for k, v in defaults.items()}
#### Part 2: Core Logic, Simulation Engine, and Sidebar UI ####

# ------------------------ HELPER FUNCTIONS ------------------------
def Rs_of_p(p, pb, Rs_pb):
    p = np.asarray(p, float)
    return np.where(p <= pb, Rs_pb, Rs_pb + 0.00012 * (p - pb) ** 1.1)


def Bo_of_p(p, pb, Bo_pb):
    p = np.asarray(p, float)
    slope = -1.0e-5
    return np.where(p <= pb, Bo_pb, Bo_pb + slope * (p - pb))


def Bg_of_p(p):
    p = np.asarray(p, float)
    return 1.2e-5 + (7.0e-6 - 1.2e-5) * (p - p.min()) / (p.max() - p.min() + 1e-12)


def mu_g_of_p(p, pb, mug_pb):
    p = np.asarray(p, float)
    peak = mug_pb * 1.03
    left = mug_pb - 0.0006
    right = mug_pb - 0.0008
    mu = np.where(
        p < pb,
        left + (peak - left) * (p - p.min()) / (pb - p.min() + 1e-9),
        peak + (right - peak) * (p - pb) / (p.max() - pb + 1e-9),
    )
    return np.clip(mu, 0.001, None)


def z_factor_approx(p_psi, p_init_psi=5800.0):
    p_norm = p_psi / p_init_psi
    return 0.95 - 0.2 * (1 - p_norm) + 0.4 * (1 - p_norm) ** 2


# --- PVT adapter: callables named exactly as the engine expects ---
class _PVTAdapter(dict):
    """Adapter that holds PVT callables and parameters; supports attribute & dict access."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)  # allows pvt.Rs(...) and pvt['Rs'](...)


def _build_pvt_payload_from_state(state):
    """Build a PVT payload the engine can use directly (with callables)."""
    pb = float(state.get('pb_psi', 1.0))
    Rs_pb = float(state.get('Rs_pb_scf_stb', 0.0))
    Bo_pb = float(state.get('Bo_pb_rb_stb', 1.0))
    mug_pb = float(state.get('mug_pb_cp', 0.020))
    muo_pb = float(state.get('muo_pb_cp', 1.20))

    def Rs(p):
        return Rs_of_p(p, pb, Rs_pb)

    def Bo(p):
        return Bo_of_p(p, pb, Bo_pb)

    def Bg(p):
        return Bg_of_p(p)

    def mu_g(p):
        return mu_g_of_p(p, pb, mug_pb)

    def mu_o(p):
        return np.full_like(np.asarray(p, float), muo_pb, dtype=float)

    return _PVTAdapter(
        Rs=Rs,
        Bo=Bo,
        Bg=Bg,
        mu_g=mu_g,
        mu_o=mu_o,
        ct_o_1psi=state.get('ct_o_1psi', 8e-6),
        ct_g_1psi=state.get('ct_g_1psi', 3e-6),
        ct_w_1psi=state.get('ct_w_1psi', 3e-6),
        include_RsP=bool(state.get('include_RsP', True)),
        pb_psi=pb,
    )


# --- Defensive monkey-patch: if engine's Fluid class lacks methods, inject thin wrappers ---
def _monkeypatch_engine_fluid_if_needed(adapter):
    """
    Some engine builds instantiate their own Fluid and expect .Rs/.Bo/.Bg/.mu_g/.mu_o.
    If missing, attach wrappers that forward to our adapter. Safe no-op if import fails.
    """
    try:
        from core.blackoil_pvt1 import Fluid as EngineFluid  # optional; may not exist in all builds

        patched = []
        if not hasattr(EngineFluid, "Rs"):
            EngineFluid.Rs = lambda self, p: adapter.Rs(p)
            patched.append("Rs")
        if not hasattr(EngineFluid, "Bo"):
            EngineFluid.Bo = lambda self, p: adapter.Bo(p)
            patched.append("Bo")
        if not hasattr(EngineFluid, "Bg"):
            EngineFluid.Bg = lambda self, p: adapter.Bg(p)
            patched.append("Bg")
        if not hasattr(EngineFluid, "mu_g"):
            EngineFluid.mu_g = lambda self, p: adapter.mu_g(p)
            patched.append("mu_g")
        if not hasattr(EngineFluid, "mu_o"):
            EngineFluid.mu_o = lambda self, p: adapter.mu_o(p)
            patched.append("mu_o")
        if patched:
            print(f"[PVT patch] Injected Fluid methods: {patched}")
    except Exception:
        pass  # safety net


# --- Public helper used by run_simulation_engine(...) ---
def _pvt_from_state(state):
    adapter = _build_pvt_payload_from_state(state)
    _monkeypatch_engine_fluid_if_needed(adapter)
    return adapter


def eur_gauges(EUR_g_BCF, EUR_o_MMBO):
    import plotly.graph_objects as go
    import numpy as np

    def g(val, label, suffix, color, vmax):
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(val),
                number={'suffix': f" {suffix}", 'font': {'size': 44, 'color': '#0b2545'}},
                title={'text': f"<b>{label}</b>", 'font': {'size': 22, 'color': '#0b2545'}},
                gauge={
                    'shape': 'angular',
                    'axis': {'range': [0, vmax], 'tickwidth': 1.2, 'tickcolor': '#0b2545'},
                    'bar': {'color': color, 'thickness': 0.28},
                    'bgcolor': 'white',
                    'borderwidth': 1,
                    'bordercolor': '#cfe0ff',  # moved inside gauge:
                    'steps': [
                        {'range': [0, 0.6 * vmax], 'color': 'rgba(0,0,0,0.04)'},
                        {'range': [0.6 * vmax, 0.85 * vmax], 'color': 'rgba(0,0,0,0.07)'}
                    ],
                    'threshold': {
                        'line': {'color': 'green' if color == '#d62728' else 'red', 'width': 4},
                        'thickness': 0.9,
                        'value': float(val)
                    },
                },
            )
        )
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="#ffffff")
        return fig

    gmax = max(1.0, np.ceil(EUR_g_BCF / 5.0) * 5.0)
    omax = max(0.5, np.ceil(EUR_o_MMBO / 0.5) * 0.5)
    return g(EUR_g_BCF, "EUR Gas", "BCF", "#d62728", gmax), g(
        EUR_o_MMBO, "EUR Oil", "MMBO", "#2ca02c", omax
    )


def semi_log_layout(title, xaxis="Day (log scale)", yaxis="Rate"):
    return dict(
        title=f"<b>{title}</b>",
        template="plotly_white",
        xaxis=dict(type="log", title=xaxis, showgrid=True, gridcolor="rgba(0,0,0,0.15)"),
        yaxis=dict(title=yaxis, showgrid=True, gridcolor="rgba(0,0,0,0.15)"),
        legend=dict(orientation="h"),
    )


def ensure_3d(arr2d_or_3d):
    a = np.asarray(arr2d_or_3d)
    return a[None, ...] if a.ndim == 2 else a

def get_k_slice(A, k):
    A3 = ensure_3d(A)
    nz = A3.shape[0]
    k = int(np.clip(k, 0, nz - 1))
    return A3[k, :, :]


def downsample_3d(A, ds):
    A3 = ensure_3d(A)
    ds = max(1, int(ds))
    return A3[::ds, ::ds, ::ds]


def parse_dfn_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    req = ["x0", "y0", "z0", "x1", "y1", "z1"]
    for c in req:
        if c not in df.columns:
            raise ValueError("DFN CSV must include columns: x0,y0,z0,x1,y1,z1[,k_mult,aperture_ft]")
    arr = df[req].to_numpy(float)
    if "k_mult" in df.columns or "aperture_ft" in df.columns:
        k_mult = df["k_mult"].to_numpy(float) if "k_mult" in df.columns else np.ones(len(df))
        ap = df["aperture_ft"].to_numpy(float) if "aperture_ft" in df.columns else np.full(len(df), np.nan)
        arr = np.column_stack([arr, k_mult, ap])
    return arr


def gen_auto_dfn_from_stages(nx, ny, nz, dx, dy, dz, L_ft, stage_spacing_ft, n_lats, hf_ft):
    n_stages = max(1, int(L_ft / max(stage_spacing_ft, 1.0)))
    Lcells = int(L_ft / max(dx, 1.0))
    xs = np.linspace(5, max(6, Lcells - 5), n_stages) * dx
    lat_rows = [ny // 3, 2 * ny // 3] if n_lats >= 2 else [ny // 2]
    segs = []
    half_h = hf_ft / 2.0
    for jr in lat_rows:
        y_ft = jr * dy
        for xcell in xs:
            x_ft = xcell
            z0 = max(0.0, (nz * dz) / 2.0 - half_h)
            z1 = min(nz * dz, (nz * dz) / 2.0 + half_h)
            segs.append([x_ft, y_ft, z0, x_ft, y_ft, z1])
    return np.array(segs, float) if segs else None


def _get_sim_preview():
    tmp = {k: st.session_state[k] for k in list(defaults.keys()) if k in st.session_state}
    rng_preview = np.random.default_rng(int(st.session_state.get("rng_seed", 1234)) + 999)
    try:
        # --- SAFETY NET FOR PREVIEW ---
        # We explicitly watch for the classic Arps power failure:
        # RuntimeWarning: invalid value encountered in power
        # That happens if (1 + b*D*t) becomes negative and is raised to 1/b.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)
            result = fallback_fast_solver(tmp, rng_preview)
            # If the solver emitted a "power" invalid warning, attempt one sanitized retry.
            bad_power = any(("invalid value encountered in power" in str(x.message)) for x in w)
            if bad_power or _looks_nan_like(result):
                tmp2 = _sanitize_decline_params(tmp.copy())
                result = fallback_fast_solver(tmp2, rng_preview)
            # Ensure no NaNs leak into charts
            result = _nan_guard_result(result)
            return result
    except Exception as e:
        st.error("ERROR IN PREVIEW SOLVER (_get_sim_preview):")
        st.exception(e)
        # Return a dummy structure to prevent crashing the UI layout
        return {'t': [0], 'qg': [0], 'qo': [0], 'EUR_g_BCF': 0, 'EUR_o_MMBO': 0}
# ------------------------ Arps/decline safety helpers (ANALYTICAL ONLY) ------------------------
def _sanitize_decline_params(state_like: dict) -> dict:
    """
    Human note: Some builds feed different names for hyperbolic parameters.
    We defensively scan keys for anything that LOOKS like a hyperbolic 'b' or a decline rate,
    clamp 'b' into a safe (0,1) interval, and force any negative declines positive.
    This keeps (1 + b*D*t) from ever going negative during power().
    """
    SAFE_B_MIN, SAFE_B_MAX = 1.0e-6, 0.95

    def _clip_b(x):
        try:
            xv = float(x)
            return min(max(xv, SAFE_B_MIN), SAFE_B_MAX)
        except Exception:
            return x

    def _abs_decline(x):
        try:
            return abs(float(x))
        except Exception:
            return x

    for k in list(state_like.keys()):
        lk = k.lower()
        # Common patterns we've seen across fast proxies
        if lk in ("b", "b_oil", "b_gas", "b_liq", "b_decline", "b_hyp", "bhyp", "bexp"):
            state_like[k] = _clip_b(state_like[k])
        # Decline rates frequently show up as D, Di, D1, decline_*, etc.
        if lk in ("d", "di", "d1", "decline", "decline_rate") or ("decline" in lk):
            state_like[k] = _abs_decline(state_like[k])

    # Optional: mark that we sanitized to help debug later
    state_like["__analytical_sanitized__"] = True
    return state_like


def _looks_nan_like(result: dict) -> bool:
    """Return True if any primary arrays contain NaNs or infs."""
    if not isinstance(result, dict):
        return True
    for key in ("t", "qg", "qo"):
        if key in result and result[key] is not None:
            arr = np.asarray(result[key], float)
            if not np.all(np.isfinite(arr)):
                return True
    return False


def _nan_guard_result(result: dict) -> dict:
    """
    Replace NaNs/infs in rate vectors so the UI can draw safely.
    We do NOT change EURs here; the engine will re-compute them later in 'Results'.
    """
    if not isinstance(result, dict):
        return result

    out = dict(result)
    for key in ("t", "qg", "qo", "qw"):
        if key in out and out[key] is not None:
            arr = np.asarray(out[key], float)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            out[key] = arr
    return out
# -----------------------------------------------------------------------------------------------

def generate_property_volumes(state):
    """Generates kx, ky, and phi volumes based on sidebar settings and stores them in session_state."""
    rng = np.random.default_rng(int(st.session_state.rng_seed))
    nz, ny, nx = int(state["nz"]), int(state["ny"]), int(state["nx"])
    # Use the facies style from the state to generate the base 2D maps
    style = state.get("facies_style", "Continuous (Gaussian)")
    if "Continuous" in style:
        kx_mid = 0.05 + state["k_stdev"] * rng.standard_normal((ny, nx))
        ky_mid = (0.05 / state["anis_kxky"]) + state["k_stdev"] * rng.standard_normal((ny, nx))
        phi_mid = 0.10 + state["phi_stdev"] * rng.standard_normal((ny, nx))
    elif "Speckled" in style:
        # High variance using log-normal distribution for more contrast
        kx_mid = np.exp(rng.normal(np.log(0.05), 1.5 + state["k_stdev"]*5, (ny, nx)))
        ky_mid = kx_mid / state["anis_kxky"]
        phi_mid = np.exp(rng.normal(np.log(0.10), 0.8 + state["phi_stdev"]*3, (ny, nx)))
    elif "Layered" in style:
        # Vertical bands (variation primarily in y-direction)
        base_profile_k = 0.05 + state["k_stdev"] * rng.standard_normal(ny)
        kx_mid = np.tile(base_profile_k[:, None], (1, nx))
        ky_mid = kx_mid / state["anis_kxky"]
        base_profile_phi = 0.10 + state["phi_stdev"] * rng.standard_normal(ny)
        phi_mid = np.tile(base_profile_phi[:, None], (1, nx))
    # Apply a slight vertical trend and store in session_state
    kz_scale = np.linspace(0.95, 1.05, nz)[:, None, None]
    st.session_state.kx = np.clip(kx_mid[None, ...] * kz_scale, 1e-4, 5.0)
    st.session_state.ky = np.clip(ky_mid[None, ...] * kz_scale, 1e-4, 5.0)
    st.session_state.phi = np.clip(phi_mid[None, ...] * kz_scale, 0.01, 0.35)
    st.success("Successfully generated 3D property volumes!")


def _sanity_bounds_for_play(play_name: str):
    s = (play_name or "").lower()
    # Default (conservative, oil-window-ish)
    bounds = dict(oil_mmbo=(0.3, 1.5), gas_bcf=(0.3, 3.0), max_eur_gor_scfstb=2000.0)
    if "midland" in s:
        # Restore realistic Midland oil-window ranges
        bounds = dict(oil_mmbo=(0.3, 2.0), gas_bcf=(0.2, 3.5), max_eur_gor_scfstb=2000.0)
    return bounds

def run_simulation_engine(state):
    """
    Main function to run the selected simulation engine, process results,
    and enforce realism for the analytical proxy.
    """
    t0 = time.time()
    chosen_engine = st.session_state.get("engine_type", "")
    current_play = st.session_state.get("play_sel", "")
    out = None

    # --- Step 1: Run the selected simulation engine ---
    try:
        if "Analytical" in chosen_engine:
            rng = np.random.default_rng(int(st.session_state.get("rng_seed", 1234)))
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", RuntimeWarning)
                out = fallback_fast_solver(state, rng)
                bad_power = any(("invalid value encountered in power" in str(x.message)) for x in w)
                if bad_power or _looks_nan_like(out):
                    st.info("Analytical model unstable. Retrying with safe decline parameters…")
                    safe_state = _sanitize_decline_params(state.copy())
                    out = fallback_fast_solver(safe_state, rng)
        else:
            st.warning("3D Implicit Engine path is not fully detailed in this stub.")
            if st.session_state.get('kx') is None:
                generate_property_volumes(state)
            out = fallback_fast_solver(state, np.random.default_rng(1))

    except Exception as e:
        st.error(f"FATAL SIMULATOR CRASH in '{chosen_engine}':")
        st.exception(e)
        return None

    if out is None or not isinstance(out, dict) or "t" not in out:
        st.error("Simulation engine failed to return valid data.")
        return None

    # --- Step 2: Extract and clean raw rate arrays ---
    t = np.nan_to_num(np.asarray(out.get("t"), float), nan=0.0)
    qg = np.nan_to_num(np.asarray(out.get("qg"), float), nan=0.0) if out.get("qg") is not None else None
    qo = np.nan_to_num(np.asarray(out.get("qo"), float), nan=0.0) if out.get("qo") is not None else None
    qw = np.nan_to_num(np.asarray(out.get("qw"), float), nan=0.0) if out.get("qw") is not None else None

    # --- Step 3: Enforce Realism for Analytical Oil Plays ---
    if "Analytical" in chosen_engine and "oil" in current_play.lower() and qo is not None and qg is not None and len(t) > 1:
        # CORRECTED: Aligned the realism cap with the warning threshold
        REALISTIC_GOR_CAP_SCFTSTB = 2000.0
        total_oil_stb = trapezoid(qo, t)
        total_gas_scf = trapezoid(qg, t) * 1000.0
        if total_oil_stb > 1e-6: # Avoid division by zero
            current_gor = total_gas_scf / total_oil_stb
            if current_gor > REALISTIC_GOR_CAP_SCFTSTB:
                st.info(f"Analytical model produced high GOR ({current_gor:,.0f} scf/STB). Scaling gas rate for realism.")
                scale_factor = REALISTIC_GOR_CAP_SCFTSTB / current_gor
                qg *= scale_factor
                out["qg"] = qg

    # --- Step 4: Calculate Cumulatives and Final EURs ---
    sim = dict(out)
    sim.update(_compute_eurs_and_cums(t, qg=qg, qo=qo, qw=qw))

    # --- Step 5: Final processing ---
    sim["runtime_s"] = time.time() - t0
    sim["_sim_signature"] = _sim_signature_from_state()
    if "eur_gas_BCF" in sim: sim["EUR_g_BCF"] = sim["eur_gas_BCF"]
    if "eur_oil_MMBO" in sim: sim["EUR_o_MMBO"] = sim["eur_oil_MMBO"]
    
    return sim
# ------------------------ Engine & Presets (SIDEBAR) ------------------------
with st.sidebar:
    st.markdown("## Simulation Setup")
    
    # --- All controls are now correctly inside the sidebar ---

    with st.expander("Engine & Presets", expanded=True):
        engine_type_ui = st.selectbox(
            "Engine Type",
            ENGINE_TYPES,
            key="engine_type_ui",
            help="Choose the calculation engine. Phase 1a/1b are the developing implicit engines; the analytical model is a fast proxy.",
        )
        st.session_state["engine_type"] = engine_type_ui
        model_choice = st.selectbox("Model Type", VALID_MODEL_TYPES, key="sim_mode")
        st.session_state.fluid_model = (
            "black_oil" if "Black Oil" in model_choice else "unconventional"
        )

    with st.expander("Shale Play Preset", expanded=True):
        _current_play = st.session_state.get("play_sel", PLAY_LIST[0])
        try:
            _default_idx = PLAY_LIST.index(_current_play)
        except ValueError:
            _default_idx = 0

        play = st.selectbox(
            "Select a Play",
            PLAY_LIST,
            index=_default_idx,
            key="play_sel",
            label_visibility="visible", # Use a visible label in the sidebar
            on_change=_on_play_change,
        )

        apply_clicked = st.button("Apply Preset", use_container_width=True, type="primary")
        if apply_clicked:
            payload = defaults.copy()
            payload.update(PLAY_PRESETS[st.session_state.play_sel])
            if st.session_state.fluid_model == "black_oil":
                payload.update(
                    dict(
                        Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00,
                        p_init_psi=max(3500.0, float(payload.get("p_init_psi", 5200.0))),
                    )
                )
            st.session_state.sim = None
            st.session_state.apply_preset_payload = payload
            _safe_rerun()

    with st.expander("Grid & Heterogeneity", expanded=False):
        st.markdown("#### Grid (ft)")
        c1, c2, c3 = st.columns(3)
        with c1: st.number_input("nx", 1, 500, key="nx")
        with c2: st.number_input("ny", 1, 500, key="ny")
        with c3: st.number_input("nz", 1, 200, key="nz")
        c1, c2, c3 = st.columns(3)
        with c1: st.number_input("dx", step=1.0, key="dx")
        with c2: st.number_input("dy", step=1.0, key="dy")
        with c3: st.number_input("dz", step=1.0, key="dz")

        st.markdown("#### Heterogeneity & Anisotropy")
        st.selectbox("Facies style", ["Continuous (Gaussian)", "Speckled (high-variance)", "Layered (vertical bands)"], key="facies_style")
        st.slider("k stdev", 0.0, 0.20, float(st.session_state.k_stdev), 0.01, key="k_stdev", help="Standard deviation for permeability field generation.")
        st.slider("ϕ stdev", 0.0, 0.20, float(st.session_state.phi_stdev), 0.01, key="phi_stdev", help="Standard deviation for porosity field generation.")
        st.slider("Anisotropy kx/ky", 0.5, 3.0, float(st.session_state.anis_kxky), 0.05, key="anis_kxky")

    with st.expander("Faults", expanded=False):
        st.checkbox("Enable fault TMULT", value=bool(st.session_state.use_fault), key="use_fault")
        fault_plane_choice = st.selectbox("Fault plane", ["i-plane (vertical)", "j-plane (vertical)"], index=0, key="fault_plane")
        max_idx = int(st.session_state.nx) - 2 if 'i-plane' in fault_plane_choice else int(st.session_state.ny) - 2
        if st.session_state.fault_index > max_idx: st.session_state.fault_index = max_idx
        st.number_input("Plane index", 1, max(1, max_idx), key="fault_index")
        st.number_input("Transmissibility multiplier", value=float(st.session_state.fault_tm), step=0.01, key="fault_tm")

    with st.expander("Pad / Wellbore & Frac", expanded=False):
        st.number_input("Laterals", 1, 6, int(st.session_state.n_laterals), 1, key="n_laterals")
        st.number_input("Lateral length (ft)", value=float(st.session_state.L_ft), step=50.0, key="L_ft")
        st.number_input("Stage spacing (ft)", value=float(st.session_state.stage_spacing_ft), step=5.0, key="stage_spacing_ft")
        st.number_input("Clusters per stage", 1, 12, int(st.session_state.clusters_per_stage), 1, key="clusters_per_stage")
        st.number_input("Δp limited-entry (psi)", value=float(st.session_state.dP_LE_psi), step=5.0, key="dP_LE_psi")
        st.number_input("Wellbore friction factor", value=float(st.session_state.f_fric), format="%.3f", step=0.005, key="f_fric")
        st.number_input("Wellbore ID (ft)", value=float(st.session_state.wellbore_ID_ft), step=0.01, key="wellbore_ID_ft")
        st.number_input("Frac half-length xf (ft)", value=float(st.session_state.xf_ft), step=5.0, key="xf_ft")
        st.number_input("Frac height hf (ft)", value=float(st.session_state.hf_ft), step=5.0, key="hf_ft")
        st.slider("Pad interference coeff.", 0.00, 0.80, float(st.session_state.pad_interf), 0.01, key="pad_interf")

    with st.expander("Controls & Boundary", expanded=False):
        st.selectbox("Pad control", ["BHP", "RATE"], key="pad_ctrl")
        st.number_input("Pad BHP (psi)", value=float(st.session_state.pad_bhp_psi), step=10.0, key="pad_bhp_psi")
        st.number_input("Pad RATE (Mscf/d)", value=float(st.session_state.pad_rate_mscfd), step=1000.0, key="pad_rate_mscfd")
        st.selectbox("Outer boundary", ["Infinite-acting", "Constant-p"], key="outer_bc")
        st.number_input("Boundary pressure (psi)", value=float(st.session_state.p_outer_psi), step=10.0, key="p_outer_psi")
    
    with st.expander("DFN (Discrete Fracture Network)", expanded=False):
        st.checkbox("Use DFN-driven sink in solver", value=bool(st.session_state.use_dfn_sink), key="use_dfn_sink")
        st.checkbox("Auto-generate DFN from stages", value=bool(st.session_state.use_auto_dfn), key="use_auto_dfn")
        st.number_input("DFN influence radius (ft)", value=float(st.session_state.dfn_radius_ft), step=5.0, key="dfn_radius_ft")
        st.number_input("DFN sink strength (psi)", value=float(st.session_state.dfn_strength_psi), step=10.0, key="dfn_strength_psi")
        dfn_up = st.file_uploader("Upload DFN CSV", type=["csv"], key="dfn_csv")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Load DFN"):
                if dfn_up:
                    st.session_state.dfn_segments = parse_dfn_csv(dfn_up)
                    st.success(f"Loaded {len(st.session_state.dfn_segments)} segments")
                else: st.warning("Please choose a CSV")
        with c2:
            if st.button("Generate DFN"):
                segs = gen_auto_dfn_from_stages(
                    int(st.session_state.nx), int(st.session_state.ny), int(st.session_state.nz),
                    float(st.session_state.dx), float(st.session_state.dy), float(st.session_state.dz),
                    float(st.session_state.L_ft), float(st.session_state.stage_spacing_ft),
                    int(st.session_state.n_laterals), float(st.session_state.hf_ft),
                )
                st.session_state.dfn_segments = segs
                st.success(f"Generated {0 if segs is None else len(segs)} segments")

    with st.expander("Solver & Profiling", expanded=False):
        st.number_input("Newton tolerance", value=float(st.session_state.newton_tol), format="%.1e", key="newton_tol")
        st.number_input("Transmissibility tolerance", value=float(st.session_state.trans_tol), format="%.1e", key="trans_tol")
        st.number_input("Max Newton iterations", value=int(st.session_state.max_newton), step=1, key="max_newton")
        st.number_input("Max linear solver iterations", value=int(st.session_state.max_lin), step=10, key="max_lin")
        st.number_input("Threads (0 for auto)", value=int(st.session_state.threads), step=1, key="threads")
        st.checkbox("Use OpenMP", value=bool(st.session_state.use_omp), key="use_omp")
        st.checkbox("Use Intel MKL", value=bool(st.session_state.use_mkl), key="use_mkl")
        st.checkbox("Use PyAMG solver", value=bool(st.session_state.use_pyamg), key="use_pyamg")
        st.checkbox("Use NVIDIA cuSPARSE", value=bool(st.session_state.use_cusparse), key="use_cusparse")
        
    st.markdown("---")
    st.markdown("##### Developed by:")
    st.markdown("##### Omar Nur, Petroleum Engineer")
    st.markdown("---")
#### Part 3: Main Application UI - Primary Workflow Tabs ####
# --- Tab list ---
tab_names = [
    "Setup Preview",
    "Generate 3D property volumes",
    "PVT (Black-Oil)",
    "MSW Wellbore",
    "RTA",
    "Results",
    "3D Viewer",
    "Slice Viewer",
    "QA / Material Balance",
    "Economics",
    "EUR vs Lateral Length",
    "Field Match (CSV)",
    "Automated Match", # <-- NEW TAB
    "Uncertainty & Monte Carlo",
    "Well Placement Optimization",
    "User’s Manual",
    "Solver & Profiling",
    "DFN Viewer",
]
st.write(
    '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} '
    '.stRadio > label {display:none;} '
    'div.row-widget.stRadio > div > div {border: 1px solid #ccc; padding: 6px 12px; border-radius: 4px; margin: 2px; background-color: #f0f2f6;} '
    'div.row-widget.stRadio > div > div[aria-checked="true"] {background-color: #e57373; color: white; border-color: #d32f2f;}</style>',
    unsafe_allow_html=True,
)
selected_tab = st.radio("Navigation", tab_names, label_visibility="collapsed")

# ------------------------ TAB CONTENT DEFINITIONS ------------------------
if selected_tab == "Setup Preview":
    st.header("Setup Preview")
    c1, c2 = st.columns([1, 1])
    # ----- LEFT COLUMN -----
    with c1:
        st.markdown("#### Grid & Rock Summary")
        grid_data = {
            "Parameter": [
                "Grid Dimensions (nx, ny, nz)",
                "Cell Size (dx, dy, dz) (ft)",
                "Total Volume (MM-ft³)",
                "Facies Style",
                "Permeability Anisotropy (kx/ky)",
            ],
            "Value": [
                f"{state['nx']} x {state['ny']} x {state['nz']}",
                f"{state['dx']} x {state['dy']} x {state['dz']}",
                f"{state['nx']*state['ny']*state['nz']*state['dx']*state['dy']*state['dz']/1e6:.1f}",
                state['facies_style'],
                f"{state['anis_kxky']:.2f}",
            ],
        }
        st.table(pd.DataFrame(grid_data))
        with st.expander("Click for details"):
            st.markdown(
                "- **Grid Dimensions**: The number of cells in the X, Y, and Z directions.\n"
                "- **Cell Size**: The physical size of each grid cell in feet.\n"
                "- **Total Volume**: The total bulk volume of the reservoir model.\n"
                "- **Facies Style**: The method used to generate geological heterogeneity.\n"
                "- **Anisotropy**: The ratio of permeability in X (kx) to Y (ky)."
            )
        with st.expander("Preset sanity check (debug)"):
            st.write({
                "Play selected": st.session_state.get("play_sel"),
                "Model Type (sim_mode)": st.session_state.get("sim_mode"),
                "fluid_model": st.session_state.get("fluid_model"),
                "Engine Type": st.session_state.get("engine_type"),
                "L_ft": state.get("L_ft"),
                "stage_spacing_ft": state.get("stage_spacing_ft"),
                "xf_ft": state.get("xf_ft"),
                "hf_ft": state.get("hf_ft"),
                "pb_psi": state.get("pb_psi"),
                "Rs_pb_scf_stb": state.get("Rs_pb_scf_stb"),
                "Bo_pb_rb_stb": state.get("Bo_pb_rb_stb"),
                "p_init_psi": state.get("p_init_psi"),
            })
        st.markdown("#### Well & Frac Summary")
        well_data = {
            "Parameter": [
                "Laterals",
                "Lateral Length (ft)",
                "Frac Half-length (ft)",
                "Frac Height (ft)",
                "Stages",
                "Clusters/Stage",
            ],
            "Value": [
                state['n_laterals'],
                state['L_ft'],
                state['xf_ft'],
                state['hf_ft'],
                int(state['L_ft'] / state['stage_spacing_ft']),
                state['clusters_per_stage'],
            ],
        }
        st.table(pd.DataFrame(well_data))
        with st.expander("Click for details"):
            st.markdown(
                "- **Laterals**: Number of horizontal wells in the pad.\n"
                "- **Lateral Length**: Length of each horizontal wellbore.\n"
                "- **Frac Half-length (xf)**: Distance a hydraulic fracture extends from the wellbore.\n"
                "- **Frac Height (hf)**: Vertical extent of the hydraulic fractures.\n"
                "- **Stages**: Number of fracturing treatments.\n"
                "- **Clusters/Stage**: Perforation clusters within each stage."
            )
    # ----- RIGHT COLUMN -----
    with c2:
        st.markdown("#### Top-Down Schematic")
        fig = go.Figure()
        nx, ny, dx, dy = state['nx'], state['ny'], state['dx'], state['dy']
        L_ft, xf_ft, ss_ft, n_lats = state['L_ft'], state['xf_ft'], state['stage_spacing_ft'], state['n_laterals']
        fig.add_shape(
            type="rect",
            x0=0,
            y0=0,
            x1=nx * dx,
            y1=ny * dy,
            line=dict(color="RoyalBlue"),
            fillcolor="lightskyblue",
            opacity=0.3,
        )
        lat_rows_y = [ny * dy / 3, 2 * ny * dy / 3] if n_lats >= 2 else [ny * dy / 2]
        n_stages = max(1, int(L_ft / max(ss_ft, 1.0)))
        for i, y_lat in enumerate(lat_rows_y):
            fig.add_trace(go.Scatter(
                x=[0, L_ft],
                y=[y_lat, y_lat],
                mode='lines',
                line=dict(color='black', width=3),
                name='Lateral',
                showlegend=(i == 0),
            ))
            for j in range(n_stages):
                x_stage = (j + 0.5) * ss_ft
                if x_stage > L_ft:
                    continue
                fig.add_trace(go.Scatter(
                    x=[x_stage, x_stage],
                    y=[y_lat - xf_ft, y_lat + xf_ft],
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Frac',
                    showlegend=(i == 0 and j == 0),
                ))
        fig.update_layout(
            title="<b>Well and Fracture Geometry</b>",
            xaxis_title="X (ft)",
            yaxis_title="Y (ft)",
            yaxis_range=[-0.1 * ny * dy, 1.1 * ny * dy],
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        with st.expander("Click for details"):
            st.markdown(
                "Bird's-eye view of the simulation model:\n"
                "- **Light blue** = reservoir boundary\n"
                "- **Black** = horizontal well laterals\n"
                "- **Red** = hydraulic fractures"
            )
    st.markdown("---")
    st.markdown("### Production Forecast Preview (Analytical Model)")
    preview = _get_sim_preview()
    p_c1, p_c2 = st.columns(2)
    with p_c1:
        fig_g = go.Figure(go.Scatter(x=preview['t'], y=preview['qg'], name="Gas Rate", line=dict(color="#d62728")))
        fig_g.update_layout(**semi_log_layout("Gas Production Preview", yaxis="Gas Rate (Mscf/d)"))
        st.plotly_chart(fig_g, use_container_width=True, theme="streamlit")
    with p_c2:
        fig_o = go.Figure(go.Scatter(x=preview['t'], y=preview['qo'], name="Oil Rate", line=dict(color="#2ca02c")))
        fig_o.update_layout(**semi_log_layout("Oil Production Preview", yaxis="Oil Rate (STB/d)"))
        st.plotly_chart(fig_o, use_container_width=True, theme="streamlit")
    with st.expander("Click for details"):
        st.markdown("These charts use a simplified analytical model for quick iteration.")

elif selected_tab == "Control Panel":
    st.header("Control Panel")
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox(
            "Well control",
            ["BHP", "RATE_GAS_MSCFD", "RATE_OIL_STBD", "RATE_LIQ_STBD"],
            index=["BHP", "RATE_GAS_MSCFD", "RATE_OIL_STBD", "RATE_LIQ_STBD"].index(st.session_state.get("control", "BHP")),
            key="control",
            help="Choose BHP control or a rate target.",
        )
        if st.session_state.control == "BHP":
            st.number_input("BHP (psi)", 500.0, 15000.0, float(st.session_state.get("bhp_psi", 2500.0)), 50.0, key="pad_bhp_psi")
        elif st.session_state.control == "RATE_GAS_MSCFD":
            st.number_input("Gas rate (Mscf/d)", 0.0, 500000.0, float(st.session_state.get("rate_mscfd", 5000.0)), 100.0, key="pad_rate_mscfd")
        elif st.session_state.control == "RATE_OIL_STBD":
            st.number_input("Oil rate (STB/d)", 0.0, 20000.0, float(st.session_state.get("rate_stbd", 800.0)), 10.0, key="pad_rate_stbd")
        elif st.session_state.control == "RATE_LIQ_STBD":
            st.number_input("Liquid rate (STB/d)", 0.0, 40000.0, float(st.session_state.get("rate_stbd", 1200.0)), 10.0, key="pad_rate_stbd")
    with c2:
        st.checkbox("Use gravity", bool(st.session_state.get("use_gravity", True)), key="use_gravity")
        st.number_input("kv/kh", 0.01, 1.0, float(st.session_state.get("kvkh", 0.10)), 0.01, "%.2f", key="kvkh")
        st.number_input("Geomech α (1/psi)", 0.0, 1e-3, float(st.session_state.get("geo_alpha", 0.0)), 1e-5, "%.5f", key="geo_alpha")
    st.markdown("#### Control Summary")
    summary = {
        "Control": st.session_state.get("pad_ctrl"),
        "BHP (psi)": st.session_state.get("pad_bhp_psi"),
        "Gas rate (Mscf/d)": st.session_state.get("pad_rate_mscfd"),
        "Oil/Liq rate (STB/d)": st.session_state.get("pad_rate_stbd"),
        "Use gravity": st.session_state.get("use_gravity"),
        "kv/kh": st.session_state.get("kvkh"),
        "Geomech α (1/psi)": st.session_state.get("geo_alpha"),
    }
    st.write(summary)
    
elif selected_tab == "Generate 3D property volumes":
    st.header("Generate 3D Property Volumes (kx, ky, ϕ)")
    st.info("Use this tab to (re)generate φ/k grids based on sidebar parameters.")
    
    if st.button("Generate New Property Volumes", use_container_width=True, type="primary"):
        generate_property_volumes(state)
        
    st.markdown("---")
    
    if st.session_state.get('kx') is not None:
        st.markdown("### Mid-Layer Property Maps")
        kx_display = get_k_slice(st.session_state.kx, state['nz'] // 2)
        ky_display = get_k_slice(st.session_state.ky, state['nz'] // 2)
        phi_display = get_k_slice(st.session_state.phi, state['nz'] // 2)
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.imshow(kx_display, origin="lower", color_continuous_scale="Viridis", labels=dict(color="mD"), title="<b>kx — mid-layer (mD)</b>"), use_container_width=True)
        with c2:
            st.plotly_chart(px.imshow(ky_display, origin="lower", color_continuous_scale="Cividis", labels=dict(color="mD"), title="<b>ky — mid-layer (mD)</b>"), use_container_width=True)
            
        st.plotly_chart(px.imshow(phi_display, origin="lower", color_continuous_scale="Magma", labels=dict(color="ϕ"), title="<b>Porosity ϕ — mid-layer (fraction)</b>"), use_container_width=True)
    else:
        st.info("Click the button above to generate initial property volumes.")


elif selected_tab == "PVT (Black-Oil)":
    st.header("PVT (Black-Oil) Analysis")
    P = np.linspace(max(1000, state["p_min_bhp_psi"]), max(2000, state["p_init_psi"] + 1000), 120)
    Rs, Bo, Bg, mug = (Rs_of_p(P, state["pb_psi"], state["Rs_pb_scf_stb"]), Bo_of_p(P, state["pb_psi"], state["Bo_pb_rb_stb"]), Bg_of_p(P), mu_g_of_p(P, state["pb_psi"], state["mug_pb_cp"]))
    f1 = go.Figure(go.Scatter(x=P, y=Rs, line=dict(color="firebrick", width=3)))
    f1.add_vline(x=state["pb_psi"], line_dash="dash", line_width=2, annotation_text="Bubble Point")
    f1.update_layout(template="plotly_white", title="<b>Solution GOR Rs vs Pressure</b>", xaxis_title="Pressure (psi)", yaxis_title="Rs (scf/STB)")
    st.plotly_chart(f1, use_container_width=True)
    f2 = go.Figure(go.Scatter(x=P, y=Bo, line=dict(color="seagreen", width=3)))
    f2.add_vline(x=state["pb_psi"], line_dash="dash", line_width=2, annotation_text="Bubble Point")
    f2.update_layout(template="plotly_white", title="<b>Oil FVF Bo vs Pressure</b>", xaxis_title="Pressure (psi)", yaxis_title="Bo (rb/STB)")
    st.plotly_chart(f2, use_container_width=True)
    f3 = go.Figure(go.Scatter(x=P, y=Bg, line=dict(color="steelblue", width=3)))
    f3.update_layout(template="plotly_white", title="<b>Gas FVF Bg vs Pressure</b>", xaxis_title="Pressure (psi)", yaxis_title="Bg (rb/scf)")
    st.plotly_chart(f3, use_container_width=True)
    f4 = go.Figure(go.Scatter(x=P, y=mug, line=dict(color="mediumpurple", width=3)))
    f4.update_layout(template="plotly_white", title="<b>Gas viscosity μg vs Pressure</b>", xaxis_title="Pressure (psi)", yaxis_title="μg (cP)")
    st.plotly_chart(f4, use_container_width=True)

elif selected_tab == "MSW Wellbore":
    st.header("MSW Wellbore Physics — Heel–Toe & Limited-Entry")
    try:
        L_ft, ss_ft = float(state['L_ft']), float(state['stage_spacing_ft'])
        n_stages = max(1, int(L_ft / ss_ft))
        well_id_ft, f_fric, dP_le = float(state['wellbore_ID_ft']), float(state['f_fric']), float(state['dP_LE_psi'])
        p_bhp, p_res = float(state['pad_bhp_psi']), float(state['p_init_psi'])
        q_oil_total_stbd = _get_sim_preview()['qo'][0]
        q_dist = np.ones(n_stages) / n_stages
        for _ in range(5):
            q_per_stage_bpd, p_wellbore_at_stage = q_dist * q_oil_total_stbd, np.zeros(n_stages)
            p_current, flow_rate_bpd = p_bhp, q_oil_total_stbd
            for i in range(n_stages):
                p_wellbore_at_stage[i] = p_current
                v_fps = (flow_rate_bpd * 5.615 / (24*3600)) / (np.pi * (well_id_ft/2)**2)
                p_current += (2 * f_fric * 50.0 * v_fps**2 * ss_ft / well_id_ft) / 144.0
                flow_rate_bpd -= q_per_stage_bpd[i]
            drawdown = p_res - p_wellbore_at_stage - dP_le
            q_new_dist_unnorm = np.sqrt(np.maximum(0, drawdown))
            if np.sum(q_new_dist_unnorm) > 1e-9:
                q_dist = q_new_dist_unnorm / np.sum(q_new_dist_unnorm)
        c1_msw, c2_msw = st.columns(2)
        with c1_msw:
            fig_p = go.Figure(go.Scatter(x=np.arange(n_stages)*ss_ft, y=p_wellbore_at_stage, mode='lines+markers'))
            fig_p.update_layout(title="<b>Wellbore Pressure Profile</b>", xaxis_title="Dist. from Heel (ft)", yaxis_title="Pressure (psi)", template="plotly_white")
            st.plotly_chart(fig_p, use_container_width=True)
        with c2_msw:
            fig_q = go.Figure(go.Bar(x=np.arange(n_stages)*ss_ft, y=q_dist * 100))
            fig_q.update_layout(title="<b>Flow Contribution per Stage</b>", xaxis_title="Dist. from Heel (ft)", yaxis_title="Contribution (%)", template="plotly_white")
            st.plotly_chart(fig_q, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute wellbore hydraulics. Error: {e}")

elif selected_tab == "RTA":
    st.header("RTA — Quick Diagnostics")
    sim_data = st.session_state.sim if st.session_state.sim is not None else _get_sim_preview()
    t, qg = sim_data["t"], sim_data["qg"]
    y_type_rta = "log" if st.radio("Rate y-axis", ["Linear", "Log"], horizontal=True) == "Log" else "linear"
    fig = go.Figure(go.Scatter(x=t, y=qg, line=dict(color="firebrick", width=3), name="Gas"))
    fig.update_layout(**semi_log_layout("R1. Gas rate (q) vs time", yaxis="q (Mscf/d)"))
    fig.update_yaxes(type=y_type_rta)
    st.plotly_chart(fig, use_container_width=True)
    slope = np.gradient(np.log(np.maximum(qg, 1e-9)), np.log(np.maximum(t, 1e-9)))
    fig2 = go.Figure(go.Scatter(x=t, y=slope, line=dict(color="teal", width=3), name="dlogq/dlogt"))
    fig2.update_layout(**semi_log_layout("R2. Log-log derivative", yaxis="Slope"))
    st.plotly_chart(fig2, use_container_width=True)

elif selected_tab == "Results":
    st.header("Simulation Results")
    
    # ---- Run button ----
    run_clicked = st.button("Run simulation", type="primary", use_container_width=True)
    if run_clicked:
        st.session_state.sim = None
        st.session_state["sanity_warned"] = False
        if "kx" not in st.session_state:
            st.info("Rock properties not found. Generating them first...")
            generate_property_volumes(state)
        with st.spinner("Running simulation..."):
            sim_out = run_simulation_engine(state)
            if sim_out is not None:
                st.session_state.sim = sim_out

    # ---- Fetch sim & guard against stale signatures ----
    sim = st.session_state.get("sim")
    cur_sig  = _sim_signature_from_state()
    prev_sig = sim.get("_sim_signature") if isinstance(sim, dict) else None
    if (sim is not None) and (prev_sig is not None) and (cur_sig != prev_sig):
        st.session_state.sim = None
        sim = None
        st.info("Inputs have changed. Please click **Run simulation** to refresh results.")

    if not isinstance(sim, dict):
        st.info("Click **Run simulation** to compute and display the full results.")
        st.stop()

    # ---- Success banner ----
    if sim.get("runtime_s") is not None:
        st.success(f"Simulation complete in {sim.get('runtime_s', 0):.2f} seconds.")

    # ---- Resolve EURs ----
    eur_g = sim.get("EUR_g_BCF", 0.0)
    eur_o = sim.get("EUR_o_MMBO", 0.0)

    # ---- Sanity messages (raw values, not clamped) ----
    OIL_MIN, OIL_MAX = 0.4, 2.0
    GAS_MIN, GAS_MAX = 0.2, 3.5
    GOR_MAX = 2000
    issues = []
    if not (GAS_MIN <= eur_g <= GAS_MAX):
        issues.append(f"Gas EUR {eur_g:.2f} BCF outside sanity ({GAS_MIN}, {GAS_MAX}) BCF.")
    
    implied_gor = (eur_g * 1e9) / (eur_o * 1e6) if eur_o > 1e-6 else 0.0
    
    # CORRECTED: Added a small tolerance to the GOR check to avoid floating-point errors
    if implied_gor > (GOR_MAX + 1.0):
        issues.append(f"Implied EUR GOR {implied_gor:,.0f} scf/STB exceeds {GOR_MAX:,} scf/STB.")
        
    if issues and not st.session_state.get("sanity_warned"):
        st.warning(
            "Sanity checks flagged issues (Analytical engine).\n\n"
            + "\n".join(f"- {m}" for m in issues)
            + "\n\n**Tip:** If gas looks too high for the oil window, try increasing **Pad BHP (psi)** "
              "toward **pb_psi** to reduce gas liberation."
        )
        st.session_state["sanity_warned"] = True

    # ---- DISPLAY: clamp to Midland bands & render compact gauges ----
    eur_o_disp, eur_g_disp, clamp_note = enforce_midland_bounds(eur_o, eur_g)
    if not st.session_state.get("eur_bounds_noted"):
        st.info(clamp_note)
        st.session_state["eur_bounds_noted"] = True

    c1, c2 = st.columns(2)
    with c1:
        fig_oil = render_semi_gauge(
            "EUR Oil", eur_o_disp, "MMBO",
            vmin=0.0, vmax=MIDLAND_BOUNDS["oil_mmbo"][1],
            bar_color="#22c55e",
        )
        st.plotly_chart(fig_oil, use_container_width=True)
    with c2:
        fig_gas = render_semi_gauge(
            "EUR Gas", eur_g_disp, "BCF",
            vmin=0.0, vmax=MIDLAND_BOUNDS["gas_bcf"][1],
            bar_color="#ef4444",
        )
        st.plotly_chart(fig_gas, use_container_width=True)
    
    st.markdown("---")
    plot_scale = st.radio(
        "Plot Scale (Time Axis)", 
        ["Semi-Log", "Linear"], 
        horizontal=True,
        label_visibility="collapsed"
    )
    xaxis_type = "log" if plot_scale == "Semi-Log" else "linear"

    # ===================== RATE & CUMULATIVE PLOTS =====================
    t  = sim.get("t"); qg = sim.get("qg"); qo = sim.get("qo"); qw = sim.get("qw")

    if t is not None and (qg is not None or qo is not None or qw is not None):
        fig_rate = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        if qg is not None: fig_rate.add_trace(go.Scatter(x=t, y=qg, name="Gas (Mscf/d)", line=dict(width=2, color=COLOR_GAS)), secondary_y=False)
        if qo is not None: fig_rate.add_trace(go.Scatter(x=t, y=qo, name="Oil (STB/d)", line=dict(width=2, color=COLOR_OIL)), secondary_y=True)
        if qw is not None: fig_rate.add_trace(go.Scatter(x=t, y=qw, name="Water (STB/d)", line=dict(width=1.8, dash="dot", color=COLOR_WATER)), secondary_y=True)
        fig_rate.update_layout(template="plotly_white", title_text="<b>Production Rate vs. Time</b>", height=460, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), font=dict(size=13), margin=dict(l=10, r=10, t=50, b=10))
        fig_rate.update_xaxes(type=xaxis_type, title="Time (days)", showgrid=True, gridcolor="rgba(0,0,0,0.12)")
        fig_rate.update_yaxes(title_text="Gas rate (Mscf/d)", secondary_y=False, showgrid=True, gridcolor="rgba(0,0,0,0.15)")
        fig_rate.update_yaxes(title_text="Liquid rates (STB/d)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_rate, use_container_width=True, theme=None)
    else:
        st.warning("Rate series not available.")

    cum_g = sim.get("cum_g_BCF"); cum_o = sim.get("cum_o_MMBO"); cum_w = sim.get("cum_w_MMBL")
    if t is not None and (cum_g is not None or cum_o is not None or cum_w is not None):
        fig_cum = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        if cum_g is not None: fig_cum.add_trace(go.Scatter(x=t, y=cum_g, name="Cum Gas (BCF)", line=dict(width=3, color=COLOR_GAS)), secondary_y=False)
        if cum_o is not None: fig_cum.add_trace(go.Scatter(x=t, y=cum_o, name="Cum Oil (MMbbl)", line=dict(width=3, color=COLOR_OIL)), secondary_y=True)
        if cum_w is not None: fig_cum.add_trace(go.Scatter(x=t, y=cum_w, name="Cum Water (MMbbl)", line=dict(width=2, dash="dot", color=COLOR_WATER)), secondary_y=True)
        fig_cum.update_layout(template="plotly_white", title_text="<b>Cumulative Production vs. Time</b>", height=460, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), font=dict(size=13), margin=dict(l=10, r=10, t=50, b=10))
        fig_cum.update_xaxes(type=xaxis_type, title="Time (days)", showgrid=True, gridcolor="rgba(0,0,0,0.12)")
        fig_cum.update_yaxes(title_text="Gas (BCF)", secondary_y=False, showgrid=True, gridcolor="rgba(0,0,0,0.15)")
        fig_cum.update_yaxes(title_text="Liquids (MMbbl)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_cum, use_container_width=True, theme=None)
    else:
        st.warning("Cumulative series not available.")

# ======== 3D Viewer tab ========
elif selected_tab == "3D Viewer":
    """
    Interactive 3D isosurface viewer for model properties and simulation results.
    - Pulls 3D arrays from session_state (kx, phi) and from the last sim (pressure, ΔP, OOIP).
    - Lets the user choose downsampling + isosurface cut to keep it fast and legible.
    """
    st.subheader("3D Viewer")

    # -- Gather inputs --
    sim     = st.session_state.get("sim") or {}
    kx_vol  = st.session_state.get("kx")           # expected shape: (nz, ny, nx)
    phi_vol = st.session_state.get("phi")          # expected shape: (nz, ny, nx)

    # If nothing is available, guide the user and exit the tab early.
    if kx_vol is None and phi_vol is None and not sim:
        st.info(
            "Render your 3D grid / fractures / saturation maps here.\n\n"
            "No 3D properties yet — generate rock properties or run a simulation."
        )
        st.stop()

    # Build the menu only with properties that actually exist.
    menu = []
    if kx_vol is not None:
        menu.append("Permeability (kx)")
    if phi_vol is not None:
        menu.append("Porosity (ϕ)")
    if sim.get("press_matrix") is not None:
        menu.append("Pressure (psi)")
    if sim.get("press_matrix") is not None and sim.get("p_init_3d") is not None:
        menu.append("Pressure Change (ΔP)")
    if sim.get("ooip_3d") is not None:
        menu.append("Original Oil In Place (OOIP)")

    if not menu:
        st.info("No 3D properties are available yet. Run a simulation to populate pressure/OOIP.")
        st.stop()

    prop_3d = st.selectbox("Select property to view:", menu, index=0)

    # Viewer controls
    c1, c2 = st.columns(2)
    with c1:
        ds = st.slider(
            "Downsample factor",
            min_value=1, max_value=10,
            value=int(st.session_state.get("vol_downsample", 2)),
            step=1,
            help="Larger values speed up rendering by skipping voxels.",
            key="vol_ds",
        )
    with c2:
        iso_rel = st.slider(
            "Isosurface value (relative)",
            min_value=0.05, max_value=0.95,
            value=float(st.session_state.get("iso_value_rel", 0.85)),
            step=0.05,
            help="Relative cut inside the min–max value range.",
            key="iso_val_rel",
        )

    # Grid spacing (accept *_ft fallbacks)
    dx = float(state.get("dx_ft", state.get("dx", 1.0)))
    dy = float(state.get("dy_ft", state.get("dy", 1.0)))
    dz = float(state.get("dz_ft", state.get("dz", 1.0)))

    # Resolve data + styling for the selected property
    data_3d = None
    colorscale = "Viridis"
    colorbar_title = ""

    if prop_3d.startswith("Permeability"):
        data_3d = kx_vol
        colorscale = "Viridis"
        colorbar_title = "kx (mD)"
    elif prop_3d.startswith("Porosity"):
        data_3d = phi_vol
        colorscale = "Magma"
        colorbar_title = "Porosity (ϕ)"
    elif prop_3d.startswith("Pressure (psi)"):
        data_3d = sim.get("press_matrix")  # (nz, ny, nx)
        colorscale = "Jet"
        colorbar_title = "Pressure (psi)"
    elif prop_3d.startswith("Pressure Change"):
        p_final = sim.get("press_matrix")
        p_init  = sim.get("p_init_3d")
        if p_final is not None and p_init is not None:
            data_3d = (np.asarray(p_init) - np.asarray(p_final))  # ΔP = Pin − Pfinal
            colorscale = "Inferno"
            colorbar_title = "ΔP (psi)"
    elif prop_3d.startswith("Original Oil"):
        data_3d = sim.get("ooip_3d")
        colorscale = "Plasma"
        colorbar_title = "OOIP (STB/cell)"

    # Basic validation
    if data_3d is None:
        st.warning(f"Data for '{prop_3d}' not found. Please run a simulation.")
        st.stop()

    data_3d = np.asarray(data_3d)
    if data_3d.ndim != 3:
        st.warning("3D data is not in the expected (nz, ny, nx) shape.")
        st.stop()

    # Downsample (prefer a helper if you have one)
    try:
        data_ds = downsample_3d(data_3d, ds)  # your helper (if present)
    except Exception:
        data_ds = data_3d[::ds, ::ds, ::ds]   # simple stride fallback

    # Isosurface cut
    vmin, vmax = float(np.nanmin(data_ds)), float(np.nanmax(data_ds))
    isoval = vmin + (vmax - vmin) * float(iso_rel)

    # Build coordinates consistent with (nz, ny, nx)
    nz, ny, nx = data_ds.shape
    z = np.arange(nz) * dz * ds
    y = np.arange(ny) * dy * ds
    x = np.arange(nx) * dx * ds
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")  # shapes (nz, ny, nx)

    # Render the 3D isosurface
    with st.spinner("Generating 3D plot..."):
        fig3d = go.Figure(
            go.Isosurface(
                x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
                value=data_ds.ravel(),
                isomin=isoval, isomax=vmax,
                surface_count=1,
                caps=dict(x_show=False, y_show=False, z_show=False),
                colorscale=colorscale,
                colorbar=dict(title=colorbar_title),
            )
        )

        # Optional: draw a horizontal well (best-effort based on state)
        try:
            L_ft   = float(state.get("L_ft", nx * dx))
            n_lat  = int(state.get("n_laterals", 1))
            y_span = ny * dy * ds
            y_positions = ([y_span/3.0, 2*y_span/3.0] if n_lat >= 2 else [y_span/2.0])
            z_mid = (nz * dz * ds) / 2.0

            for i, y_pos in enumerate(y_positions):
                fig3d.add_trace(
                    go.Scatter3d(
                        x=[0.0, L_ft], y=[y_pos, y_pos], z=[z_mid, z_mid],
                        mode="lines",
                        line=dict(width=8),
                        name=("Well" if i == 0 else None),
                        showlegend=(i == 0),
                    )
                )
        except Exception:
            # If we can't compute the overlay, silently skip it.
            pass

        fig3d.update_layout(
            title=f"<b>3D Isosurface: {prop_3d}</b>",
            scene=dict(
                xaxis_title="X (ft)",
                yaxis_title="Y (ft)",
                zaxis_title="Z (ft)",
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            template="plotly_white",
        )

        st.plotly_chart(fig3d, use_container_width=True)

# ======== Debug tab ========
elif selected_tab == "Debug":
    """Raw JSON dump of the latest simulation dictionary for quick inspection."""
    st.subheader("Debug")
    st.json(st.session_state.get("sim"))

# ======== Slice Viewer ========
elif selected_tab == "Slice Viewer":
    """
    2D slice explorer for any (nz, ny, nx) volume.
    Lets you scroll through i / j / k planes with live imshow rendering.
    """
    st.header("Slice Viewer")

    sim_data = st.session_state.get("sim")
    have_kx  = st.session_state.get('kx') is not None
    have_ky  = st.session_state.get('ky') is not None
    have_phi = st.session_state.get('phi') is not None

    if sim_data is None and not (have_kx or have_ky or have_phi):
        st.warning("Please generate rock properties or run a simulation to enable the slice viewer.")
        st.stop()

    # Build property options based on availability
    prop_list = []
    if have_kx:  prop_list.append('Permeability (kx)')
    if have_ky:  prop_list.append('Permeability (ky)')
    if have_phi: prop_list.append('Porosity (ϕ)')
    if sim_data and sim_data.get('press_matrix') is not None:
        prop_list.append('Pressure (psi)')

    c1, c2 = st.columns(2)
    with c1:
        prop_slice = st.selectbox("Select property:", prop_list)
    with c2:
        plane_slice = st.selectbox(
            "Select plane:",
            ["k-plane (z, top-down)", "j-plane (y, side-view)", "i-plane (x, end-view)"]
        )

    # Choose the volume to slice
    if 'kx' in prop_slice:
        data_3d = st.session_state.get('kx')
    elif 'ky' in prop_slice:
        data_3d = st.session_state.get('ky')
    elif 'ϕ' in prop_slice:
        data_3d = st.session_state.get('phi')
    else:
        data_3d = sim_data.get('press_matrix') if sim_data else None

    if data_3d is not None:
        data_3d = np.asarray(data_3d)
        if data_3d.ndim != 3:
            st.warning("3D data is not in the expected (nz, ny, nx) shape.")
            st.stop()

        nz, ny, nx = data_3d.shape

        # Plane selection + index slider
        if "k-plane" in plane_slice:
            idx, axis_name = st.slider("k-index (z-layer)", 0, nz - 1, nz // 2), "k"
            data_2d, labels = data_3d[idx, :, :], dict(x="i-index", y="j-index")
        elif "j-plane" in plane_slice:
            idx, axis_name = st.slider("j-index (y-layer)", 0, ny - 1, ny // 2), "j"
            data_2d, labels = data_3d[:, idx, :], dict(x="i-index", y="k-index")
        else:
            idx, axis_name = st.slider("i-index (x-layer)", 0, nx - 1, nx // 2), "i"
            data_2d, labels = data_3d[:, :, idx], dict(x="j-index", y="k-index")

        # Render the slice
        fig = px.imshow(
            data_2d, origin="lower", aspect='equal',
            labels=labels, color_continuous_scale='viridis'
        )
        fig.update_layout(title=f"<b>{prop_slice} @ {axis_name} = {idx}</b>", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Data for '{prop_slice}' not found.")

# ======== QA / Material Balance ========
elif selected_tab == "QA / Material Balance":
    """
    Quick material-balance sanity plots.
    - Generates synthetic pressure for analytical models to enable MB plots.
    """
    st.header("QA / Material Balance")
    sim = st.session_state.get("sim")

    if sim is None:
        st.warning("Run a simulation on the 'Results' tab to view QA plots.")
        st.stop()

    pavg = sim.get("p_avg_psi")
    
    # --- Check for missing pressure and generate a synthetic profile if needed ---
    if pavg is None and "Analytical" in st.session_state.get("engine_type", ""):
        st.info("Analytical model does not compute average pressure. Generating a synthetic pressure profile using material balance principles...")
        try:
            # Estimate total pore volume and compressibility
            phi_avg = 0.10 # Assume average porosity for bulk calculation
            ct = state.get('ct_1_over_psi', 15e-6)
            Vp = state['nx'] * state['ny'] * state['nz'] * state['dx'] * state['dy'] * state['dz'] * phi_avg
            
            # Get production volumes
            t = np.asarray(sim['t'], float)
            qo = np.asarray(sim['qo'], float)
            qg = np.asarray(sim['qg'], float)
            Np = cumulative_trapezoid(qo, t, initial=0.0)
            Gp = cumulative_trapezoid(qg * 1000.0, t, initial=0.0) # Gp in scf
            
            # Use PVT functions
            p_init = state["p_init_psi"]
            pb = state["pb_psi"]
            Rs_pb = state["Rs_pb_scf_stb"]
            Bo_pb = state["Bo_pb_rb_stb"]
            
            # Underground withdrawal in reservoir bbls
            Bo = Bo_of_p(p_init, pb, Bo_pb) # Simplified: use initial Bo for withdrawal calc
            Bg = Bg_of_p(p_init) # Simplified: use initial Bg
            Rs = Rs_of_p(p_init, pb, Rs_pb) # Simplified
            underground_withdrawal = Np * Bo + (Gp - Np * Rs) * Bg
            
            # Calculate pressure drop and the synthetic profile
            delta_p = underground_withdrawal / (Vp * ct)
            pavg = p_init - delta_p
            sim['p_avg_psi'] = pavg # Store it for this session
            st.success("Synthetic pressure profile generated successfully!")
        except Exception as e:
            st.error(f"Failed to generate synthetic pressure profile: {e}")
            st.stop()
            
    elif pavg is None:
        st.error("Average reservoir pressure time series was not returned by the solver. Cannot generate Material Balance plots.")
        st.stop()

    # --- Plotting Section ---
    st.markdown("#### Figure 1: Average Reservoir Pressure vs. Time")
    fig_p = go.Figure(go.Scatter(x=sim["t"], y=pavg, name="p̄ reservoir (psi)"))
    fig_p.update_layout(
        template="plotly_white",
        xaxis_title="Time (days)", yaxis_title="Pressure (psi)"
    )
    st.plotly_chart(fig_p, use_container_width=True, theme=None)
    with st.expander("About this chart"):
        st.markdown("This plot shows the depletion of the average reservoir pressure over time. For analytical models, this profile is estimated from production volumes.")

    t  = np.asarray(sim["t"], float); qg = np.asarray(sim["qg"], float); qo = np.asarray(sim["qo"], float)
    
    # --- Gas Material Balance ---
    st.markdown("---")
    st.markdown("### Gas Material Balance (P/Z vs. Gp)")
    Gp_MMscf  = cumulative_trapezoid(qg, t, initial=0.0) / 1e3
    z_factors = z_factor_approx(np.asarray(pavg), p_init_psi=state["p_init_psi"])
    p_over_z  = np.asarray(pavg) / np.maximum(z_factors, 1e-12)
    fit_start = max(1, len(Gp_MMscf) // 4)

    if len(Gp_MMscf[fit_start:]) > 1:
        slope, intercept, _, _, _ = stats.linregress(Gp_MMscf[fit_start:], p_over_z[fit_start:])
        giip_bcf = max(0.0, -intercept / slope / 1000.0) if slope != 0 else 0.0
        
        st.markdown("#### Figure 2: Gas P/Z Plot")
        fig_pz_gas = go.Figure()
        fig_pz_gas.add_trace(go.Scatter(x=Gp_MMscf, y=p_over_z, mode="markers", name="P/Z Data"))
        x_fit = np.array([0.0, giip_bcf * 1000.0]); y_fit = slope * x_fit + intercept
        fig_pz_gas.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines", name="Linear Extrapolation", line=dict(dash="dash")))
        fig_pz_gas.update_layout(xaxis_title="Gp - Cumulative Gas Production (MMscf)", yaxis_title="P/Z", template="plotly_white")
        st.plotly_chart(fig_pz_gas, use_container_width=True)
        with st.expander("About this chart"):
            st.markdown("This is a classic material balance plot for gas reservoirs. For a volumetric reservoir, the data should follow a straight line. Extrapolating this line to a P/Z of zero gives an estimate of the Gas-Initially-In-Place (GIIP).")
        st.metric("Material Balance GIIP (from P/Z)", f"{giip_bcf:.2f} BCF")

    # --- Oil Material Balance (Havlena-Odeh) ---
    st.markdown("---")
    st.markdown("### Oil Material Balance (Havlena-Odeh)")
    Np_STB = cumulative_trapezoid(qo, t, initial=0.0)
    Gp_scf = cumulative_trapezoid(qg * 1_000.0, t, initial=0.0)
    Rp     = np.divide(Gp_scf, Np_STB, out=np.zeros_like(Gp_scf), where=Np_STB > 1e-3)
    Bo     = Bo_of_p(pavg, state["pb_psi"], state["Bo_pb_rb_stb"])
    Rs     = Rs_of_p(pavg, state["pb_psi"], state["Rs_pb_scf_stb"])
    Bg     = Bg_of_p(pavg)
    p_init = state["p_init_psi"]
    Boi    = Bo_of_p(p_init, state["pb_psi"], state["Bo_pb_rb_stb"])
    Rsi    = Rs_of_p(p_init, state["pb_psi"], state["Rs_pb_scf_stb"])
    F  = Np_STB * (Bo + (Rp - Rs) * Bg)
    Et = (Bo - Boi) + (Rsi - Rs) * Bg
    
    fit_start_oil = max(1, len(F) // 4)
    if len(F[fit_start_oil:]) > 1:
        slope_oil, _, _, _, _ = stats.linregress(Et[fit_start_oil:], F[fit_start_oil:])
        ooip_mmstb = max(0.0, slope_oil / 1e6)
        
        st.markdown("#### Figure 3: Havlena-Odeh F vs. Et Plot")
        fig_mbe_oil = go.Figure()
        fig_mbe_oil.add_trace(go.Scatter(x=Et, y=F, mode="markers", name="F vs Et Data"))
        x_fit_oil = np.array([0.0, np.nanmax(Et)]); y_fit_oil = slope_oil * x_fit_oil
        fig_mbe_oil.add_trace(go.Scatter(x=x_fit_oil, y=y_fit_oil, mode="lines", name=f"Slope (OOIP) = {ooip_mmstb:.2f} MMSTB", line=dict(dash="dash")))
        fig_mbe_oil.update_layout(xaxis_title="Et - Total Expansion (rb/STB)", yaxis_title="F - Underground Withdrawal (rb)", template="plotly_white")
        st.plotly_chart(fig_mbe_oil, use_container_width=True)
        with st.expander("About this chart"):
            st.markdown("The Havlena-Odeh method linearizes the material balance equation. Plotting underground withdrawal (F) against total fluid expansion (Et) should yield a straight line whose slope is the Oil-Initially-In-Place (OOIP).")
        st.metric("Material Balance OOIP (from F vs Et)", f"{ooip_mmstb:.2f} MMSTB")

# ======== Economics ========
elif selected_tab == "Economics":
    """
    Simple yearly cash-flow model driven by the latest simulation.
    Includes NPV/IRR/payout and a formatted table of yearly metrics.
    """
    st.header("Financial Model")

    if st.session_state.get("sim") is None:
        st.info("Run a simulation on the 'Results' tab first to populate the financial model.")
    else:
        sim = st.session_state["sim"]
        t = np.asarray(sim.get("t", []), float)

        # Safe extraction for rate series (fall back to zeros of length t)
        qo = np.nan_to_num(np.asarray(sim.get("qo")), nan=0.0) if sim.get("qo") is not None else np.zeros_like(t)
        qg = np.nan_to_num(np.asarray(sim.get("qg")), nan=0.0) if sim.get("qg") is not None else np.zeros_like(t)
        qw = np.nan_to_num(np.asarray(sim.get("qw")), nan=0.0) if sim.get("qw") is not None else np.zeros_like(t)

        st.subheader("Economic Assumptions")
        c1, c2, c3, c4 = st.columns(4)
        with c1: capex      = st.number_input("CAPEX ($MM)", 1.0, 100.0, 15.0, 0.5, key="econ_capex") * 1e6
        with c2: oil_price  = st.number_input("Oil price ($/bbl)", 0.0, 500.0, 75.0, 1.0, key="econ_oil_price")
        with c3: gas_price  = st.number_input("Gas price ($/Mcf)", 0.0, 50.0, 2.50, 0.1, key="econ_gas_price")
        with c4: disc_rate  = st.number_input("Discount rate (fraction)", 0.0, 1.0, 0.10, 0.01, key="econ_disc")

        c1, c2, c3, c4 = st.columns(4)
        with c1: royalty = st.number_input("Royalty (fraction)", 0.0, 0.99, 0.20, 0.01, key="econ_royalty")
        with c2: tax     = st.number_input("Severance tax (fraction)", 0.0, 0.99, 0.045, 0.005, key="econ_tax")
        with c3: opex_bpd = st.number_input("OPEX ($/bbl liquids)", 0.0, 200.0, 6.0, 0.5, key="econ_opex")
        with c4: wd_cost = st.number_input("Water disposal ($/bbl)", 0.0, 50.0, 1.5, 0.1, key="econ_wd")

        # ---- Aggregate to yearly totals (trap small series gracefully) ----
        df_yearly = pd.DataFrame()
        if len(t) > 1:
            df = pd.DataFrame({'days': t, 'oil_stb_d': qo, 'gas_mscf_d': qg, 'water_stb_d': qw})
            df['year'] = (df['days'] / 365.25).astype(int)

            rows = []
            for year, g in df.groupby('year'):
                d = g['days'].values
                if len(d) > 1:
                    # CORRECTED: Replaced deprecated np.trapz with trapezoid
                    rows.append({
                        'year': year,
                        'oil_stb':   float(trapezoid(g['oil_stb_d'].values, d)),
                        'gas_mscf':  float(trapezoid(g['gas_mscf_d'].values, d)),
                        'water_stb': float(trapezoid(g['water_stb_d'].values, d)),
                    })
            if rows:
                df_yearly = pd.DataFrame(rows)

        # ---- Build cash flows ----
        if not df_yearly.empty:
            df_yearly['Revenue'] = (df_yearly['oil_stb'] * oil_price) + (df_yearly['gas_mscf'] * gas_price)
            df_yearly['Royalty'] = df_yearly['Revenue'] * royalty
            df_yearly['Taxes']   = (df_yearly['Revenue'] - df_yearly['Royalty']) * tax
            df_yearly['OPEX']    = (df_yearly['oil_stb'] + df_yearly['water_stb']) * opex_bpd + (df_yearly['water_stb'] * wd_cost)
            df_yearly['Net Cash Flow'] = df_yearly['Revenue'] - df_yearly['Royalty'] - df_yearly['Taxes'] - df_yearly['OPEX']
            cash_flows = [-capex] + df_yearly['Net Cash Flow'].tolist()
        else:
            cash_flows = [-capex]

        # ---- Financial KPIs ----
        npv = npf.npv(disc_rate, cash_flows) if cash_flows else 0.0
        try:
            irr = npf.irr(cash_flows) if len(cash_flows) > 1 else np.nan
        except ValueError:
            irr = np.nan

        display_df = pd.DataFrame({'year': range(-1, len(cash_flows)-1), 'Net Cash Flow': cash_flows})
        display_df['Cumulative Cash Flow'] = display_df['Net Cash Flow'].cumsum()

        # Payout = interpolation within the first positive year
        payout_period = np.nan
        if (display_df['Cumulative Cash Flow'] > 0).any():
            first_pos = display_df['Cumulative Cash Flow'].gt(0).idxmax()
            if first_pos > 0 and display_df['Cumulative Cash Flow'].iloc[first_pos-1] < 0:
                last_neg = display_df['Cumulative Cash Flow'].iloc[first_pos - 1]
                ncf_year = display_df['Net Cash Flow'].iloc[first_pos]
                payout_period = (display_df['year'].iloc[first_pos-1]) + (-last_neg / ncf_year if ncf_year > 0 else np.inf)

        st.subheader("Key Financial Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("NPV", f"${npv/1e6:,.2f} MM", help="Net Present Value at the specified discount rate.")
        m2.metric("IRR", f"{irr:.1%}" if pd.notna(irr) and np.isfinite(irr) else "N/A", help="Internal Rate of Return.")
        m3.metric("Payout Period (Years)", f"{payout_period:.2f}" if pd.notna(payout_period) else "N/A", help="Time until initial investment is recovered.")

        st.subheader("Cash Flow Details")
        c1, c2 = st.columns(2)
        with c1:
            fig_ncf = px.bar(display_df, x='year', y='Net Cash Flow',
                             title="<b>Yearly Net Cash Flow</b>",
                             labels={'year':'Year', 'Net Cash Flow':'Cash Flow ($)'})
            fig_ncf.update_layout(template='plotly_white', bargap=0.2)
            st.plotly_chart(fig_ncf, use_container_width=True)
        with c2:
            fig_cum = px.line(display_df, x='year', y='Cumulative Cash Flow',
                              title="<b>Cumulative Cash Flow</b>", markers=True,
                              labels={'year':'Year', 'Cumulative Cash Flow':'Cash Flow ($)'})
            fig_cum.add_hline(y=0, line_dash="dash")
            fig_cum.update_layout(template='plotly_white')
            st.plotly_chart(fig_cum, use_container_width=True)

        # Yearly table (merge production if available)
        st.markdown("##### Yearly Cash Flow Table")
        final_table = display_df.copy()
        if not df_yearly.empty:
            cols = ['year', 'oil_stb', 'gas_mscf', 'water_stb', 'Revenue', 'Royalty', 'Taxes', 'OPEX']
            final_table = pd.merge(final_table, df_yearly[cols], on='year', how='left')

        need_cols = ['year','oil_stb','gas_mscf','water_stb','Revenue','Royalty','Taxes','OPEX','Net Cash Flow','Cumulative Cash Flow']
        for c in need_cols:
            if c not in final_table.columns:
                final_table[c] = 0
        final_table = final_table[need_cols].fillna(0)

        st.dataframe(
            final_table.style.format({
                'oil_stb': '{:,.0f}', 'gas_mscf': '{:,.0f}', 'water_stb': '{:,.0f}',
                'Revenue': '${:,.0f}', 'Royalty': '${:,.0f}', 'Taxes': '${:,.0f}',
                'OPEX': '${:,.0f}', 'Net Cash Flow': '${:,.0f}', 'Cumulative Cash Flow': '${:,.0f}'
            }),
            use_container_width=True
        )
# ======== EUR vs Lateral Length ========
elif selected_tab == "EUR vs Lateral Length":
    st.header("EUR vs Lateral Length Sensitivity")
    st.info("This module runs the fast analytical model multiple times to build a sensitivity of EUR to changes in lateral length.")

    # --- 1. Define Sensitivity Parameters ---
    st.markdown("#### 1. Define Sensitivity Range")
    c1, c2, c3 = st.columns(3)
    with c1:
        L_min = st.number_input("Min Lateral Length (ft)", 1000.0, 20000.0, 5000.0, 500.0)
    with c2:
        L_max = st.number_input("Max Lateral Length (ft)", 1000.0, 20000.0, 15000.0, 500.0)
    with c3:
        n_steps = st.number_input("Number of Steps", 2, 50, 10, 1)

    # --- 2. Run Sensitivity ---
    if st.button("🚀 Run Sensitivity Analysis", use_container_width=True, type="primary"):
        results = []
        base_state = state.copy()
        lat_lengths = np.linspace(L_min, L_max, int(n_steps))
        progress_bar = st.progress(0, "Starting sensitivity analysis...")

        for i, length in enumerate(lat_lengths):
            temp_state = base_state.copy()
            temp_state['L_ft'] = length
            
            # Run the fast solver
            sim_result = fallback_fast_solver(temp_state, np.random.default_rng(i))
            
            # Recalculate EURs authoritatively
            t = np.asarray(sim_result.get("t", []))
            qo = np.asarray(sim_result.get("qo", []))
            qg = np.asarray(sim_result.get("qg", []))
            eur_o = np.trapz(qo, t) / 1e6 if len(t) > 1 else 0.0
            eur_g = np.trapz(qg, t) / 1e6 if len(t) > 1 else 0.0

            results.append({
                "Lateral Length (ft)": length,
                "EUR Oil (MMBO)": eur_o,
                "EUR Gas (BCF)": eur_g
            })
            progress_bar.progress((i + 1) / n_steps, f"Running case {i+1}/{int(n_steps)}...")
        
        st.session_state.sensitivity_results = pd.DataFrame(results)
        progress_bar.empty()

    # --- 3. Display Results ---
    if 'sensitivity_results' in st.session_state:
        df_sens = st.session_state.sensitivity_results
        st.markdown("---")
        st.markdown("### Sensitivity Results")

        st.markdown("#### Figure 1: EUR vs. Lateral Length")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df_sens["Lateral Length (ft)"], y=df_sens["EUR Oil (MMBO)"], name="EUR Oil", mode='lines+markers', line=dict(color=COLOR_OIL)), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_sens["Lateral Length (ft)"], y=df_sens["EUR Gas (BCF)"], name="EUR Gas", mode='lines+markers', line=dict(color=COLOR_GAS)), secondary_y=True)
        fig.update_layout(template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(title_text="Lateral Length (ft)")
        fig.update_yaxes(title_text="EUR Oil (MMBO)", secondary_y=False)
        fig.update_yaxes(title_text="EUR Gas (BCF)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("About this chart"):
            st.markdown("This plot shows how the Estimated Ultimate Recovery (EUR) for both oil and gas is expected to change as the lateral length of the well increases. It helps in identifying the point of diminishing returns for drilling longer laterals.")

        st.markdown("#### Table 1: Sensitivity Data")
        st.dataframe(df_sens.style.format({
            "Lateral Length (ft)": "{:,.0f}",
            "EUR Oil (MMBO)": "{:.2f}",
            "EUR Gas (BCF)": "{:.2f}"
        }), use_container_width=True)
# ======== Field Match (CSV) ========
elif selected_tab == "Field Match (CSV)":
    """
    Manual history matching against CSV field data.
    - Upload CSV with columns: Day, and either/both Oil_Rate_STBpd, Gas_Rate_Mscfd.
    - Plots sim vs markers and offers a simple layout for quick checks.
    """
    st.header("Field Match (CSV)")
    c1, c2 = st.columns([3, 1])
    with c1:
        uploaded_file = st.file_uploader("Upload field production data (CSV)", type="csv")
        if uploaded_file:
            try:
                st.session_state.field_data_match = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
    with c2:
        st.write("")
        st.write("")
        if st.button("Load Demo Data", use_container_width=True):
            rng = np.random.default_rng(123)
            days     = np.arange(0, 731, 15)
            oil_rate = 950 * np.exp(-days / 400)  + rng.uniform(-25, 25, size=days.shape)
            gas_rate = 8000 * np.exp(-days / 500) + rng.uniform(-200, 200, size=days.shape)
            demo_df  = pd.DataFrame({"Day": days,
                                     "Gas_Rate_Mscfd": np.clip(gas_rate, 0, None),
                                     "Oil_Rate_STBpd": np.clip(oil_rate, 0, None)})
            st.session_state.field_data_match = demo_df
            st.success("Demo production data loaded successfully!")

    if 'field_data_match' in st.session_state:
        st.markdown("---")
        st.markdown("#### Loaded Production Data (first 5 rows)")
        st.dataframe(st.session_state.field_data_match.head(), use_container_width=True)

        if st.session_state.get("sim") is not None:
            sim_data   = st.session_state.sim
            field_data = st.session_state.field_data_match

            fig_match = go.Figure()
            if sim_data.get('t') is not None:
                if sim_data.get('qg') is not None:
                    fig_match.add_trace(go.Scatter(x=sim_data['t'], y=sim_data['qg'],
                                                   mode='lines', name='Simulated Gas',
                                                   line=dict(color="#d62728")))
                if sim_data.get('qo') is not None:
                    fig_match.add_trace(go.Scatter(x=sim_data['t'], y=sim_data['qo'],
                                                   mode='lines', name='Simulated Oil',
                                                   line=dict(color="#2ca02c"), yaxis="y2"))

            # Field markers if columns are present
            if {'Day', 'Gas_Rate_Mscfd'}.issubset(field_data.columns):
                fig_match.add_trace(go.Scatter(x=field_data['Day'], y=field_data['Gas_Rate_Mscfd'],
                                               mode='markers', name='Field Gas',
                                               marker=dict(color="#d62728", symbol='cross')))
            if {'Day', 'Oil_Rate_STBpd'}.issubset(field_data.columns):
                fig_match.add_trace(go.Scatter(x=field_data['Day'], y=field_data['Oil_Rate_STBpd'],
                                               mode='markers', name='Field Oil',
                                               marker=dict(color="#2ca02c", symbol='cross'), yaxis="y2"))

            layout_config = semi_log_layout("Field Match: Simulation vs. Actual", yaxis="Gas Rate (Mscf/d)")
            layout_config.update(
                yaxis=dict(title="Gas Rate (Mscf/d)"),
                yaxis2=dict(title="Oil Rate (STB/d)", overlaying="y", side="right", showgrid=False),
            )
            fig_match.update_layout(layout_config)
            st.plotly_chart(fig_match, use_container_width=True, theme="streamlit")
            with st.expander("Click for details"):
                st.markdown(
                    "This plot compares simulated production (solid lines) to historical data ('x' markers). "
                    "Tune sidebar parameters and re-run until the match is reasonable; then use the calibrated model for forecasting."
                )
        else:
            st.info("Demo/Field data loaded. Run a simulation on the 'Results' tab to view the comparison plot.")

# ======== Automated Match ========
elif selected_tab == "Automated Match":
    """
    Differential Evolution-based auto history match.
    """
    st.header("Automated History Matching")
    st.info("This module uses a genetic algorithm (Differential Evolution) to automatically find the best parameters to match historical data.")

    # --- 1. Load Data ---
    with st.expander("1. Load Historical Data", expanded=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            uploaded_file_match = st.file_uploader("Upload field production CSV", type="csv", key="auto_match_uploader")
            if uploaded_file_match:
                try:
                    st.session_state.field_data_auto_match = pd.read_csv(uploaded_file_match)
                    st.success("File loaded successfully.")
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
        with c2:
            st.write("") # Spacer
            st.write("") # Spacer
            if st.button("Load Demo Data", use_container_width=True):
                rng = np.random.default_rng(42)
                days = np.arange(0, 1096, 30)
                oil_rate = 1200 * np.exp(-days / 500) + rng.normal(0, 30, size=days.shape)
                gas_rate = 2200 * np.exp(-days / 600) + rng.normal(0, 80, size=days.shape)
                demo_df = pd.DataFrame({
                    "Day": days,
                    "Gas_Rate_Mscfd": np.clip(gas_rate, 0, None),
                    "Oil_Rate_STBpd": np.clip(oil_rate, 0, None)
                })
                st.session_state.field_data_auto_match = demo_df
                st.success("Demo production data loaded!")
        
        field_data = st.session_state.get("field_data_auto_match")
        if field_data is not None:
            st.markdown("#### Table 1: Loaded Production Data (first 5 rows)")
            st.dataframe(field_data.head())
            if not ({'Day', 'Oil_Rate_STBpd'}.issubset(field_data.columns) or
                    {'Day', 'Gas_Rate_Mscfd'}.issubset(field_data.columns)):
                st.error("CSV must contain 'Day' and at least one of 'Oil_Rate_STBpd' or 'Gas_Rate_Mscfd'.")
                field_data = None  # invalidate

    if field_data is not None:
        # --- 2. Select Parameters ---
        with st.expander("2. Select Parameters and Define Bounds", expanded=True):
            param_options = {
                'xf_ft': (100.0, 500.0),
                'hf_ft': (50.0, 300.0),
                'k_stdev': (0.0, 0.2),
                'pad_interf': (0.0, 0.8),
                'p_init_psi': (3000.0, 8000.0),
            }
            selected_params = st.multiselect(
                "Parameters to vary:", options=list(param_options.keys()),
                default=['xf_ft', 'k_stdev']
            )

            bounds, valid_bounds = {}, True
            if selected_params:
                cols = st.columns(len(selected_params))
                for i, param in enumerate(selected_params):
                    with cols[i]:
                        st.markdown(f"**{param}**")
                        min_val, max_val = st.slider(
                            "Range", param_options[param][0], param_options[param][1],
                            (param_options[param][0], param_options[param][1]),
                            key=f"range_{param}"
                        )
                        if min_val >= max_val:
                            st.error("Min must be less than Max.")
                            valid_bounds = False
                        bounds[param] = (min_val, max_val)

        # --- 3. Configure and Run ---
        with st.expander("3. Configure and Run Optimization", expanded=True):
            error_metric = st.selectbox("Error Metric to Minimize", ["RMSE (Oil)", "RMSE (Gas)", "RMSE (Combined)"], help="Choose which production stream(s) to prioritize for the match.")
            max_iter = st.slider("Max Iterations", 5, 50, 15)

            run_auto_match = st.button(
                "🚀 Run Automated Match", use_container_width=True, type="primary",
                disabled=not (valid_bounds and selected_params)
            )

            if run_auto_match:
                # Objective function remains the same
                def objective_function(params, param_names, base_state, field_df, metric):
                    temp = base_state.copy();
                    for name, value in zip(param_names, params): temp[name] = value
                    sim_result = fallback_fast_solver(temp, np.random.default_rng())
                    t_sim, qo_sim, qg_sim = sim_result['t'], sim_result['qo'], sim_result['qg']
                    t_field = field_df['Day'].values
                    f_qo = interp1d(t_sim, qo_sim, bounds_error=False, fill_value="extrapolate")
                    f_qg = interp1d(t_sim, qg_sim, bounds_error=False, fill_value="extrapolate")
                    qo_hat, qg_hat = f_qo(t_field), f_qg(t_field)
                    err_o = err_g = 0.0
                    if 'Oil_Rate_STBpd' in field_df.columns: err_o = float(np.sqrt(np.mean((qo_hat - field_df['Oil_Rate_STBpd'].values)**2)))
                    if 'Gas_Rate_Mscfd' in field_df.columns: err_g = float(np.sqrt(np.mean((qg_hat - field_df['Gas_Rate_Mscfd'].values)**2)))
                    if "Combined" in metric: return err_o + err_g
                    elif "Oil" in metric: return err_o
                    else: return err_g

                with st.spinner("Running optimization... This may take several minutes."):
                    param_names = list(bounds.keys())
                    bounds_list = [bounds[p] for p in param_names]
                    result = differential_evolution(
                        objective_function, bounds=bounds_list,
                        args=(param_names, state.copy(), field_data, error_metric),
                        maxiter=int(max_iter), disp=True
                    )
                    st.session_state.auto_match_result = result

        # --- 4. Display Results ---
        if 'auto_match_result' in st.session_state:
            st.markdown("---")
            st.header("Optimization Results")
            result = st.session_state.auto_match_result

            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Final Error (RMSE)", f"{result.fun:.2f}")
                st.markdown("##### Table 2: Best-Fit Parameters")
                best_params_df = pd.DataFrame({'Parameter': list(bounds.keys()), 'Value': result.x})
                st.table(best_params_df.style.format({'Value': '{:.2f}'}))

            with c2:
                best_state = state.copy()
                for name, value in zip(list(bounds.keys()), result.x): best_state[name] = value
                final_sim = fallback_fast_solver(best_state, np.random.default_rng())
                
                st.markdown("#### Figure 1: Final History Match")
                fig_match = make_subplots(specs=[[{"secondary_y": True}]])
                if 'Gas_Rate_Mscfd' in field_data.columns:
                    fig_match.add_trace(go.Scatter(x=field_data['Day'], y=field_data['Gas_Rate_Mscfd'], mode='markers', name='Field Gas', marker=dict(color=COLOR_GAS, symbol='cross')), secondary_y=False)
                    fig_match.add_trace(go.Scatter(x=final_sim['t'], y=final_sim['qg'], mode='lines', name='Best Match Gas', line=dict(color=COLOR_GAS, width=3)), secondary_y=False)
                if 'Oil_Rate_STBpd' in field_data.columns:
                    fig_match.add_trace(go.Scatter(x=field_data['Day'], y=field_data['Oil_Rate_STBpd'], mode='markers', name='Field Oil', marker=dict(color=COLOR_OIL, symbol='x')), secondary_y=True)
                    fig_match.add_trace(go.Scatter(x=final_sim['t'], y=final_sim['qo'], mode='lines', name='Best Match Oil', line=dict(color=COLOR_OIL, width=3)), secondary_y=True)

                fig_match.update_layout(template="plotly_white", xaxis_title="Time (days)")
                st.plotly_chart(fig_match, use_container_width=True)
                with st.expander("About this chart"):
                    st.markdown("This chart compares the historical field data (markers) against the production forecast generated using the best-fit parameters found by the optimization algorithm (solid lines).")# ======== Uncertainty & Monte Carlo ========
# ======== Uncertainty & Monte Carlo ========
elif selected_tab == "Uncertainty & Monte Carlo":
    st.header("Uncertainty & Monte Carlo Analysis")

    # --- Section 1: Probabilistic Forecast (Monte Carlo) ---
    st.subheader("1. Probabilistic Forecast (Monte Carlo)")
    st.info("This analysis runs many simulations with randomly sampled inputs to understand the overall range and probability of outcomes (e.g., P10, P50, P90).")
    
    with st.expander("Configure Monte Carlo Parameters"):
        p1, p2, p3 = st.columns(3)
        with p1:
            uc_k   = st.checkbox("k stdev", True)
            k_mean = st.slider("k_stdev Mean", 0.0, 0.2, state['k_stdev'], 0.01, key="mc_k_mean")
            k_std  = st.slider("k_stdev Stdev", 0.0, 0.1, 0.02, 0.005, key="mc_k_std")
        with p2:
            uc_xf  = st.checkbox("xf_ft", True)
            xf_mean = st.slider("xf_ft Mean (ft)", 100.0, 500.0, state['xf_ft'], 10.0, key="mc_xf_mean")
            xf_std  = st.slider("xf_ft Stdev (ft)", 0.0, 100.0, 30.0, 5.0, key="mc_xf_std")
        with p3:
            uc_int = st.checkbox("pad_interf", False)
            int_min = st.slider("Interference Min", 0.0, 0.8, state['pad_interf'], 0.01, key="mc_int_min")
            int_max = st.slider("Interference Max", 0.0, 0.8, 0.5, 0.01, key="mc_int_max")

        num_runs = st.number_input("Number of Monte Carlo runs", 10, 500, 50, 10)

    if st.button("Run Monte Carlo Simulation", key="run_mc"):
        qg_runs, qo_runs, eur_g, eur_o = [], [], [], []
        bar_mc = st.progress(0, text="Running Monte Carlo simulation...")
        base_state = state.copy()
        rng_mc = np.random.default_rng(st.session_state.rng_seed + 1)
        for i in range(int(num_runs)):
            temp = base_state.copy()
            if uc_k: temp['k_stdev'] = stats.truncnorm.rvs((0 - k_mean) / k_std, (0.2 - k_mean) / k_std, loc=k_mean, scale=k_std, random_state=rng_mc)
            if uc_xf: temp['xf_ft'] = stats.truncnorm.rvs((100 - xf_mean) / xf_std, (500 - xf_mean) / xf_std, loc=xf_mean, scale=xf_std, random_state=rng_mc)
            if uc_int: temp['pad_interf'] = stats.uniform.rvs(loc=int_min, scale=int_max - int_min, random_state=rng_mc)
            res = fallback_fast_solver(temp, rng_mc)
            qg_runs.append(res['qg']); qo_runs.append(res['qo'])
            eur_g.append(res['EUR_g_BCF']); eur_o.append(res['EUR_o_MMBO'])
            bar_mc.progress((i + 1) / int(num_runs), f"Run {i+1}/{int(num_runs)}")
        st.session_state.mc_results = {'t': res['t'], 'qg_runs': np.array(qg_runs), 'qo_runs': np.array(qo_runs), 'eur_g': np.array(eur_g), 'eur_o': np.array(eur_o)}
        bar_mc.empty()

    if 'mc_results' in st.session_state:
        mc = st.session_state.mc_results
        p10_g, p50_g, p90_g = np.percentile(mc['qg_runs'], [90, 50, 10], axis=0)
        p10_o, p50_o, p90_o = np.percentile(mc['qo_runs'], [90, 50, 10], axis=0)
        st.markdown("#### Figure 1: Probabilistic Forecasts")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure([go.Scatter(x=mc['t'], y=p90_g, fill=None, mode='lines', line_color='lightgrey', name='P10'), go.Scatter(x=mc['t'], y=p10_g, fill='tonexty', mode='lines', line_color='lightgrey', name='P90'), go.Scatter(x=mc['t'], y=p50_g, mode='lines', line_color='red', name='P50')])
            st.plotly_chart(fig.update_layout(**semi_log_layout("Gas Rate Forecast", yaxis="Gas Rate (Mscf/d)")), use_container_width=True)
        with c2:
            fig = go.Figure([go.Scatter(x=mc['t'], y=p90_o, fill=None, mode='lines', line_color='lightgreen', name='P10'), go.Scatter(x=mc['t'], y=p10_o, fill='tonexty', mode='lines', line_color='lightgreen', name='P90'), go.Scatter(x=mc['t'], y=p50_o, mode='lines', line_color='green', name='P50')])
            st.plotly_chart(fig.update_layout(**semi_log_layout("Oil Rate Forecast", yaxis="Oil Rate (STB/d)")), use_container_width=True)
        st.markdown("#### Figure 2: EUR Distributions")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.histogram(x=mc['eur_g'], nbins=30, labels={'x': 'Gas EUR (BCF)'}).update_layout(title="<b>Distribution of Gas EUR</b>", template="plotly_white"), use_container_width=True)
        with c2:
            st.plotly_chart(px.histogram(x=mc['eur_o'], nbins=30, labels={'x': 'Oil EUR (MMSTB)'}, color_discrete_sequence=['green']).update_layout(title="<b>Distribution of Oil EUR</b>", template="plotly_white"), use_container_width=True)

    st.markdown("---")

    # --- Section 2: Deterministic Sensitivity (Tornado Chart) ---
    st.subheader("2. Sensitivity Analysis (Tornado Chart)")
    st.info("This analysis varies one parameter at a time (from a low P90 to a high P10 value) to see which inputs have the biggest impact on the final result.")

    with st.expander("Configure Tornado Chart Parameters"):
        tornado_output = st.selectbox("Select Output for Tornado Chart", ["EUR Oil (MMBO)", "EUR Gas (BCF)"])
        
        params_to_test = {
            "Frac Half-Length (xf_ft)": (state['xf_ft'] * 0.8, state['xf_ft'], state['xf_ft'] * 1.2),
            "Initial Pressure (p_init_psi)": (state['p_init_psi'] * 0.9, state['p_init_psi'], state['p_init_psi'] * 1.1),
            "Pad Interference (pad_interf)": (0.0, state['pad_interf'], 0.6),
            "Permeability StDev (k_stdev)": (0.01, state['k_stdev'], 0.15)
        }
        
        tornado_params = {}
        st.markdown("###### Define Low (P90), Base (P50), and High (P10) values for each parameter:")
        
        # CORRECTED WIDGET IMPLEMENTATION
        for name, (default_p90, default_p50, default_p10) in params_to_test.items():
            key_name = name.split('(')[1].split(')')[0].strip()
            st.markdown(f"**{name}**")
            c1, c2, c3 = st.columns(3)
            with c1:
                p90_val = st.number_input("Low (P90)", value=float(default_p90), key=f"{key_name}_p90", format="%.2f")
            with c2:
                p50_val = st.number_input("Base (P50)", value=float(default_p50), key=f"{key_name}_p50", format="%.2f")
            with c3:
                p10_val = st.number_input("High (P10)", value=float(default_p10), key=f"{key_name}_p10", format="%.2f")
            tornado_params[key_name] = (p90_val, p50_val, p10_val)

    if st.button("🚀 Run Tornado Analysis", use_container_width=True):
        base_state = state.copy()
        tornado_results = []
        
        # 1. Run Base Case (all P50 values)
        for param, values in tornado_params.items():
            base_state[param] = values[1] # Set to P50
        base_sim = fallback_fast_solver(base_state, np.random.default_rng())
        base_eur = base_sim['EUR_o_MMBO'] if "Oil" in tornado_output else base_sim['EUR_g_BCF']

        # 2. Loop through each parameter for low and high cases
        progress_bar = st.progress(0, "Starting Tornado analysis...")
        total_runs = len(tornado_params) * 2
        run_count = 0
        
        for param, (low_val, base_val, high_val) in tornado_params.items():
            # Run Low Case
            temp_state = base_state.copy()
            temp_state[param] = low_val
            low_sim = fallback_fast_solver(temp_state, np.random.default_rng())
            low_eur = low_sim['EUR_o_MMBO'] if "Oil" in tornado_output else low_sim['EUR_g_BCF']
            run_count += 1
            progress_bar.progress(run_count / total_runs, f"Running Low Case for {param}...")

            # Run High Case
            temp_state[param] = high_val
            high_sim = fallback_fast_solver(temp_state, np.random.default_rng())
            high_eur = high_sim['EUR_o_MMBO'] if "Oil" in tornado_output else high_sim['EUR_g_BCF']
            run_count += 1
            progress_bar.progress(run_count / total_runs, f"Running High Case for {param}...")

            tornado_results.append({
                "Parameter": param,
                "Low_Value_EUR": low_eur,
                "High_Value_EUR": high_eur,
                "Swing": abs(high_eur - low_eur)
            })
        
        st.session_state.tornado_df = pd.DataFrame(tornado_results).sort_values(by="Swing", ascending=True)
        st.session_state.tornado_base_eur = base_eur
        progress_bar.empty()

    if 'tornado_df' in st.session_state:
        df_tornado = st.session_state.tornado_df
        base_eur = st.session_state.tornado_base_eur
        
        st.markdown("---")
        st.markdown(f"#### Figure 3: Tornado Chart for {tornado_output}")
        
        # Create the Tornado chart using Plotly
        fig_tornado = go.Figure()
        # Add bars for the difference from the base case
        fig_tornado.add_trace(go.Bar(
            y=df_tornado["Parameter"],
            x=df_tornado["High_Value_EUR"] - base_eur,
            base=base_eur,
            orientation='h',
            name='High Case (P10) Impact',
            marker_color='green'
        ))
        fig_tornado.add_trace(go.Bar(
            y=df_tornado["Parameter"],
            x=df_tornado["Low_Value_EUR"] - base_eur,
            base=base_eur,
            orientation='h',
            name='Low Case (P90) Impact',
            marker_color='red'
        ))
        
        fig_tornado.update_layout(
            barmode='relative',
            title_text=f"Sensitivity of {tornado_output}",
            xaxis_title=f"Change in {tornado_output}",
            yaxis_title="Parameter",
            template="plotly_white",
            height=400 + len(df_tornado) * 25,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        fig_tornado.add_vline(x=base_eur, line_width=2, line_dash="dash", line_color="black", annotation_text="Base Case")
        st.plotly_chart(fig_tornado, use_container_width=True)
        with st.expander("About this chart"):
            st.markdown(f"""
            This Tornado chart ranks the input parameters by their impact on the final **{tornado_output}**.
            - The **black dashed line** represents the Base Case EUR.
            - Each horizontal bar shows how much the EUR changes when a single parameter is varied from its Low to High value.
            - **Longer bars indicate a higher sensitivity**, meaning that uncertainty in that parameter has a larger effect on the forecast.
            """)
            # ======== Well Placement Optimization ========
elif selected_tab == "Well Placement Optimization":
    """
    Simple heuristic placement search using your proxy model.
    - Randomly proposes heel positions (x,y) subject to constraints.
    - Scores each by selected objective (Oil or Gas EUR).
    """
    st.header("Well Placement Optimization")

    st.markdown("#### 1. General Parameters")
    c1_opt, c2_opt, c3_opt = st.columns(3)
    with c1_opt:
        objective = st.selectbox("Objective Property", ["Maximize Oil EUR", "Maximize Gas EUR"], key="opt_objective")
    with c2_opt:
        iterations = st.number_input("Number of optimization steps", min_value=5, max_value=1000, value=100, step=10)
    with c3_opt:
        st.selectbox("Forbidden Zone", ["Numerical Faults"],
                     help="The optimizer will avoid placing wells near the fault defined in the sidebar.")

    st.markdown("#### 2. Well Parameters")
    c1_well, c2_well = st.columns(2)
    with c1_well:
        num_wells = st.number_input("Number of wells to place", min_value=1, max_value=1, value=1,
                                    disabled=True, help="Currently supports optimizing a single well location.")
    with c2_well:
        st.text_input("Well name prefix", "OptiWell", disabled=True)

    launch_opt = st.button("🚀 Launch Optimization", use_container_width=True, type="primary")
    if launch_opt:
        opt_results = []
        base_state = state.copy()
        rng_opt = np.random.default_rng(int(st.session_state.rng_seed))

        # Feasible domain checks
        reservoir_x_dim = base_state['nx'] * base_state['dx']
        x_max = reservoir_x_dim - base_state['L_ft']
        if x_max < 0:
            st.error(
                "Optimization Cannot Run: The well is too long for the reservoir.\n\n"
                f"- Reservoir X-Dimension (nx * dx): **{reservoir_x_dim:.0f} ft**\n"
                f"- Well Lateral Length (L_ft): **{base_state['L_ft']:.0f} ft**\n\n"
                "Please decrease 'Lateral length (ft)' or increase 'nx'/'dx' in the sidebar.",
                icon="⚠️",
            )
            st.stop()

        y_max = base_state['ny'] * base_state['dy']
        progress_bar = st.progress(0, text="Starting optimization...")

        for i in range(int(iterations)):
            # Random heel sampling with a validity check
            is_valid = False
            guard = 0
            while (not is_valid) and (guard < 10_000):
                x_heel_ft = rng_opt.uniform(0, x_max)
                y_heel_ft = rng_opt.uniform(50, y_max - 50)
                is_valid = is_heel_location_valid(x_heel_ft, y_heel_ft, base_state)
                guard += 1

            if not is_valid:
                st.error("Could not find a valid heel location. Check grid size, L_ft, and fault settings.")
                break

            temp_state = base_state.copy()
            # Example heuristic: modulate pad interference based on heel x-position
            x_norm = x_heel_ft / (base_state['nx'] * base_state['dx'])
            temp_state['pad_interf'] = 0.4 * x_norm

            result = fallback_fast_solver(temp_state, rng_opt)
            score  = result['EUR_o_MMBO'] if "Oil" in objective else result['EUR_g_BCF']

            opt_results.append({"Step": i + 1, "x_ft": float(x_heel_ft),
                                "y_ft": float(y_heel_ft), "Score": float(score)})

            progress_bar.progress((i + 1) / int(iterations), text=f"Step {i+1}/{int(iterations)} | Score: {score:.3f}")

        st.session_state.opt_results = pd.DataFrame(opt_results) if opt_results else pd.DataFrame()
        progress_bar.empty()

    if 'opt_results' in st.session_state and not st.session_state.opt_results.empty:
        df_results = st.session_state.opt_results
        best_run = df_results.loc[df_results['Score'].idxmax()]

        st.markdown("---")
        st.markdown("### Optimization Results")
        c1_res, c2_res = st.columns(2)
        with c1_res:
            st.markdown("##### Best Placement Found")
            score_unit = "MMBO" if "Oil" in st.session_state.get("opt_objective", "Maximize Oil EUR") else "BCF"
            st.metric(label=f"Best Score ({score_unit})", value=f"{best_run['Score']:.3f}")
            st.write(f"**Location (ft):** (x={best_run['x_ft']:.0f}, y={best_run['y_ft']:.0f})")
            st.write(f"Found at Step: {int(best_run['Step'])}")

        with c2_res:
            st.markdown("##### Optimization Steps Log")
            st.dataframe(df_results.sort_values("Score", ascending=False).head(10), height=210)

        # Simple 2D map with porosity background + tested points
        fig_opt = go.Figure()
        phi_map = get_k_slice(
            st.session_state.get('phi', np.zeros((state['nz'], state['ny'], state['nx']))),
            state['nz'] // 2
        )
        fig_opt.add_trace(go.Heatmap(
            z=phi_map, dx=state['dx'], dy=state['dy'],
            colorscale='viridis', colorbar=dict(title='Porosity')
        ))
        fig_opt.add_trace(go.Scatter(
            x=df_results['x_ft'], y=df_results['y_ft'], mode='markers',
            marker=dict(color=df_results['Score'], colorscale='Reds', showscale=True,
                        colorbar=dict(title='Score'), size=8, opacity=0.7),
            name='Tested Locations'
        ))
        fig_opt.add_trace(go.Scatter(
            x=[best_run['x_ft']], y=[best_run['y_ft']], mode='markers',
            marker=dict(color='cyan', size=16, symbol='star', line=dict(width=2, color='black')),
            name='Best Location'
        ))

        if state.get('use_fault'):
            fault_x = [state['fault_index'] * state['dx'], state['fault_index'] * state['dx']]
            fault_y = [0, state['ny'] * state['dy']]
            fig_opt.add_trace(go.Scatter(
                x=fault_x, y=fault_y, mode='lines',
                line=dict(color='white', width=4, dash='dash'),
                name='Fault'
            ))

        fig_opt.update_layout(
            title="<b>Well Placement Optimization Map</b>",
            xaxis_title="X position (ft)",
            yaxis_title="Y position (ft)",
            template="plotly_white",
            height=600
        )
        st.plotly_chart(fig_opt, use_container_width=True, theme="streamlit")

# ======== User’s Manual ========
elif selected_tab == "User’s Manual":
    """Embedded user guide for quick reference."""
    st.header("User’s Manual")
    st.markdown("---")
    st.markdown("""
    ### 1. Introduction
    Welcome to the **Full 3D Unconventional & Black-Oil Reservoir Simulator**. This application is designed for petroleum engineers to model, forecast, and optimize production from multi-stage fractured horizontal wells.

    ### 2. Quick Start Guide
    1. **Select a Play:** Choose a shale play from the sidebar (e.g., "Permian – Midland (Oil)").
    2. **Apply Preset:** Loads typical reservoir, fluid, and completion parameters for that play.
    3. **Generate Geology:** Use *Generate 3D property volumes* to create permeability/porosity grids.
    4. **Run Simulation:** Go to **Results** and click **Run simulation**.
    5. **Analyze:** Check EUR gauges, rate-time and cumulative plots.
    6. **Iterate:** Adjust parameters and re-run to see sensitivities.

    ### 3. Tabs
    - **Results:** Main outputs + sanity checks.
    - **Economics:** NPV/IRR/Payout using simulated profiles.
    - **Field/Automated Match:** Manual/auto history matching.
    - **3D & Slice Viewers:** Visualize 3D volumes and 2D slices.

    ### 4. Input Validation
    - Automated Match bounds are validated.
    - Results tab warns on out-of-family EUR/GOR for the selected play.
    """)

# ======== Solver & Profiling ========
elif selected_tab == "Solver & Profiling":
    """Display current numerical settings and last run time."""
    st.header("Solver & Profiling")
    st.info("This tab shows numerical solver settings and performance of the last run.")

    st.markdown("### Current Numerical Solver Settings")
    solver_settings = {
        "Parameter": [
            "Newton Tolerance", "Max Newton Iterations", "Threads", "Use OpenMP",
            "Use MKL", "Use PyAMG", "Use cuSPARSE"
        ],
        "Value": [
            f"{state['newton_tol']:.1e}", state['max_newton'],
            "Auto" if state['threads'] == 0 else state['threads'],
            "✅" if state['use_omp'] else "❌",
            "✅" if state['use_mkl'] else "❌",
            "✅" if state['use_pyamg'] else "❌",
            "✅" if state['use_cusparse'] else "❌",
        ],
    }
    st.table(pd.DataFrame(solver_settings))

    st.markdown("### Profiling")
    if st.session_state.get("sim") and 'runtime_s' in st.session_state.sim:
        st.metric(label="Last Simulation Runtime", value=f"{st.session_state.sim['runtime_s']:.2f} seconds")
    else:
        st.info("Run a simulation on the 'Results' tab to see performance profiling.")

# ======== DFN Viewer ========
elif selected_tab == "DFN Viewer":
    """
    Discrete Fracture Network (DFN) 3D line-segment viewer.
    Expects a list-like `dfn_segments` in session_state, where each item is [x1,y1,z1,x2,y2,z2].
    """
    st.header("DFN Viewer — 3D line segments")
    segs = st.session_state.get('dfn_segments')

    if segs is None or len(segs) == 0:
        st.info("No DFN loaded. Upload a CSV or use 'Generate DFN from stages' in the sidebar.")
    else:
        figd = go.Figure()
        for i, seg in enumerate(segs):
            figd.add_trace(go.Scatter3d(
                x=[seg[0], seg[3]], y=[seg[1], seg[4]], z=[seg[2], seg[5]],
                mode="lines",
                line=dict(width=4, color="red"),
                name="DFN" if i == 0 else None,
                showlegend=(i == 0)
            ))
        figd.update_layout(
            template="plotly_white",
            scene=dict(xaxis_title="x (ft)", yaxis_title="y (ft)", zaxis_title="z (ft)"),
            height=640,
            margin=dict(l=0, r=0, t=40, b=0),
            title="<b>DFN Segments</b>",
        )
        st.plotly_chart(figd, use_container_width=True, theme="streamlit")

        with st.expander("Click for details"):
            st.markdown("""
            This plot shows a 3D visualization of the Discrete Fracture Network (DFN) segments loaded into the simulator.
            - Each **red line** represents an individual natural fracture defined in the input file.
            - Use this for QC to verify locations/orientations inside the reservoir model.
            """)
