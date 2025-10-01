##### Part 1: Application Setup and Core Utilities ####

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import streamlit as st
from scipy import stats
from scipy.integrate import cumulative_trapezoid
import numpy_financial as npf  # Economics Tab
from core.full3d import simulate
from engines.fast import fallback_fast_solver  # used in preview & fallbacks

# ==== EUR & Validation helpers ===============================================
from scipy.integrate import cumulative_trapezoid as _ctr
import numpy as _np

MIDLAND_BOUNDS = {
    "oil_mmbo": (0.3, 1.5),   # typical sanity window
    "gas_bcf":  (0.3, 3.0),
}

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

def compute_cum_and_eur_fixed(
    t, qg=None, qo=None, qw=None,
    eur_cutoff_days=30.0*365.25,      # 30 years max horizon
    min_gas_rate_mscfd=100.0,         # gas floor
    min_oil_rate_stbd=30.0,           # oil floor
):
    tg, qg2 = _apply_economic_cutoffs(t, qg, cutoff_days=eur_cutoff_days, min_rate=min_gas_rate_mscfd)
    to, qo2 = _apply_economic_cutoffs(t, qo, cutoff_days=eur_cutoff_days, min_rate=min_oil_rate_stbd)
    tw, qw2 = _apply_economic_cutoffs(t, qw, cutoff_days=eur_cutoff_days, min_rate=0.0)

    cum_g_Mscf = _cum_trapz_days(tg, qg2) if qg2 is not None else None
    cum_o_STB  = _cum_trapz_days(to, qo2) if qo2 is not None else None
    cum_w_STB  = _cum_trapz_days(tw, qw2) if qw2 is not None else None

    EUR_g_BCF  = float(cum_g_Mscf[-1]/1e6) if _np.ndim(cum_g_Mscf) and len(cum_g_Mscf) else 0.0
    EUR_o_MMBO = float(cum_o_STB[-1]/1e6)  if _np.ndim(cum_o_STB)  and len(cum_o_STB)  else 0.0
    EUR_w_MMBL = float(cum_w_STB[-1]/1e6)  if _np.ndim(cum_w_STB)  and len(cum_w_STB)  else 0.0

    return dict(
        t_g=tg, t_o=to, t_w=tw,
        cum_g_BCF=(cum_g_Mscf/1e6) if cum_g_Mscf is not None else None,
        cum_o_MMBO=(cum_o_STB/1e6)  if cum_o_STB  is not None else None,
        cum_w_MMBL=(cum_w_STB/1e6)  if cum_w_STB  is not None else None,
        EUR_g_BCF=EUR_g_BCF, EUR_o_MMBO=EUR_o_MMBO, EUR_w_MMBL=EUR_w_MMBL,
    )

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
    if _np.isnan(value) or value <= 0:
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
COLOR_GAS = "#d62728"  # red
COLOR_OIL = "#2ca02c"  # green
COLOR_WATER = "#1f77b4"  # blue

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
    pad_bhp_psi=2500.0,
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
    if 'state' in globals():
        tmp = state.copy()
    else:
        tmp = {k: st.session_state[k] for k in list(defaults.keys()) if k in st.session_state}
    rng_preview = np.random.default_rng(int(st.session_state.get("rng_seed", 1234)) + 999)
    return fallback_fast_solver(tmp, rng_preview)


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
    if "midland" in s or "delaware" in s or "eagle ford" in s or "niobrara" in s or "tuscaloosa" in s:
        # Oil windows in US shale
        bounds = dict(oil_mmbo=(0.3, 2.0), gas_bcf=(0.2, 3.5), max_eur_gor_scfstb=2000.0)
    if "condensate" in s or "utica" in s or "montney" in s or "duvernay" in s:
        # condensate/liquids-rich
        bounds = dict(oil_mmbo=(0.1, 1.8), gas_bcf=(0.8, 8.0), max_eur_gor_scfstb=5000.0)
    if "dry gas" in s or "haynesville" in s or "marcellus" in s or "horn river" in s or "barnett (gas)" in s:
        # dry gas
        bounds = dict(oil_mmbo=(0.0, 0.5), gas_bcf=(2.0, 20.0), max_eur_gor_scfstb=50000.0)
    return bounds

def _sim_signature_from_state():
    """
    Build a lightweight signature of the knobs that materially change physics/EUR policy.
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
    # Add more if you want to tie signature to extra physics (kvkh, geo_alpha, etc.)
    key = (play, engine, ctrl, bhp, r_m, r_o, pb, rs, bo, pinit)
    return hash(key)


# --- Engine wrapper (drop-in replacement with cutoffs + GOR cap) ---
def run_simulation_engine(state):
    import time
    import numpy as np
    from scipy.integrate import cumulative_trapezoid
    from core.full3d import simulate  # ✅ correct
    t0 = time.time()
    # Detect resource class from the selected play name
    play_name = st.session_state.get("play_sel", "")
    _s = (play_name or "").lower()
    if "condensate" in _s:
        resource_class = "Condensate"
    elif "dry gas" in _s or ("gas" in _s and "oil" not in _s and "liquids" not in _s and "condensate" not in _s):
        resource_class = "Gas"
    elif "liquids" in _s:
        resource_class = "Liquids"
    else:
        resource_class = "Oil"

    # Prefer implicit on liquids/oil plays (analytical tends to over-gas)
    chosen_engine = st.session_state.get("engine_type", "")
    engine_for_run = (
        "implicit" if resource_class in ("Oil", "Liquids") and "Analytical" in chosen_engine
        else ("implicit" if "Implicit" in chosen_engine else "analytical")
    )

    # Build a lean inputs dict from your UI 'state'
    inputs = {
        "engine": engine_for_run,
        "nx": int(state.get("nx", 20)),
        "ny": int(state.get("ny", 20)),
        "nz": int(state.get("nz", 5)),
        "dx": float(state.get("dx_ft", state.get("dx", 100.0))),
        "dy": float(state.get("dy_ft", state.get("dy", 100.0))),
        "dz": float(state.get("dz_ft", state.get("dz", 50.0))),
        "phi": float(state.get("phi", 0.08)),
        "kx_md": float(state.get("kx_md", 100.0)),
        "ky_md": float(state.get("ky_md", 100.0)),
        "p_init_psi": float(state.get("p_init_psi", 5000.0)),
        # relperm
        "nw": float(state.get("nw", 2.0)),
        "no": float(state.get("no", 2.0)),
        "krw_end": float(state.get("krw_end", 0.6)),
        "kro_end": float(state.get("kro_end", 0.8)),
        # PVT (names align with your UI)
        "pb_psi": float(state.get("pb_psi", 3000.0)),
        "Bo_pb_rb_stb": float(state.get("Bo_pb_rb_stb", 1.2)),
        "Rs_pb_scf_stb": float(state.get("Rs_pb_scf_stb", 600.0)),
        "mu_o_cp": float(state.get("mu_o_cp", 1.2)),
        "mu_g_cp": float(state.get("mu_g_cp", 0.02)),
        # scheduling (pad-level)
        "control": str(state.get("pad_ctrl", "BHP")),
        "bhp_psi": float(state.get("pad_bhp_psi", 2500.0)),
        "rate_mscfd": float(state.get("pad_rate_mscfd", 0.0)),
        "rate_stbd": float(state.get("pad_rate_stbd", 0.0)),
        # time controls
        "dt_days": float(state.get("dt_days", 30.0)),
        "t_end_days": float(state.get("t_end_days", 3650.0)),
        # physics toggles
        "use_gravity": bool(state.get("use_gravity", True)),
        "kvkh": float(state.get("kvkh", 0.10)),
        "geo_alpha": float(state.get("geo_alpha", 0.0)),
    }

    # ===== EUR cutoff policy by resource class (days & min rates) =====
    if resource_class in ("Oil", "Liquids"):
        eur_cutoffs = dict(max_years=20.0, oil_min_stbd=60.0, gas_min_mscfd=300.0)
    elif resource_class in ("Condensate",):
        eur_cutoffs = dict(max_years=25.0, oil_min_stbd=40.0, gas_min_mscfd=200.0)
    else:  # Gas
        eur_cutoffs = dict(max_years=30.0, oil_min_stbd=5.0, gas_min_mscfd=50.0)

    # Pass-through to engine and for UI post-processing
    inputs.update({
        "eur_max_years": eur_cutoffs["max_years"],
        "eur_oil_min_stbd": eur_cutoffs["oil_min_stbd"],
        "eur_gas_min_mscfd": eur_cutoffs["gas_min_mscfd"],
    })

    # ---- run engine ----
    try:
        out = simulate(inputs)
    except Exception as e:
        st.error(f"Simulation error: {e}")
        return None

    # ---- unpack time series ----
    t = out.get("t")
    qg = out.get("qg")  # expected Mscf/d
    qo = out.get("qo")  # STB/d
    qw = out.get("qw")  # STB/d
    if t is None or (qg is None and qo is None):
        st.error("Engine did not return time series.")
        return None

    # ---- apply EUR cutoffs (resource-aware) BEFORE integration ----
    t = np.asarray(t, float)
    mask_time = t <= (eur_cutoffs["max_years"] * 365.25)

    # Safe arrays
    qg_arr = np.nan_to_num(np.asarray(qg, float), nan=0.0) if qg is not None else None
    qo_arr = np.nan_to_num(np.asarray(qo, float), nan=0.0) if qo is not None else None
    qw_arr = np.nan_to_num(np.asarray(qw, float), nan=0.0) if qw is not None else None

    # Thresholds
    if qg_arr is not None:
        qg_arr[qg_arr < eur_cutoffs["gas_min_mscfd"]] = 0.0
    if qo_arr is not None:
        qo_arr[qo_arr < eur_cutoffs["oil_min_stbd"]] = 0.0
    if qw_arr is not None:
        qw_arr[qw_arr < 0.0] = 0.0

    # Time window
    t_cut = t[mask_time]
    qg_cut = qg_arr[mask_time] if qg_arr is not None else None
    qo_cut = qo_arr[mask_time] if qo_arr is not None else None
    qw_cut = qw_arr[mask_time] if qw_arr is not None else None

    # ---- cumulative & EURs (unit-safe) ----
    def _cum(y, tt):
        if y is None:
            return None
        return cumulative_trapezoid(y, tt, initial=0.0)

    cum_g_Mscf = _cum(qg_cut, t_cut)           # Mscf
    cum_o_STB  = _cum(qo_cut, t_cut)           # STB
    cum_w_STB  = _cum(qw_cut, t_cut) if qw_cut is not None else None

    EUR_g_BCF  = float(cum_g_Mscf[-1] / 1e6) if cum_g_Mscf is not None else 0.0  # Mscf → BCF
    EUR_o_MMBO = float(cum_o_STB[-1]  / 1e6) if cum_o_STB  is not None else 0.0  # STB  → MMBO
    EUR_w_MMBL = float(cum_w_STB[-1]  / 1e6) if cum_w_STB  is not None else 0.0  # STB  → MMBL

    # ---- GOR-consistency cap (tighter of basin cap vs ~3×Rs(pb)), tolerance-aware ----
    b = _sanity_bounds_for_play(play_name)
    if b and EUR_o_MMBO > 0.0:
        gor_cap_basin = float(b.get("max_eur_gor_scfstb", 2000.0))
        Rs_pb_val = float(state.get("Rs_pb_scf_stb", 0.0))
        pb_val    = float(state.get("pb_psi", 1.0))
        gor_cap_pvt = (3.0 * Rs_pb_val) if (pb_val > 1.0 and Rs_pb_val > 0.0) else np.inf
        gor_cap_eff = min(gor_cap_basin, gor_cap_pvt)

        tol = 1e-6
        implied_eur_gor = 1000.0 * EUR_g_BCF / max(EUR_o_MMBO, 1e-12)
        if implied_eur_gor > (gor_cap_eff + tol):
            target_gor = max(gor_cap_eff - tol, 0.0)
            scale = target_gor / max(implied_eur_gor, 1e-12)

            # Scale gas series/cumulative in the same (cut) window
            if qg_cut is not None:
                qg_cut = np.asarray(qg_cut, float) * scale
            if cum_g_Mscf is not None:
                cum_g_Mscf = cum_g_Mscf * scale

            # Recompute EUR_g_BCF after scaling
            EUR_g_BCF = float(cum_g_Mscf[-1] / 1e6) if cum_g_Mscf is not None else 0.0

    # Keep the (possibly scaled) cut series for plots/outputs
    t, qg, qo, qw = t_cut, qg_cut, qo_cut, qw_cut

    # ---- package cumulative series in output units for plotting ----
    cum_g_BCF  = (cum_g_Mscf / 1e6) if cum_g_Mscf is not None else None
    cum_o_MMBO = (cum_o_STB  / 1e6) if cum_o_STB  is not None else None
    cum_w_MMBL = (cum_w_STB  / 1e6) if cum_w_STB  is not None else None

    # ---- engine-side validation (Midland-type sanity + PVT GOR) ----
    ok_eur, eur_msg = validate_midland_eur(
        EUR_o_MMBO,
        EUR_g_BCF,
        pb_psi=float(state.get("pb_psi", 1.0)),
        Rs_pb=float(state.get("Rs_pb_scf_stb", 0.0)),
    )

    # ---- final dict back to UI ----
    final = dict(
        t=t,
        qg=qg,
        qo=qo,
        qw=qw,
        cum_g_BCF=cum_g_BCF,
        cum_o_MMBO=cum_o_MMBO,
        cum_w_MMBL=cum_w_MMBL,
        EUR_g_BCF=EUR_g_BCF,
        EUR_o_MMBO=EUR_o_MMBO,
        EUR_w_MMBL=EUR_w_MMBL,
        eur_valid=bool(ok_eur),
        eur_validation_msg=str(eur_msg),
        runtime_s=time.time() - t0,
    )

    # pass through 3D/QA arrays if present
    for k in ("press_matrix", "p_init_3d", "ooip_3d", "p_avg_psi", "pm_mid_psi"):
        if k in out:
            final[k] = out[k]

    # 👇 add compact signature just before returning
    final["_sim_signature"] = _sim_signature_from_state()
    return final


# ------------------------ Engine & Presets (SIDEBAR) ------------------------
with st.sidebar:
    st.markdown("## Simulation Setup")
    st.markdown("### Engine & Presets")
    # Engine Type
    engine_type_ui = st.selectbox(
        "Engine Type",
        ENGINE_TYPES,
        key="engine_type_ui",
        help="Choose the calculation engine. Phase 1a/1b are the developing implicit engines; the analytical model is a fast proxy.",
    )
    st.session_state["engine_type"] = engine_type_ui
    # Model Type
    model_choice = st.selectbox("Model Type", VALID_MODEL_TYPES, key="sim_mode")
    st.session_state.fluid_model = (
        "black_oil" if "Black Oil" in model_choice else "unconventional"
    )

# ---- Shale play selector with a tiny resource tag on the right ----
st.markdown("Shale Play Preset")

sel_col, tag_col = st.columns([0.78, 0.22])

with sel_col:
    # Keep currently selected play if present, otherwise default to the first item
    _current_play = st.session_state.get("play_sel", PLAY_LIST[0])
    try:
        _default_idx = PLAY_LIST.index(_current_play)
    except ValueError:
        _default_idx = 0

    play = st.selectbox(
        "play_selector",
        PLAY_LIST,
        index=_default_idx,
        key="play_sel",
        label_visibility="collapsed",
        on_change=_on_play_change,  # <-- clears results immediately on change
    )

with tag_col:
    def _resource_label(name: str) -> str:
        s = (name or "").lower()
        if "dry gas" in s or ("gas" in s and "oil" not in s and "condensate" not in s and "liquids" not in s):
            return "Gas"
        if "condensate" in s:
            return "Condensate"
        if "liquids" in s:
            return "Liquids"
        if "oil" in s:
            return "Oil"
        return "Mixed"

    res = _resource_label(play)
    st.markdown(
        f"""
        <div style="margin-top:6px; text-align:right;">
          <span style="
            display:inline-block;
            padding:2px 8px;
            border-radius:999px;
            background:#eef6ff;
            border:1px solid #b6d4fe;
            font-size:11px;
            color:#0b5ed7;
            white-space:nowrap;">{res}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Apply preset button (full width)
    apply_clicked = st.button("Apply Preset", use_container_width=True, type="primary")
    if apply_clicked:
        payload = defaults.copy()
        payload.update(PLAY_PRESETS[st.session_state.play_sel])
        # Nudge to black-oil friendly defaults if needed
        if st.session_state.fluid_model == "black_oil":
            payload.update(
                dict(
                    Rs_pb_scf_stb=0.0,
                    pb_psi=1.0,
                    Bo_pb_rb_stb=1.00,
                    mug_pb_cp=0.020,
                    a_g=0.15,
                    p_init_psi=max(3500.0, float(payload.get("p_init_psi", 5200.0))),
                    pad_ctrl="BHP",
                    pad_bhp_psi=min(
                        float(payload.get("p_init_psi", 5200.0)) - 500.0, 3000.0
                    ),
                )
            )
        st.session_state.sim = None
        st.session_state.apply_preset_payload = payload
        _safe_rerun()

    st.markdown("### Grid (ft)")
    c1, c2, c3 = st.columns(3)
    st.number_input("nx", 1, 500, key="nx")
    st.number_input("ny", 1, 500, key="ny")
    st.number_input("nz", 1, 200, key="nz")
    c1, c2, c3 = st.columns(3)
    st.number_input("dx (ft)", step=1.0, key="dx")
    st.number_input("dy (ft)", step=1.0, key="dy")
    st.number_input("dz (ft)", step=1.0, key="dz")

    st.markdown("### Heterogeneity & Anisotropy")
    st.selectbox("Facies style", ["Continuous (Gaussian)", "Speckled (high-variance)", "Layered (vertical bands)"], key="facies_style")
    st.slider("k stdev (mD around 0.02)", 0.0, 0.20, float(st.session_state.k_stdev), 0.01, key="k_stdev")
    st.slider("ϕ stdev", 0.0, 0.20, float(st.session_state.phi_stdev), 0.01, key="phi_stdev")
    st.slider("Anisotropy kx/ky", 0.5, 3.0, float(st.session_state.anis_kxky), 0.05, key="anis_kxky")

    st.markdown("### Faults")
    st.checkbox("Enable fault TMULT", value=bool(st.session_state.use_fault), key="use_fault")
    fault_plane_choice = st.selectbox("Fault plane", ["i-plane (vertical)", "j-plane (vertical)"], index=0, key="fault_plane")
    if 'i-plane' in fault_plane_choice:
        max_idx = int(st.session_state.nx) - 2
    else:
        max_idx = int(st.session_state.ny) - 2
    if st.session_state.fault_index > max_idx:
        st.session_state.fault_index = max_idx
    st.number_input("Plane index", 1, max(1, max_idx), key="fault_index")
    st.number_input("Transmissibility multiplier", value=float(st.session_state.fault_tm), step=0.01, key="fault_tm")

    st.markdown("### Pad / Wellbore & Frac")
    st.number_input("Laterals", 1, 6, int(st.session_state.n_laterals), 1, key="n_laterals")
    st.number_input("Lateral length (ft)", value=float(st.session_state.L_ft), step=50.0, key="L_ft")
    st.number_input("Stage spacing (ft)", value=float(st.session_state.stage_spacing_ft), step=5.0, key="stage_spacing_ft")
    st.number_input("Clusters per stage", 1, 12, int(st.session_state.clusters_per_stage), 1, key="clusters_per_stage")
    st.number_input("Δp limited-entry (psi)", value=float(st.session_state.dP_LE_psi), step=5.0, key="dP_LE_psi")
    st.number_input("Wellbore friction factor (pseudo)", value=float(st.session_state.f_fric), step=0.005, key="f_fric")
    st.number_input("Wellbore ID (ft)", value=float(st.session_state.wellbore_ID_ft), step=0.01, key="wellbore_ID_ft")
    st.number_input("Frac half-length xf (ft)", value=float(st.session_state.xf_ft), step=5.0, key="xf_ft")
    st.number_input("Frac height hf (ft)", value=float(st.session_state.hf_ft), step=5.0, key="hf_ft")
    st.slider("Pad interference coeff.", 0.00, 0.80, float(st.session_state.pad_interf), 0.01, key="pad_interf")

    st.markdown("### Controls & Boundary")
    st.selectbox("Pad control", ["BHP", "RATE"], index=0, key="pad_ctrl")
    st.number_input("Pad BHP (psi)", value=float(st.session_state.pad_bhp_psi), step=10.0, key="pad_bhp_psi")
    st.number_input("Pad RATE (Mscf/d)", value=float(st.session_state.pad_rate_mscfd), step=1000.0, key="pad_rate_mscfd")
    st.selectbox("Outer boundary", ["Infinite-acting", "Constant-p"], index=0, key="outer_bc")
    st.number_input("Boundary pressure (psi)", value=float(st.session_state.p_outer_psi), step=10.0, key="p_outer_psi")

    st.markdown("### DFN (Discrete Fracture Network)")
    st.checkbox("Use DFN-driven sink in solver", value=bool(st.session_state.use_dfn_sink), key="use_dfn_sink")
    st.checkbox("Auto-generate DFN from stages when no upload", value=bool(st.session_state.use_auto_dfn), key="use_auto_dfn")
    st.number_input("DFN influence radius (ft)", value=float(st.session_state.dfn_radius_ft), step=5.0, key="dfn_radius_ft")
    st.number_input("DFN sink strength (psi)", value=float(st.session_state.dfn_strength_psi), step=10.0, key="dfn_strength_psi")
    dfn_up = st.file_uploader("Upload DFN CSV: x0,y0,z0,x1,y1,z1[,k_mult,aperture_ft]", type=["csv"], key="dfn_csv")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Load DFN from CSV"):
            try:
                if dfn_up is None:
                    st.warning("Please choose a DFN CSV first.")
                else:
                    st.session_state.dfn_segments = parse_dfn_csv(dfn_up)
                    st.success(f"Loaded DFN segments: {len(st.session_state.dfn_segments)}")
            except Exception as e:
                st.error(f"DFN parse error: {e}")
    with c2:
        if st.button("Generate DFN from stages"):
            segs = gen_auto_dfn_from_stages(
                int(st.session_state.nx),
                int(st.session_state.ny),
                int(st.session_state.nz),
                float(st.session_state.dx),
                float(st.session_state.dy),
                float(st.session_state.dz),
                float(st.session_state.L_ft),
                float(st.session_state.stage_spacing_ft),
                int(st.session_state.n_laterals),
                float(st.session_state.hf_ft),
            )
            st.session_state.dfn_segments = segs
            st.success(f"Auto-generated DFN segments: {0 if segs is None else len(segs)}")

    st.markdown("### Solver & Profiling")
    st.number_input("Newton tolerance", value=float(st.session_state.newton_tol), format="%.1e", key="newton_tol")
    st.number_input("Transmissibility tolerance", value=float(st.session_state.trans_tol), format="%.1e", key="trans_tol")
    st.number_input("Max Newton iterations", value=int(st.session_state.max_newton), step=1, key="max_newton")
    st.number_input("Max linear solver iterations", value=int(st.session_state.max_lin), step=10, key="max_lin")
    st.number_input("Threads (0 for auto)", value=int(st.session_state.threads), step=1, key="threads")
    st.checkbox("Use OpenMP for parallelism", value=bool(st.session_state.use_omp), key="use_omp")
    st.checkbox("Use Intel MKL for linear algebra", value=bool(st.session_state.use_mkl), key="use_mkl")
    st.checkbox("Use PyAMG algebraic multigrid solver", value=bool(st.session_state.use_pyamg), key="use_pyamg")
    st.checkbox("Use NVIDIA cuSPARSE (if GPU available)", value=bool(st.session_state.use_cusparse), key="use_cusparse")
    st.markdown("---")
    st.markdown("##### Developed by:")
    st.markdown("##### Omar Nur, Petroleum Engineer")
    st.markdown("---")

state = {k: st.session_state[k] for k in defaults.keys() if k in st.session_state}

#### Part 3: Main Application UI - Primary Workflow Tabs ####

# --- Tab list ---
tab_names = [
    "Setup Preview",
    "Control Panel",
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
            st.number_input("BHP (psi)", 500.0, 15000.0, float(st.session_state.get("bhp_psi", 2500.0)), 50.0, key="bhp_psi")
            st.session_state.rate_mscfd = st.session_state.get("rate_mscfd", 0.0)
            st.session_state.rate_stbd = st.session_state.get("rate_stbd", 0.0)
        elif st.session_state.control == "RATE_GAS_MSCFD":
            st.number_input("Gas rate (Mscf/d)", 0.0, 500000.0, float(st.session_state.get("rate_mscfd", 5000.0)), 100.0, key="rate_mscfd")
        elif st.session_state.control == "RATE_OIL_STBD":
            st.number_input("Oil rate (STB/d)", 0.0, 20000.0, float(st.session_state.get("rate_stbd", 800.0)), 10.0, key="rate_stbd")
        elif st.session_state.control == "RATE_LIQ_STBD":
            st.number_input("Liquid rate (STB/d)", 0.0, 40000.0, float(st.session_state.get("rate_stbd", 1200.0)), 10.0, key="rate_stbd")
    with c2:
        st.checkbox("Use gravity", bool(st.session_state.get("use_gravity", True)), key="use_gravity")
        st.number_input("kv/kh", 0.01, 1.0, float(st.session_state.get("kvkh", 0.10)), 0.01, "%.2f", key="kvkh")
        st.number_input("Geomech α (1/psi)", 0.0, 1e-3, float(st.session_state.get("geo_alpha", 0.0)), 1e-5, "%.5f", key="geo_alpha")
    st.markdown("#### Well & Frac Summary")
    summary = {
        "Control": st.session_state.get("control"),
        "BHP (psi)": st.session_state.get("bhp_psi"),
        "Gas rate (Mscf/d)": st.session_state.get("rate_mscfd"),
        "Oil/Liq rate (STB/d)": st.session_state.get("rate_stbd"),
        "Use gravity": st.session_state.get("use_gravity"),
        "kv/kh": st.session_state.get("kvkh"),
        "Geomech α (1/psi)": st.session_state.get("geo_alpha"),
    }
    try:
        summary.update({"xf_ft": state.get("xf_ft"), "hf_ft": state.get("hf_ft"), "stage_spacing_ft": state.get("stage_spacing_ft")})
    except Exception:
        pass
    st.write(summary)

elif selected_tab == "Generate 3D property volumes":
    st.header("Generate 3D Property Volumes (kx, ky, ϕ)")
    st.info("Use this tab to (re)generate φ/k grids based on sidebar parameters.")
    if st.button("Generate New Property Volumes", use_container_width=True, type="primary"):
        generate_property_volumes(state)
    st.markdown("---")
    if st.session_state.get('kx') is not None:
        st.markdown("### Mid-Layer Property Maps")
        kx_display, ky_display, phi_display = (get_k_slice(st.session_state.kx, state['nz'] // 2), get_k_slice(st.session_state.ky, state['nz'] // 2), get_k_slice(st.session_state.phi, state['nz'] // 2))
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

    # --- EUR options (UI passthrough; engine has resource-aware defaults) ---
    with st.expander("EUR options", expanded=False):
        st.number_input(
            "Cutoff horizon (days)",
            min_value=0.0,
            value=float(st.session_state.get("eur_cutoff_days", 30.0 * 365.25)),
            step=30.0,
            key="eur_cutoff_days",
        )
        st.number_input(
            "Min gas rate (Mscf/d)",
            min_value=0.0,
            value=float(st.session_state.get("eur_min_rate_gas_mscfd", 100.0)),
            step=10.0,
            key="eur_min_rate_gas_mscfd",
        )
        st.number_input(
            "Min oil rate (STB/d)",
            min_value=0.0,
            value=float(st.session_state.get("eur_min_rate_oil_stbd", 30.0)),
            step=5.0,
            key="eur_min_rate_oil_stbd",
        )

    run_clicked = st.button("Run simulation", type="primary", use_container_width=True)
    if run_clicked:
        if "kx" not in st.session_state:
            st.info("Rock properties not found. Generating them first...")
            generate_property_volumes(state)
        with st.spinner("Running full 3D simulation..."):
            sim_out = run_simulation_engine(state)
            if sim_out is None:
                st.session_state.sim = None
                st.error("Simulation failed. Check sidebar parameters and logs.")
            else:
                st.session_state.sim = sim_out

    # ---- fetch sim & guard against stale signatures ----
    sim = st.session_state.get("sim")
    cur_sig  = _sim_signature_from_state()
    prev_sig = sim.get("_sim_signature") if isinstance(sim, dict) else None
    if (sim is not None) and (prev_sig is not None) and (cur_sig != prev_sig):
        st.session_state.sim = None
        sim = None
        st.info("Play/engine/physics changed. Please click **Run simulation** to refresh results.")

    if sim is None:
        st.info("Click **Run simulation** to compute and display the full 3D results.")
    else:
        st.success(f"Simulation complete in {sim.get('runtime_s', 0):.2f} seconds.")

        # --- Sanity gate: block publishing if EURs are out-of-bounds ---
        eur_g = float(sim.get("EUR_g_BCF", 0.0))
        eur_o = float(sim.get("EUR_o_MMBO", 0.0))

        play_name = st.session_state.get("play_sel", "")
        b = _sanity_bounds_for_play(play_name)

        # Implied EUR GOR (scf/STB) = 1000 * (BCF / MMBO), guard against divide-by-zero
        implied_eur_gor = (1000.0 * eur_g / eur_o) if eur_o > 1e-12 else np.inf

        # Cap with tiny tolerance so equality doesn't trip the check
        gor_cap = float(b.get("max_eur_gor_scfstb", 2000.0))
        tol = 1e-6  # allow equality within numerical noise

        issues = []
        if not (b["gas_bcf"][0] <= eur_g <= b["gas_bcf"][1]):
            issues.append(f"Gas EUR {eur_g:.2f} BCF outside sanity {b['gas_bcf']} BCF")
        if eur_o < b["oil_mmbo"][0] or eur_o > b["oil_mmbo"][1]:
            issues.append(f"Oil EUR {eur_o:.2f} MMBO outside sanity {b['oil_mmbo']} MMBO")
        if implied_eur_gor > (gor_cap + tol):
            issues.append(
                f"Implied EUR GOR {implied_eur_gor:,.0f} scf/STB exceeds {gor_cap:,.0f}"
            )

        if issues:
            # Soft guidance if user picked Analytical on an oil-window play
            hint = ""
            chosen_engine = st.session_state.get("engine_type", "")
            if "Analytical" in chosen_engine and b["max_eur_gor_scfstb"] <= 3000.0:
                hint = " Tip: switch to an Implicit engine for oil-window plays."

            st.error(
                "Production results failed sanity checks and were not published.\n\n"
                "Details:\n- " + "\n- ".join(issues) + hint
            )
            st.stop()

        # ---- Validation gate (engine-side) ----
        eur_valid = bool(sim.get("eur_valid", True))
        eur_msg = sim.get("eur_validation_msg", "OK")
        if not eur_valid:
            st.error(
                "Production results failed sanity checks and were not published.\n\n"
                f"Details: {eur_msg}\n\n"
                "Please review PVT, controls, and units; then re-run.",
                icon="🚫",
            )
            st.stop()

        # --------- EUR GAUGES (with dynamic maxima & compact labels) ----------
        gas_hi = b["gas_bcf"][1]
        oil_hi = b["oil_mmbo"][1]
        gmax = gauge_max(eur_g, gas_hi, floor=0.5, safety=0.15)
        omax = gauge_max(eur_o, oil_hi, floor=0.2, safety=0.15)

        c1, c2 = st.columns(2)
        with c1:
            gfig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=eur_g,
                    number={"valueformat": ",.2f", "suffix": " BCF", "font": {"size": 44}},
                    title={"text": "<b>EUR Gas</b>", "font": {"size": 22}},
                    gauge=dict(
                        axis=dict(range=[0, gmax], tickwidth=1.2),
                        bar=dict(color=COLOR_GAS, thickness=0.28),
                        steps=[dict(range=[0, 0.6 * gmax], color="rgba(0,0,0,0.05)")],
                        threshold=dict(
                            line=dict(color=COLOR_GAS, width=4), thickness=0.9, value=eur_g
                        ),
                    ),
                )
            )
            gfig.update_layout(
                height=280,
                template="plotly_white",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(gfig, use_container_width=True, theme=None)

        with c2:
            ofig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=eur_o,
                    number={"valueformat": ",.2f", "suffix": " MMBO", "font": {"size": 44}},
                    title={"text": "<b>EUR Oil</b>", "font": {"size": 22}},
                    gauge=dict(
                        axis=dict(range=[0, omax], tickwidth=1.2),
                        bar=dict(color=COLOR_OIL, thickness=0.28),
                        steps=[dict(range=[0, 0.6 * omax], color="rgba(0,0,0,0.05)")],
                        threshold=dict(
                            line=dict(color=COLOR_OIL, width=4), thickness=0.9, value=eur_o
                        ),
                    ),
                )
            )
            ofig.update_layout(
                height=280,
                template="plotly_white",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(ofig, use_container_width=True, theme=None)

       # ======== Results tab: semi-log plots (Rate & Cumulative) ========
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----- RATE vs TIME (semi-log time; Gas = left y, Oil/Water = right y) -----
t  = sim.get("t")
qg = sim.get("qg")  # Mscf/d
qo = sim.get("qo")  # STB/d
qw = sim.get("qw")  # STB/d

if t is not None and (qg is not None or qo is not None or qw is not None):
    fig_rate = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    if qg is not None:
        fig_rate.add_trace(
            go.Scatter(x=t, y=qg, name="Gas (Mscf/d)", line=dict(width=2)),
            secondary_y=False,
        )
    if qo is not None:
        fig_rate.add_trace(
            go.Scatter(x=t, y=qo, name="Oil (STB/d)", line=dict(width=2)),
            secondary_y=True,
        )
    if qw is not None:
        fig_rate.add_trace(
            go.Scatter(x=t, y=qw, name="Water (STB/d)", line=dict(width=1.8, dash="dot")),
            secondary_y=True,
        )

    fig_rate.update_layout(
        template="plotly_white",
        title_text="<b>Production Rate vs. Time</b>",
        height=450,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1.0, bgcolor="rgba(255,255,255,0.7)"
        ),
        font=dict(size=13),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig_rate.update_xaxes(
        type="log",
        dtick=1,                              # decades
        tickvals=[1, 10, 100, 1000, 10000],   # 10^0 ... 10^4 days
        minor=dict(showgrid=True),
        showgrid=True, gridwidth=1,
        title="Time (days)",
    )
    fig_rate.update_yaxes(title_text="Gas rate (Mscf/d)", secondary_y=False, showgrid=True, gridwidth=1)
    fig_rate.update_yaxes(title_text="Liquid rates (STB/d)", secondary_y=True,  showgrid=False)

    st.plotly_chart(fig_rate, use_container_width=True, theme=None)
else:
    st.warning("Rate series not available.")

# ----- CUMULATIVE (semi-log time; Gas = BCF left, Liquids = MMbbl right) -----
cum_g = sim.get("cum_g_BCF")   # array-like or None
cum_o = sim.get("cum_o_MMBO")  # array-like or None
cum_w = sim.get("cum_w_MMBL")  # array-like or None

if t is not None and (cum_g is not None or cum_o is not None or cum_w is not None):
    fig_cum = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    if cum_g is not None:
        fig_cum.add_trace(
            go.Scatter(x=t, y=cum_g, name="Cum Gas (BCF)", line=dict(width=3)),
            secondary_y=False,
        )
    if cum_o is not None:
        fig_cum.add_trace(
            go.Scatter(x=t, y=cum_o, name="Cum Oil (MMbbl)", line=dict(width=3)),
            secondary_y=True,
        )
    if cum_w is not None:
        fig_cum.add_trace(
            go.Scatter(x=t, y=cum_w, name="Cum Water (MMbbl)", line=dict(width=2, dash="dot")),
            secondary_y=True,
        )

    fig_cum.update_layout(
        template="plotly_white",
        title_text="<b>Cumulative Production</b>",
        height=450,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1.0, bgcolor="rgba(255,255,255,0.7)"
        ),
        font=dict(size=13),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig_cum.update_xaxes(
        type="log",
        dtick=1,
        tickvals=[1, 10, 100, 1000, 10000],
        minor=dict(showgrid=True),
        showgrid=True, gridwidth=1,
        title="Time (days)",
    )
    fig_cum.update_yaxes(title_text="Gas (BCF)",        secondary_y=False, showgrid=True,  gridwidth=1)
    fig_cum.update_yaxes(title_text="Liquids (MMbbl)",  secondary_y=True,  showgrid=False)

    st.plotly_chart(fig_cum, use_container_width=True, theme=None)
else:
    st.warning("Cumulative series not available.")
# ======== end Results tab semi-log plots ========

# #### Part 4: Main Application UI - Advanced and Visualization Tabs ####

elif selected_tab == "3D Viewer":
    st.header("3D Viewer")

    sim = st.session_state.get("sim") or {}
    kx_vol = st.session_state.get("kx")   # expected (nz, ny, nx)
    phi_vol = st.session_state.get("phi") # expected (nz, ny, nx)

    # If nothing at all is available, bail early
    if kx_vol is None and phi_vol is None and not sim:
        st.warning("Please generate rock properties or run a simulation to enable the 3D viewer.")
        st.stop()

    # Build the property list only from fields that actually exist
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

    c1, c2 = st.columns(2)
    with c1:
        ds = st.slider(
            "Downsample factor",
            1, 10,
            int(st.session_state.get("vol_downsample", 2)),
            1,
            key="vol_ds"
        )
    with c2:
        iso_rel = st.slider(
            "Isosurface value (relative)",
            0.05, 0.95,
            float(st.session_state.get("iso_value_rel", 0.85)),
            0.05,
            key="iso_val_rel"
        )

    # Resolve grid spacing (accept *_ft or raw)
    dx = float(state.get("dx_ft", state.get("dx", 1.0)))
    dy = float(state.get("dy_ft", state.get("dy", 1.0)))
    dz = float(state.get("dz_ft", state.get("dz", 1.0)))

    # Select data and styling
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
        p_init = sim.get("p_init_3d")
        if p_final is not None and p_init is not None:
            data_3d = (np.asarray(p_init) - np.asarray(p_final))  # ΔP = Pin − Pfinal
            colorscale = "Inferno"
            colorbar_title = "ΔP (psi)"
    elif prop_3d.startswith("Original Oil"):
        data_3d = sim.get("ooip_3d")
        colorscale = "Plasma"
        colorbar_title = "OOIP (STB/cell)"

    # Validate
    if data_3d is None:
        st.warning(f"Data for '{prop_3d}' not found. Please run a simulation.")
        st.stop()

    data_3d = np.asarray(data_3d)
    if data_3d.ndim != 3:
        st.warning("3D data is not in the expected (nz, ny, nx) shape.")
        st.stop()

    # Downsample (use your helper if available)
    try:
        data_ds = downsample_3d(data_3d, ds)
    except Exception:
        # simple stride fallback
        data_ds = data_3d[::ds, ::ds, ::ds]

    vmin, vmax = float(np.nanmin(data_ds)), float(np.nanmax(data_ds))
    isoval = vmin + (vmax - vmin) * iso_rel

    # Build coordinates consistent with (nz, ny, nx)
    nz, ny, nx = data_ds.shape
    z = np.arange(nz) * dz * ds
    y = np.arange(ny) * dy * ds
    x = np.arange(nx) * dx * ds
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")  # shapes (nz, ny, nx)

    with st.spinner("Generating 3D plot..."):
        fig3d = go.Figure(
            go.Isosurface(
                x=X.ravel(),
                y=Y.ravel(),
                z=Z.ravel(),
                value=data_ds.ravel(),
                isomin=isoval,
                isomax=vmax,
                surface_count=1,
                caps=dict(x_show=False, y_show=False, z_show=False),
                colorscale=colorscale,
                colorbar=dict(title=colorbar_title),
            )
        )

        # Optional horizontal well overlay (best-effort)
        try:
            L_ft = float(state.get("L_ft", nx * dx))
            n_lat = int(state.get("n_laterals", 1))
            y_span = ny * dy * ds
            y_positions = ([y_span/3.0, 2*y_span/3.0] if n_lat >= 2 else [y_span/2.0])
            z_mid = (nz * dz * ds) / 2.0

            for i, y_pos in enumerate(y_positions):
                fig3d.add_trace(
                    go.Scatter3d(
                        x=[0.0, L_ft],
                        y=[y_pos, y_pos],
                        z=[z_mid, z_mid],
                        mode="lines",
                        line=dict(width=8),
                        name=("Well" if i == 0 else ""),
                        showlegend=(i == 0),
                    )
                )
        except Exception:
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
        )

        st.plotly_chart(fig3d, use_container_width=True)


elif selected_tab == "Slice Viewer":
    st.header("Slice Viewer")
    sim_data = st.session_state.get("sim")
    if sim_data is None and st.session_state.get('kx') is None:
        st.warning("Please generate rock properties or run a simulation to enable the slice viewer.")
    else:
        prop_list = ['Permeability (kx)', 'Permeability (ky)', 'Porosity (ϕ)']
        if sim_data and sim_data.get('press_matrix') is not None:
            prop_list.append('Pressure (psi)')
        c1, c2 = st.columns(2)
        with c1:
            prop_slice = st.selectbox("Select property:", prop_list)
        with c2:
            plane_slice = st.selectbox("Select plane:", ["k-plane (z, top-down)", "j-plane (y, side-view)", "i-plane (x, end-view)"])
        data_3d = (st.session_state.get('kx') if 'kx' in prop_slice else st.session_state.get('ky') if 'ky' in prop_slice else st.session_state.get('phi') if 'ϕ' in prop_slice else sim_data.get('press_matrix'))
        if data_3d is not None:
            nz, ny, nx = data_3d.shape
            if "k-plane" in plane_slice:
                idx, axis_name = st.slider("k-index (z-layer)", 0, nz - 1, nz // 2), "k"
                data_2d, labels = data_3d[idx, :, :], dict(x="i-index", y="j-index")
            elif "j-plane" in plane_slice:
                idx, axis_name = st.slider("j-index (y-layer)", 0, ny - 1, ny // 2), "j"
                data_2d, labels = data_3d[:, idx, :], dict(x="i-index", y="k-index")
            else:
                idx, axis_name = st.slider("i-index (x-layer)", 0, nx - 1, nx // 2), "i"
                data_2d, labels = data_3d[:, :, idx], dict(x="j-index", y="k-index")
            fig = px.imshow(data_2d, origin="lower", aspect='equal', labels=labels, color_continuous_scale='viridis')
            fig.update_layout(title=f"<b>{prop_slice} @ {axis_name} = {idx}</b>")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Data for '{prop_slice}' not found.")

elif selected_tab == "QA / Material Balance":
    st.header("QA / Material Balance")
    sim = st.session_state.get("sim")
    if sim is None:
        st.warning("Run a simulation on the 'Results' tab to view QA plots.")
        st.stop()
    # --- Average reservoir pressure series (new: p_avg_psi fallback to pm_mid_psi)
    pavg = sim.get("p_avg_psi") or sim.get("pm_mid_psi")
    if pavg is None:
        st.info("Average reservoir pressure time series not returned by solver.")
        st.stop()
    # Quick pressure vs time plot
    if "t" in sim and len(sim["t"]) == len(pavg):
        fig_p = go.Figure(go.Scatter(x=sim["t"], y=pavg, name="p̄ reservoir (psi)"))
        fig_p.update_layout(template="plotly_white", title_text="<b>Average Reservoir Pressure</b>", xaxis_title="Time (days)", yaxis_title="Pressure (psi)")
        st.plotly_chart(fig_p, use_container_width=True, theme=None)
    # --- Guard for rates needed below
    if not all(k in sim for k in ("t", "qg", "qo")) or len(sim["t"]) < 2:
        st.warning("Simulation data is missing required arrays ('t', 'qg', 'qo') for this analysis.")
        st.stop()
    t = np.asarray(sim["t"], float)
    qg = np.asarray(sim["qg"], float)  # Mscf/d
    qo = np.asarray(sim["qo"], float)  # STB/d
    # ============================== GAS MATERIAL BALANCE ==============================
    st.markdown("### Gas Material Balance")
    # Cumulative gas on x-axis in MMscf (units: qg[Mscf/d] ⨉ dt[d] → Mscf → /1e3 → MMscf)
    Gp_Mscf = cumulative_trapezoid(qg, t, initial=0.0)  # Mscf
    Gp_MMscf = Gp_Mscf / 1e3  # MMscf
    # P/Z
    z_factors = z_factor_approx(np.asarray(pavg), p_init_psi=state["p_init_psi"])
    p_over_z = np.asarray(pavg) / np.maximum(z_factors, 1e-12)
    # Linear fit (ignore early transients: start at 25% of points)
    fit_start = max(1, len(Gp_MMscf) // 4)
    slope, intercept, _, _, _ = stats.linregress(Gp_MMscf[fit_start:], p_over_z[fit_start:])
    # GIIP from P/Z line intercept (Gp at P/Z = 0). Convert MMscf→BCF (/1000).
    giip_bcf = max(0.0, -intercept / slope / 1000.0) if slope != 0 else 0.0
    # Simulator EUR (fallback compute if not provided)
    sim_eur_g_bcf = sim.get("EUR_g_BCF")
    if sim_eur_g_bcf is None:
        sim_eur_g_bcf = np.trapz(qg, t) / 1e6  # Mscf → BCF
    c1, c2 = st.columns(2)
    c1.metric("Simulator Gas EUR", f"{sim_eur_g_bcf:.2f} BCF")
    c2.metric("Material Balance GIIP (from P/Z)", f"{giip_bcf:.2f} BCF", delta=(f"{(giip_bcf - sim_eur_g_bcf)/sim_eur_g_bcf:.1%} vs Sim" if sim_eur_g_bcf > 0 else None))
    # Plot P/Z vs cumulative gas (MMscf)
    fig_pz_gas = go.Figure()
    fig_pz_gas.add_trace(go.Scatter(x=Gp_MMscf, y=p_over_z, mode="markers", name="P/Z Data"))
    x_fit = np.array([0.0, giip_bcf * 1000.0])  # MMscf
    y_fit = slope * x_fit + intercept
    fig_pz_gas.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines", name="Linear Extrapolation", line=dict(dash="dash")))
    xmax = max(Gp_MMscf.max(), (giip_bcf * 1000.0) * 1.05)
    fig_pz_gas.update_layout(
        title="<b>P/Z vs. Cumulative Gas Production</b>",
        xaxis_title="Gp - Cumulative Gas Production (MMscf)",
        yaxis_title="P/Z",
        template="plotly_white",
        xaxis_range=[0, xmax]
    )
    st.plotly_chart(fig_pz_gas, use_container_width=True, theme="streamlit")
    st.markdown("---")
    # ============================== OIL MATERIAL BALANCE ==============================
    st.markdown("### Oil Material Balance")
    # Cumulative oil (STB) and gas (scf)
    Np_STB = cumulative_trapezoid(qo, t, initial=0.0)  # STB
    Gp_scf = cumulative_trapezoid(qg * 1_000.0, t, initial=0.0)  # scf (qg in Mscf/d)
    # Instantaneous produced GOR Rp (scf/STB)
    Rp = np.divide(Gp_scf, Np_STB, out=np.zeros_like(Gp_scf), where=Np_STB > 1e-3)  # safe divide
    # PVT from average reservoir pressure series
    Bo = Bo_of_p(pavg, state["pb_psi"], state["Bo_pb_rb_stb"])  # rb/STB
    Rs = Rs_of_p(pavg, state["pb_psi"], state["Rs_pb_scf_stb"])  # scf/STB
    Bg = Bg_of_p(pavg)  # rb/scf
    # Initial PVT at p_init
    p_init = state["p_init_psi"]
    Boi = Bo_of_p(p_init, state["pb_psi"], state["Bo_pb_rb_stb"])
    Rsi = Rs_of_p(p_init, state["pb_psi"], state["Rs_pb_scf_stb"])
    # Havlena–Odeh: F = Np_STB * (Bo + (Rp - Rs) * Bg) # rb
    F = Np_STB * (Bo + (Rp - Rs) * Bg)
    Et = (Bo - Boi) + (Rsi - Rs) * Bg  # rb/STB
    fit_start_oil = max(1, len(F) // 4)
    slope_oil, _, _, _, _ = stats.linregress(Et[fit_start_oil:], F[fit_start_oil:])
    ooip_mmstb = max(0.0, slope_oil / 1e6)  # rb / (rb/STB) -> STB; /1e6 for MMSTB
    sim_eur_o_mmstb = sim.get("EUR_o_MMBO")
    if sim_eur_o_mmstb is None:
        sim_eur_o_mmstb = np.trapz(qo, t) / 1e6  # STB → MMSTB
    rec_factor = (sim_eur_o_mmstb / ooip_mmstb * 100.0) if ooip_mmstb > 0 else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Simulator Oil EUR", f"{sim_eur_o_mmstb:.2f} MMSTB")
    c2.metric("Material Balance OOIP (F vs Et)", f"{ooip_mmstb:.2f} MMSTB")
    c3.metric("Implied Recovery Factor", f"{rec_factor:.1f}%")
    fig_mbe_oil = go.Figure()
    fig_mbe_oil.add_trace(go.Scatter(x=Et, y=F, mode="markers", name="F vs Et Data"))
    x_fit_oil = np.array([0.0, np.nanmax(Et)])
    y_fit_oil = slope_oil * x_fit_oil
    fig_mbe_oil.add_trace(go.Scatter(x=x_fit_oil, y=y_fit_oil, mode="lines", name=f"Slope (OOIP) = {ooip_mmstb:.2f} MMSTB", line=dict(dash="dash")))
    fig_mbe_oil.update_layout(title="<b>F vs. Et (Havlena–Odeh)</b>", xaxis_title="Et - Total Expansion (rb/STB)", yaxis_title="F - Underground Withdrawal (rb)", template="plotly_white")
    st.plotly_chart(fig_mbe_oil, use_container_width=True, theme="streamlit")

elif selected_tab == "Economics":
    st.header("Economics")
    if st.session_state.get("sim") is None:
        st.info("Run a simulation first to populate economics.")
    else:
        sim = st.session_state["sim"]
        t = np.asarray(sim["t"], float)
        qo = np.nan_to_num(np.asarray(sim.get("qo")), nan=0.0)
        qg = np.nan_to_num(np.asarray(sim.get("qg")), nan=0.0)
        qw = np.nan_to_num(np.asarray(sim.get("qw")) if sim.get("qw") is not None else np.zeros_like(qo), nan=0.0)
        st.subheader("Assumptions")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            oil_price = st.number_input("Oil price ($/bbl)", 0.0, 500.0, 75.0, 1.0)
        with c2:
            gas_price = st.number_input("Gas price ($/Mcf)", 0.0, 50.0, 2.50, 0.1)
        with c3:
            opex_bpd = st.number_input("OPEX ($/bbl liquids)", 0.0, 200.0, 6.0, 0.5)
        with c4:
            wd_cost = st.number_input("Water disposal ($/bbl)", 0.0, 50.0, 1.5, 0.1)
        c1, c2, c3 = st.columns(3)
        with c1:
            royalty = st.number_input("Royalty (fraction)", 0.0, 0.99, 0.20, 0.01)
        with c2:
            tax = st.number_input("Severance tax (fraction)", 0.0, 0.99, 0.045, 0.005)
        with c3:
            disc = st.number_input("Discount rate (APR)", 0.0, 1.0, 0.10, 0.01)
        # monthly cash flow from daily rates
        days = np.diff(t, prepend=t[0])
        oil_rev = qo * days * oil_price
        gas_rev = qg * days * gas_price
        gross = oil_rev + gas_rev
        # costs
        liquids = (qo + qw) * days
        opex = liquids * opex_bpd
        w_disp = qw * days * wd_cost
        # royalty & tax
        net_rev_int = gross * (1.0 - royalty)
        taxes = net_rev_int * tax
        cf = net_rev_int - (opex + w_disp) - taxes
        # NPV (daily to years)
        years = t / 365.25
        disc_fac = (1.0 + disc) ** (years - years[0])
        npv = float(np.nansum(cf / np.maximum(disc_fac, 1e-12)))
        st.metric("NPV (approx.)", f"${npv:,.0f}")
        fig_cf = go.Figure(go.Bar(x=t, y=cf/1e3, name="Cash Flow (k$)"))
        fig_cf.update_layout(template="plotly_white", title_text="<b>Cash Flow (Approx.)</b>", xaxis_title="Time (days)", yaxis_title="k$")
        st.plotly_chart(fig_cf, use_container_width=True, theme=None)

elif selected_tab == "Field Match (CSV)":
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
            days = np.arange(0, 731, 15)
            oil_rate = 950 * np.exp(-days / 400) + rng.uniform(-25, 25, size=days.shape)
            gas_rate = 8000 * np.exp(-days / 500) + rng.uniform(-200, 200, size=days.shape)
            oil_rate = np.clip(oil_rate, 0, None)
            gas_rate = np.clip(gas_rate, 0, None)
            demo_df = pd.DataFrame({"Day": days, "Gas_Rate_Mscfd": gas_rate, "Oil_Rate_STBpd": oil_rate})
            st.session_state.field_data_match = demo_df
            st.success("Demo production data loaded successfully!")
    if 'field_data_match' in st.session_state:
        st.markdown("---")
        st.markdown("#### Loaded Production Data (first 5 rows)")
        st.dataframe(st.session_state.field_data_match.head(), use_container_width=True)
        if st.session_state.get("sim") is not None and st.session_state.get("field_data_match") is not None:
            sim_data = st.session_state.sim
            field_data = st.session_state.field_data_match
            fig_match = go.Figure()
            fig_match.add_trace(go.Scatter(x=sim_data['t'], y=sim_data['qg'], mode='lines', name='Simulated Gas', line=dict(color="#d62728")))
            fig_match.add_trace(go.Scatter(x=sim_data['t'], y=sim_data['qo'], mode='lines', name='Simulated Oil', line=dict(color="#2ca02c"), yaxis="y2"))
            if {'Day', 'Gas_Rate_Mscfd'}.issubset(field_data.columns):
                fig_match.add_trace(go.Scatter(x=field_data['Day'], y=field_data['Gas_Rate_Mscfd'], mode='markers', name='Field Gas', marker=dict(color="#d62728", symbol='cross')))
            if {'Day', 'Oil_Rate_STBpd'}.issubset(field_data.columns):
                fig_match.add_trace(go.Scatter(x=field_data['Day'], y=field_data['Oil_Rate_STBpd'], mode='markers', name='Field Oil', marker=dict(color="#2ca02c", symbol='cross'), yaxis="y2"))
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
        elif st.session_state.get("sim") is None and st.session_state.get("field_data_match") is not None:
            st.info("Demo/Field data loaded. Run a simulation on the 'Results' tab to view the comparison plot.")

elif selected_tab == "Uncertainty & Monte Carlo":
    st.header("Uncertainty & Monte Carlo")
    p1, p2, p3 = st.columns(3)
    with p1:
        uc_k = st.checkbox("k stdev", True)
        k_mean = st.slider("k_stdev Mean", 0.0, 0.2, state['k_stdev'], 0.01)
        k_std = st.slider("k_stdev Stdev", 0.0, 0.1, 0.02, 0.005)
    with p2:
        uc_xf = st.checkbox("xf_ft", True)
        xf_mean = st.slider("xf_ft Mean (ft)", 100.0, 500.0, state['xf_ft'], 10.0)
        xf_std = st.slider("xf_ft Stdev (ft)", 0.0, 100.0, 30.0, 5.0)
    with p3:
        uc_int = st.checkbox("pad_interf", False)
        int_min = st.slider("Interference Min", 0.0, 0.8, state['pad_interf'], 0.01)
        int_max = st.slider("Interference Max", 0.0, 0.8, 0.5, 0.01)
    num_runs = st.number_input("Number of Monte Carlo runs", 10, 500, 50, 10)
    if st.button("Run Monte Carlo Simulation", key="run_mc"):
        qg_runs, qo_runs, eur_g, eur_o = [], [], [], []
        bar_mc = st.progress(0, text="Running Monte Carlo simulation...")
        base_state = state.copy()
        rng_mc = np.random.default_rng(st.session_state.rng_seed + 1)
        for i in range(num_runs):
            temp_state = base_state.copy()
            if uc_k:
                temp_state['k_stdev'] = stats.truncnorm.rvs(
                    (0 - k_mean) / k_std, (0.2 - k_mean) / k_std, loc=k_mean, scale=k_std, random_state=rng_mc
                )
            if uc_xf:
                temp_state['xf_ft'] = stats.truncnorm.rvs(
                    (100 - xf_mean) / xf_std, (500 - xf_mean) / xf_std, loc=xf_mean, scale=xf_std, random_state=rng_mc
                )
            if uc_int:
                temp_state['pad_interf'] = stats.uniform.rvs(
                    loc=int_min, scale=int_max - int_min, random_state=rng_mc
                )
            res = fallback_fast_solver(temp_state, rng_mc)
            qg_runs.append(res['qg'])
            qo_runs.append(res['qo'])
            eur_g.append(res['EUR_g_BCF'])
            eur_o.append(res['EUR_o_MMBO'])
            bar_mc.progress((i + 1) / num_runs, f"Run {i+1}/{num_runs}")
        st.session_state.mc_results = {
            't': res['t'],
            'qg_runs': np.array(qg_runs),
            'qo_runs': np.array(qo_runs),
            'eur_g': np.array(eur_g),
            'eur_o': np.array(eur_o),
        }
        bar_mc.empty()
    if 'mc_results' in st.session_state:
        mc = st.session_state.mc_results
        p10_g, p50_g, p90_g = np.percentile(mc['qg_runs'], [90, 50, 10], axis=0)
        p10_o, p50_o, p90_o = np.percentile(mc['qo_runs'], [90, 50, 10], axis=0)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure([
                go.Scatter(x=mc['t'], y=p90_g, fill=None, mode='lines', line_color='lightgrey', name='P10'),
                go.Scatter(x=mc['t'], y=p10_g, fill='tonexty', mode='lines', line_color='lightgrey', name='P90'),
                go.Scatter(x=mc['t'], y=p50_g, mode='lines', line_color='red', name='P50'),
            ])
            st.plotly_chart(fig.update_layout(**semi_log_layout("Gas Rate Probabilistic Forecast", yaxis="Gas Rate (Mscf/d)")), use_container_width=True, theme="streamlit")
            st.plotly_chart(
                px.histogram(x=mc['eur_g'], nbins=30, labels={'x': 'Gas EUR (BCF)'}).update_layout(
                    title="<b>Distribution of Gas EUR</b>", template="plotly_white"
                ), use_container_width=True, theme="streamlit"
            )
        with c2:
            fig = go.Figure([
                go.Scatter(x=mc['t'], y=p90_o, fill=None, mode='lines', line_color='lightgreen', name='P10'),
                go.Scatter(x=mc['t'], y=p10_o, fill='tonexty', mode='lines', line_color='lightgreen', name='P90'),
                go.Scatter(x=mc['t'], y=p50_o, mode='lines', line_color='green', name='P50'),
            ])
            st.plotly_chart(fig.update_layout(**semi_log_layout("Oil Rate Probabilistic Forecast", yaxis="Oil Rate (STB/d)")), use_container_width=True, theme="streamlit")
            st.plotly_chart(
                px.histogram(x=mc['eur_o'], nbins=30, labels={'x': 'Oil EUR (MMSTB)'}, color_discrete_sequence=['green']).update_layout(
                    title="<b>Distribution of Oil EUR</b>", template="plotly_white"
                ), use_container_width=True, theme="streamlit"
            )

elif selected_tab == "Well Placement Optimization":
    st.header("Well Placement Optimization")
    st.markdown("#### 1. General Parameters")
    c1_opt, c2_opt, c3_opt = st.columns(3)
    with c1_opt:
        objective = st.selectbox(
            "Objective Property", ["Maximize Oil EUR", "Maximize Gas EUR"], key="opt_objective"
        )
    with c2_opt:
        iterations = st.number_input(
            "Number of optimization steps", min_value=5, max_value=1000, value=100, step=10
        )
    with c3_opt:
        st.selectbox(
            "Forbidden Zone", ["Numerical Faults"], help="The optimizer will avoid placing wells near the fault defined in the sidebar."
        )
    st.markdown("#### 2. Well Parameters")
    c1_well, c2_well = st.columns(2)
    with c1_well:
        num_wells = st.number_input(
            "Number of wells to place", min_value=1, max_value=1, value=1, disabled=True, help="Currently supports optimizing a single well location."
        )
    with c2_well:
        st.text_input("Well name prefix", "OptiWell", disabled=True)
    launch_opt = st.button("🚀 Launch Optimization", use_container_width=True, type="primary")
    if launch_opt:
        opt_results = []
        base_state = state.copy()
        rng_opt = np.random.default_rng(int(st.session_state.rng_seed))
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
            is_valid = False
            guard = 0
            while (not is_valid) and (guard < 10000):
                x_heel_ft = rng_opt.uniform(0, x_max)
                y_heel_ft = rng_opt.uniform(50, y_max - 50)
                is_valid = is_heel_location_valid(x_heel_ft, y_heel_ft, base_state)
                guard += 1
            if not is_valid:
                st.error("Could not find a valid heel location. Check grid size, L_ft, and fault settings.")
                break
            temp_state = base_state.copy()
            x_norm = x_heel_ft / (base_state['nx'] * base_state['dx'])
            temp_state['pad_interf'] = 0.4 * x_norm
            result = fallback_fast_solver(temp_state, rng_opt)
            score = result['EUR_o_MMBO'] if "Oil" in objective else result['EUR_g_BCF']
            opt_results.append({
                "Step": i + 1, "x_ft": float(x_heel_ft), "y_ft": float(y_heel_ft), "Score": float(score),
            })
            progress_bar.progress(
                (i + 1) / int(iterations), text=f"Step {i+1}/{int(iterations)} | Score: {score:.3f}"
            )
        st.session_state.opt_results = pd.DataFrame(opt_results)
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
        fig_opt = go.Figure()
        phi_map = get_k_slice(
            st.session_state.get('phi', np.zeros((state['nz'], state['ny'], state['nx']))),
            state['nz'] // 2
        )
        fig_opt.add_trace(go.Heatmap(
            z=phi_map, dx=state['dx'], dy=state['dy'], colorscale='viridis', colorbar=dict(title='Porosity')
        ))
        fig_opt.add_trace(go.Scatter(
            x=df_results['x_ft'], y=df_results['y_ft'], mode='markers',
            marker=dict(
                color=df_results['Score'], colorscale='Reds', showscale=True,
                colorbar=dict(title='Score'), size=8, opacity=0.7
            ),
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
                x=fault_x, y=fault_y, mode='lines', line=dict(color='white', width=4, dash='dash'), name='Fault'
            ))
        fig_opt.update_layout(
            title="<b>Well Placement Optimization Map</b>",
            xaxis_title="X position (ft)",
            yaxis_title="Y position (ft)",
            template="plotly_white",
            height=600
        )
        st.plotly_chart(fig_opt, use_container_width=True, theme="streamlit")

elif selected_tab == "User’s Manual":
    st.header("User’s Manual")
    st.markdown("---")
    st.markdown("""
    ### 1. Introduction
    Welcome to the **Full 3D Unconventional & Black-Oil Reservoir Simulator**. This tool helps you explore reservoir behavior and forecast production for MSW wells.
    """)
    st.markdown("---")
    st.markdown("""
    ### 2. Quick Start
    1) Pick a **Preset** in the sidebar and click **Apply Preset**
    2) **Generate 3D property volumes** (Tab: "Generate 3D property volumes")
    3) Run **Results → Run simulation**
    4) Review plots; iterate parameters as needed
    """)
    st.markdown("---")
    st.markdown("""
    ### 3. History Matching Workflow
    Use **Field Match (CSV)** to load historical data, adjust parameters in the sidebar, and rerun until simulated rates align with measured points.
    """)

elif selected_tab == "Solver & Profiling":
    st.header("Solver & Profiling")
    st.info("This tab shows numerical solver settings and performance of the last run.")
    st.markdown("### Current Numerical Solver Settings")
    solver_settings = {
        "Parameter": [
            "Newton Tolerance", "Max Newton Iterations", "Threads", "Use OpenMP",
            "Use MKL", "Use PyAMG", "Use cuSPARSE"
        ],
        "Value": [
            f"{state['newton_tol']:.1e}", state['max_newton'], "Auto" if state['threads'] == 0 else state['threads'],
            "✅" if state['use_omp'] else "❌", "✅" if state['use_mkl'] else "❌",
            "✅" if state['use_pyamg'] else "❌", "✅" if state['use_cusparse'] else "❌",
        ],
    }
    st.table(pd.DataFrame(solver_settings))
    st.markdown("### Profiling")
    if st.session_state.get("sim") and 'runtime_s' in st.session_state.sim:
        st.metric(label="Last Simulation Runtime", value=f"{st.session_state.sim['runtime_s']:.2f} seconds")
    else:
        st.info("Run a simulation on the 'Results' tab to see performance profiling.")

elif selected_tab == "DFN Viewer":
    st.header("DFN Viewer — 3D line segments")
    segs = st.session_state.get('dfn_segments')
    if segs is None or len(segs) == 0:
        st.info("No DFN loaded. Upload a CSV or use 'Generate DFN from stages' in the sidebar.")
    else:
        figd = go.Figure()
        for i, seg in enumerate(segs):
            figd.add_trace(go.Scatter3d(
                x=[seg[0], seg[3]],
                y=[seg[1], seg[4]],
                z=[seg[2], seg[5]],
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
