# aap.py — Full 3D Unconventional / Black-Oil Reservoir Simulator (patched + polished)
# Streamlit main app (monolithic). The modular split version is provided below.

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

# External engines you already have
from core.full3d import simulate
from engines.fast import fallback_fast_solver

# ---------------------- Global Style ----------------------
COLOR_GAS   = "#d62728"  # red
COLOR_OIL   = "#2ca02c"  # green
COLOR_WATER = "#1f77b4"  # blue
pio.templates.default = "plotly_white"

PLOT_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png", "filename": "plot", "height": 720, "width": 1280, "scale": 3
    },
}

def decorate_semilog_time(fig, t):
    """Make x a semi-log axis with decade 'cycles' labeled & minor grid on."""
    t = np.asarray(t, float)
    t = np.where(t <= 0, 1e-6, t)
    fig.update_xaxes(type="log", dtick=1, minor=dict(showgrid=True), showgrid=True, gridwidth=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, minor=dict(showgrid=True))
    nmin = int(np.floor(np.log10(np.nanmin(t))))
    nmax = int(np.ceil(np.log10(np.nanmax(t))))
    cycle = 1
    for n in range(nmin, nmax + 1):
        x = 10 ** n
        fig.add_vline(x=x, line_width=1, line_dash="dot", line_color="rgba(0,0,0,0.35)")
        fig.add_annotation(x=x, y=1.02, yref="paper", xanchor="left",
                           showarrow=False, text=f"Cycle {cycle}",
                           font=dict(size=10, color="#444"))
        cycle += 1

def nice_gauge_limits(val, step):
    if not np.isfinite(val) or val <= 0: return step
    import math
    return max(step, math.ceil(val / step) * step)

def pro_eur_gauges(eur_g_bcf, eur_o_mmbo):
    gmax  = nice_gauge_limits(float(eur_g_bcf), 5.0)
    omax  = nice_gauge_limits(float(eur_o_mmbo), 0.5)

    g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(eur_g_bcf),
        number={'suffix': " BCF", 'font': {'size': 44, 'color': '#0b2545'}},
        title={'text': "<b>EUR Gas</b>", 'font': {'size': 22, 'color': '#0b2545'}},
        gauge={
            'axis': {'range': [0, gmax], 'tickwidth': 1.2, 'tickcolor': '#0b2545'},
            'bar': {'color': COLOR_GAS, 'thickness': 0.28},
            'bgcolor': 'white',
            'borderwidth': 1, 'bordercolor': '#cfe0ff',
            'steps': [
                {'range': [0, 0.6*gmax], 'color': 'rgba(0,0,0,0.04)'},
                {'range': [0.6*gmax, 0.85*gmax], 'color': 'rgba(0,0,0,0.07)'}
            ],
            'threshold': {'line': {'color': 'green', 'width': 4},
                          'thickness': 0.9, 'value': float(eur_g_bcf)},
        }
    ))
    g.update_layout(height=260, margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="#ffffff")

    o = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(eur_o_mmbo),
        number={'suffix': " MMBO", 'font': {'size': 44, 'color': '#0b2545'}},
        title={'text': "<b>EUR Oil</b>", 'font': {'size': 22, 'color': '#0b2545'}},
        gauge={
            'axis': {'range': [0, omax], 'tickwidth': 1.2, 'tickcolor': '#0b2545'},
            'bar': {'color': COLOR_OIL, 'thickness': 0.28},
            'bgcolor': 'white',
            'borderwidth': 1, 'bordercolor': '#cfe0ff',
            'steps': [
                {'range': [0, 0.6*omax], 'color': 'rgba(0,0,0,0.04)'},
                {'range': [0.6*omax, 0.85*omax], 'color': 'rgba(0,0,0,0.07)'}
            ],
            'threshold': {'line': {'color': 'red', 'width': 4},
                          'thickness': 0.9, 'value': float(eur_o_mmbo)},
        }
    ))
    o.update_layout(height=260, margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="#ffffff")
    return g, o

# ---------------------- PVT + RTA Helpers ----------------------
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
    p_norm = np.asarray(p_psi, float) / p_init_psi
    return 0.95 - 0.2 * (1 - p_norm) + 0.4 * (1 - p_norm) ** 2

# ---------------------- Rock / DFN Helpers ----------------------
def ensure_3d(a):
    A = np.asarray(a)
    return A[None, ...] if A.ndim == 2 else A

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

def generate_property_volumes(state):
    rng = np.random.default_rng(int(st.session_state.rng_seed))
    nz, ny, nx = int(state["nz"]), int(state["ny"]), int(state["nx"])
    style = state.get("facies_style", "Continuous (Gaussian)")
    if "Continuous" in style:
        kx_mid = 0.05 + state["k_stdev"] * rng.standard_normal((ny, nx))
        ky_mid = (0.05 / state["anis_kxky"]) + state["k_stdev"] * rng.standard_normal((ny, nx))
        phi_mid = 0.10 + state["phi_stdev"] * rng.standard_normal((ny, nx))
    elif "Speckled" in style:
        kx_mid = np.exp(rng.normal(np.log(0.05), 1.5 + state["k_stdev"]*5, (ny, nx)))
        ky_mid = kx_mid / state["anis_kxky"]
        phi_mid = np.exp(rng.normal(np.log(0.10), 0.8 + state["phi_stdev"]*3, (ny, nx)))
    else:  # Layered
        base_profile_k = 0.05 + state["k_stdev"] * rng.standard_normal(ny)
        kx_mid = np.tile(base_profile_k[:, None], (1, nx))
        ky_mid = kx_mid / state["anis_kxky"]
        base_profile_phi = 0.10 + state["phi_stdev"] * rng.standard_normal(ny)
        phi_mid = np.tile(base_profile_phi[:, None], (1, nx))

    kz_scale = np.linspace(0.95, 1.05, nz)[:, None, None]
    st.session_state.kx  = np.clip(kx_mid[None, ...] * kz_scale, 1e-4, 5.0)
    st.session_state.ky  = np.clip(ky_mid[None, ...] * kz_scale, 1e-4, 5.0)
    st.session_state.phi = np.clip(phi_mid[None, ...] * kz_scale, 0.01, 0.35)
    st.success("Successfully generated 3D property volumes!")

# ---------------------- Simulation Wrapper ----------------------
def _normalize_rates(qg, qo, qw):
    def arr(a): return None if a is None else np.asarray(a, float)
    qg, qo, qw = arr(qg), arr(qo), arr(qw)
    if qg is not None and np.nanmax(qg) > 2e5:  # likely scf/d → convert to Mscf/d
        qg = qg / 1e3
    return qg, qo, qw

def _get_sim_preview(state):
    tmp = state.copy()
    rng_preview = np.random.default_rng(int(st.session_state.get("rng_seed", 1234)) + 999)
    return fallback_fast_solver(tmp, rng_preview)

def run_simulation_engine(state):
    t0 = time.time()
    inputs = {
        "engine": "implicit" if "Implicit" in str(state.get("engine_type", "")) else "analytical",
        "nx": int(state.get("nx", 20)), "ny": int(state.get("ny", 20)), "nz": int(state.get("nz", 5)),
        "dx": float(state.get("dx_ft", state.get("dx", 100.0))),
        "dy": float(state.get("dy_ft", state.get("dy", 100.0))),
        "dz": float(state.get("dz_ft", state.get("dz", 50.0))),
        "phi": float(state.get("phi", 0.08)),
        "kx_md": float(state.get("kx_md", 100.0)),
        "ky_md": float(state.get("ky_md", 100.0)),
        "p_init_psi": float(state.get("p_init_psi", 5000.0)),
        "nw": float(state.get("nw", 2.0)), "no": float(state.get("no", 2.0)),
        "krw_end": float(state.get("krw_end", 0.6)), "kro_end": float(state.get("kro_end", 0.8)),
        "pb_psi": float(state.get("pb_psi", 3000.0)),
        "Bo_pb_rb_stb": float(state.get("Bo_pb_rb_stb", 1.2)),
        "Rs_pb_scf_stb": float(state.get("Rs_pb_scf_stb", 600.0)),
        "mu_o_cp": float(state.get("mu_o_cp", 1.2)),
        "mu_g_cp": float(state.get("mu_g_cp", 0.02)),
        "control": str(state.get("pad_ctrl", "BHP")),
        "bhp_psi": float(state.get("pad_bhp_psi", 2500.0)),
        "rate_mscfd": float(state.get("pad_rate_mscfd", 0.0)),
        "rate_stbd": float(state.get("pad_rate_stbd", 0.0)),
        "dt_days": float(state.get("dt_days", 30.0)),
        "t_end_days": float(state.get("t_end_days", 3650.0)),
        "use_gravity": bool(state.get("use_gravity", True)),
        "kvkh": float(state.get("kvkh", 0.10)),
        "geo_alpha": float(state.get("geo_alpha", 0.0)),
    }
    try:
        out = simulate(inputs)
    except Exception as e:
        st.error(f"Simulation error: {e}")
        return None

    t = out.get("t")
    qg, qo, qw = _normalize_rates(out.get("qg"), out.get("qo"), out.get("qw"))
    if t is None or (qg is None and qo is None):
        st.error("Engine did not return time series.")
        return None

    t = np.asarray(t, float)

    def _cum(y):
        if y is None: return None
        y = np.nan_to_num(np.asarray(y, float), nan=0.0)
        return cumulative_trapezoid(y, t, initial=0.0)

    cum_g_Mscf = _cum(qg)
    cum_o_STB  = _cum(qo)
    cum_w_STB  = _cum(qw)

    EUR_g_BCF  = float((cum_g_Mscf[-1] / 1e6) if cum_g_Mscf is not None else 0.0)
    EUR_o_MMBO = float((cum_o_STB[-1]  / 1e6) if cum_o_STB  is not None else 0.0)
    EUR_w_MMBL = float((cum_w_STB[-1]  / 1e6) if cum_w_STB  is not None else 0.0)

    p_avg_psi = out.get("p_avg_psi")
    press_matrix = out.get("press_matrix")

    if p_avg_psi is None:
        if isinstance(press_matrix, np.ndarray) and press_matrix.ndim == 4:
            p_avg_psi = np.nanmean(press_matrix, axis=(1, 2, 3))
        elif isinstance(press_matrix, np.ndarray) and press_matrix.ndim == 3:
            p0 = float(state.get("p_init_psi", 5000.0))
            pf = float(np.nanmean(press_matrix))
            w  = (t - t[0]) / max(t[-1] - t[0], 1e-9)
            p_avg_psi = p0 + (pf - p0) * w
        else:
            p0 = float(state.get("p_init_psi", 5000.0))
            pmin = float(state.get("p_min_bhp_psi", 2500.0))
            w  = (t - t[0]) / max(t[-1] - t[0], 1e-9)
            p_avg_psi = p0 - (p0 - pmin) * (0.6 * w)

    final = dict(
        t=t, qg=qg, qo=qo, qw=qw,
        cum_g_BCF=(cum_g_Mscf/1e6) if cum_g_Mscf is not None else None,
        cum_o_MMBO=(cum_o_STB/1e6) if cum_o_STB is not None else None,
        cum_w_MMBL=(cum_w_STB/1e6) if cum_w_STB is not None else None,
        EUR_g_BCF=EUR_g_BCF, EUR_o_MMBO=EUR_o_MMBO, EUR_w_MMBL=EUR_w_MMBL,
        runtime_s=time.time() - t0,
        p_avg_psi=p_avg_psi
    )

    if out.get("press_matrix") is not None:
        pm = np.asarray(out["press_matrix"])
        final["press_matrix"] = pm
        if out.get("p_init_3d") is not None:
            final["p_init_3d"] = out["p_init_3d"]
        else:
            nz, ny, nx = pm.shape[-3:]
            final["p_init_3d"] = np.full((nz, ny, nx), float(state.get("p_init_psi", 5000.0)), dtype=float)

    for k in ("ooip_3d", "pm_mid_psi"):
        if k in out: final[k] = out[k]

    return final

# ---------------------- App Defaults & Presets ----------------------
st.set_page_config(page_title="3D Unconventional / Black-Oil Reservoir Simulator", layout="wide")

def _setdefault(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

def _safe_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

def is_heel_location_valid(x_heel_ft, y_heel_ft, state):
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

ENGINE_TYPES = [
    "Analytical Model (Fast Proxy)",
    "3D Three-Phase Implicit (Phase 1a)",
    "3D Three-Phase Implicit (Phase 1b)",
]
VALID_MODEL_TYPES = ["Unconventional Reservoir", "Black Oil Reservoir"]

_setdefault("apply_preset_payload", None)
_setdefault("sim", None)
_setdefault("rng_seed", 1234)

defaults = dict(
    nx=300, ny=60, nz=12, dx=40.0, dy=40.0, dz=15.0,
    k_stdev=0.02, phi_stdev=0.02, anis_kxky=1.0,
    facies_style="Continuous (Gaussian)",
    use_fault=False, fault_plane="i-plane (vertical)", fault_index=60, fault_tm=0.10,
    n_laterals=2, L_ft=10000.0, stage_spacing_ft=250.0, clusters_per_stage=3,
    dP_LE_psi=200.0, f_fric=0.02, wellbore_ID_ft=0.30,
    xf_ft=300.0, hf_ft=180.0, pad_interf=0.20,
    pad_ctrl="BHP", pad_bhp_psi=2500.0, pad_rate_mscfd=100000.0,
    outer_bc="Infinite-acting", p_outer_psi=7950.0,
    pb_psi=5200.0, Rs_pb_scf_stb=650.0, Bo_pb_rb_stb=1.35,
    muo_pb_cp=1.20, mug_pb_cp=0.020, a_g=0.15, z_g=0.90,
    p_init_psi=5800.0, p_min_bhp_psi=2500.0, ct_1_over_psi=0.000015, include_RsP=True,
    krw_end=0.6, kro_end=0.8, nw=2.0, no=2.0, Swc=0.15, Sor=0.25, pc_slope_psi=0.0,
    ct_o_1psi=8e-6, ct_g_1psi=3e-6, ct_w_1psi=3e-6,
    newton_tol=1e-6, trans_tol=1e-7, max_newton=12, max_lin=200, threads=0,
    use_omp=False, use_mkl=False, use_pyamg=False, use_cusparse=False,
    dfn_radius_ft=60.0, dfn_strength_psi=500.0,
    engine_type="Analytical Model (Fast Proxy)"
)
for k, v in defaults.items(): _setdefault(k, v)

# ------------------------ PRESETS (US + Canada Shale Plays) ------------------------
# Typical, rounded values for quick-start modeling. Tune as needed per asset.
PLAY_PRESETS = {
    # --- UNITED STATES ---
    "Permian – Midland (Oil)": dict(
        L_ft=10000.0, stage_spacing_ft=250.0, xf_ft=300.0, hf_ft=180.0,
        Rs_pb_scf_stb=650.0, pb_psi=5200.0, Bo_pb_rb_stb=1.35, p_init_psi=5800.0
    ),
    "Permian – Delaware (Oil/Gas)": dict(
        L_ft=10000.0, stage_spacing_ft=225.0, xf_ft=320.0, hf_ft=200.0,
        Rs_pb_scf_stb=700.0, pb_psi=5400.0, Bo_pb_rb_stb=1.36, p_init_psi=6000.0
    ),
    "Eagle Ford (Oil Window)": dict(
        L_ft=9000.0, stage_spacing_ft=225.0, xf_ft=270.0, hf_ft=150.0,
        Rs_pb_scf_stb=700.0, pb_psi=5400.0, Bo_pb_rb_stb=1.34, p_init_psi=5600.0
    ),
    "Eagle Ford (Condensate)": dict(
        L_ft=9000.0, stage_spacing_ft=225.0, xf_ft=300.0, hf_ft=160.0,
        Rs_pb_scf_stb=900.0, pb_psi=5600.0, Bo_pb_rb_stb=1.30, p_init_psi=5800.0
    ),
    "Bakken / Three Forks (Oil)": dict(
        L_ft=10000.0, stage_spacing_ft=240.0, xf_ft=280.0, hf_ft=160.0,
        Rs_pb_scf_stb=350.0, pb_psi=4300.0, Bo_pb_rb_stb=1.20, p_init_psi=4700.0
    ),
    "Haynesville (Dry Gas)": dict(
        L_ft=10000.0, stage_spacing_ft=200.0, xf_ft=350.0, hf_ft=180.0,
        Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, p_init_psi=7000.0
    ),
    "Marcellus (Dry Gas)": dict(
        L_ft=9000.0, stage_spacing_ft=225.0, xf_ft=300.0, hf_ft=150.0,
        Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, p_init_psi=5200.0
    ),
    "Utica (Liquids-Rich)": dict(
        L_ft=10000.0, stage_spacing_ft=225.0, xf_ft=320.0, hf_ft=180.0,
        Rs_pb_scf_stb=400.0, pb_psi=5000.0, Bo_pb_rb_stb=1.22, p_init_psi=5500.0
    ),
    "Barnett (Gas)": dict(
        L_ft=6500.0, stage_spacing_ft=200.0, xf_ft=250.0, hf_ft=120.0,
        Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, p_init_psi=4200.0
    ),
    "Niobrara / DJ (Oil)": dict(
        L_ft=9000.0, stage_spacing_ft=225.0, xf_ft=280.0, hf_ft=140.0,
        Rs_pb_scf_stb=250.0, pb_psi=3800.0, Bo_pb_rb_stb=1.18, p_init_psi=4200.0
    ),
    "Anadarko – Woodford": dict(
        L_ft=10000.0, stage_spacing_ft=225.0, xf_ft=300.0, hf_ft=160.0,
        Rs_pb_scf_stb=300.0, pb_psi=4600.0, Bo_pb_rb_stb=1.20, p_init_psi=5000.0
    ),
    "Granite Wash": dict(
        L_ft=8000.0, stage_spacing_ft=225.0, xf_ft=280.0, hf_ft=150.0,
        Rs_pb_scf_stb=200.0, pb_psi=4200.0, Bo_pb_rb_stb=1.15, p_init_psi=4600.0
    ),
    "Fayetteville (Gas)": dict(
        L_ft=6000.0, stage_spacing_ft=200.0, xf_ft=240.0, hf_ft=120.0,
        Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, p_init_psi=3500.0
    ),
    "Tuscaloosa Marine (Oil)": dict(
        L_ft=10000.0, stage_spacing_ft=250.0, xf_ft=300.0, hf_ft=160.0,
        Rs_pb_scf_stb=450.0, pb_psi=5000.0, Bo_pb_rb_stb=1.25, p_init_psi=5400.0
    ),
    # --- CANADA ---
    "Montney (Condensate-Rich)": dict(
        L_ft=10000.0, stage_spacing_ft=225.0, xf_ft=330.0, hf_ft=180.0,
        Rs_pb_scf_stb=600.0, pb_psi=5200.0, Bo_pb_rb_stb=1.28, p_init_psi=5600.0
    ),
    "Duvernay (Liquids)": dict(
        L_ft=9500.0, stage_spacing_ft=225.0, xf_ft=320.0, hf_ft=180.0,
        Rs_pb_scf_stb=700.0, pb_psi=5400.0, Bo_pb_rb_stb=1.32, p_init_psi=5800.0
    ),
    "Horn River (Dry Gas)": dict(
        L_ft=9000.0, stage_spacing_ft=225.0, xf_ft=320.0, hf_ft=170.0,
        Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, p_init_psi=6500.0
    ),
}

PLAY_LIST = list(PLAY_PRESETS.keys())
# Apply queued preset payload
if st.session_state.apply_preset_payload is not None:
    for k, v in st.session_state.apply_preset_payload.items():
        st.session_state[k] = v
    st.session_state.apply_preset_payload = None
    _safe_rerun()

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.markdown("## Simulation Setup")
    st.markdown("### Engine & Presets")

    engine_type_ui = st.selectbox("Engine Type", ENGINE_TYPES, key="engine_type_ui")
    st.session_state["engine_type"] = engine_type_ui

    model_choice = st.selectbox("Model Type", VALID_MODEL_TYPES, key="sim_mode")
    st.session_state.fluid_model = "black_oil" if "Black Oil" in model_choice else "unconventional"

    st.markdown("Shale Play Preset")
    sel_col, tag_col = st.columns([0.78, 0.22])

    def _resource_label(name: str) -> str:
        s = name.lower()
        if "dry gas" in s or ("gas" in s and "oil" not in s and "condensate" not in s and "liquids" not in s):
            return "Gas"
        if "condensate" in s: return "Condensate"
        if "liquids" in s: return "Liquids"
        if "oil" in s: return "Oil"
        return "Mixed"

    with sel_col:
        PLAY_LIST = list(PLAY_PRESETS.keys())
        play = st.selectbox("play_selector", PLAY_LIST, index=0, key="play_sel", label_visibility="collapsed")
    with tag_col:
        res = _resource_label(play)
        st.markdown(
            f"""<div style="margin-top:6px; text-align:right;">
                 <span style="display:inline-block; padding:2px 8px; border-radius:999px;
                 background:#eef6ff; border:1px solid #b6d4fe; font-size:11px;
                 color:#0b5ed7; white-space:nowrap;">{res}</span></div>""",
            unsafe_allow_html=True,
        )

    if st.button("Apply Preset", use_container_width=True, type="primary"):
        payload = defaults.copy()
        payload.update(PLAY_PRESETS[st.session_state.play_sel])
        if st.session_state.fluid_model == "black_oil":
            payload.update(dict(
                Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, mug_pb_cp=0.020, a_g=0.15,
                p_init_psi=max(3500.0, float(payload.get("p_init_psi", 5200.0))),
                pad_ctrl="BHP",
                pad_bhp_psi=min(float(payload.get("p_init_psi", 5200.0)) - 500.0, 3000.0),
            ))
        st.session_state.sim = None
        st.session_state.apply_preset_payload = payload
        _safe_rerun()

    st.markdown("### Grid (ft)")
    st.number_input("nx", 1, 500, key="nx")
    st.number_input("ny", 1, 500, key="ny")
    st.number_input("nz", 1, 200, key="nz")
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
    max_idx = int(st.session_state.nx) - 2 if 'i-plane' in fault_plane_choice else int(st.session_state.ny) - 2
    if st.session_state.fault_index > max_idx: st.session_state.fault_index = max_idx
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
    st.checkbox("Use DFN-driven sink in solver", value=bool(st.session_state.get("use_dfn_sink", True)), key="use_dfn_sink")
    st.checkbox("Auto-generate DFN from stages when no upload", value=bool(st.session_state.get("use_auto_dfn", True)), key="use_auto_dfn")
    st.number_input("DFN influence radius (ft)", value=float(st.session_state.get("dfn_radius_ft", 60.0)), step=5.0, key="dfn_radius_ft")
    st.number_input("DFN sink strength (psi)", value=float(st.session_state.get("dfn_strength_psi", 500.0)), step=10.0, key="dfn_strength_psi")
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
                int(st.session_state.nx), int(st.session_state.ny), int(st.session_state.nz),
                float(st.session_state.dx), float(st.session_state.dy), float(st.session_state.dz),
                float(st.session_state.L_ft), float(st.session_state.stage_spacing_ft),
                int(st.session_state.n_laterals), float(st.session_state.hf_ft)
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

# ---------------------- Tabs ----------------------
state = {k: st.session_state[k] for k in defaults.keys() if k in st.session_state}

tabs = [
    "Setup Preview",
    "Generate 3D property volumes",
    "Results",
    "3D Viewer",
    "QA / Material Balance",
    "EUR vs Lateral Length",
    "User’s Manual",
]
st.write(
    '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content:center;}'
    '.stRadio > label {display:none;}'
    'div.row-widget.stRadio > div > div {border:1px solid #ccc; padding:6px 12px; border-radius:4px; margin:2px; background:#f0f2f6;}'
    'div.row-widget.stRadio > div > div[aria-checked="true"] {background:#e57373; color:white; border-color:#d32f2f;}</style>',
    unsafe_allow_html=True
)
selected_tab = st.radio("Navigation", tabs, label_visibility="collapsed")

# ---------------------- Tab: Setup Preview ----------------------
if selected_tab == "Setup Preview":
    st.header("Setup Preview")
    c1, c2 = st.columns([1, 1])

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
        with st.expander("Preset details"):
            st.write({
                "Play": st.session_state.get("play_sel"),
                "Model Type": st.session_state.get("sim_mode"),
                "Engine Type": st.session_state.get("engine_type"),
                "xf_ft": state.get("xf_ft"), "hf_ft": state.get("hf_ft"),
                "pb_psi": state.get("pb_psi"), "Rs_pb_scf_stb": state.get("Rs_pb_scf_stb"),
                "Bo_pb_rb_stb": state.get("Bo_pb_rb_stb"), "p_init_psi": state.get("p_init_psi"),
            })

        st.markdown("#### Well & Frac Summary")
        well_data = {
            "Parameter": ["Laterals", "Lateral Length (ft)", "Frac Half-length (ft)",
                          "Frac Height (ft)", "Stages", "Clusters/Stage"],
            "Value": [state['n_laterals'], state['L_ft'], state['xf_ft'], state['hf_ft'],
                      int(state['L_ft'] / state['stage_spacing_ft']), state['clusters_per_stage']],
        }
        st.table(pd.DataFrame(well_data))

    with c2:
        st.markdown("#### Top-Down Schematic")
        fig = go.Figure()
        nx, ny, dx, dy = state['nx'], state['ny'], state['dx'], state['dy']
        L_ft, xf_ft, ss_ft, n_lats = state['L_ft'], state['xf_ft'], state['stage_spacing_ft'], state['n_laterals']
        fig.add_shape(type="rect", x0=0, y0=0, x1=nx*dx, y1=ny*dy, line=dict(color="RoyalBlue"),
                      fillcolor="lightskyblue", opacity=0.3)
        lat_rows_y = [ny*dy/3, 2*ny*dy/3] if n_lats >= 2 else [ny*dy/2]
        n_stages = max(1, int(L_ft / max(ss_ft, 1.0)))
        for i, y_lat in enumerate(lat_rows_y):
            fig.add_trace(go.Scatter(x=[0, L_ft], y=[y_lat, y_lat], mode='lines',
                                     line=dict(color='black', width=3), name='Lateral',
                                     showlegend=(i == 0)))
            for j in range(n_stages):
                x_stage = (j + 0.5) * ss_ft
                if x_stage > L_ft: continue
                fig.add_trace(go.Scatter(x=[x_stage, x_stage], y=[y_lat - xf_ft, y_lat + xf_ft],
                                         mode='lines', line=dict(color='red', width=2), name='Frac',
                                         showlegend=(i == 0 and j == 0)))
        fig.update_layout(title="<b>Well and Fracture Geometry</b>", xaxis_title="X (ft)", yaxis_title="Y (ft)")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    st.markdown("---")
    st.markdown("### Production Forecast Preview (Analytical Model)")
    preview = _get_sim_preview(state)
    p_c1, p_c2 = st.columns(2)
    with p_c1:
        fig_g = go.Figure(go.Scatter(x=preview['t'], y=preview['qg'], name="Gas Rate", line=dict(color=COLOR_GAS)))
        fig_g.update_layout(title="<b>Gas Production Preview</b>", yaxis_title="Gas Rate (Mscf/d)")
        decorate_semilog_time(fig_g, preview['t'])
        st.plotly_chart(fig_g, use_container_width=True, config=PLOT_CONFIG)
    with p_c2:
        fig_o = go.Figure(go.Scatter(x=preview['t'], y=preview['qo'], name="Oil Rate", line=dict(color=COLOR_OIL)))
        fig_o.update_layout(title="<b>Oil Production Preview</b>", yaxis_title="Oil Rate (STB/d)")
        decorate_semilog_time(fig_o, preview['t'])
        st.plotly_chart(fig_o, use_container_width=True, config=PLOT_CONFIG)

# ---------------------- Tab: Generate 3D property volumes ----------------------
elif selected_tab == "Generate 3D property volumes":
    st.header("Generate 3D Property Volumes (kx, ky, ϕ)")
    st.info("Use this tab to (re)generate φ/k grids based on sidebar parameters.")
    if st.button("Generate New Property Volumes", use_container_width=True, type="primary"):
        generate_property_volumes(state)
    st.markdown("---")
    if st.session_state.get('kx') is not None:
        kx_display = get_k_slice(st.session_state.kx, state['nz'] // 2)
        ky_display = get_k_slice(st.session_state.ky, state['nz'] // 2)
        phi_display = get_k_slice(st.session_state.phi, state['nz'] // 2)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.imshow(kx_display, origin="lower", color_continuous_scale="Viridis",
                                      labels=dict(color="mD"), title="<b>kx — mid-layer (mD)</b>"),
                            use_container_width=True)
        with c2:
            st.plotly_chart(px.imshow(ky_display, origin="lower", color_continuous_scale="Cividis",
                                      labels=dict(color="mD"), title="<b>ky — mid-layer (mD)</b>"),
                            use_container_width=True)
        st.plotly_chart(px.imshow(phi_display, origin="lower", color_continuous_scale="Magma",
                                  labels=dict(color="ϕ"), title="<b>Porosity ϕ — mid-layer (fraction)</b>"),
                        use_container_width=True)
    else:
        st.info("Click the button above to generate initial property volumes.")

# ---------------------- Tab: Results ----------------------
elif selected_tab == "Results":
    st.header("Simulation Results")
    if st.button("Run simulation", type="primary", use_container_width=True):
        if 'kx' not in st.session_state:
            st.info("Rock properties not found. Generating them first...")
            generate_property_volumes(state)
        with st.spinner("Running full 3D simulation..."):
            sim_out = run_simulation_engine(state)
        st.session_state.sim = sim_out

    sim = st.session_state.get("sim")
    if not sim:
        st.info("Click **Run simulation** to compute and display the full 3D results.")
    else:
        st.success(f"Simulation complete in {sim.get('runtime_s', 0):.2f} seconds.")

        # Gauges
        c1, c2 = st.columns(2)
        gfig, ofig = pro_eur_gauges(sim.get("EUR_g_BCF", 0.0), sim.get("EUR_o_MMBO", 0.0))
        with c1: st.plotly_chart(gfig, use_container_width=True, config=PLOT_CONFIG)
        with c2: st.plotly_chart(ofig,  use_container_width=True, config=PLOT_CONFIG)

        # Rate vs Time (semi-log + cycles)
        t, qg, qo, qw = sim.get("t"), sim.get("qg"), sim.get("qo"), sim.get("qw")
        if t is not None and any(v is not None for v in (qg, qo, qw)):
            fig_rate = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
            if qg is not None:
                fig_rate.add_trace(go.Scatter(x=t, y=qg, name="Gas (Mscf/d)", line=dict(color=COLOR_GAS, width=2)),
                                   secondary_y=False)
            if qo is not None:
                fig_rate.add_trace(go.Scatter(x=t, y=qo, name="Oil (STB/d)",  line=dict(color=COLOR_OIL, width=2)),
                                   secondary_y=True)
            if qw is not None:
                fig_rate.add_trace(go.Scatter(x=t, y=qw, name="Water (STB/d)",
                                              line=dict(color=COLOR_WATER, width=1.5, dash="dot")),
                                   secondary_y=True)
            fig_rate.update_layout(template="plotly_white",
                                   title_text="<b>Production Rate vs. Time (Semi-log)</b>",
                                   height=480,
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                                   font=dict(size=13))
            decorate_semilog_time(fig_rate, t)
            fig_rate.update_yaxes(title_text="Gas Rate (Mscf/d)",  secondary_y=False)
            fig_rate.update_yaxes(title_text="Liquid Rate (STB/d)", secondary_y=True)
            st.markdown("### Production Profiles")
            st.plotly_chart(fig_rate, use_container_width=True, config=PLOT_CONFIG)

        # Cumulative (semi-log X)
        cum_g, cum_o, cum_w = sim.get("cum_g_BCF"), sim.get("cum_o_MMBO"), sim.get("cum_w_MMBL")
        if t is not None and (cum_g is not None or cum_o is not None or cum_w is not None):
            fig_cum = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
            if cum_g is not None:
                fig_cum.add_trace(go.Scatter(x=t, y=cum_g, name="Cum Gas (BCF)",   line=dict(color=COLOR_GAS,  width=2)),
                                  secondary_y=False)
            if cum_o is not None:
                fig_cum.add_trace(go.Scatter(x=t, y=cum_o, name="Cum Oil (MMbbl)", line=dict(color=COLOR_OIL,  width=2)),
                                  secondary_y=True)
            if cum_w is not None:
                fig_cum.add_trace(go.Scatter(x=t, y=cum_w, name="Cum Water (MMbbl)",
                                             line=dict(color=COLOR_WATER, width=1.5, dash="dot")),
                                  secondary_y=True)
            fig_cum.update_layout(template="plotly_white",
                                  title_text="<b>Cumulative Production (Semi-log X)</b>",
                                  height=460,
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                                  font=dict(size=13))
            decorate_semilog_time(fig_cum, t)
            fig_cum.update_yaxes(title_text="Gas (BCF)", secondary_y=False)
            fig_cum.update_yaxes(title_text="Liquids (MMbbl)", secondary_y=True)
            st.plotly_chart(fig_cum, use_container_width=True, config=PLOT_CONFIG)

# ---------------------- Tab: 3D Viewer ----------------------
elif selected_tab == "3D Viewer":
    st.header("3D Viewer")
    sim = st.session_state.get("sim") or {}
    kx_vol  = st.session_state.get("kx")
    phi_vol = st.session_state.get("phi")

    if sim.get("press_matrix") is not None and sim.get("p_init_3d") is None:
        pm = np.asarray(sim["press_matrix"])
        nz, ny, nx = pm.shape[-3:]
        sim["p_init_3d"] = np.full((nz, ny, nx), float(state.get("p_init_psi", 5000.0)))

    if kx_vol is None and phi_vol is None and not sim:
        st.warning("Please generate rock properties or run a simulation to enable the 3D viewer.")
        st.stop()

    menu = []
    if kx_vol is not None:                       menu.append("Permeability (kx)")
    if phi_vol is not None:                      menu.append("Porosity (ϕ)")
    if sim.get("press_matrix") is not None:      menu.append("Pressure (psi)")
    if sim.get("press_matrix") is not None and sim.get("p_init_3d") is not None:
        menu.append("Pressure Change (ΔP)")
    if sim.get("ooip_3d") is not None:           menu.append("Original Oil In Place (OOIP)")
    if not menu:
        st.info("No 3D properties are available yet. Run a simulation to populate pressure/OOIP.")
        st.stop()

    prop_3d = st.selectbox("Select property to view:", menu, index=0)
    c1, c2 = st.columns(2)
    with c1:
        ds = st.slider("Downsample factor", 1, 10, int(st.session_state.get("vol_downsample", 2)), 1, key="vol_ds")
    with c2:
        iso_rel = st.slider("Isosurface value (relative)", 0.05, 0.95,
                            float(st.session_state.get("iso_value_rel", 0.85)), 0.05, key="iso_val_rel")

    dx = float(state.get("dx", 1.0)); dy = float(state.get("dy", 1.0)); dz = float(state.get("dz", 1.0))
    colorscale = "Viridis"; colorbar_title = ""; data_3d = None

    if prop_3d.startswith("Permeability"):
        data_3d = kx_vol; colorscale = "Viridis"; colorbar_title = "kx (mD)"
    elif prop_3d.startswith("Porosity"):
        data_3d = phi_vol; colorscale = "Magma"; colorbar_title = "Porosity (ϕ)"
    elif prop_3d.startswith("Pressure (psi)"):
        data_3d = sim.get("press_matrix"); colorscale = "Jet"; colorbar_title = "Pressure (psi)"
    elif prop_3d.startswith("Pressure Change"):
        p_final = sim.get("press_matrix"); p_init = sim.get("p_init_3d")
        if p_final is not None and p_init is not None:
            data_3d = (np.asarray(p_init) - np.asarray(p_final)); colorscale = "Inferno"; colorbar_title = "ΔP (psi)"
    elif prop_3d.startswith("Original Oil"):
        data_3d = sim.get("ooip_3d"); colorscale = "Plasma"; colorbar_title = "OOIP (STB/cell)"

    if data_3d is None:
        st.warning(f"Data for '{prop_3d}' not found. Please run a simulation.")
        st.stop()

    data_3d = np.asarray(data_3d)
    if data_3d.ndim != 3:
        st.warning("3D data is not in the expected (nz, ny, nx) shape.")
        st.stop()

    try:
        data_ds = downsample_3d(data_3d, ds)
    except Exception:
        data_ds = data_3d[::ds, ::ds, ::ds]

    vmin, vmax = float(np.nanmin(data_ds)), float(np.nanmax(data_ds))
    isoval = vmin + (vmax - vmin) * iso_rel
    nz, ny, nx = data_ds.shape
    z = np.arange(nz) * dz * ds; y = np.arange(ny) * dy * ds; x = np.arange(nx) * dx * ds
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    with st.spinner("Generating 3D plot..."):
        fig3d = go.Figure(go.Isosurface(
            x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=data_ds.ravel(),
            isomin=isoval, isomax=vmax, surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale=colorscale, colorbar=dict(title=colorbar_title)
        ))
        # Optional lateral overlay
        try:
            L_ft = float(state.get("L_ft", nx * dx)); n_lat = int(state.get("n_laterals", 1))
            y_span = ny * dy * ds; y_positions = ([y_span/3.0, 2*y_span/3.0] if n_lat >= 2 else [y_span/2.0])
            z_mid = (nz * dz * ds) / 2.0
            for i, y_pos in enumerate(y_positions):
                fig3d.add_trace(go.Scatter3d(x=[0.0, L_ft], y=[y_pos, y_pos], z=[z_mid, z_mid],
                                             mode="lines", line=dict(width=8),
                                             name=("Well" if i == 0 else ""), showlegend=(i == 0)))
        except Exception:
            pass
        fig3d.update_layout(title=f"<b>3D Isosurface: {prop_3d}</b>",
                            scene=dict(xaxis_title="X (ft)", yaxis_title="Y (ft)", zaxis_title="Z (ft)",
                                       aspectmode="data"),
                            margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig3d, use_container_width=True, config=PLOT_CONFIG)

# ---------------------- Tab: QA / Material Balance ----------------------
elif selected_tab == "QA / Material Balance":
    st.header("QA / Material Balance")
    sim = st.session_state.get("sim")
    if not sim:
        st.warning("Run a simulation on the 'Results' tab to view QA plots.")
        st.stop()

    pavg = sim.get("p_avg_psi") or sim.get("pm_mid_psi")
    if pavg is None:
        st.info("Average reservoir pressure time series not available.")
        st.stop()

    # Pressure vs time
    if "t" in sim and len(sim["t"]) == len(pavg):
        fig_p = go.Figure(go.Scatter(x=sim["t"], y=pavg, name="p̄ reservoir (psi)"))
        fig_p.update_layout(title_text="<b>Average Reservoir Pressure</b>",
                            xaxis_title="Time (days)", yaxis_title="Pressure (psi)")
        st.plotly_chart(fig_p, use_container_width=True, config=PLOT_CONFIG)

    if not all(k in sim for k in ("t", "qg", "qo")) or len(sim["t"]) < 2:
        st.warning("Simulation missing arrays ('t','qg','qo') needed for MB analysis.")
        st.stop()

    t = np.asarray(sim["t"], float)
    qg = np.asarray(sim["qg"], float)  # Mscf/d
    qo = np.asarray(sim["qo"], float)  # STB/d

    # --- Gas Material Balance (P/Z)
    st.markdown("### Gas Material Balance")
    Gp_Mscf  = cumulative_trapezoid(qg, t, initial=0.0)      # Mscf
    Gp_MMscf = Gp_Mscf / 1e3
    zf = z_factor_approx(np.asarray(pavg), p_init_psi=state["p_init_psi"])
    p_over_z = np.asarray(pavg) / np.maximum(zf, 1e-12)
    fit_start = max(1, len(Gp_MMscf) // 4)
    slope, intercept, _, _, _ = stats.linregress(Gp_MMscf[fit_start:], p_over_z[fit_start:])
    giip_bcf = max(0.0, -intercept / slope / 1000.0) if slope != 0 else 0.0
    sim_eur_g_bcf = sim.get("EUR_g_BCF", float(np.trapz(qg, t) / 1e6))
    c1, c2 = st.columns(2)
    c1.metric("Simulator Gas EUR", f"{sim_eur_g_bcf:.2f} BCF")
    c2.metric("Material Balance GIIP (from P/Z)", f"{giip_bcf:.2f} BCF",
              delta=(f"{(giip_bcf - sim_eur_g_bcf)/sim_eur_g_bcf:.1%} vs Sim" if sim_eur_g_bcf > 0 else None))
    fig_pz = go.Figure()
    fig_pz.add_trace(go.Scatter(x=Gp_MMscf, y=p_over_z, mode="markers", name="P/Z Data"))
    x_fit = np.array([0.0, giip_bcf * 1000.0]); y_fit = slope * x_fit + intercept
    fig_pz.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines", name="Linear Extrapolation", line=dict(dash="dash")))
    xmax = max(Gp_MMscf.max(), (giip_bcf * 1000.0) * 1.05)
    fig_pz.update_layout(title="<b>P/Z vs. Cumulative Gas Production</b>",
                         xaxis_title="Gp - Cumulative Gas (MMscf)", yaxis_title="P/Z",
                         xaxis_range=[0, xmax])
    st.plotly_chart(fig_pz, use_container_width=True, config=PLOT_CONFIG)
    st.markdown("---")

    # --- Oil Material Balance (Havlena–Odeh)
    st.markdown("### Oil Material Balance")
    Np_STB = cumulative_trapezoid(qo, t, initial=0.0)             # STB
    Gp_scf = cumulative_trapezoid(qg * 1_000.0, t, initial=0.0)   # scf
    Rp = np.divide(Gp_scf, Np_STB, out=np.zeros_like(Gp_scf), where=Np_STB > 1e-3)
    Bo = Bo_of_p(pavg, state["pb_psi"], state["Bo_pb_rb_stb"])
    Rs = Rs_of_p(pavg, state["pb_psi"], state["Rs_pb_scf_stb"])
    Bg = Bg_of_p(pavg)
    p_init = state["p_init_psi"]
    Boi = Bo_of_p(p_init, state["pb_psi"], state["Bo_pb_rb_stb"])
    Rsi = Rs_of_p(p_init, state["pb_psi"], state["Rs_pb_scf_stb"])
    F  = Np_STB * (Bo + (Rp - Rs) * Bg)
    Et = (Bo - Boi) + (Rsi - Rs) * Bg
    fit_start_oil = max(1, len(F) // 4)
    slope_oil, _, _, _, _ = stats.linregress(Et[fit_start_oil:], F[fit_start_oil:])
    ooip_mmstb = max(0.0, slope_oil / 1e6)
    sim_eur_o_mmstb = sim.get("EUR_o_MMBO", float(np.trapz(qo, t) / 1e6))
    rec_factor = (sim_eur_o_mmstb / ooip_mmstb * 100.0) if ooip_mmstb > 0 else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Simulator Oil EUR", f"{sim_eur_o_mmstb:.2f} MMSTB")
    c2.metric("Material Balance OOIP (F vs Et)", f"{ooip_mmstb:.2f} MMSTB")
    c3.metric("Implied Recovery Factor", f"{rec_factor:.1f}%")
    fig_mbe = go.Figure()
    fig_mbe.add_trace(go.Scatter(x=Et, y=F, mode="markers", name="F vs Et Data"))
    x_fit_oil = np.array([0.0, np.nanmax(Et)]); y_fit_oil = slope_oil * x_fit_oil
    fig_mbe.add_trace(go.Scatter(x=x_fit_oil, y=y_fit_oil, mode="lines",
                                 name=f"Slope (OOIP) = {ooip_mmstb:.2f} MMSTB", line=dict(dash="dash")))
    fig_mbe.update_layout(title="<b>F vs. Et (Havlena–Odeh)</b>",
                          xaxis_title="Et - Total Expansion (rb/STB)",
                          yaxis_title="F - Underground Withdrawal (rb)")
    st.plotly_chart(fig_mbe, use_container_width=True, config=PLOT_CONFIG)

# ---------------------- Tab: EUR vs Lateral Length ----------------------
elif selected_tab == "EUR vs Lateral Length":
    st.header("EUR vs Lateral Length")
    base = state.copy()
    Lmin, Lmax = st.slider("Lateral length range (ft)", 4000, 15000, (7000, 12000), 250)
    step = st.number_input("Step (ft)", 100, 1000, 500, 50)
    cents = np.arange(Lmin, Lmax + 1, step, dtype=int)
    results = []
    rng = np.random.default_rng(int(st.session_state.get("rng_seed", 1234)) + 555)
    for L in cents:
        trial = base.copy()
        trial["L_ft"] = float(L)
        out = fallback_fast_solver(trial, rng)
        results.append((L, float(out["EUR_g_BCF"]), float(out["EUR_o_MMBO"])))
    df = pd.DataFrame(results, columns=["Lateral_ft", "EUR_g_BCF", "EUR_o_MMBO"])
    c1, c2 = st.columns(2)
    with c1:
        f = go.Figure(go.Scatter(x=df["Lateral_ft"], y=df["EUR_g_BCF"], mode="lines+markers", name="Gas EUR"))
        f.update_layout(title="<b>Gas EUR vs Lateral Length</b>", xaxis_title="Lateral (ft)", yaxis_title="EUR Gas (BCF)")
        st.plotly_chart(f, use_container_width=True, config=PLOT_CONFIG)
    with c2:
        f = go.Figure(go.Scatter(x=df["Lateral_ft"], y=df["EUR_o_MMBO"], mode="lines+markers", name="Oil EUR"))
        f.update_layout(title="<b>Oil EUR vs Lateral Length</b>", xaxis_title="Lateral (ft)", yaxis_title="EUR Oil (MMBO)")
        st.plotly_chart(f, use_container_width=True, config=PLOT_CONFIG)
    st.dataframe(df, use_container_width=True)

# ---------------------- Tab: User’s Manual ----------------------
elif selected_tab == "User’s Manual":
    st.header("User’s Manual")
    st.markdown("---")
    st.markdown("""
### Overview
This app provides two engines:
- **Analytical Model (Fast Proxy)** for rapid preview, Monte Carlo, and optimization.
- **3D Three-Phase Implicit** (Phase 1a/1b) for full-physics runs (developing).

### Quick Start
1. Select a **Shale Play Preset** in the sidebar and click **Apply Preset**.  
2. Open **Generate 3D property volumes** and click **Generate** (kx, ky, ϕ).  
3. Go to **Results** and click **Run simulation**.  
4. Review **EUR Gauges**, **Rate (semi-log)**, and **Cumulative (semi-log)**.  
5. Use **3D Viewer** to inspect Permeability, Porosity, Pressure, and **ΔP**.  
6. For diagnostics, see **QA / Material Balance** (P/Z and F vs Et).  
7. Explore **EUR vs Lateral Length** for design sensitivity.

### Units & Conventions
- Gas rate **Mscf/d**, cumulative gas **BCF**.  
- Oil/Water rate **STB/d**; cumulative **MMbbl**.  
- Semi-log plots show **decade cycles** with vertical markers and minor grids.
- All charts support high-res export (camera icon).
""")
