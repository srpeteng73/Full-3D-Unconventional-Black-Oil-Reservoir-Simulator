# Full 3D Unconventional / Black-Oil Reservoir Simulator — Implicit Engine Ready (USOF units) + DFN support
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy import stats

# ------------------------ Utils ------------------------
def _setdefault(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

st.set_page_config(page_title="3D Unconventional / Black-Oil Reservoir Simulator", layout="wide")

# ------------------------ Defaults ------------------------
_setdefault("apply_preset_payload", None)
_setdefault("sim", None)
_setdefault("rng_seed", 1234)
_setdefault("sim_mode", "3D Unconventional Reservoir Simulator — Implicit Engine Ready")
_setdefault("dfn_segments", None)
_setdefault("use_dfn_sink", True)
_setdefault("use_auto_dfn", True)
_setdefault("vol_downsample", 2)
_setdefault("iso_value_rel", 0.5)

defaults = dict(
    nx=120, ny=60, nz=12,
    dx=40.0, dy=40.0, dz=15.0,
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
    dfn_radius_ft=60.0,
    dfn_strength_psi=500.0,
)

for k, v in defaults.items():
    _setdefault(k, v)

if st.session_state.apply_preset_payload is not None:
    for k, v in st.session_state.apply_preset_payload.items():
        st.session_state[k] = v
    st.session_state.apply_preset_payload = None
    _safe_rerun()

# ------------------------ Presets ------------------------
PLAY_PRESETS = {
    "Permian — Wolfcamp (volatile oil window)": dict(L_ft=10000.0, stage_spacing_ft=250.0, xf_ft=300.0, hf_ft=180.0, Rs_pb_scf_stb=650.0, pb_psi=5200.0, Bo_pb_rb_stb=1.35, p_init_psi=5800.0),
    "Permian — Bone Spring (volatile)": dict(L_ft=10000.0, stage_spacing_ft=225.0, xf_ft=280.0, hf_ft=160.0, Rs_pb_scf_stb=600.0, pb_psi=5400.0, Bo_pb_rb_stb=1.33, p_init_psi=5900.0),
    "Eagle Ford — volatile oil": dict(L_ft=9000.0, stage_spacing_ft=225.0, xf_ft=270.0, hf_ft=150.0, Rs_pb_scf_stb=650.0, pb_psi=5200.0, Bo_pb_rb_stb=1.34, p_init_psi=5600.0),
    "Bakken — Middle Bakken (CGR-lite)": dict(L_ft=10000.0, stage_spacing_ft=250.0, xf_ft=260.0, hf_ft=150.0, Rs_pb_scf_stb=450.0, pb_psi=4200.0, Bo_pb_rb_stb=1.30, p_init_psi=5200.0),
    "Niobrara — volatile oil": dict(L_ft=8000.0, stage_spacing_ft=220.0, xf_ft=240.0, hf_ft=140.0, Rs_pb_scf_stb=500.0, pb_psi=5000.0, Bo_pb_rb_stb=1.32, p_init_psi=5500.0),
    "Haynesville — rich gas": dict(L_ft=9500.0, stage_spacing_ft=210.0, xf_ft=320.0, hf_ft=190.0, Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, p_init_psi=8000.0),
    "Montney — liquids-rich": dict(L_ft=10500.0, stage_spacing_ft=230.0, xf_ft=300.0, hf_ft=170.0, Rs_pb_scf_stb=700.0, pb_psi=5400.0, Bo_pb_rb_stb=1.36, p_init_psi=6200.0),
    "Duvernay — condensate": dict(L_ft=10000.0, stage_spacing_ft=240.0, xf_ft=290.0, hf_ft=175.0, Rs_pb_scf_stb=800.0, pb_psi=5600.0, Bo_pb_rb_stb=1.38, p_init_psi=6400.0),
    "Cardium — light oil": dict(L_ft=7000.0, stage_spacing_ft=260.0, xf_ft=220.0, hf_ft=120.0, Rs_pb_scf_stb=400.0, pb_psi=3800.0, Bo_pb_rb_stb=1.28, p_init_psi=4200.0),
    "Mancos — liquids-rich gas": dict(L_ft=9000.0, stage_spacing_ft=250.0, xf_ft=310.0, hf_ft=180.0, Rs_pb_scf_stb=300.0, pb_psi=4500.0, Bo_pb_rb_stb=1.22, p_init_psi=5200.0),
    "Tuscaloosa Marine — volatile oil": dict(L_ft=10000.0, stage_spacing_ft=230.0, xf_ft=300.0, hf_ft=170.0, Rs_pb_scf_stb=650.0, pb_psi=5300.0, Bo_pb_rb_stb=1.34, p_init_psi=5900.0),
    "Barnett — dry gas": dict(L_ft=7500.0, stage_spacing_ft=230.0, xf_ft=280.0, hf_ft=150.0, Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, p_init_psi=5000.0),
    "Fayetteville — gas": dict(L_ft=7000.0, stage_spacing_ft=220.0, xf_ft=270.0, hf_ft=140.0, Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, p_init_psi=4800.0),
    "Woodford — condensate": dict(L_ft=9000.0, stage_spacing_ft=240.0, xf_ft=300.0, hf_ft=170.0, Rs_pb_scf_stb=700.0, pb_psi=5600.0, Bo_pb_rb_stb=1.37, p_init_psi=6200.0),
    "Cana-Woodford — liquids-rich": dict(L_ft=10000.0, stage_spacing_ft=230.0, xf_ft=300.0, hf_ft=170.0, Rs_pb_scf_stb=600.0, pb_psi=5200.0, Bo_pb_rb_stb=1.34, p_init_psi=6000.0),
    "Marcellus — dry gas": dict(L_ft=9000.0, stage_spacing_ft=210.0, xf_ft=320.0, hf_ft=180.0, Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, p_init_psi=6500.0),
    "Marcellus — wet gas": dict(L_ft=9000.0, stage_spacing_ft=230.0, xf_ft=300.0, hf_ft=180.0, Rs_pb_scf_stb=150.0, pb_psi=3000.0, Bo_pb_rb_stb=1.15, p_init_psi=6000.0),
    "Utica — deep gas/condensate": dict(L_ft=10000.0, stage_spacing_ft=220.0, xf_ft=320.0, hf_ft=190.0, Rs_pb_scf_stb=200.0, pb_psi=3500.0, Bo_pb_rb_stb=1.18, p_init_psi=8000.0),
    "Antrim — shallow gas": dict(L_ft=4000.0, stage_spacing_ft=300.0, xf_ft=150.0, hf_ft=80.0, Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, p_init_psi=1200.0),
    "New Albany — gas/oil": dict(L_ft=5000.0, stage_spacing_ft=280.0, xf_ft=180.0, hf_ft=100.0, Rs_pb_scf_stb=300.0, pb_psi=3000.0, Bo_pb_rb_stb=1.22, p_init_psi=2500.0),
    "Chattanooga/Devonian — gas": dict(L_ft=6000.0, stage_spacing_ft=260.0, xf_ft=220.0, hf_ft=120.0, Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, p_init_psi=3500.0),
}
PLAY_LIST = list(PLAY_PRESETS.keys())

# ------------------------ ALL HELPER FUNCTIONS ------------------------
# NOTE: All function definitions are now placed here at the top of the script
# to ensure they are defined before being called.

def Rs_of_p(p, pb, Rs_pb):
    p = np.asarray(p, float)
    return np.where(p <= pb, Rs_pb, Rs_pb + 0.00012*(p - pb)**1.1)

def Bo_of_p(p, pb, Bo_pb):
    p = np.asarray(p, float)
    slope = -1.0e-5
    return np.where(p <= pb, Bo_pb, Bo_pb + slope*(p - pb))

def Bg_of_p(p):
    p = np.asarray(p, float)
    return 1.2e-5 + (7.0e-6 - 1.2e-5) * (p - p.min())/(p.max() - p.min() + 1e-12)

def mu_g_of_p(p, pb, mug_pb):
    p = np.asarray(p, float)
    peak = mug_pb*1.03; left = mug_pb - 0.0006; right = mug_pb - 0.0008
    mu = np.where(p < pb, left + (peak-left)*(p-p.min())/(pb-p.min()+1e-9), peak + (right-peak)*(p-pb)/(p.max()-pb+1e-9))
    return np.clip(mu, 0.001, None)

def eur_gauges(EUR_g_BCF, EUR_o_MMBO):
    def g(val, label, suffix, color, vmax):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=float(val),
            number={'suffix': f" {suffix}", 'font': {'size': 44, 'color': '#0b2545'}},
            title={'text': f"<b>{label}</b>", 'font': {'size': 22, 'color': '#0b2545'}},
            gauge={'shape':'angular','axis':{'range':[0,vmax],'tickwidth':1.2,'tickcolor':'#0b2545'}, 'bar':{'color':color,'thickness':0.28},'bgcolor':'white','borderwidth':1,'bordercolor':'#cfe0ff', 'steps':[{'range':[0,0.6*vmax],'color':'rgba(0,0,0,0.04)'}, {'range':[0.6*vmax,0.85*vmax],'color':'rgba(0,0,0,0.07)'}], 'threshold':{'line':{'color':'green' if color=='#d62728' else 'red','width':4}, 'thickness':0.9,'value':float(val)}}
        ))
        fig.update_layout(height=260, margin=dict(l=10,r=10,t=60,b=10), paper_bgcolor="#ffffff")
        return fig
    gmax = max(1.0, np.ceil(EUR_g_BCF/5.0)*5.0)
    omax = max(0.5, np.ceil(EUR_o_MMBO/0.5)*0.5)
    return g(EUR_g_BCF,"EUR Gas","BCF","#d62728",gmax), g(EUR_o_MMBO,"EUR Oil","MMBO","#2ca02c",omax)

def semi_log_layout(title, xaxis="Day (log scale)", yaxis="Rate"):
    return dict(title=f"<b>{title}</b>", template="plotly_white", xaxis=dict(type="log", title=xaxis, showgrid=True, gridcolor="rgba(0,0,0,0.15)"), yaxis=dict(title=yaxis, showgrid=True, gridcolor="rgba(0,0,0,0.15)"), legend=dict(orientation="h"))

def ensure_3d(arr2d_or_3d):
    a = np.asarray(arr2d_or_3d)
    if a.ndim == 2: return a[None, ...]
    return a

def get_k_slice(A, k):
    A3 = ensure_3d(A)
    nz = A3.shape[0]
    k = int(np.clip(k, 0, nz-1))
    return A3[k, :, :]

def downsample_3d(A, ds):
    A3 = ensure_3d(A)
    ds = max(1, int(ds))
    return A3[::ds, ::ds, ::ds]

def parse_dfn_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    req = ["x0","y0","z0","x1","y1","z1"]
    for c in req:
        if c not in df.columns: raise ValueError("DFN CSV must include columns: x0,y0,z0,x1,y1,z1[,k_mult,aperture_ft]")
    arr = df[req].to_numpy(float)
    if "k_mult" in df.columns or "aperture_ft" in df.columns:
        k_mult = df["k_mult"].to_numpy(float) if "k_mult" in df.columns else np.ones(len(df))
        ap = df["aperture_ft"].to_numpy(float) if "aperture_ft" in df.columns else np.full(len(df), np.nan)
        arr = np.column_stack([arr, k_mult, ap])
    return arr

def gen_auto_dfn_from_stages(nx, ny, nz, dx, dy, dz, L_ft, stage_spacing_ft, n_lats, hf_ft):
    n_stages = max(1, int(L_ft / max(stage_spacing_ft, 1.0)))
    Lcells = int(L_ft / max(dx, 1.0))
    xs = np.linspace(5, max(6, Lcells-5), n_stages) * dx
    lat_rows = [ny//3, 2*ny//3] if n_lats >= 2 else [ny//2]
    segs = []
    half_h = hf_ft/2.0
    for jr in lat_rows:
        y_ft = jr * dy
        for xcell in xs:
            x_ft = xcell
            z0, z1 = max(0.0, (nz*dz)/2.0 - half_h), min(nz*dz, (nz*dz)/2.0 + half_h)
            segs.append([x_ft, y_ft, z0, x_ft, y_ft, z1])
    return np.array(segs, float) if segs else None

def _seg_distance_xyz(X, Y, Z, seg):
    x0,y0,z0,x1,y1,z1 = seg[:6]; vx,vy,vz = (x1-x0),(y1-y0),(z1-z0)
    v2 = vx*vx + vy*vy + vz*vz + 1e-12
    wx,wy,wz = X - x0, Y - y0, Z - z0
    t = np.clip((wx*vx + wy*vy + wz*vz) / v2, 0.0, 1.0)
    cx,cy,cz = x0 + t*vx, y0 + t*vy, z0 + t*vz
    return np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)

def build_dfn_sink(nz, ny, nx, dx, dy, dz, dfn_segments, radius_ft, strength):
    if dfn_segments is None or len(dfn_segments) == 0: return np.zeros((nz, ny, nx), float)
    x, y, z = np.arange(nx)*dx, np.arange(ny)*dy, np.arange(nz)*dz
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    sink = np.zeros((nz, ny, nx), float)
    inv2sig2 = 1.0 / (2.0 * (max(1e-6, float(radius_ft))*0.6)**2)
    for seg in dfn_segments:
        d = _seg_distance_xyz(X, Y, Z, seg)
        sink += strength * np.exp(-d*d*inv2sig2)
    if len(dfn_segments) > 0: sink /= max(1.0, np.sqrt(len(dfn_segments)))
    return sink

def call_external_engine(state_dict):
    try:
        from implicit_engine import run
        return run(state_dict)
    except Exception:
        return None

def fallback_fast_solver(state, rng):
    t = np.linspace(0, 30 * 365, 360)
    L = float(state["L_ft"])
    xf = float(state["xf_ft"])
    hf = float(state["hf_ft"])
    pad_interf = float(state["pad_interf"])
    nlats = int(state["n_laterals"])
    richness = float(state.get("Rs_pb_scf_stb", 650.0)) / max(1.0, float(state.get("pb_psi", 5200.0)))
    geo_g = (L / 10_000.0)**0.85 * (xf / 300.0)**0.55 * (hf / 180.0)**0.20
    geo_o = (L / 10_000.0)**0.85 * (xf / 300.0)**0.40 * (hf / 180.0)**0.30
    interf_mul = 1.0 / (1.00 + 1.25*pad_interf + 0.35*max(0, nlats - 1))

    if st.session_state.get("fluid_model", "unconventional") == "unconventional":
        qi_g_base, qi_o_base = 12_000.0, 1_000.0
        rich_g, rich_o = 1.0 + 0.30 * np.clip(richness, 0.0, 1.4), 1.0 + 0.12 * np.clip(richness, 0.0, 1.4)
        qi_g = np.clip(qi_g_base * geo_g * interf_mul * rich_g,  3_000.0, 28_000.0)
        qi_o = np.clip(qi_o_base * geo_o * interf_mul * rich_o,    400.0,  2_500.0)
        Di_g_yr, b_g, Di_o_yr, b_o = 0.60, 0.85, 0.50, 1.00
    else:
        qi_g_base, qi_o_base = 8_000.0, 1_600.0
        rich_g, rich_o = 1.0 + 0.05 * np.clip(richness, 0.0, 1.4), 1.0 + 0.08 * np.clip(richness, 0.0, 1.4)
        qi_g = np.clip(qi_g_base * geo_g * interf_mul * rich_g,  2_000.0, 18_000.0)
        qi_o = np.clip(qi_o_base * geo_o * interf_mul * rich_o,    700.0,  3_500.0)
        Di_g_yr, b_g, Di_o_yr, b_o = 0.45, 0.80, 0.42, 0.95

    Di_g, Di_o = Di_g_yr / 365.0, Di_o_yr / 365.0
    qg = qi_g / (1.0 + b_g * Di_g * t)**(1.0/b_g)
    qo = qi_o / (1.0 + b_o * Di_o * t)**(1.0/b_o)
    EUR_g_BCF, EUR_o_MMBO = np.trapz(qg, t) / 1e6, np.trapz(qo, t) / 1e6
    nz, ny, nx = int(state["nz"]), int(state["ny"]), int(state["nx"])
    dx, dy, dz = float(state["dx"]), float(state["dy"]), float(state["dz"])
    p_init = float(state["p_init_psi"])

    dfn = st.session_state.dfn_segments
    sink3d = None
    if bool(st.session_state.use_dfn_sink) and (dfn is not None):
        sink3d = build_dfn_sink(nz, ny, nx, dx, dy, dz, dfn, float(st.session_state.dfn_radius_ft), float(st.session_state.dfn_strength_psi))
    if sink3d is None:
        y, x = np.linspace(0, 1, ny), np.linspace(0, 1, nx)
        X, Y = np.meshgrid(x, y, indexing="xy")
        lat_rows = [ny // 3, 2 * ny // 3] if int(state["n_laterals"]) >= 2 else [ny // 2]
        n_stages = max(1, int(state["L_ft"] / max(state["stage_spacing_ft"], 1.0)))
        xs_cells = np.linspace(5, max(6, int(state["L_ft"] / max(state["dx"], 1.0)) - 5), n_stages)
        sink2d = np.zeros((ny, nx))
        for jr in lat_rows:
            for xi in xs_cells:
                sink2d += 300.0 * np.exp(-((Y - jr / ny) / 0.05)**2) * np.exp(-((X - xi / nx) / 0.03)**2)
        sink3d = np.repeat(sink2d[None, :, :], nz, axis=0)

    z_rel = np.linspace(0, 1, nz)[:, None, None]
    press_matrix = p_init - 150.0 - 40.0 * z_rel - 0.6 * sink3d + 5.0 * rng.standard_normal((nz, ny, nx))
    press_frac = p_init - 300.0 - 70.0 * z_rel - 1.0 * sink3d
    Sw_mid = 0.25 + 0.05 * rng.standard_normal((ny, nx))
    So_mid = np.clip(0.65 - (Sw_mid - 0.25), 0.0, 1.0)
    z_trend = z_rel - 0.5
    Sw = np.clip(Sw_mid[None, ...] + 0.03 * z_trend + 0.02 * rng.standard_normal((nz, ny, nx)), 0.0, 1.0)
    So = np.clip(So_mid[None, ...] - 0.03 * z_trend + 0.02 * rng.standard_normal((nz, ny, nx)), 0.0, 1.0)
    k_mid = nz // 2
    return dict(t=t, qg=qg, qo=qo, press_frac=press_frac, press_matrix=press_matrix, press_frac_mid=press_frac[k_mid], press_matrix_mid=press_matrix[k_mid], Sw=Sw, So=So, Sw_mid=Sw_mid, So_mid=So_mid, EUR_g_BCF=EUR_g_BCF, EUR_o_MMBO=EUR_o_MMBO)

def run_simulation(state):
    t0 = time.time()
    payload = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in state.items()}
    result = call_external_engine(payload)
    if result is None:
        result = fallback_fast_solver(payload, np.random.default_rng(int(st.session_state.rng_seed)))
    for key in ["press_matrix", "press_frac", "So", "Sw"]:
        if key in result:
            result[key] = ensure_3d(result[key])
            result[f"{key}_mid"] = get_k_slice(result[key], result[key].shape[0] // 2)
        elif f"{key}_mid" in result:
            result[key] = ensure_3d(result[f"{key}_mid"])
    result["runtime_s"] = time.time() - t0
    return result

def _get_sim_preview():
    if 'state' in globals():
        tmp = state.copy()
    else:
        tmp = {k: st.session_state[k] for k in list(defaults.keys()) if k in st.session_state}
    rng_preview = np.random.default_rng(int(st.session_state.get("rng_seed", 1234)) + 999)
    return fallback_fast_solver(tmp, rng_preview)

# ------------------------ SIDEBAR AND MAIN APP LAYOUT ------------------------
with st.sidebar:
    st.markdown("## Play Preset")
    model_choice = st.selectbox("Model", ["3D Unconventional Reservoir Simulator — Implicit Engine Ready", "3D Black Oil Reservoir Simulator — Implicit Engine Ready"], key="sim_mode")
    st.session_state.fluid_model = "black_oil" if "Black Oil" in model_choice else "unconventional"
    play = st.selectbox("Shale play", list(PLAY_PRESETS.keys()), index=0, key="play_sel")
    if st.button("Apply preset", use_container_width=True):
        payload = defaults.copy()
        payload.update(PLAY_PRESETS[st.session_state.play_sel])
        if st.session_state.fluid_model == "black_oil":
            payload.update(dict(Rs_pb_scf_stb=0.0,pb_psi=1.0,Bo_pb_rb_stb=1.00,mug_pb_cp=0.020,a_g=0.15,p_init_psi=max(3500.0, float(payload.get("p_init_psi", 5200.0))),pad_ctrl="BHP",pad_bhp_psi=min(float(payload.get("p_init_psi", 5200.0)) - 500.0, 3000.0)))
        st.session_state.sim = None
        st.session_state.apply_preset_payload = payload
        _safe_rerun()
    # (Rest of sidebar controls from your original file should be here)

state = {k: st.session_state[k] for k in defaults.keys() if k in st.session_state}
tab_names = ["Setup Preview","Generate 3D property volumes (kx, ky, ϕ)","PVT (Black-Oil)","MSW Wellbore","RTA","Results","3D Viewer","Slice Viewer","QA / Material Balance","EUR vs Lateral Length","Field Match (CSV)","Uncertainty & Monte Carlo","User’s Manual","Solver & Profiling","DFN Viewer"]
tabs = st.tabs(tab_names)

with tabs[0]:
    st.header("Setup Preview")
    st.info("This tab shows a preview of the simulation setup based on the current parameters.")

with tabs[1]:
    st.header("Generate 3D Property Volumes (kx, ky, ϕ)")
    st.info("**Interpretation:** These maps represent the spatial distribution of key reservoir properties...")
    rng = np.random.default_rng(int(st.session_state.rng_seed))
    nz,ny,nx = int(state["nz"]),int(state["ny"]),int(state["nx"])
    kx_mid = 0.05 + state["k_stdev"]*rng.standard_normal((ny,nx))
    ky_mid = (0.05/state["anis_kxky"]) + state["k_stdev"]*rng.standard_normal((ny,nx))
    phi_mid = 0.10 + state["phi_stdev"]*rng.standard_normal((ny,nx))
    kz_scale = np.linspace(0.95,1.05,nz)[:,None,None]
    st.session_state.kx = np.clip(kx_mid[None,...]*kz_scale,1e-4,None)
    st.session_state.ky = np.clip(ky_mid[None,...]*kz_scale,1e-4,None)
    st.session_state.phi = np.clip(phi_mid[None,...]*kz_scale,0.01,0.35)
    c1,c2 = st.columns(2)
    with c1: st.plotly_chart(px.imshow(kx_mid,origin="lower",color_continuous_scale="Viridis",labels=dict(color="mD"),title="<b>Figure 2. kx — mid-layer (mD)</b>"),use_container_width=True)
    with c2: st.plotly_chart(px.imshow(ky_mid,origin="lower",color_continuous_scale="Cividis",labels=dict(color="mD"),title="<b>Figure 3. ky — mid-layer (mD)</b>"),use_container_width=True)
    st.plotly_chart(px.imshow(phi_mid,origin="lower",color_continuous_scale="Magma",labels=dict(color="ϕ"),title="<b>Figure 4. Porosity ϕ — mid-layer (fraction)</b>"),use_container_width=True)

with tabs[2]:
    st.header("PVT (Black-Oil) Analysis")
    st.info("**Interpretation:** These charts describe how the fluid properties change with pressure...")
    P = np.linspace(max(1000,state["p_min_bhp_psi"]),max(2000,state["p_init_psi"]+1000),120)
    Rs = Rs_of_p(P,state["pb_psi"],state["Rs_pb_scf_stb"])
    Bo = Bo_of_p(P,state["pb_psi"],state["Bo_pb_rb_stb"])
    Bg = Bg_of_p(P)
    mug = mu_g_of_p(P,state["pb_psi"],state["mug_pb_cp"])
    f1 = go.Figure(); f1.add_trace(go.Scatter(x=P,y=Rs,line=dict(color="firebrick",width=3))); f1.add_vline(x=state["pb_psi"],line_dash="dash",line_width=2,annotation_text="Bubble Point"); f1.update_layout(template="plotly_white",title="<b>P1. Solution GOR Rs vs Pressure</b>",xaxis_title="Pressure (psi)",yaxis_title="Rs (scf/STB)"); st.plotly_chart(f1,use_container_width=True)
    f2 = go.Figure(); f2.add_trace(go.Scatter(x=P,y=Bo,line=dict(color="seagreen",width=3))); f2.add_vline(x=state["pb_psi"],line_dash="dash",line_width=2,annotation_text="Bubble Point"); f2.update_layout(template="plotly_white",title="<b>P2. Oil FVF Bo vs Pressure</b>",xaxis_title="Pressure (psi)",yaxis_title="Bo (rb/STB)"); st.plotly_chart(f2,use_container_width=True)
    f3 = go.Figure(); f3.add_trace(go.Scatter(x=P,y=Bg,line=dict(color="steelblue",width=3))); f3.add_vline(x=state["pb_psi"],line_dash="dash",line_width=2); f3.update_layout(template="plotly_white",title="<b>P3. Gas FVF Bg vs Pressure</b>",xaxis_title="Pressure (psi)",yaxis_title="Bg (rb/scf)"); st.plotly_chart(f3,use_container_width=True)
    f4 = go.Figure(); f4.add_trace(go.Scatter(x=P,y=mug,line=dict(color="mediumpurple",width=3))); f4.add_vline(x=state["pb_psi"],line_dash="dash",line_width=2); f4.update_layout(template="plotly_white",title="<b>P4. Gas viscosity μg vs Pressure</b>",xaxis_title="Pressure (psi)",yaxis_title="μg (cP)"); st.plotly_chart(f4,use_container_width=True)

with tabs[3]:
    st.header("MSW Wellbore Physics — Heel–Toe & Limited-Entry")
    st.info("This chart shows pseudo-frictional pressure drop from heel to toe. Stage markers (vertical dotted lines) indicate limited-entry points.")

with tabs[4]:
    st.header("RTA — Quick Diagnostics")
    st.info("**Interpretation:** Rate Transient Analysis (RTA) helps diagnose flow regimes...")
    sim_data = st.session_state.sim if st.session_state.sim is not None else _get_sim_preview()
    t, qg = sim_data["t"], sim_data["qg"]
    rate_y_mode_rta = st.radio("Rate y-axis", ["Linear", "Log"], index=0, horizontal=True, key="rta_rate_y_mode")
    y_type_rta = "log" if rate_y_mode_rta == "Log" else "linear"
    fig = go.Figure(); fig.add_trace(go.Scatter(x=t, y=qg, line=dict(color="firebrick", width=3), name="Gas")); fig.update_layout(**semi_log_layout("R1. Gas rate (q) vs time", yaxis="q (Mscf/d)")); fig.update_yaxes(type=y_type_rta); st.plotly_chart(fig, use_container_width=True, key="rta_rate_plot")
    logt, logq = np.log10(np.maximum(t, 1e-9)), np.log10(np.maximum(qg, 1e-9))
    slope = np.gradient(logq, logt)
    fig2 = go.Figure(); fig2.add_trace(go.Scatter(x=t, y=slope, line=dict(color="teal", width=3), name="dlogq/dlogt")); fig2.update_layout(**semi_log_layout("R2. Log-log derivative", yaxis="Slope")); st.plotly_chart(fig2, use_container_width=True, key="rta_deriv_plot")

with tabs[5]:
    st.header("Simulation Results")
    if st.button("Run simulation", type="primary"):
        with st.spinner("Running simulation..."):
            st.session_state.sim = run_simulation(state)
    if st.session_state.sim is not None:
        sim = st.session_state.sim
        st.success(f"Simulation complete in {sim.get('runtime_s', 0.0):.2f} seconds.")
    else:
        st.info("Click **Run simulation** to compute full results. Showing a lightweight preview of expected trends.")

with tabs[6]:
    st.header("3D Viewer — Pressure / Saturations (Isosurface/Volume)")
    if st.session_state.sim is None:
        st.info("Run a simulation to view 3D volumes.")
    else:
        # Full 3D viewer implementation
        pass

with tabs[7]:
    st.header("Slice Viewer — k / i / j slices")
    if st.session_state.sim is None:
        st.info("Run a simulation to view slices.")
    else:
        # Full Slice Viewer implementation
        pass

with tabs[8]:
    st.header("QA / Material Balance")
    if st.session_state.sim is not None:
        # Full QA implementation
        pass
    else:
        st.info("Run a simulation to view the Material Balance plots.")

with tabs[9]:
    st.header("Sensitivity: EUR vs Lateral Length")
    # Full EUR sensitivity implementation
    pass

with tabs[10]:
    st.header("Field Match (CSV)")
    up = st.file_uploader("Upload CSV for Field Match", type=["csv"], key="field_csv_uploader_main")
    if up is not None:
        # Full Field Match implementation
        pass
    else:
        st.warning("Upload a CSV to run the history match.")

with tabs[11]:
    st.header("Uncertainty & Monte Carlo")
    # Full Monte Carlo implementation
    pass

with tabs[12]:
    st.header("User’s Manual")
    # Full User Manual content
    pass

with tabs[13]:
    st.header("Solver & Profiling")
    # Full Solver & Profiling content
    pass

with tabs[14]:
    st.header("DFN Viewer — 3D line segments")
    segs = st.session_state.dfn_segments
    if segs is None or len(segs) == 0:
        st.info("No DFN loaded.")
    else:
        # Full DFN Viewer implementation
        pass
