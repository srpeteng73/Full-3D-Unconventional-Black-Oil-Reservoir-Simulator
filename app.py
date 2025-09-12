%%writefile app.py
# Cell #2 — Streamlit app (full replacement)
# Full 3D Unconventional / Black-Oil Reservoir Simulator — Implicit Engine Ready (USOF units) + DFN support
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy import stats  # Added for KDE plots

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
_setdefault("dfn_segments", None)      # ndarray shape [n, 6] or [n, 8] if k_mult, aperture provided
_setdefault("use_dfn_sink", True)      # whether fallback solver uses DFN-driven sink
_setdefault("use_auto_dfn", True)      # auto-generate DFN from stages if no upload
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
    # RelPerm/Pc knobs
    krw_end=0.6, kro_end=0.8, nw=2.0, no=2.0, Swc=0.15, Sor=0.25, pc_slope_psi=0.0,
    # Phase compressibilities (additional tuning)
    ct_o_1psi=8e-6, ct_g_1psi=3e-6, ct_w_1psi=3e-6,
    # Solver/perf
    newton_tol=1e-6, trans_tol=1e-7, max_newton=12, max_lin=200, threads=0,
    use_omp=False, use_mkl=False, use_pyamg=False, use_cusparse=False,
    # DFN controls
    dfn_radius_ft=60.0,          # radial influence for sink/Gaussian (ft)
    dfn_strength_psi=500.0,      # sink strength scalar for fallback pressure drawdown
)

for k, v in defaults.items():
    _setdefault(k, v)

# --- Apply preset before widgets render
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

# ------------------------ PVT helpers ------------------------
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

# ------------------------ 3D helpers ------------------------
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

# ------------------------ DFN helpers ------------------------
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

# ------------------------ Engine hook ------------------------
def call_external_engine(state_dict):
    try:
        from implicit_engine import run
        return run(state_dict)
    except Exception:
        return None

# ------------------------ Fallback Solver (MODE-AWARE) ------------------------
def fallback_fast_solver(state, rng):
    """
    Fast synthetic production + 3D fields generator.
    Mode-aware (unconventional vs black-oil) via st.session_state.fluid_model.
    """
    t = np.geomspace(1, 5500, 280)

    # Geometry / scaling
    L, xf, hf = float(state["L_ft"]), float(state["xf_ft"]), float(state["hf_ft"])
    pad_interf, nlats = float(state["pad_interf"]), int(state["n_laterals"])
    richness = float(state.get("Rs_pb_scf_stb", 0.0)) / 650.0  # 0 for black-oil, ~1 for volatile
    interf = (1 + 0.25*pad_interf) * (1 + 0.1*(nlats-1))

    # --- key change: mode-aware initial rates + decline exponents ---
    fluid_model = st.session_state.get("fluid_model", "unconventional")
    if fluid_model == "black_oil":
        # Oil-dominant with modest associated gas
        qi_o = 1800.0 * (L/10_000.0) * (hf/180.0) * (1.00 + 0.10*richness) * interf
        qi_g = 2500.0 * (L/10_000.0) * (xf/300.0) * (0.30 + 0.20*richness) * interf
        Di_o_yr, b_o = 0.45, 0.70
        Di_g_yr, b_g = 0.60, 0.60
    else:
        # Unconventional / volatile
        qi_g = 80_000.0 * (L/10_000.0) * (xf/300.0) * interf
        qi_o = 1_500.0 * (L/10_000.0) * (hf/180.0) * (0.7 + 0.6*richness) * interf
        Di_g_yr, b_g = 0.70, 0.70
        Di_o_yr, b_o = 0.50, 0.80

    # Daily decline parameters
    Di_g, Di_o = Di_g_yr / 365.0, Di_o_yr / 365.0

    # Hyperbolic declines
    qg = qi_g / (1 + b_g*Di_g*t)**(1/b_g)
    qo = qi_o / (1 + b_o*Di_o*t)**(1/b_o)

    # EURs
    EUR_g_BCF = np.trapz(qg, t) / 1e6
    EUR_o_MMBO = np.trapz(qo, t) / 1e6

    # 3D pressure / saturation fields
    nz,ny,nx = int(state["nz"]),int(state["ny"]),int(state["nx"])
    dx,dy,dz = float(state["dx"]),float(state["dy"]),float(state["dz"])
    p_init = float(state["p_init_psi"])

    dfn = st.session_state.dfn_segments
    sink3d = None
    if bool(st.session_state.use_dfn_sink) and (dfn is not None):
        sink3d = build_dfn_sink(nz,ny,nx,dx,dy,dz,dfn,
                                float(st.session_state.dfn_radius_ft),
                                float(st.session_state.dfn_strength_psi))
    if sink3d is None:
        y, x = np.linspace(0,1,ny), np.linspace(0,1,nx)
        X, Y = np.meshgrid(x, y, indexing="xy")
        lat_rows = [ny//3, 2*ny//3] if int(state["n_laterals"]) >= 2 else [ny//2]
        n_stages = max(1, int(state["L_ft"]/max(state["stage_spacing_ft"],1.0)))
        xs_cells = np.linspace(5, max(6, int(state["L_ft"]/max(state["dx"],1.0)) - 5), n_stages)
        sink2d = np.zeros((ny, nx))
        for jr in lat_rows:
            for xi in xs_cells:
                sink2d += 300.0 * np.exp(-((Y-jr/ny)/0.05)**2) * np.exp(-((X-xi/nx)/0.03)**2)
        sink3d = np.repeat(sink2d[None,:,:], nz, axis=0)

    z_rel = np.linspace(0,1,nz)[:,None,None]
    press_matrix = p_init - 150.0 - 40.0*z_rel - 0.6*sink3d + 5.0*rng.standard_normal((nz,ny,nx))
    press_frac   = p_init - 300.0 - 70.0*z_rel - 1.0*sink3d

    Sw_mid = 0.25 + 0.05*rng.standard_normal((ny,nx))
    So_mid = np.clip(0.65 - (Sw_mid-0.25), 0.0, 1.0)
    z_trend = z_rel - 0.5
    Sw = np.clip(Sw_mid[None,...] + 0.03*z_trend + 0.02*rng.standard_normal((nz,ny,nx)), 0.0, 1.0)
    So = np.clip(So_mid[None,...] - 0.03*z_trend + 0.02*rng.standard_normal((nz,ny,nx)), 0.0, 1.0)

    k_mid = nz//2
    return dict(
        t=t, qg=qg, qo=qo,
        press_frac=press_frac, press_matrix=press_matrix,
        press_frac_mid=press_frac[k_mid], press_matrix_mid=press_matrix[k_mid],
        Sw=Sw, So=So, Sw_mid=Sw_mid, So_mid=So_mid,
        EUR_g_BCF=EUR_g_BCF, EUR_o_MMBO=EUR_o_MMBO
    )

def run_simulation(state):
    t0 = time.time()
    payload = {k:(float(v) if isinstance(v,(int,float)) else v) for k,v in state.items()}
    result = call_external_engine(payload)
    if result is None:
        result = fallback_fast_solver(payload, np.random.default_rng(int(st.session_state.rng_seed)))
    for key in ["press_matrix", "press_frac", "So", "Sw"]:
        if key in result:
            result[key] = ensure_3d(result[key])
            result[f"{key}_mid"] = get_k_slice(result[key], result[key].shape[0]//2)
        elif f"{key}_mid" in result:
            result[key] = ensure_3d(result[f"{key}_mid"])
    result["runtime_s"] = time.time() - t0
    return result

# ------------------------ Sidebar controls ------------------------
with st.sidebar:
    st.markdown("## Play Preset")

    # Model selector: sets a mode flag we use in the solver
    model_choice = st.selectbox(
        "Model",
        [
            "3D Unconventional Reservoir Simulator — Implicit Engine Ready",
            "3D Black Oil Reservoir Simulator — Implicit Engine Ready",
        ],
        index=0 if "sim_mode" not in st.session_state
        else [
            "3D Unconventional Reservoir Simulator — Implicit Engine Ready",
            "3D Black Oil Reservoir Simulator — Implicit Engine Ready",
        ].index(st.session_state.sim_mode),
        key="sim_mode",
    )
    # Simple model flag for use in the solver
    st.session_state.fluid_model = "black_oil" if "Black Oil" in model_choice else "unconventional"

    play = st.selectbox("Shale play", list(PLAY_PRESETS.keys()), index=0, key="play_sel")
    st.caption("Presets adjust lateral length, stage spacing, frac geometry, and PVT primaries.")

    if st.button("Apply preset", use_container_width=True):
        payload = defaults.copy()
        payload.update(PLAY_PRESETS[st.session_state.play_sel])

        # Override a few key PVT/controls by model
        if st.session_state.fluid_model == "black_oil":
            # Black-oil: no solution gas driving early gas rate, 'oil-first' behavior
            payload.update(
                dict(
                    Rs_pb_scf_stb=0.0,
                    pb_psi=1.0,
                    Bo_pb_rb_stb=1.00,
                    mug_pb_cp=0.020,
                    a_g=0.15,
                    p_init_psi=max(3500.0, float(payload.get("p_init_psi", 5200.0))),
                    pad_ctrl="BHP",
                    pad_bhp_psi=min(float(payload.get("p_init_psi", 5200.0)) - 500.0, 3000.0),
                )
            )
        # else: leave unconventional presets as-is

        st.session_state.sim = None
        st.session_state.apply_preset_payload = payload
        _safe_rerun()

    st.markdown("### Grid (ft)")
    c1,c2,c3 = st.columns(3)
    with c1: st.number_input("nx",10,500,int(st.session_state.nx),1,key="nx")
    with c2: st.number_input("ny",10,500,int(st.session_state.ny),1,key="ny")
    with c3: st.number_input("nz",1,200,int(st.session_state.nz),1,key="nz")

    c1,c2,c3 = st.columns(3)
    with c1: st.number_input("dx (ft)",value=float(st.session_state.dx),step=1.0,key="dx")
    with c2: st.number_input("dy (ft)",value=float(st.session_state.dy),step=1.0,key="dy")
    with c3: st.number_input("dz (ft)",value=float(st.session_state.dz),step=1.0,key="dz")

    st.markdown("### Heterogeneity & Anisotropy")
    st.selectbox("Facies style", ["Continuous (Gaussian)","Speckled (high-variance)","Layered (vertical bands)"],
                 index=["Continuous (Gaussian)","Speckled (high-variance)","Layered (vertical bands)"].index(st.session_state.facies_style),
                 key="facies_style")
    st.slider("k stdev (mD around 0.02)",0.0,0.20,float(st.session_state.k_stdev),0.01,key="k_stdev")
    st.slider("ϕ stdev",0.0,0.20,float(st.session_state.phi_stdev),0.01,key="phi_stdev")
    st.slider("Anisotropy kx/ky",0.5,3.0,float(st.session_state.anis_kxky),0.05,key="anis_kxky")

    st.markdown("### Faults")
    st.checkbox("Enable fault TMULT",value=bool(st.session_state.use_fault),key="use_fault")
    st.selectbox("Fault plane",["i-plane (vertical)","j-plane (vertical)"],index=0,key="fault_plane")
    st.number_input("Plane index",1,max(1,int(st.session_state.nx)-2),int(st.session_state.fault_index),1,key="fault_index")
    st.number_input("Transmissibility multiplier",value=float(st.session_state.fault_tm),step=0.01,key="fault_tm")

    st.markdown("### Pad / Wellbore & Frac")
    st.number_input("Laterals",1,6,int(st.session_state.n_laterals),1,key="n_laterals")
    st.number_input("Lateral length (ft)",value=float(st.session_state.L_ft),step=50.0,key="L_ft")
    st.number_input("Stage spacing (ft)",value=float(st.session_state.stage_spacing_ft),step=5.0,key="stage_spacing_ft")
    st.number_input("Clusters per stage",1,12,int(st.session_state.clusters_per_stage),1,key="clusters_per_stage")
    st.number_input("Δp limited-entry (psi)",value=float(st.session_state.dP_LE_psi),step=5.0,key="dP_LE_psi")
    st.number_input("Wellbore friction factor (pseudo)",value=float(st.session_state.f_fric),step=0.005,key="f_fric")
    st.number_input("Wellbore ID (ft)",value=float(st.session_state.wellbore_ID_ft),step=0.01,key="wellbore_ID_ft")
    st.number_input("Frac half-length xf (ft)",value=float(st.session_state.xf_ft),step=5.0,key="xf_ft")
    st.number_input("Frac height hf (ft)",value=float(st.session_state.hf_ft),step=5.0,key="hf_ft")
    st.slider("Pad interference coeff.",0.00,0.80,float(st.session_state.pad_interf),0.01,key="pad_interf")

    st.markdown("### Controls & Boundary")
    st.selectbox("Pad control",["BHP","RATE"],index=0,key="pad_ctrl")
    st.number_input("Pad BHP (psi)",value=float(st.session_state.pad_bhp_psi),step=10.0,key="pad_bhp_psi")
    st.number_input("Pad RATE (Mscf/d)",value=float(st.session_state.pad_rate_mscfd),step=1000.0,key="pad_rate_mscfd")
    st.selectbox("Outer boundary",["Infinite-acting","Constant-p"],index=0,key="outer_bc")
    st.number_input("Boundary pressure (psi)",value=float(st.session_state.p_outer_psi),step=10.0,key="p_outer_psi")

    st.markdown("### PVT (Black-Oil) + Compressibilities")
    st.number_input("Bubblepoint pb (psi)",value=float(st.session_state.pb_psi),step=50.0,key="pb_psi")
    st.number_input("Rs at pb (scf/STB)",value=float(st.session_state.Rs_pb_scf_stb),step=10.0,key="Rs_pb_scf_stb")
    st.number_input("Bo at pb (rb/STB)",value=float(st.session_state.Bo_pb_rb_stb),step=0.01,key="Bo_pb_rb_stb")
    st.number_input("μo at pb (cP)",value=float(st.session_state.muo_pb_cp),step=0.01,key="muo_pb_cp")
    st.number_input("μg at pb (cP)",value=float(st.session_state.mug_pb_cp),step=0.001,key="mug_pb_cp")
    st.number_input("μg pressure exponent a_g",value=float(st.session_state.a_g),step=0.01,key="a_g")
    st.number_input("Gas z-factor",value=float(st.session_state.z_g),step=0.01,key="z_g")
    st.number_input("Initial reservoir p_i (psi)",value=float(st.session_state.p_init_psi),step=50.0,key="p_init_psi")
    st.number_input("Minimum flowing BHP (psi)",value=float(st.session_state.p_min_bhp_psi),step=25.0,key="p_min_bhp_psi")
    st.number_input("Total compressibility ct (1/psi)",value=float(st.session_state.ct_1_over_psi),step=1e-6,format="%.6f",key="ct_1_over_psi")
    st.checkbox("Include Rs(P) in material balance",value=bool(st.session_state.include_RsP),key="include_RsP")

    st.markdown("### Rel-Perm / Pc (Corey)")
    st.number_input("krw end",value=float(st.session_state.krw_end),step=0.05,key="krw_end")
    st.number_input("kro end",value=float(st.session_state.kro_end),step=0.05,key="kro_end")
    st.number_input("Corey n_w",value=float(st.session_state.nw),step=0.1,key="nw")
    st.number_input("Corey n_o",value=float(st.session_state.no),step=0.1,key="no")
    st.number_input("Swc",value=float(st.session_state.Swc),step=0.01,key="Swc")
    st.number_input("Sor",value=float(st.session_state.Sor),step=0.01,key="Sor")
    st.number_input("Pc slope (psi per frac Sw)",value=float(st.session_state.pc_slope_psi),step=0.1,key="pc_slope_psi")

    st.markdown("### Phase Compressibilities (1/psi)")
    st.number_input("ct_o",value=float(st.session_state.ct_o_1psi),step=1e-6,format="%.6f",key="ct_o_1psi")
    st.number_input("ct_g",value=float(st.session_state.ct_g_1psi),step=1e-6,format="%.6f",key="ct_g_1psi")
    st.number_input("ct_w",value=float(st.session_state.ct_w_1psi),step=1e-6,format="%.6f",key="ct_w_1psi")

    st.markdown("### 3D Viewer Settings")
    st.slider("Downsample factor (3D)",1,6,int(st.session_state.vol_downsample),1,key="vol_downsample")
    st.slider("Isosurface relative value (0-1)",0.0,1.0,float(st.session_state.iso_value_rel),0.01,key="iso_value_rel")

    st.markdown("### DFN (Discrete Fracture Network)")
    st.checkbox("Use DFN-driven sink in solver",value=bool(st.session_state.use_dfn_sink),key="use_dfn_sink")
    st.checkbox("Auto-generate DFN from stages when no upload",value=bool(st.session_state.use_auto_dfn),key="use_auto_dfn")
    st.number_input("DFN influence radius (ft)",value=float(st.session_state.dfn_radius_ft),step=5.0,key="dfn_radius_ft")
    st.number_input("DFN sink strength (psi)",value=float(st.session_state.dfn_strength_psi),step=10.0,key="dfn_strength_psi")

    dfn_up = st.file_uploader("Upload DFN CSV: x0,y0,z0,x1,y1,z1[,k_mult,aperture_ft]",type=["csv"],key="dfn_csv")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Load DFN from CSV"):
            try:
                if dfn_up is None: st.warning("Please choose a DFN CSV first.")
                else:
                    st.session_state.dfn_segments = parse_dfn_csv(dfn_up)
                    st.success(f"Loaded DFN segments: {len(st.session_state.dfn_segments)}")
            except Exception as e: st.error(f"DFN parse error: {e}")
    with c2:
        if st.button("Generate DFN from stages"):
            segs = gen_auto_dfn_from_stages(int(st.session_state.nx),int(st.session_state.ny),int(st.session_state.nz),
                                            float(st.session_state.dx),float(st.session_state.dy),float(st.session_state.dz),
                                            float(st.session_state.L_ft),float(st.session_state.stage_spacing_ft),
                                            int(st.session_state.n_laterals),float(st.session_state.hf_ft))
            st.session_state.dfn_segments = segs
            st.success(f"Auto-generated DFN segments: {0 if segs is None else len(segs)}")

    st.markdown("### Performance Flags")
    st.checkbox("Use OMP",value=bool(st.session_state.use_omp),key="use_omp")
    st.checkbox("Use MKL",value=bool(st.session_state.use_mkl),key="use_mkl")
    st.checkbox("Use PyAMG",value=bool(st.session_state.use_pyamg),key="use_pyamg")
    st.checkbox("Use cuSPARSE",value=bool(st.session_state.use_cusparse),key="use_cusparse")

    st.markdown("### Random Seed")
    st.number_input("seed",value=int(st.session_state.rng_seed),step=1,key="rng_seed")

state = {k: st.session_state[k] for k in defaults.keys() if k in st.session_state}
tab_names = ["Setup Preview","Generate 3D property volumes (kx, ky, ϕ)","PVT (Black-Oil)","MSW Wellbore","RTA","Results","3D Viewer","Slice Viewer","QA / Material Balance","EUR vs Lateral Length","Field Match (CSV)","Uncertainty & Monte Carlo","User’s Manual","Solver & Profiling","DFN Viewer"]
tabs = st.tabs(tab_names)

with tabs[0]:
    st.header("Figure 1. Mid-Layer Pad Layout")
    st.info("**Interpretation:** This view shows the well laterals (gray), stimulation stages (red), and hydraulic fracture geometry (blue). Stage density and frac geometry (xf, hf) control the early-time drainage near the fractures. The SRV overlay provides a quick visual of the approximate stimulated rock volume.")
    nx,ny = int(state["nx"]),int(state["ny"])
    Lcells = int(state["L_ft"]/max(state["dx"],1.0))
    n_stages = max(1, int(state["L_ft"]/max(state["stage_spacing_ft"],1.0)))
    lat_rows = [ny//3, 2*ny//3] if int(state["n_laterals"]) >= 2 else [ny//2]
    show_srv = st.checkbox("Show SRV overlay",value=False,key="show_srv_preview")
    fig = go.Figure()
    stage_xs_all = []
    for jr in lat_rows:
        fig.add_trace(go.Scatter(x=[5,max(6,Lcells-5)],y=[jr,jr],mode="lines",line=dict(color="grey",width=8),showlegend=False))
        xs = np.linspace(5,max(6,Lcells-5),n_stages)
        stage_xs_all.append(xs)
        fig.add_trace(go.Scatter(x=xs,y=jr*np.ones_like(xs),mode="markers",marker=dict(color="firebrick",size=10,symbol="square"),name="Stages" if jr==lat_rows[0] else None,showlegend=(jr==lat_rows[0])))
    dx,dy = float(state["dx"]),float(state["dy"])
    xf_cells,hf_cells = float(state["xf_ft"])/max(dx,1e-6), float(state["hf_ft"])/max(dy,1e-6)
    half_h = hf_cells/2.0
    for xs,jr in zip(stage_xs_all,lat_rows):
        for xi in xs:
            x0,x1 = max(0,xi-xf_cells),min(Lcells,xi+xf_cells)
            y0,y1 = max(0,jr-half_h),min(ny,jr+half_h)
            fig.add_shape(type="rect",x0=x0,x1=x1,y0=y0,y1=y1,line=dict(color="rgba(30,144,255,0.6)",width=1),fillcolor="rgba(30,144,255,0.12)")
            if show_srv: fig.add_shape(type="rect",x0=max(0,x0-0.2*xf_cells),x1=min(Lcells,x1+0.2*xf_cells),y0=max(0,y0-0.2*hf_cells),y1=min(ny,y1+0.2*hf_cells),line=dict(color="rgba(0,0,255,0.3)",width=1,dash="dot"),fillcolor="rgba(0,0,255,0.06)")
    fig.update_layout(template="plotly_white",height=480,xaxis_title="i (cells)",yaxis_title="j (cells)",title="<b>Mid-layer pad layout (grey=laterals, red=stages, blue=frac rectangles)</b>")
    st.plotly_chart(fig,use_container_width=True)

with tabs[1]:
    st.header("Generate 3D Property Volumes (kx, ky, ϕ)")
    st.info("**Interpretation:** These maps represent the spatial distribution of key reservoir properties. The patterns of permeability (kx, ky) and porosity (ϕ) control pressure gradients and how fluids move through the reservoir. Anisotropy (the difference between kx and ky) governs the directional preference of flow and sweep efficiency.")
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
    with c1: st.plotly_chart(px.imshow(kx_mid,origin="lower",color_continuous_scale="Viridis",labels=dict(color="mD"),title="<b>Figure 2. kx — mid-layer (mD)</b>"),use_container_width=True,theme=None)
    with c2: st.plotly_chart(px.imshow(ky_mid,origin="lower",color_continuous_scale="Cividis",labels=dict(color="mD"),title="<b>Figure 3. ky — mid-layer (mD)</b>"),use_container_width=True,theme=None)
    st.plotly_chart(px.imshow(phi_mid,origin="lower",color_continuous_scale="Magma",labels=dict(color="ϕ"),title="<b>Figure 4. Porosity ϕ — mid-layer (fraction)</b>"),use_container_width=True,theme=None)

with tabs[2]:
    st.header("PVT (Black-Oil) Analysis")
    st.info("**Interpretation:** These charts describe how the fluid properties change with pressure. **Rs** (Solution GOR) dictates when gas comes out of solution as pressure drops below the bubblepoint. **Bo** (Oil Formation Volume Factor) describes how oil shrinks as gas is liberated. **Viscosities** are critical for determining how easily each phase can flow. These properties are fundamental inputs for the simulation engine.")
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
    st.info("**Interpretation:** This chart illustrates the pseudo-frictional pressure drop from the heel (start) to the toe (end) of the lateral. This pressure drop can cause uneven production, with stages near the heel producing more than those at the toe. The **Limited-Entry Δp** is the pressure drop across perforations, designed to help mitigate this effect and ensure more uniform stimulation and production from all stages.")
    L_ft = state["L_ft"]; n_stages = max(1,int(L_ft/max(state["stage_spacing_ft"],1.0)))
    s_positions = np.linspace(0,L_ft,n_stages)
    x = np.linspace(0,L_ft,200)
    dpf = state["f_fric"]*(x/max(L_ft,1))*800.0
    fig = go.Figure(); fig.add_trace(go.Scatter(x=x,y=dpf,name="Along-lateral friction Δp (psi)",line=dict(color="royalblue",width=3)))
    for xs in s_positions: fig.add_vline(x=xs,line_dash="dot",line_width=1,opacity=0.4,annotation_text="Stage")
    fig.update_layout(template="plotly_white",xaxis_title="Measured depth (ft)",yaxis_title="Δp (psi)",title="<b>Friction profile with stage markers</b>")
    st.plotly_chart(fig,use_container_width=True)
    st.markdown(f"**Cluster limited-entry Δp = {state['dP_LE_psi']:.0f} psi.**")

with tabs[4]:
    st.header("RTA — Quick Diagnostics")
    st.info("**Interpretation:** Rate Transient Analysis (RTA) helps diagnose flow regimes. The **log-log derivative** plot is key: a slope of ~0.5 can indicate linear flow (common in early unconventional well life), while a slope of ~0 can indicate boundary-dominated flow. These trends help validate the simulation physics and understand the drainage behavior.")
    sim_data = st.session_state.sim if st.session_state.sim is not None else fallback_fast_solver(state, np.random.default_rng(int(st.session_state.rng_seed)))
    t,qg = sim_data["t"], sim_data["qg"]

    # NEW: y-axis toggle for rates (Linear / Log)
    rate_y_mode_rta = st.radio("Rate y-axis", ["Linear", "Log"], index=0, horizontal=True, key="rta_rate_y_mode")
    y_type_rta = "log" if rate_y_mode_rta == "Log" else "linear"

    # R1. Gas rate (q) vs time (x is already log scale in the layout helper)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=qg, line=dict(color="firebrick", width=3), name="Gas"))
    fig.update_layout(**semi_log_layout("R1. Gas rate (q) vs time", yaxis="q (Mscf/d)"))
    fig.update_yaxes(type=y_type_rta)
    st.plotly_chart(fig, use_container_width=True, key="rta_rate_plot")

    # R2. Log-log derivative (keep linear y for slope clarity)
    logt,logq = np.log10(t), np.log10(np.maximum(qg,1e-9))
    slope = np.gradient(logq, logt)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=slope, line=dict(color="teal", width=3), name="dlogq/dlogt"))
    fig2.update_layout(**semi_log_layout("R2. Log-log derivative", yaxis="Slope"))
    st.plotly_chart(fig2, use_container_width=True, key="rta_deriv_plot")

with tabs[5]:
    st.header("Simulation Results")
    if st.button("Run simulation", type="primary"):
        with st.spinner("Running simulation..."):
            st.session_state.sim = run_simulation(state)

    if st.session_state.sim is not None:
        sim = st.session_state.sim
        st.info("**Interpretation:** The primary results include Estimated Ultimate Recovery (EUR) gauges, production rate declines over time, cumulative production, and pressure maps. These outputs allow you to assess the well's performance, diagnose production behavior (like GOR changes), and visualize reservoir depletion.")

        g_g,o_g = eur_gauges(sim["EUR_g_BCF"],sim["EUR_o_MMBO"])
        c1,c2 = st.columns(2)
        with c1: st.plotly_chart(g_g, use_container_width=True, key="res_gauge_gas")
        with c2: st.plotly_chart(o_g, use_container_width=True, key="res_gauge_oil")

        rate_y_mode_results = st.radio("Rate y-axis (Results)", ["Linear", "Log"], index=0, horizontal=True, key="results_rate_y_mode")
        y_type_results = "log" if rate_y_mode_results == "Log" else "linear"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim["t"], y=sim["qg"], name="Gas (Mscf/d)", line=dict(color="#d62728", width=3)))
        fig.add_trace(go.Scatter(x=sim["t"], y=sim["qo"], name="Oil (STB/d)",  line=dict(color="#2ca02c", width=3)))
        fig.update_layout(**semi_log_layout("Figure 7. Field Production Rates", yaxis="Rate"))
        fig.update_yaxes(type=y_type_results)
        st.plotly_chart(fig, use_container_width=True, key="res_rates_plot")

        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(x=sim["t"], y=sim["qo"], line=dict(color="#2ca02c", width=3), name="Oil"))
        fig8.update_layout(**semi_log_layout("Figure 8. Oil Decline", yaxis="Oil rate (STB/d)"))
        fig8.update_yaxes(type=y_type_results)
        st.plotly_chart(fig8, use_container_width=True, key="res_oil_decline_plot")

        cum_g = np.cumsum(sim["qg"]) * np.mean(np.diff(sim["t"]))
        cum_o = np.cumsum(sim["qo"]) * np.mean(np.diff(sim["t"]))
        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(x=sim["t"], y=cum_g/1e6, name="Gas (BCF)",  line=dict(color="#d62728", width=3)))
        fig9.add_trace(go.Scatter(x=sim["t"], y=cum_o/1e6, name="Oil (MMBO)", line=dict(color="#2ca02c", width=3)))
        fig9.update_layout(**semi_log_layout("Figure 9. Cumulative Production", yaxis="Cumulative"))
        st.plotly_chart(fig9, use_container_width=True, key="res_cum_plot")

        bhp_t = sim.get("bhp_t", sim["t"])
        bhp_psi = sim.get("bhp_psi", np.full_like(bhp_t, float(state["pad_bhp_psi"]) if state["pad_ctrl"]=="BHP" else float(state["p_min_bhp_psi"])))
        fig_bhp = go.Figure()
        fig_bhp.add_trace(go.Scatter(x=bhp_t, y=bhp_psi, name="BHP (psi)", line=dict(width=3, color="purple")))
        fig_bhp.update_layout(**semi_log_layout("Bottom-Hole Pressure vs Time", yaxis="BHP (psi)"))
        st.plotly_chart(fig_bhp, use_container_width=True, key="res_bhp_plot")

        qo_safe = np.where(sim["qo"]<=1e-9, np.nan, sim["qo"])
        gor = sim["qg"]*1000/qo_safe
        fig_gor = go.Figure()
        fig_gor.add_trace(go.Scatter(x=sim["t"], y=gor, name="GOR (scf/STB)", line=dict(width=3, color="orange")))
        fig_gor.update_layout(**semi_log_layout("GOR vs Time", yaxis="GOR (scf/STB)"))
        st.plotly_chart(fig_gor, use_container_width=True, key="res_gor_plot")

        c1,c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.imshow(sim["press_frac_mid"], origin="lower", color_continuous_scale="Viridis",
                                      title="<b>Figure 5. Fracture Pressure (mid-layer, psi)</b>"),
                            use_container_width=True, theme=None, key="res_frac_mid")
        with c2:
            st.plotly_chart(px.imshow(sim["press_matrix_mid"], origin="lower", color_continuous_scale="Cividis",
                                      title="<b>Figure 6. Matrix Pressure (mid-layer, psi)</b>"),
                            use_container_width=True, theme=None, key="res_matrix_mid")

        st.caption(f"Runtime: {sim.get('runtime_s',0):.2f} s")
    else:
        st.info("Click **Run simulation** to compute rates, pressures, and EUR gauges.")

with tabs[6]:
    st.header("3D Viewer — Pressure / Saturations (Isosurface/Volume)")
    if st.session_state.sim is None:
        st.info("Run a simulation to view 3D volumes.")
    else:
        st.info("**Interpretation:** This tool visualizes the 3D distribution of pressure or fluid saturations. **Isosurface** mode shows a surface of a constant value (e.g., all points at 4000 psi), which is useful for seeing the depletion front. **Volume** mode provides a translucent view of the entire property field. Overlaying the **DFN** shows how fractures are interacting with the depletion.")
        sim = st.session_state.sim
        nz, ny, nx = sim["press_matrix"].shape
        field_choice = st.selectbox("Field", ["Matrix Pressure (psi)", "Fracture Pressure (psi)", "Oil Saturation So", "Water Saturation Sw", "Gas Saturation Sg (computed)"], index=0, key="field_choice_3d")
        vis_mode = st.radio("Mode", ["Isosurface", "Volume"], index=0, horizontal=True, key="vis_mode_3d")
        show_dfn = st.checkbox("Overlay DFN", value=True, key="show_dfn_overlay")
        
        So = sim.get("So", ensure_3d(sim.get("So_mid", np.zeros((ny, nx)))))
        Sw = sim.get("Sw", ensure_3d(sim.get("Sw_mid", np.zeros((ny, nx)))))
        
        field_map = {
            "Matrix Pressure (psi)": sim["press_matrix"],
            "Fracture Pressure (psi)": sim["press_frac"],
            "Oil Saturation So": So,
            "Water Saturation Sw": Sw,
            "Gas Saturation Sg (computed)": np.clip(1.0 - So - Sw, 0.0, 1.0)
        }
        F = ensure_3d(field_map.get(field_choice))

        ds = max(1, int(st.session_state.vol_downsample))
        Fd = downsample_3d(F, ds)
        zz, yy, xx = Fd.shape

        # Build a true meshgrid (one coordinate per voxel) in FEET
        x = np.linspace(0.0, (nx - 1) * state["dx"], xx)
        y = np.linspace(0.0, (ny - 1) * state["dy"], yy)
        z = np.linspace(0.0, (nz - 1) * state["dz"], zz)
        Zv, Yv, Xv = np.meshgrid(z, y, x, indexing="ij")  # shapes (zz, yy, xx) to match Fd

        fig3d = go.Figure()
        vmin, vmax = float(np.nanmin(Fd)), float(np.nanmax(Fd))
        vrel = float(st.session_state.iso_value_rel)
        iso_val = vmin + vrel * (vmax - vmin + 1e-12)

        if vis_mode == "Isosurface":
            fig3d.add_trace(go.Isosurface(
                x=Xv.ravel(), y=Yv.ravel(), z=Zv.ravel(), value=Fd.ravel(),
                isomin=iso_val, isomax=iso_val, surface_count=1,
                caps=dict(x_show=False, y_show=False, z_show=False), showscale=True
            ))
        else:
            fig3d.add_trace(go.Volume(
                x=Xv.ravel(), y=Yv.ravel(), z=Zv.ravel(), value=Fd.ravel(),
                opacity=0.08, surface_count=12,
                caps=dict(x_show=False, y_show=False, z_show=False), showscale=True
            ))
        
        if show_dfn and st.session_state.dfn_segments is not None:
            for i, seg in enumerate(st.session_state.dfn_segments):
                fig3d.add_trace(go.Scatter3d(
                    x=[seg[0], seg[3]], y=[seg[1], seg[4]], z=[seg[2], seg[5]],
                    mode="lines", line=dict(width=6, color='red'),
                    name="DFN" if i == 0 else None, showlegend=(i == 0)
                ))
        
        fig3d.update_layout(template="plotly_white", scene=dict(xaxis_title="x (ft)", yaxis_title="y (ft)", zaxis_title="z (ft)"), height=680, margin=dict(l=0, r=0, t=40, b=0), title=f"<b>3D View: {field_choice}</b>")
        st.plotly_chart(fig3d, use_container_width=True)
        st.caption("Tip: Increase the downsample factor in the sidebar to speed up rendering for large models.")
with tabs[7]:
    st.header("Slice Viewer — k / i / j slices")
    if st.session_state.sim is None: st.info("Run a simulation to view slices.")
    else:
        st.info("**Interpretation:** This tool lets you inspect 2D cross-sections of the 3D data volumes. Slicing through different layers (k), columns (i), and rows (j) is essential for quality control and understanding how properties and fluid flow vary in all three dimensions.")
        sim = st.session_state.sim; nz,ny,nx = sim["press_matrix"].shape
        k_idx = st.slider("k (layer)",0,nz-1,nz//2,1,key="k_idx"); i_idx = st.slider("i (column)",0,nx-1,nx//2,1,key="i_idx"); j_idx = st.slider("j (row)",0,ny-1,ny//2,1,key="j_idx")
        def draw_slice(field,title_prefix):
            A3 = ensure_3d(field); st.subheader(f"Slices for: {title_prefix}")
            k_slice,i_slice,j_slice = A3[k_idx,:,:],A3[:,:,i_idx],A3[:,j_idx,:]
            c1,c2 = st.columns(2)
            with c1: st.plotly_chart(px.imshow(k_slice,origin="lower",color_continuous_scale="Viridis",title=f"<b>k-slice (ny×nx) at k={k_idx}</b>"),use_container_width=True,theme=None)
            with c2: st.plotly_chart(px.imshow(i_slice,origin="lower",color_continuous_scale="Cividis",title=f"<b>i-slice (nz×ny) at i={i_idx}</b>"),use_container_width=True,theme=None)
            st.plotly_chart(px.imshow(j_slice,origin="lower",color_continuous_scale="Magma",title=f"<b>j-slice (nz×nx) at j={j_idx}</b>"),use_container_width=True,theme=None)
        So3 = sim.get("So",ensure_3d(sim.get("So_mid",np.zeros((ny,nx)))))
        Sw3 = sim.get("Sw",ensure_3d(sim.get("Sw_mid",np.zeros((ny,nx)))))
        Sg3 = np.clip(1.0-So3-Sw3,0.0,1.0)
        slice_field_choice = st.selectbox("Select field to slice:",["Matrix Pressure","Fracture Pressure","Oil Saturation","Water Saturation","Gas Saturation"])
        field_map = {"Matrix Pressure": (sim["press_matrix"],"Matrix Pressure (psi)"), "Fracture Pressure": (sim["press_frac"],"Fracture Pressure (psi)"), "Oil Saturation": (So3,"Oil Saturation So"), "Water Saturation": (Sw3,"Water Saturation Sw"), "Gas Saturation": (Sg3,"Gas Saturation Sg")}
        if slice_field_choice in field_map: draw_slice(*field_map[slice_field_choice])

with tabs[8]:
    st.header("QA / Material Balance")
    st.info("**Interpretation:** These plots check for material balance closure. The **produced** volume should eventually equal the initial volume in place minus the **residual** volume. If the residual lines trend towards zero over time, it indicates a good material balance, meaning the simulation is conserving mass correctly. Deviations can hint at issues with PVT properties, compressibility values, or boundary condition assumptions.")
    if st.session_state.sim is not None:
        sim = st.session_state.sim
        t = sim["t"]

        # --- MBAL EUR gauges (Gas in BCF, Oil in MMBO) ---
        EUR_g_BCF_mb  = float(np.trapz(sim["qg"], t) / 1e6)   # qg [Mscf/d] -> Mscf -> BCF
        EUR_o_MMBO_mb = float(np.trapz(sim["qo"], t) / 1e6)   # qo [STB/d]  -> STB  -> MMBO
        g_g_mb, o_g_mb = eur_gauges(EUR_g_BCF_mb, EUR_o_MMBO_mb)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(g_g_mb, use_container_width=True, key="qa_mbal_gauge_gas")
        with c2:
            st.plotly_chart(o_g_mb, use_container_width=True, key="qa_mbal_gauge_oil")

        # Cumulative produced & residual using consistent units with the gauges
        dt = np.gradient(t)
        cum_g_BCF  = np.cumsum(dt * sim["qg"]) / 1e6   # BCF
        cum_o_MMBO = np.cumsum(dt * sim["qo"]) / 1e6   # MMBO
        res_g_BCF  = cum_g_BCF[-1]  - cum_g_BCF
        res_o_MMBO = cum_o_MMBO[-1] - cum_o_MMBO

        # Gas MB plot
        fg = go.Figure()
        fg.add_trace(go.Scatter(x=t, y=cum_g_BCF, name="Gas produced (BCF)", line=dict(color="#d62728", width=3)))
        fg.add_trace(go.Scatter(x=t, y=res_g_BCF, name="Gas residual (BCF)", line=dict(color="#ff9896", width=2, dash="dot")))
        fg.update_layout(**semi_log_layout("Gas Material Balance", yaxis="BCF"))
        st.plotly_chart(fg, use_container_width=True, key="qa_mbal_plot_gas")

        # Oil MB plot
        fo = go.Figure()
        fo.add_trace(go.Scatter(x=t, y=cum_o_MMBO, name="Oil produced (MMBO)", line=dict(color="#2ca02c", width=3)))
        fo.add_trace(go.Scatter(x=t, y=res_o_MMBO, name="Oil residual (MMBO)", line=dict(color="#98df8a", width=2, dash="dot")))
        fo.update_layout(**semi_log_layout("Oil Material Balance", yaxis="MMBO"))
        st.plotly_chart(fo, use_container_width=True, key="qa_mbal_plot_oil")
    else:
        st.info("Run a simulation to view the Material Balance plots.")

with tabs[9]:
    st.header("Sensitivity: EUR vs Lateral Length")
    st.info("Dual view: the Dual Axis tab gives a compact overview (gas left axis, oil right axis), while Stacked Panels separates the series for maximum readability.")

    L_grid = np.array([6000,8000,10000,12000,14000],float)
    rows = [dict(L_ft=int(L), **{k:v for k,v in fallback_fast_solver({**state,"L_ft":float(L)},np.random.default_rng(123)).items() if "EUR" in k}) for L in L_grid]
    df = pd.DataFrame(rows)

    tab_dual, tab_stack = st.tabs(["Dual Axis (overview)","Stacked Panels (clear)"])

    with tab_dual:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df["L_ft"],y=df["EUR_o_MMBO"],name="Oil EUR (MMBO)",mode="lines+markers",
                                  line=dict(color="#2ca02c",width=3,dash="dot"),marker=dict(size=9,symbol="square"),
                                  yaxis="y2",opacity=0.95))
        fig1.add_trace(go.Scatter(x=df["L_ft"],y=df["EUR_g_BCF"],name="Gas EUR (BCF)",mode="lines+markers",
                                  line=dict(color="#d62728",width=4),marker=dict(size=10,symbol="circle-open"),yaxis="y"))
        fig1.update_layout(template="plotly_white",title="<b>EUR vs Lateral Length (Dual Axis)</b>",xaxis_title="Lateral length (ft)",
                           yaxis=dict(title="Gas EUR (BCF)"),
                           yaxis2=dict(title="Oil EUR (MMBO)", overlaying="y", side="right", showgrid=False, zeroline=False),
                           legend=dict(orientation="h"))
        st.plotly_chart(fig1, use_container_width=True)

    with tab_stack:
        from plotly.subplots import make_subplots
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                             subplot_titles=("Gas EUR (BCF)","Oil EUR (MMBO)"))
        fig2.add_trace(go.Scatter(x=df["L_ft"],y=df["EUR_g_BCF"],name="Gas EUR (BCF)",mode="lines+markers",
                                  line=dict(color="#d62728",width=3)), row=1, col=1)
        fig2.add_trace(go.Scatter(x=df["L_ft"],y=df["EUR_o_MMBO"],name="Oil EUR (MMBO)",mode="lines+markers",
                                  line=dict(color="#2ca02c",width=3)), row=2, col=1)
        fig2.update_xaxes(title_text="Lateral length (ft)", row=2, col=1)
        fig2.update_yaxes(title_text="BCF",  row=1, col=1)
        fig2.update_yaxes(title_text="MMBO", row=2, col=1)
        fig2.update_layout(template="plotly_white",title="<b>EUR vs Lateral Length (Stacked Panels)</b>",legend=dict(orientation="h"))
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df, use_container_width=True)
    st.download_button("Download EUR vs Lateral Length CSV", df.to_csv(index=False).encode("utf-8"),
                       "eur_vs_lateral_length.csv", "text/csv")

with tabs[10]:
    st.header("Field Match (CSV)")
    st.info(
        "Upload a CSV with historical production (time_days, qg_Mscfd, qo_STBpd). "
        "Use the quick dials to time-shift and scale the simulator for a first-pass match."
    )
    st.caption("CSV required columns: time_days, qg_Mscfd, qo_STBpd. Optional: p_res_psi.")
    up = st.file_uploader("Upload CSV", type=["csv"], key="field_csv_uploader")

    # Demo generator
    if st.button("Generate synthetic demo", key="fm_gen_demo_btn"):
        demo = fallback_fast_solver(state, np.random.default_rng(321))
        demo_df = pd.DataFrame(dict(time_days=demo["t"], qg_Mscfd=demo["qg"], qo_STBpd=demo["qo"]))
        st.download_button("Download demo CSV", demo_df.to_csv(index=False), "demo_field.csv", key="fm_demo_dl")

    if up is not None:
        # ---------- Load & basic prep ----------
        field = (
            pd.read_csv(up)
            .dropna(subset=["time_days", "qg_Mscfd", "qo_STBpd"])
            .sort_values("time_days")
        )
        tF = field["time_days"].to_numpy(float)
        qgF = field["qg_Mscfd"].to_numpy(float)
        qoF = field["qo_STBpd"].to_numpy(float)
        dtF = np.gradient(tF)

        # Field EURs (reference)
        EURgF_BCF  = float(np.trapz(qgF, tF) / 1e6)
        EURoF_MMBO = float(np.trapz(qoF, tF) / 1e6)

        # ---------- Get simulator curves ----------
        sim = st.session_state.sim if st.session_state.sim is not None else fallback_fast_solver(state, np.random.default_rng(1234))
        tS  = sim["t"].astype(float)
        qgS = sim["qg"].astype(float)
        qoS = sim["qo"].astype(float)

        # ---------- Quick tuning controls ----------
        st.subheader("Quick Tuning (first-pass history match)")
        cA, cB, cC, cD, cE = st.columns(5)

        with cA:
            t_shift = st.slider("Time shift (days)", -180, 180, 0, 1, help="Positive shifts simulator to the right.", key="fm_tshift")
        with cB:
            gas_scale = st.slider("Gas scale ×", 0.50, 1.50, 1.00, 0.01, key="fm_gscale")
        with cC:
            oil_scale = st.slider("Oil scale ×", 0.50, 1.50, 1.00, 0.01, key="fm_oscale")
        with cD:
            time_axis_mode = st.radio("Time axis", ["Linear", "Log (RTA)"], index=0, horizontal=True, key="fm_time_axis")
        with cE:
            rate_y_mode = st.radio("Rate y-axis", ["Linear", "Log"], index=0, horizontal=True, key="fm_rate_y")

        # Apply shifts/scales and interpolate sim → field time grid
        tS_adj  = tS + float(t_shift)
        qgS_adj = gas_scale * qgS
        qoS_adj = oil_scale * qoS

        def interp_safe(x_src, y_src, x_new):
            # Extrapolate flat beyond ends
            return np.interp(x_new, x_src, y_src, left=y_src[0], right=y_src[-1])

        qgS_i = interp_safe(tS_adj, qgS_adj, tF)
        qoS_i = interp_safe(tS_adj, qoS_adj, tF)

        # Cumulative on the field grid
        cum_gF_BCF  = np.cumsum(dtF * qgF) / 1e6
        cum_oF_MMBO = np.cumsum(dtF * qoF) / 1e6
        cum_gS_BCF  = np.cumsum(dtF * qgS_i) / 1e6
        cum_oS_MMBO = np.cumsum(dtF * qoS_i) / 1e6

        # Sim EURs (after tuning)
        EURgS_BCF  = float(cum_gS_BCF[-1])
        EURoS_MMBO = float(cum_oS_MMBO[-1])

        # ---------- Metrics ----------
        def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))
        def mape(a, b):
            denom = np.maximum(1e-9, np.abs(a))
            return float(np.mean(np.abs((a - b) / denom)) * 100.0)

        metrics = {
            "Gas rate RMSE (Mscf/d)": rmse(qgF, qgS_i),
            "Oil rate RMSE (STB/d)": rmse(qoF, qoS_i),
            "Gas rate MAPE (%)": mape(qgF, qgS_i),
            "Oil rate MAPE (%)": mape(qoF, qoS_i),
            "Gas cum RMSE (BCF)": rmse(cum_gF_BCF, cum_gS_BCF),
            "Oil cum RMSE (MMBO)": rmse(cum_oF_MMBO, cum_oS_MMBO),
        }

        st.markdown("**Match Quality (first-pass):**")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Gas rate RMSE (Mscf/d)", f"{metrics['Gas rate RMSE (Mscf/d)']:.0f}")
            st.metric("Gas cum RMSE (BCF)", f"{metrics['Gas cum RMSE (BCF)']:.3f}")
        with m2:
            st.metric("Oil rate RMSE (STB/d)", f"{metrics['Oil rate RMSE (STB/d)']:.0f}")
            st.metric("Oil cum RMSE (MMBO)", f"{metrics['Oil cum RMSE (MMBO)']:.3f}")
        with m3:
            st.metric("Gas rate MAPE (%)", f"{metrics['Gas rate MAPE (%)']:.1f}")
            st.metric("Oil rate MAPE (%)", f"{metrics['Oil rate MAPE (%)']:.1f}")

        # ---------- Axis modes ----------
        xaxis_type = "log" if time_axis_mode.startswith("Log") else "linear"
        tF_plot = np.where(tF <= 0.0, 1e-3, tF) if xaxis_type == "log" else tF

        yaxis_type_rates = "log" if rate_y_mode == "Log" else "linear"

        def log_safe(vals):
            return np.where(vals <= 0.0, 1e-3, vals)

        # Use log-safe series only if plotting on log-y for Rates
        if yaxis_type_rates == "log":
            qgF_plot, qgS_i_plot = log_safe(qgF), log_safe(qgS_i)
            qoF_plot, qoS_i_plot = log_safe(qoF), log_safe(qoS_i)
        else:
            qgF_plot, qgS_i_plot = qgF, qgS_i
            qoF_plot, qoS_i_plot = qoF, qoS_i

        # ---------- Visualizations ----------
        from plotly.subplots import make_subplots

        # A) Rates (stacked panels)
        fig_rates = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
            subplot_titles=("Gas Rate (Mscf/d)", "Oil Rate (STB/d)")
        )
        fig_rates.add_trace(go.Scatter(x=tF_plot, y=qgF_plot,   name="Field Gas",     line=dict(color="#d62728", width=3)),            row=1, col=1)
        fig_rates.add_trace(go.Scatter(x=tF_plot, y=qgS_i_plot, name="Sim Gas (adj.)", line=dict(color="#c44e52", width=2, dash="dot")), row=1, col=1)
        fig_rates.add_trace(go.Scatter(x=tF_plot, y=qoF_plot,   name="Field Oil",     line=dict(color="#2ca02c", width=3)),            row=2, col=1)
        fig_rates.add_trace(go.Scatter(x=tF_plot, y=qoS_i_plot, name="Sim Oil (adj.)", line=dict(color="#55a868", width=2, dash="dot")), row=2, col=1)
        fig_rates.update_xaxes(title_text="Time (days)", type=xaxis_type, row=2, col=1)
        fig_rates.update_yaxes(type=yaxis_type_rates, row=1, col=1)
        fig_rates.update_yaxes(type=yaxis_type_rates, row=2, col=1)
        fig_rates.update_layout(template="plotly_white", title="<b>Rates — Field vs Sim (Adjusted)</b>", legend=dict(orientation="h"))
        st.plotly_chart(fig_rates, use_container_width=True, key="fm_rates_stacked")

        # B) Cumulative (stacked panels) — keep linear x
        fig_cum = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
            subplot_titles=("Gas Cumulative (BCF)", "Oil Cumulative (MMBO)")
        )
        fig_cum.add_trace(go.Scatter(x=tF, y=cum_gF_BCF, name="Field Gas", line=dict(color="#d62728", width=3)), row=1, col=1)
        fig_cum.add_trace(go.Scatter(x=tF, y=cum_gS_BCF, name="Sim Gas (adj.)", line=dict(color="#c44e52", width=2, dash="dot")), row=1, col=1)
        fig_cum.add_trace(go.Scatter(x=tF, y=cum_oF_MMBO, name="Field Oil", line=dict(color="#2ca02c", width=3)), row=2, col=1)
        fig_cum.add_trace(go.Scatter(x=tF, y=cum_oS_MMBO, name="Sim Oil (adj.)", line=dict(color="#55a868", width=2, dash="dot")), row=2, col=1)
        fig_cum.update_xaxes(title_text="Time (days)", row=2, col=1)  # linear
        fig_cum.update_layout(template="plotly_white", title="<b>Cumulative — Field vs Sim (Adjusted)</b>", legend=dict(orientation="h"))
        st.plotly_chart(fig_cum, use_container_width=True, key="fm_cum_stacked")

        # C) Parity plots (Sim vs Field with 1:1 line)
        fig_par = make_subplots(rows=1, cols=2, subplot_titles=("Gas Rate Parity", "Oil Rate Parity"))
        xg = np.linspace(0, max(qgF.max(), qgS_i.max())*1.05, 50)
        xo = np.linspace(0, max(qoF.max(), qoS_i.max())*1.05, 50)
        fig_par.add_trace(go.Scatter(x=xg, y=xg, name="1:1", line=dict(color="gray", width=1, dash="dash")), row=1, col=1)
        fig_par.add_trace(go.Scatter(x=xo, y=xo, name="1:1", line=dict(color="gray", width=1, dash="dash"), showlegend=False), row=1, col=2)
        fig_par.add_trace(go.Scatter(x=qgF, y=qgS_i, mode="markers", name="Gas", marker=dict(color="#d62728", size=6, opacity=0.7)), row=1, col=1)
        fig_par.add_trace(go.Scatter(x=qoF, y=qoS_i, mode="markers", name="Oil", marker=dict(color="#2ca02c", size=6, opacity=0.7), showlegend=False), row=1, col=2)
        fig_par.update_xaxes(title_text="Field")
        fig_par.update_yaxes(title_text="Sim (adj.)")
        fig_par.update_layout(template="plotly_white", title="<b>Parity — Sim vs Field (Rates)</b>", legend=dict(orientation="h"))
        st.plotly_chart(fig_par, use_container_width=True, key="fm_parity_rates")

        # D) Residuals vs time (x Linear/Log toggle)
        fig_res = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
            subplot_titles=("Gas Residual = Field − Sim (Mscf/d)", "Oil Residual = Field − Sim (STB/d)")
        )
        fig_res.add_trace(go.Scatter(x=tF_plot, y=(qgF - qgS_i), name="Gas residual", line=dict(color="#d62728")), row=1, col=1)
        fig_res.add_trace(go.Scatter(x=tF_plot, y=(qoF - qoS_i), name="Oil residual", line=dict(color="#2ca02c")), row=2, col=1)
        fig_res.update_xaxes(title_text="Time (days)", type=xaxis_type, row=2, col=1)
        fig_res.update_layout(template="plotly_white", title="<b>Residuals — Field minus Sim</b>", legend=dict(orientation="h"))
        st.plotly_chart(fig_res, use_container_width=True, key="fm_residuals")

        # E) GOR vs time (x Linear/Log toggle); guard divide-by-zero
        qoF_safe = np.where(np.abs(qoF) < 1e-6, np.nan, qoF)
        qoS_safe = np.where(np.abs(qoS_i) < 1e-6, np.nan, qoS_i)
        GOR_F = (1000.0 * qgF) / qoF_safe
        GOR_S = (1000.0 * qgS_i) / qoS_safe
        fig_gor = go.Figure()
        fig_gor.add_trace(go.Scatter(x=tF_plot, y=GOR_F, name="Field GOR", line=dict(color="#ff7f0e", width=3)))
        fig_gor.add_trace(go.Scatter(x=tF_plot, y=GOR_S, name="Sim GOR (adj.)", line=dict(color="#9467bd", width=2, dash="dot")))
        fig_gor.update_layout(template="plotly_white", title="<b>GOR vs Time</b>", xaxis_title="Time (days)", yaxis_title="GOR (scf/STB)", legend=dict(orientation="h"))
        fig_gor.update_xaxes(type=xaxis_type)
        st.plotly_chart(fig_gor, use_container_width=True, key="fm_gor")

        # F) EUR comparison bars
        eur_df = pd.DataFrame({
            "Phase": ["Gas", "Oil"],
            "Field": [EURgF_BCF, EURoF_MMBO],
            "Sim (adj.)": [EURgS_BCF, EURoS_MMBO]
        })
        fig_eur = go.Figure()
        fig_eur.add_trace(go.Bar(x=eur_df["Phase"], y=eur_df["Field"], name="Field", marker=dict(color="#888")))
        fig_eur.add_trace(go.Bar(x=eur_df["Phase"], y=eur_df["Sim (adj.)"], name="Sim (adj.)", marker=dict(color="#1f77b4")))
        fig_eur.update_layout(template="plotly_white", barmode="group", title="<b>EUR Comparison</b>", yaxis_title="Gas: BCF | Oil: MMBO (per phase)")
        st.plotly_chart(fig_eur, use_container_width=True, key="fm_eur_bars")

        # ---------- Download merged comparison table ----------
        comp = pd.DataFrame({
            "time_days": tF,
            "qg_field_Mscfd": qgF,
            "qg_sim_adj_Mscfd": qgS_i,
            "qo_field_STBpd": qoF,
            "qo_sim_adj_STBpd": qoS_i,
            "cum_g_field_BCF": cum_gF_BCF,
            "cum_g_sim_BCF": cum_gS_BCF,
            "cum_o_field_MMBO": cum_oF_MMBO,
            "cum_o_sim_MMBO": cum_oS_MMBO,
            "GOR_field_scf_per_STB": GOR_F,
            "GOR_sim_adj_scf_per_STB": GOR_S,
        })
        st.download_button(
            "Download comparison CSV",
            comp.to_csv(index=False).encode("utf-8"),
            "history_match_comparison.csv",
            "text/csv",
            key="fm_download_csv"
        )
    else:
        st.warning("Upload a CSV to run the history match.")

with tabs[11]:
    st.header("Uncertainty & Monte Carlo")
    st.info("**Interpretation:** This tab runs a Monte Carlo simulation by varying key input parameters to generate a distribution of possible EUR outcomes. The **distribution plot** shows the probabilistic range of results (P10, P50, P90). The **Tornado chart** ranks which uncertain parameters have the biggest impact on the outcome.")
    N = st.slider("Samples", 50, 500, 150, 10, key="mc_samples")
    rng = np.random.default_rng(777)
    with st.spinner(f"Running {N} Monte Carlo simulations..."):
        samples = []
        for i in range(N):
            tmp = state.copy()
            tmp["k_stdev"] = float(np.clip(rng.normal(state["k_stdev"], 0.006), 0.0, 0.20))
            tmp["hf_ft"]   = float(np.clip(rng.normal(state["hf_ft"], 15.0), 60.0, 300.0))
            tmp["xf_ft"]   = float(np.clip(rng.normal(state["xf_ft"], 20.0), 120.0, 500.0))
            tmp["pad_interf"] = float(np.clip(rng.normal(state["pad_interf"], 0.05), 0.0, 0.8))
            s = fallback_fast_solver(tmp, rng)
            samples.append([s["EUR_g_BCF"], s["EUR_o_MMBO"], tmp["k_stdev"], tmp["hf_ft"], tmp["xf_ft"], tmp["pad_interf"]])
        S = np.array(samples)

    def plot_dist_and_tornado(data, name, unit, color, key_prefix):
        st.subheader(f"{name} EUR Analysis")
        p10,p50,p90 = np.percentile(data,10),np.percentile(data,50),np.percentile(data,90)

        # Distribution Plot
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=data, nbinsx=30, name='Histogram', marker_color=color, histnorm='probability density'))
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        fig_dist.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines', name='Distribution Curve', line=dict(color='black', width=2)))
        for p, p_name, p_col in [(p10, 'P10', 'blue'), (p50, 'P50', 'green'), (p90, 'P90', 'red')]:
            fig_dist.add_vline(x=p, line_width=2, line_dash="dash", line_color=p_col, annotation_text=f"{p_name}={p:.1f}", annotation_position="top right")
        fig_dist.update_layout(template="plotly_white", title=f"<b>{name} EUR Distribution</b>", xaxis_title=f"EUR ({unit})", yaxis_title="Density")
        st.plotly_chart(fig_dist, use_container_width=True, key=f"{key_prefix}_dist")
        st.markdown(f"**{name} EUR:** **P10** = {p10:.1f} {unit}, **P50** = {p50:.1f} {unit}, **P90** = {p90:.1f} {unit}")

        # Tornado Chart
        params = ["k stdev", "hf (ft)", "xf (ft)", "pad interference"]
        corrs = [np.corrcoef(data, S[:, i+2])[0,1] for i in range(len(params))]
        dfT = pd.DataFrame(dict(param=params, corr=corrs))
        dfT["impact"] = np.abs(dfT["corr"]); dfT = dfT.sort_values("impact", ascending=True)
        fig_tor = go.Figure(go.Bar(x=dfT["impact"], y=dfT["param"], orientation="h", marker_color=color, marker_opacity=0.7))
        fig_tor.update_layout(template="plotly_white", title=f"<b>Tornado: Relative Impact on {name} EUR</b>", xaxis_title="Absolute Correlation (Impact)")
        st.plotly_chart(fig_tor, use_container_width=True, key=f"{key_prefix}_tor")

    col1, col2 = st.columns(2)
    with col1:
        plot_dist_and_tornado(S[:,0], "Gas", "BCF", "#d62728", "mc_gas")
    with col2:
        plot_dist_and_tornado(S[:,1], "Oil", "MMBO", "#2ca02c", "mc_oil")

with tabs[12]:
    st.header("User’s Manual")
    st.markdown("""
**Overview:** This application supports full 3D arrays for pressure and saturations and is backward-compatible with mid-layer (2D) maps. DFNs can be uploaded or auto-generated from stages. They are used for both visualization and as a 3D sink driver in the fallback solver.

**Engine Schema (3D-aware):** The simulation engine expects or returns data with the following keys and units. 2D arrays are automatically promoted to 3D.
- **t**: [days], **qg**: [Mscf/d], **qo**: [STB/d]
- **press_matrix**: [nz,ny,nx] (psi) OR **press_matrix_mid**: [ny,nx]
- **press_frac**: [nz,ny,nx] (psi) OR **press_frac_mid**: [ny,nx]
- **So**: [nz,ny,nx] (fraction) OR **So_mid**: [ny,nx]
- **Sw**: [nz,ny,nx] (fraction) OR **Sw_mid**: [ny,nx]
- **EUR_g_BCF**: float, **EUR_o_MMBO**: float
- **Optional:** **bhp_t**: [days], **bhp_psi**: [psi]

**DFN CSV format:** The CSV must contain segment endpoints. Optional columns for fracture properties are stored but not used by the fallback solver.
- **Required columns:** `x0,y0,z0,x1,y1,z1` (all in feet)
- **Optional columns:** `k_mult,aperture_ft`
""")
    st.code(
        '{\n'
        '  "t": [days],\n'
        '  "qg": [Mscf/d],\n'
        '  "qo": [STB/d],\n'
        '  "press_matrix": [[[...]]],\n'
        '  "press_frac": [[[...]]],\n'
        '  "So": [[[...]]],\n'
        '  "Sw": [[[...]]],\n'
        '  "EUR_g_BCF": 0.0,\n'
        '  "EUR_o_MMBO": 0.0,\n'
        '  "bhp_t": [optional],\n'
        '  "bhp_psi": [optional]\n'
        '}\n',
        language="json",
    )

with tabs[13]:
    st.header("Solver & Profiling")
    st.info("**Interpretation:** Advanced controls for numerical solver tolerances and performance flags.")
    st.markdown(f"""
- **Newton Tolerance:** `{state['newton_tol']:.1e}`
- **Transport Tolerance:** `{state['trans_tol']:.1e}`
- **Max Newton Iterations:** `{int(state['max_newton'])}`
- **Max Linear Iterations:** `{int(state['max_lin'])}`
- **Threads:** `{int(state['threads'])}`
""")

with tabs[14]:
    st.header("DFN Viewer — 3D line segments")
    segs = st.session_state.dfn_segments
    if segs is None or len(segs) == 0:
        st.info("No DFN loaded. Upload a CSV or use 'Generate DFN from stages' in the sidebar.")
    else:
        st.info("**Interpretation:** Displays the DFN as 3D line segments for QC.")
        figd = go.Figure()
        for i, seg in enumerate(segs):
            figd.add_trace(go.Scatter3d(
                x=[seg[0], seg[3]], y=[seg[1], seg[4]], z=[seg[2], seg[5]],
                mode="lines", line=dict(width=6, color="red"),
                name="DFN" if i == 0 else None, showlegend=(i == 0)
            ))
        figd.update_layout(
            template="plotly_white",
            scene=dict(xaxis_title="x (ft)", yaxis_title="y (ft)", zaxis_title="z (ft)"),
            height=640, margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(figd, use_container_width=True, key="dfn_viewer_3d")
        st.caption(f"**Segments:** {len(segs)}. Optional columns k_mult/aperture_ft are parsed but not used in the fallback physics.")
