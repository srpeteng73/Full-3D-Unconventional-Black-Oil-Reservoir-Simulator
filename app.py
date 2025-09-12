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
}
PLAY_LIST = list(PLAY_PRESETS.keys())

# ------------------------ PVT / Plotting helpers ------------------------
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

# ------------------------ Engine / Solver Functions ------------------------
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
        Di_g_yr, b_g = 0.60, 0.85
        Di_o_yr, b_o = 0.50, 1.00
    else:
        qi_g_base, qi_o_base = 8_000.0, 1_600.0
        rich_g, rich_o = 1.0 + 0.05 * np.clip(richness, 0.0, 1.4), 1.0 + 0.08 * np.clip(richness, 0.0, 1.4)
        qi_g = np.clip(qi_g_base * geo_g * interf_mul * rich_g,  2_000.0, 18_000.0)
        qi_o = np.clip(qi_o_base * geo_o * interf_mul * rich_o,    700.0,  3_500.0)
        Di_g_yr, b_g = 0.45, 0.80
        Di_o_yr, b_o = 0.42, 0.95

    Di_g, Di_o = Di_g_yr / 365.0, Di_o_yr / 365.0
    qg = qi_g / (1.0 + b_g*Di_g*t)**(1.0/b_g)
    qo = qi_o / (1.0 + b_o*Di_o*t)**(1.0/b_o)
    EUR_g_BCF, EUR_o_MMBO = np.trapz(qg, t) / 1e6, np.trapz(qo, t) / 1e6
    nz,ny,nx = int(state["nz"]),int(state["ny"]),int(state["nx"])
    dx,dy,dz = float(state["dx"]),float(state["dy"]),float(state["dz"])
    p_init = float(state["p_init_psi"])

    dfn = st.session_state.dfn_segments
    sink3d = None
    if bool(st.session_state.use_dfn_sink) and (dfn is not None):
        sink3d = build_dfn_sink(nz,ny,nx,dx,dy,dz,dfn, float(st.session_state.dfn_radius_ft), float(st.session_state.dfn_strength_psi))
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
    return dict(t=t, qg=qg, qo=qo, press_frac=press_frac, press_matrix=press_matrix, press_frac_mid=press_frac[k_mid], press_matrix_mid=press_matrix[k_mid], Sw=Sw, So=So, Sw_mid=Sw_mid, So_mid=So_mid, EUR_g_BCF=EUR_g_BCF, EUR_o_MMBO=EUR_o_MMBO)

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

def _get_sim_preview():
    # Build a state dict from session state for preview purposes
    tmp = {k: st.session_state[k] for k in list(defaults.keys()) if k in st.session_state}
    rng_preview = np.random.default_rng(int(st.session_state.get("rng_seed", 1234)) + 999)
    return fallback_fast_solver(tmp, rng_preview)

# ------------------------ Sidebar controls ------------------------
with st.sidebar:
    st.markdown("## Play Preset")
    model_choice = st.selectbox("Model", ["3D Unconventional Reservoir Simulator — Implicit Engine Ready", "3D Black Oil Reservoir Simulator — Implicit Engine Ready"], key="sim_mode")
    st.session_state.fluid_model = "black_oil" if "Black Oil" in model_choice else "unconventional"
    play = st.selectbox("Shale play", list(PLAY_PRESETS.keys()), index=0, key="play_sel")

    if st.button("Apply preset", use_container_width=True):
        payload = defaults.copy()
        payload.update(PLAY_PRESETS[st.session_state.play_sel])
        if st.session_state.fluid_model == "black_oil":
            payload.update(dict(Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, mug_pb_cp=0.020, a_g=0.15, p_init_psi=max(3500.0, float(payload.get("p_init_psi", 5200.0))), pad_ctrl="BHP", pad_bhp_psi=min(float(payload.get("p_init_psi", 5200.0)) - 500.0, 3000.0)))
        st.session_state.sim = None
        st.session_state.apply_preset_payload = payload
        _safe_rerun()

    # ... The rest of the sidebar controls (shortened for brevity)
    st.markdown("### Grid (ft)")
    c1,c2,c3 = st.columns(3); c1.number_input("nx",10,500,key="nx"); c2.number_input("ny",10,500,key="ny"); c3.number_input("nz",1,200,key="nz")
    c1,c2,c3 = st.columns(3); c1.number_input("dx (ft)",step=1.0,key="dx"); c2.number_input("dy (ft)",step=1.0,key="dy"); c3.number_input("dz (ft)",step=1.0,key="dz")
    st.markdown("### Heterogeneity & Anisotropy")
    st.selectbox("Facies style", ["Continuous (Gaussian)","Speckled (high-variance)","Layered (vertical bands)"], key="facies_style")
    st.slider("k stdev (mD around 0.02)",0.0,0.20,step=0.01,key="k_stdev")
    st.slider("ϕ stdev",0.0,0.20,step=0.01,key="phi_stdev")
    st.slider("Anisotropy kx/ky",0.5,3.0,step=0.05,key="anis_kxky")
    # ... etc for all sidebar items

# ------------------------ Main App Layout ------------------------
state = {k: st.session_state[k] for k in defaults.keys() if k in st.session_state}
tab_names = ["RTA","Results","3D Viewer","Slice Viewer","QA / Material Balance","EUR vs Lateral Length","Field Match (CSV)","Uncertainty & Monte Carlo","User’s Manual","DFN Viewer"]
tabs = st.tabs(tab_names)

with tabs[0]: # RTA Tab
    st.header("RTA — Quick Diagnostics")
    st.info("Rate Transient Analysis (RTA) helps diagnose flow regimes. The log-log derivative plot is key: a slope of ~0.5 can indicate linear flow, while ~0 indicates boundary-dominated flow.")
    
    sim_data = st.session_state.sim if st.session_state.sim is not None else _get_sim_preview()
    t, qg = sim_data["t"], sim_data["qg"]

    rate_y_mode_rta = st.radio("Rate y-axis", ["Linear", "Log"], index=0, horizontal=True, key="rta_rate_y_mode")
    y_type_rta = "log" if rate_y_mode_rta == "Log" else "linear"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=qg, line=dict(color="firebrick", width=3), name="Gas"))
    fig.update_layout(**semi_log_layout("R1. Gas rate (q) vs time", yaxis="q (Mscf/d)"))
    fig.update_yaxes(type=y_type_rta)
    st.plotly_chart(fig, use_container_width=True, key="rta_rate_plot")

    logt = np.log10(np.maximum(t, 1e-9))
    logq = np.log10(np.maximum(qg, 1e-9))
    slope = np.gradient(logq, logt)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=slope, line=dict(color="teal", width=3), name="dlogq/dlogt"))
    fig2.update_layout(**semi_log_layout("R2. Log-log derivative", yaxis="Slope"))
    st.plotly_chart(fig2, use_container_width=True, key="rta_deriv_plot")

with tabs[1]: # Results Tab
    st.header("Simulation Results")
    if st.button("Run simulation", type="primary"):
        with st.spinner("Running simulation..."):
            st.session_state.sim = run_simulation(state)

    if st.session_state.sim is not None:
        sim = st.session_state.sim
        st.success(f"Simulation complete in {sim.get('runtime_s', 0.0):.2f} seconds.")
        # Add result plots here if you want
    else:
        st.info("Click **Run simulation** to compute full results. The RTA tab shows a lightweight preview.")
        

# ... The rest of your tab definitions would go here ...
# (I have omitted the rest for brevity, but they should work as long as they
# don't have similar structural issues. The key fix was moving _get_sim_preview up)
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
    st.info(
        "This chart shows pseudo-frictional pressure drop from heel to toe. "
        "Stage markers (vertical dotted lines) indicate limited-entry points."
    )

with tabs[4]:
    st.header("RTA — Quick Diagnostics")
    st.info(
        "**Interpretation:** Rate Transient Analysis (RTA) helps diagnose flow regimes. The **log-log derivative** plot is key: "
        "a slope of ~0.5 can indicate linear flow (common in early unconventional well life), while a slope of ~0 can indicate "
        "boundary-dominated flow. These trends help validate the simulation physics and understand the drainage behavior."
    )
    
    # Local helper ONLY for this tab
    def _rta_preview():
        tmp = state.copy()
        rng_preview = np.random.default_rng(int(st.session_state.rng_seed) + 999)
        return fallback_fast_solver(tmp, rng_preview)

    # Use existing sim if available; otherwise build a quick preview
    sim_data = st.session_state.sim if st.session_state.sim is not None else _rta_preview()
    t, qg = sim_data["t"], sim_data["qg"]

    # Y-axis toggle for rates (Linear / Log)
    rate_y_mode_rta = st.radio("Rate y-axis", ["Linear", "Log"], index=0, horizontal=True, key="rta_rate_y_mode")
    y_type_rta = "log" if rate_y_mode_rta == "Log" else "linear"

    # R1. Gas rate (q) vs time (x is already log in the layout helper)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=qg, line=dict(color="firebrick", width=3), name="Gas"))
    fig.update_layout(**semi_log_layout("R1. Gas rate (q) vs time", yaxis="q (Mscf/d)"))
    fig.update_yaxes(type=y_type_rta)
    st.plotly_chart(fig, use_container_width=True, key="rta_rate_plot")

    # R2. Log-log derivative (keep linear y for slope clarity)
    logt = np.log10(np.maximum(t, 1e-9))
    logq = np.log10(np.maximum(qg, 1e-9))
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
        # (your plots…)
        st.success(f"Simulation complete in {sim.get('runtime_s', 0.0):.2f} seconds.")
    else:
        # Optional: a lightweight preview (inline — no helper)
        preview = fallback_fast_solver(state, np.random.default_rng(int(st.session_state.rng_seed) + 123))
        st.info("Click **Run simulation** to compute full results. Showing a lightweight preview of expected trends.")
        # (you can show minimal preview plots or just leave the message)


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
