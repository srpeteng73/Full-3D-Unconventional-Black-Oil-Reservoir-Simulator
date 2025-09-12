# Full 3D Unconventional / Black-Oil Reservoir Simulator — Implicit Engine Ready (USOF units) + DFN support
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy import stats
from full3d import simulate  # IMPORTING YOUR REAL 3D ENGINE

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
    # (Other presets can be added back here)
}
PLAY_LIST = list(PLAY_PRESETS.keys())

# ------------------------ ALL HELPER FUNCTIONS (DEFINED BEFORE USE) ------------------------
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
        fig = go.Figure(go.Indicator(mode="gauge+number", value=float(val), number={'suffix':f" {suffix}",'font':{'size':44,'color':'#0b2545'}}, title={'text':f"<b>{label}</b>",'font':{'size':22,'color':'#0b2545'}}, gauge={'shape':'angular','axis':{'range':[0,vmax],'tickwidth':1.2,'tickcolor':'#0b2545'},'bar':{'color':color,'thickness':0.28},'bgcolor':'white','borderwidth':1,'bordercolor':'#cfe0ff','steps':[{'range':[0,0.6*vmax],'color':'rgba(0,0,0,0.04)'},{'range':[0.6*vmax,0.85*vmax],'color':'rgba(0,0,0,0.07)'}],'threshold':{'line':{'color':'green' if color=='#d62728' else 'red','width':4},'thickness':0.9,'value':float(val)}}))
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

def fallback_fast_solver(state, rng):
    t = np.linspace(0, 30 * 365, 360)
    L, xf, hf, pad_interf, nlats = float(state["L_ft"]), float(state["xf_ft"]), float(state["hf_ft"]), float(state["pad_interf"]), int(state["n_laterals"])
    richness = float(state.get("Rs_pb_scf_stb", 650.0)) / max(1.0, float(state.get("pb_psi", 5200.0)))
    geo_g = (L / 10000.0)**0.85 * (xf / 300.0)**0.55 * (hf / 180.0)**0.20
    geo_o = (L / 10000.0)**0.85 * (xf / 300.0)**0.40 * (hf / 180.0)**0.30
    interf_mul = 1.0 / (1.00 + 1.25*pad_interf + 0.35*max(0, nlats - 1))
    if st.session_state.get("fluid_model", "unconventional") == "unconventional":
        qi_g_base, qi_o_base = 12000.0, 1000.0
        rich_g, rich_o = 1.0 + 0.30 * np.clip(richness, 0.0, 1.4), 1.0 + 0.12 * np.clip(richness, 0.0, 1.4)
        qi_g, qi_o = np.clip(qi_g_base * geo_g * interf_mul * rich_g, 3000.0, 28000.0), np.clip(qi_o_base * geo_o * interf_mul * rich_o, 400.0, 2500.0)
        Di_g_yr, b_g, Di_o_yr, b_o = 0.60, 0.85, 0.50, 1.00
    else:
        qi_g_base, qi_o_base = 8000.0, 1600.0
        rich_g, rich_o = 1.0 + 0.05 * np.clip(richness, 0.0, 1.4), 1.0 + 0.08 * np.clip(richness, 0.0, 1.4)
        qi_g, qi_o = np.clip(qi_g_base * geo_g * interf_mul * rich_g, 2000.0, 18000.0), np.clip(qi_o_base * geo_o * interf_mul * rich_o, 700.0, 3500.0)
        Di_g_yr, b_g, Di_o_yr, b_o = 0.45, 0.80, 0.42, 0.95
    Di_g, Di_o = Di_g_yr / 365.0, Di_o_yr / 365.0
    qg, qo = qi_g / (1.0 + b_g * Di_g * t)**(1.0/b_g), qi_o / (1.0 + b_o * Di_o * t)**(1.0/b_o)
    EUR_g_BCF, EUR_o_MMBO = np.trapz(qg, t) / 1e6, np.trapz(qo, t) / 1e6
    return dict(t=t, qg=qg, qo=qo, EUR_g_BCF=EUR_g_BCF, EUR_o_MMBO=EUR_o_MMBO)

def _get_sim_preview():
    if 'state' in globals(): tmp = state.copy()
    else: tmp = {k: st.session_state[k] for k in list(defaults.keys()) if k in st.session_state}
    rng_preview = np.random.default_rng(int(st.session_state.get("rng_seed", 1234)) + 999)
    return fallback_fast_solver(tmp, rng_preview)

def run_full_3d_simulation(state):
    t0 = time.time()
    inputs = {'grid':{'nx':int(state['nx']),'ny':int(state['ny']),'nz':int(state['nz']),'dx':float(state['dx']),'dy':float(state['dy']),'dz':float(state['dz'])},'rock':{'kx_md':st.session_state.get('kx'),'ky_md':st.session_state.get('ky'),'phi':st.session_state.get('phi'),'Ti_mult':1.0,'Tj_mult':1.0},'pvt':{'pb_psi':float(state['pb_psi']),'Rs_pb_scf_stb':float(state['Rs_pb_scf_stb']),'Bo_pb_rb_stb':float(state['Bo_pb_rb_stb']),'muo_pb_cp':float(state['muo_pb_cp']),'mug_pb_cp':float(state['mug_pb_cp']),'a_g':float(state['a_g']),'z_g':float(state['z_g']),'ct_o_1psi':float(state['ct_o_1psi']),'ct_g_1psi':float(state['ct_g_1psi']),'ct_w_1psi':float(state['ct_w_1psi'])},'relperm':{'krw_end':float(state['krw_end']),'kro_end':float(state['kro_end']),'nw':float(state['nw']),'no':float(state['no']),'Swc':float(state['Swc']),'Sor':float(state['Sor'])},'schedule':{'control':state['pad_ctrl'],'bhp_psi':float(state['pad_bhp_psi']),'rate_mscfd':float(state['pad_rate_mscfd'])},'msw':{'laterals':int(state['n_laterals']),'L_ft':float(state['L_ft']),'stage_spacing_ft':float(state['stage_spacing_ft']),'clusters_per_stage':int(state['clusters_per_stage']),'dp_limited_entry_psi':float(state['dP_LE_psi']),'friction_factor':float(state['f_fric']),'well_ID_ft':float(state['wellbore_ID_ft']),'xf_ft':float(state['xf_ft']),'hf_ft':float(state['hf_ft']),'weights':[]},'stress':{'CfD0':0.0,'alpha_sigma':0.0,'sigma_overburden_psi':8500.0,'refrac_day':0,'refrac_recovery':0},'init':{'p_init_psi':float(state['p_init_psi']),'pwf_min_psi':float(state['p_min_bhp_psi']),'Sw_init':float(state['Swc'])},'include_rs_in_mb': bool(state['include_RsP'])}
    try: engine_results = simulate(inputs)
    except Exception as e: st.error(f"Error in full3d.py engine: {e}"); return None
    t, qg, qo = engine_results.get('t_days'), engine_results.get('qg_Mscfd'), engine_results.get('qo_STBpd')
    if t is None or qg is None or qo is None: st.error("Engine missing required data (t_days, qg_Mscfd, qo_STBpd)."); return None
    EUR_g_BCF, EUR_o_MMBO = np.trapz(qg, t)/1e6, np.trapz(qo, t)/1e6
    return {'t':t,'qg':qg,'qo':qo,'press_matrix':engine_results.get('p3d_psi'),'press_frac_mid':engine_results.get('pf_mid_psi'),'press_matrix_mid':engine_results.get('pm_mid_psi'),'Sw_mid':engine_results.get('Sw_mid'),'EUR_g_BCF':EUR_g_BCF,'EUR_o_MMBO':EUR_o_MMBO,'runtime_s':time.time()-t0}

def run_simulation(state):
    if st.session_state.get('kx') is None:
        rng = np.random.default_rng(int(st.session_state.rng_seed))
        nz,ny,nx = int(state["nz"]),int(state["ny"]),int(state["nx"])
        kx_mid, ky_mid, phi_mid = 0.05+state["k_stdev"]*rng.standard_normal((ny,nx)), (0.05/state["anis_kxky"])+state["k_stdev"]*rng.standard_normal((ny,nx)), 0.10+state["phi_stdev"]*rng.standard_normal((ny,nx))
        kz_scale = np.linspace(0.95,1.05,nz)[:,None,None]
        st.session_state.kx, st.session_state.ky, st.session_state.phi = np.clip(kx_mid[None,...]*kz_scale,1e-4,None), np.clip(ky_mid[None,...]*kz_scale,1e-4,None), np.clip(phi_mid[None,...]*kz_scale,0.01,0.35)
        st.info("Generated 3D rock properties for the simulation.")
    result = run_full_3d_simulation(state)
    if result is None:
        st.warning("Full 3D simulation failed. Showing results from fast preview solver.")
        result = fallback_fast_solver(state, np.random.default_rng(int(st.session_state.rng_seed)))
    for key in ["press_matrix", "press_frac", "So", "Sw"]:
        if key in result and result.get(key) is not None:
            result[key], result[f"{key}_mid"] = ensure_3d(result[key]), get_k_slice(result[key], result[key].shape[0]//2)
        elif f"{key}_mid" in result and result.get(f"{key}_mid") is not None:
            result[key] = ensure_3d(result[f"{key}_mid"])
    return result

# ------------------------ SIDEBAR AND MAIN APP LAYOUT ------------------------
with st.sidebar:
    st.markdown("## Play Preset")
    model_choice = st.selectbox("Model", ["3D Unconventional Reservoir Simulator — Implicit Engine Ready","3D Black Oil Reservoir Simulator — Implicit Engine Ready"], key="sim_mode")
    st.session_state.fluid_model = "black_oil" if "Black Oil" in model_choice else "unconventional"
    play = st.selectbox("Shale play", list(PLAY_PRESETS.keys()), index=0, key="play_sel")
    if st.button("Apply preset", use_container_width=True):
        payload = defaults.copy(); payload.update(PLAY_PRESETS[st.session_state.play_sel])
        if st.session_state.fluid_model == "black_oil": payload.update(dict(Rs_pb_scf_stb=0.0,pb_psi=1.0,Bo_pb_rb_stb=1.00,mug_pb_cp=0.020,a_g=0.15,p_init_psi=max(3500.0, float(payload.get("p_init_psi", 5200.0))),pad_ctrl="BHP",pad_bhp_psi=min(float(payload.get("p_init_psi", 5200.0)) - 500.0, 3000.0)))
        st.session_state.sim, st.session_state.apply_preset_payload = None, payload
        _safe_rerun()

    st.markdown("### Grid (ft)")
    c1,c2,c3 = st.columns(3); c1.number_input("nx",10,500,key="nx"); c2.number_input("ny",10,500,key="ny"); c3.number_input("nz",1,200,key="nz")
    c1,c2,c3 = st.columns(3); c1.number_input("dx (ft)",step=1.0,key="dx"); c2.number_input("dy (ft)",step=1.0,key="dy"); c3.number_input("dz (ft)",step=1.0,key="dz")
    # ... (all other sidebar controls from the original file)

state = {k: st.session_state[k] for k in defaults.keys() if k in st.session_state}
tab_names = ["Setup Preview","Generate 3D property volumes (kx, ky, ϕ)","PVT (Black-Oil)","MSW Wellbore","RTA","Results","3D Viewer","Slice Viewer","QA / Material Balance","EUR vs Lateral Length","Field Match (CSV)","Uncertainty & Monte Carlo","User’s Manual","Solver & Profiling","DFN Viewer"]
tabs = st.tabs(tab_names)

with tabs[0]: st.header("Setup Preview")

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
    Rs, Bo, Bg, mug = Rs_of_p(P,state["pb_psi"],state["Rs_pb_scf_stb"]), Bo_of_p(P,state["pb_psi"],state["Bo_pb_rb_stb"]), Bg_of_p(P), mu_g_of_p(P,state["pb_psi"],state["mug_pb_cp"])
    f1=go.Figure(); f1.add_trace(go.Scatter(x=P,y=Rs,line=dict(color="firebrick",width=3))); f1.add_vline(x=state["pb_psi"],line_dash="dash",line_width=2,annotation_text="Bubble Point"); f1.update_layout(template="plotly_white",title="<b>P1. Solution GOR Rs vs Pressure</b>",xaxis_title="Pressure (psi)",yaxis_title="Rs (scf/STB)"); st.plotly_chart(f1,use_container_width=True)
    f2=go.Figure(); f2.add_trace(go.Scatter(x=P,y=Bo,line=dict(color="seagreen",width=3))); f2.add_vline(x=state["pb_psi"],line_dash="dash",line_width=2,annotation_text="Bubble Point"); f2.update_layout(template="plotly_white",title="<b>P2. Oil FVF Bo vs Pressure</b>",xaxis_title="Pressure (psi)",yaxis_title="Bo (rb/STB)"); st.plotly_chart(f2,use_container_width=True)
    f3=go.Figure(); f3.add_trace(go.Scatter(x=P,y=Bg,line=dict(color="steelblue",width=3))); f3.add_vline(x=state["pb_psi"],line_dash="dash",line_width=2); f3.update_layout(template="plotly_white",title="<b>P3. Gas FVF Bg vs Pressure</b>",xaxis_title="Pressure (psi)",yaxis_title="Bg (rb/scf)"); st.plotly_chart(f3,use_container_width=True)
    f4=go.Figure(); f4.add_trace(go.Scatter(x=P,y=mug,line=dict(color="mediumpurple",width=3))); f4.add_vline(x=state["pb_psi"],line_dash="dash",line_width=2); f4.update_layout(template="plotly_white",title="<b>P4. Gas viscosity μg vs Pressure</b>",xaxis_title="Pressure (psi)",yaxis_title="μg (cP)"); st.plotly_chart(f4,use_container_width=True)

with tabs[3]:
    st.header("MSW Wellbore Physics — Heel–Toe & Limited-Entry")
    st.info("This chart shows pseudo-frictional pressure drop from heel to toe. Stage markers (vertical dotted lines) indicate limited-entry points.")

with tabs[4]:
    st.header("RTA — Quick Diagnostics")
    st.info("**Interpretation:** Rate Transient Analysis (RTA) helps diagnose flow regimes...")
    sim_data = st.session_state.sim if st.session_state.sim is not None else _get_sim_preview()
    t, qg = sim_data["t"], sim_data["qg"]
    rate_y_mode_rta = st.radio("Rate y-axis", ["Linear", "Log"], index=0, horizontal=True, key="rta_rate_y_mode_unique") # UNIQUE KEY
    y_type_rta = "log" if rate_y_mode_rta == "Log" else "linear"
    fig = go.Figure(); fig.add_trace(go.Scatter(x=t, y=qg, line=dict(color="firebrick", width=3), name="Gas")); fig.update_layout(**semi_log_layout("R1. Gas rate (q) vs time", yaxis="q (Mscf/d)")); fig.update_yaxes(type=y_type_rta); st.plotly_chart(fig, use_container_width=True)
    logt, logq = np.log10(np.maximum(t, 1e-9)), np.log10(np.maximum(qg, 1e-9))
    slope = np.gradient(logq, logt)
    fig2 = go.Figure(); fig2.add_trace(go.Scatter(x=t, y=slope, line=dict(color="teal", width=3), name="dlogq/dlogt")); fig2.update_layout(**semi_log_layout("R2. Log-log derivative", yaxis="Slope")); st.plotly_chart(fig2, use_container_width=True)

with tabs[5]:
    st.header("Simulation Results")
    if st.button("Run simulation", type="primary"):
        with st.spinner("Running full 3D simulation..."): st.session_state.sim = run_simulation(state)
    if st.session_state.sim: st.success(f"Simulation complete in {st.session_state.sim.get('runtime_s', 0):.2f} seconds.")
    else: st.info("Click **Run simulation** to compute full results.")

with tabs[6]:
    st.header("3D Viewer — Pressure / Saturations (Isosurface/Volume)")
    if st.session_state.sim is None: st.info("Run a simulation to view 3D volumes.")
    else: st.info("**Interpretation:** This tool visualizes the 3D distribution of pressure or fluid saturations...")

with tabs[7]:
    st.header("Slice Viewer — k / i / j slices")
    if st.session_state.sim is None: st.info("Run a simulation to view slices.")
    else: st.info("**Interpretation:** This tool lets you inspect 2D cross-sections of the 3D data volumes...")

with tabs[8]:
    st.header("QA / Material Balance")
    st.info("**Interpretation:** These plots check for material balance closure...")
    if st.session_state.sim is not None:
        sim = st.session_state.sim
        t = sim["t"]
        EUR_g_BCF_mb, EUR_o_MMBO_mb = float(np.trapz(sim["qg"], t)/1e6), float(np.trapz(sim["qo"], t)/1e6)
        g_g_mb, o_g_mb = eur_gauges(EUR_g_BCF_mb, EUR_o_MMBO_mb)
        c1,c2 = st.columns(2)
        with c1: st.plotly_chart(g_g_mb, use_container_width=True)
        with c2: st.plotly_chart(o_g_mb, use_container_width=True)
        dt = np.gradient(t)
        cum_g_BCF, cum_o_MMBO = np.cumsum(dt*sim["qg"])/1e6, np.cumsum(dt*sim["qo"])/1e6
        res_g_BCF, res_o_MMBO = cum_g_BCF[-1]-cum_g_BCF, cum_o_MMBO[-1]-cum_o_MMBO
        fg = go.Figure(); fg.add_trace(go.Scatter(x=t, y=cum_g_BCF, name="Gas produced (BCF)", line=dict(color="#d62728",width=3))); fg.add_trace(go.Scatter(x=t,y=res_g_BCF,name="Gas residual (BCF)",line=dict(color="#ff9896",width=2,dash="dot"))); fg.update_layout(**semi_log_layout("Gas Material Balance", yaxis="BCF")); st.plotly_chart(fg,use_container_width=True)
        fo = go.Figure(); fo.add_trace(go.Scatter(x=t, y=cum_o_MMBO, name="Oil produced (MMBO)", line=dict(color="#2ca02c",width=3))); fo.add_trace(go.Scatter(x=t,y=res_o_MMBO,name="Oil residual (MMBO)",line=dict(color="#98df8a",width=2,dash="dot"))); fo.update_layout(**semi_log_layout("Oil Material Balance", yaxis="MMBO")); st.plotly_chart(fo,use_container_width=True)
    else: st.info("Run a simulation to view the Material Balance plots.")

with tabs[9]:
    st.header("Sensitivity: EUR vs Lateral Length")
    st.info("Dual view: the Dual Axis tab gives a compact overview, while Stacked Panels separates the series for maximum readability.")
    L_grid = np.array([6000,8000,10000,12000,14000],float)
    rows = [dict(L_ft=int(L), **{k:v for k,v in fallback_fast_solver({**state,"L_ft":float(L)},np.random.default_rng(123)).items() if "EUR" in k}) for L in L_grid]
    df = pd.DataFrame(rows)
    # ... (plotting logic from original file)
    st.dataframe(df, use_container_width=True)

with tabs[10]:
    st.header("Field Match (CSV)")
    st.info("Upload a CSV with historical production (time_days, qg_Mscfd, qo_STBpd)...")
    up = st.file_uploader("Upload CSV", type=["csv"], key="field_csv_uploader")
    if up is not None:
        # ... (full field match logic from original file)
        pass
    else:
        st.warning("Upload a CSV to run the history match.")

with tabs[11]:
    st.header("Uncertainty & Monte Carlo")
    st.info("**Interpretation:** This tab runs a Monte Carlo simulation...")
    N = st.slider("Samples", 50, 500, 150, 10, key="mc_samples")
    # ... (full monte carlo logic from original file)

with tabs[12]:
    st.header("User’s Manual")
    st.markdown("""**Overview:** This application supports full 3D arrays...""")
    st.code('{\n  "t": [days],\n  ...\n}\n', language="json")

with tabs[13]:
    st.header("Solver & Profiling")
    st.info("**Interpretation:** Advanced controls for numerical solver tolerances and performance flags.")
    st.markdown(f"**Newton Tolerance:** `{state['newton_tol']:.1e}`\n...")

with tabs[14]:
    st.header("DFN Viewer — 3D line segments")
    segs = st.session_state.dfn_segments
    if segs is None or len(segs) == 0:
        st.info("No DFN loaded. Upload a CSV or use 'Generate DFN from stages' in the sidebar.")
    else:
        st.info("**Interpretation:** Displays the DFN as 3D line segments for QC.")
        figd = go.Figure()
        for i, seg in enumerate(segs):
            figd.add_trace(go.Scatter3d(x=[seg[0], seg[3]], y=[seg[1], seg[4]], z=[seg[2], seg[5]], mode="lines", line=dict(width=6, color="red"), name="DFN" if i == 0 else None, showlegend=(i == 0)))
        figd.update_layout(template="plotly_white", scene=dict(xaxis_title="x (ft)", yaxis_title="y (ft)", zaxis_title="z (ft)"), height=640, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(figd, use_container_width=True)
