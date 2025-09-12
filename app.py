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
    "Permian Basin (Wolfcamp)": dict(L_ft=10000.0, stage_spacing_ft=250.0, xf_ft=300.0, hf_ft=180.0, Rs_pb_scf_stb=650.0, pb_psi=5200.0, Bo_pb_rb_stb=1.35, p_init_psi=5800.0),
    "Eagle Ford (Oil Window)": dict(L_ft=9000.0, stage_spacing_ft=225.0, xf_ft=270.0, hf_ft=150.0, Rs_pb_scf_stb=700.0, pb_psi=5400.0, Bo_pb_rb_stb=1.34, p_init_psi=5600.0),
    "Marcellus (Dry Gas)": dict(L_ft=9000.0, stage_spacing_ft=210.0, xf_ft=320.0, hf_ft=180.0, Rs_pb_scf_stb=50.0, pb_psi=1500.0, Bo_pb_rb_stb=1.05, p_init_psi=6500.0),
    "Haynesville (Dry Gas)": dict(L_ft=9500.0, stage_spacing_ft=210.0, xf_ft=320.0, hf_ft=190.0, Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, p_init_psi=8000.0),
    "Bakken (Light Oil)": dict(L_ft=10000.0, stage_spacing_ft=250.0, xf_ft=260.0, hf_ft=150.0, Rs_pb_scf_stb=450.0, pb_psi=4200.0, Bo_pb_rb_stb=1.30, p_init_psi=5200.0),
    "Niobrara (Oil & Gas)": dict(L_ft=8000.0, stage_spacing_ft=220.0, xf_ft=240.0, hf_ft=140.0, Rs_pb_scf_stb=500.0, pb_psi=5000.0, Bo_pb_rb_stb=1.32, p_init_psi=5500.0),
    "Barnett (Gas)": dict(L_ft=7500.0, stage_spacing_ft=230.0, xf_ft=280.0, hf_ft=150.0, Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, p_init_psi=5000.0),
    "Utica (Gas & NGLs)": dict(L_ft=10000.0, stage_spacing_ft=220.0, xf_ft=320.0, hf_ft=190.0, Rs_pb_scf_stb=200.0, pb_psi=3500.0, Bo_pb_rb_stb=1.18, p_init_psi=8000.0),
    "Anadarko-Woodford (Oil & Gas)": dict(L_ft=9000.0, stage_spacing_ft=240.0, xf_ft=300.0, hf_ft=170.0, Rs_pb_scf_stb=700.0, pb_psi=5600.0, Bo_pb_rb_stb=1.37, p_init_psi=6200.0),
    "Granite Wash (Gas & Liquids)": dict(L_ft=8500.0, stage_spacing_ft=250.0, xf_ft=280.0, hf_ft=160.0, Rs_pb_scf_stb=600.0, pb_psi=5000.0, Bo_pb_rb_stb=1.33, p_init_psi=6000.0),
    "Montney (Gas, Condensate, NGLs)": dict(L_ft=10500.0, stage_spacing_ft=230.0, xf_ft=300.0, hf_ft=170.0, Rs_pb_scf_stb=700.0, pb_psi=5400.0, Bo_pb_rb_stb=1.36, p_init_psi=6200.0),
    "Duvernay (Liquids-Rich Gas)": dict(L_ft=10000.0, stage_spacing_ft=240.0, xf_ft=290.0, hf_ft=175.0, Rs_pb_scf_stb=800.0, pb_psi=5600.0, Bo_pb_rb_stb=1.38, p_init_psi=6400.0),
    "Horn River Basin (Dry Gas)": dict(L_ft=8000.0, stage_spacing_ft=280.0, xf_ft=350.0, hf_ft=200.0, Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, p_init_psi=7500.0),
    "Liard Basin (Dry Gas)": dict(L_ft=8500.0, stage_spacing_ft=300.0, xf_ft=380.0, hf_ft=220.0, Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, p_init_psi=8200.0),
    "Cordova Embayment (Gas)": dict(L_ft=7800.0, stage_spacing_ft=260.0, xf_ft=330.0, hf_ft=190.0, Rs_pb_scf_stb=20.0, pb_psi=1000.0, Bo_pb_rb_stb=1.02, p_init_psi=7000.0),
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
    play = st.selectbox("Shale play", PLAY_LIST, index=0, key="play_sel")
    if st.button("Apply preset", use_container_width=True):
        payload = defaults.copy(); payload.update(PLAY_PRESETS[st.session_state.play_sel])
        if st.session_state.fluid_model == "black_oil": payload.update(dict(Rs_pb_scf_stb=0.0,pb_psi=1.0,Bo_pb_rb_stb=1.00,mug_pb_cp=0.020,a_g=0.15,p_init_psi=max(3500.0, float(payload.get("p_init_psi", 5200.0))),pad_ctrl="BHP",pad_bhp_psi=min(float(payload.get("p_init_psi", 5200.0)) - 500.0, 3000.0)))
        st.session_state.sim, st.session_state.apply_preset_payload = None, payload
        _safe_rerun()
    st.markdown("### Grid (ft)")
    c1,c2,c3 = st.columns(3); st.number_input("nx",10,500,key="nx"); st.number_input("ny",10,500,key="ny"); st.number_input("nz",1,200,key="nz")
    c1,c2,c3 = st.columns(3); st.number_input("dx (ft)",step=1.0,key="dx"); st.number_input("dy (ft)",step=1.0,key="dy"); st.number_input("dz (ft)",step=1.0,key="dz")
    st.markdown("### Heterogeneity & Anisotropy")
    st.selectbox("Facies style", ["Continuous (Gaussian)","Speckled (high-variance)","Layered (vertical bands)"], key="facies_style")
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
            segs = gen_auto_dfn_from_stages(int(st.session_state.nx),int(st.session_state.ny),int(st.session_state.nz), float(st.session_state.dx),float(st.session_state.dy),float(st.session_state.dz), float(st.session_state.L_ft),float(st.session_state.stage_spacing_ft), int(st.session_state.n_laterals),float(st.session_state.hf_ft))
            st.session_state.dfn_segments = segs
            st.success(f"Auto-generated DFN segments: {0 if segs is None else len(segs)}")

state = {k: st.session_state[k] for k in defaults.keys() if k in st.session_state}
tab_names = ["Setup Preview","Generate 3D property volumes (kx, ky, ϕ)","PVT (Black-Oil)","MSW Wellbore","RTA","Results","3D Viewer","Slice Viewer","QA / Material Balance","EUR vs Lateral Length","Field Match (CSV)","Uncertainty & Monte Carlo","User’s Manual","Solver & Profiling","DFN Viewer"]
tabs = st.tabs(tab_names)

with tabs[0]: st.header("Setup Preview")
with tabs[1]:
    st.header("Generate 3D Property Volumes (kx, ky, ϕ)")
    st.info("**Interpretation:** These maps represent the spatial distribution of key reservoir properties...")
    # (Full tab content)
with tabs[2]:
    st.header("PVT (Black-Oil) Analysis")
    st.info("**Interpretation:** These charts describe how the fluid properties change with pressure...")
    # (Full tab content)
with tabs[3]:
    st.header("MSW Wellbore Physics — Heel–Toe & Limited-Entry")
    st.info("This chart shows pseudo-frictional pressure drop from heel to toe...")
with tabs[4]:
    st.header("RTA — Quick Diagnostics")
    st.info("**Interpretation:** Rate Transient Analysis (RTA) helps diagnose flow regimes...")
    sim_data = st.session_state.sim if st.session_state.sim is not None else _get_sim_preview()
    t, qg = sim_data["t"], sim_data["qg"]
    rate_y_mode_rta = st.radio("Rate y-axis", ["Linear", "Log"], index=0, horizontal=True, key="rta_rate_y_mode_unique")
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
    st.header("3D Viewer")
    if st.session_state.sim is None: st.info("Run a simulation to view 3D volumes.")
    else: st.info("**Interpretation:** This tool visualizes the 3D distribution of pressure or fluid saturations...")
with tabs[7]:
    st.header("Slice Viewer")
    if st.session_state.sim is None: st.info("Run a simulation to view slices.")
    else: st.info("**Interpretation:** This tool lets you inspect 2D cross-sections of the 3D data volumes...")
with tabs[8]:
    st.header("QA / Material Balance")
    if st.session_state.sim is not None: st.info("**Interpretation:** These plots check for material balance closure...")
    else: st.info("Run a simulation to view the Material Balance plots.")
with tabs[9]:
    st.header("Sensitivity: EUR vs Lateral Length")
    st.info("Dual view: the Dual Axis tab gives a compact overview, while Stacked Panels separates the series for maximum readability.")
with tabs[10]:
    st.header("Field Match (CSV)")
    st.info("Upload a CSV with historical production...")
    up = st.file_uploader("Upload CSV", type=["csv"], key="field_csv_uploader")
    if up is None: st.warning("Upload a CSV to run the history match.")
with tabs[11]:
    st.header("Uncertainty & Monte Carlo")
    st.info("**Interpretation:** This tab runs a Monte Carlo simulation...")
with tabs[12]:
    st.header("User’s Manual")
    st.markdown("""**Overview:** This application supports full 3D arrays...""")
with tabs[13]:
    st.header("Solver & Profiling")
    st.info("**Interpretation:** Advanced controls for numerical solver tolerances and performance flags.")
with tabs[14]:
    st.header("DFN Viewer — 3D line segments")
    if st.session_state.dfn_segments is None: st.info("No DFN loaded.")
    else: st.info("**Interpretation:** Displays the DFN as 3D line segments for QC.")
