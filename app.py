# Full 3D Unconventional / Black-Oil Reservoir Simulator — Implicit Engine Ready (USOF units) + DFN support
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy import stats
from scipy.integrate import cumulative_trapezoid
import numpy_financial as npf  # Economics Tab
from core.full3d import simulate
from engines.fast import fallback_fast_solver  # used in preview & fallbacks

# ------------------------ Utils ------------------------
def _setdefault(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

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
# --- Model Type options (must match the sidebar selectbox) ---
VALID_MODEL_TYPES = ["Unconventional Reservoir", "Black Oil Reservoir"]
# Default to the first allowed value
_setdefault("sim_mode", VALID_MODEL_TYPES[0])
_setdefault("dfn_segments", None)
_setdefault("use_dfn_sink", True)
_setdefault("use_auto_dfn", True)
_setdefault("vol_downsample", 2)
_setdefault("iso_value_rel", 0.5)
defaults = dict(
    nx=300, ny=60, nz=12,
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
    engine_type="Analytical Model (Fast Proxy)"  # Set stable engine as default
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
    "Permian Basin (Wolfcamp)": dict(L_ft=10000.0, stage_spacing_ft=250.0, xf_ft=300.0, hf_ft=180.0,
                                     Rs_pb_scf_stb=650.0, pb_psi=5200.0, Bo_pb_rb_stb=1.35, p_init_psi=5800.0),
    "Eagle Ford (Oil Window)": dict(L_ft=9000.0, stage_spacing_ft=225.0, xf_ft=270.0, hf_ft=150.0,
                                    Rs_pb_scf_stb=700.0, pb_psi=5400.0, Bo_pb_rb_stb=1.34, p_init_psi=5600.0),
}
PLAY_LIST = list(PLAY_PRESETS.keys())

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
    pb     = float(state.get('pb_psi', 1.0))
    Rs_pb  = float(state.get('Rs_pb_scf_stb', 0.0))
    Bo_pb  = float(state.get('Bo_pb_rb_stb', 1.0))
    mug_pb = float(state.get('mug_pb_cp', 0.020))
    muo_pb = float(state.get('muo_pb_cp', 1.20))

    def Rs(p):   return Rs_of_p(p, pb, Rs_pb)
    def Bo(p):   return Bo_of_p(p, pb, Bo_pb)
    def Bg(p):   return Bg_of_p(p)
    def mu_g(p): return mu_g_of_p(p, pb, mug_pb)
    def mu_o(p): return np.full_like(np.asarray(p, float), muo_pb, dtype=float)

    return _PVTAdapter(
        Rs=Rs, Bo=Bo, Bg=Bg, mu_g=mu_g, mu_o=mu_o,
        ct_o_1psi=state.get('ct_o_1psi', 8e-6),
        ct_g_1psi=state.get('ct_g_1psi', 3e-6),
        ct_w_1psi=state.get('ct_w_1psi', 3e-6),
        include_RsP=bool(state.get('include_RsP', True)),
        pb_psi=pb
    )

# --- Defensive monkey-patch: if engine's Fluid class lacks methods, inject thin wrappers ---
def _monkeypatch_engine_fluid_if_needed(adapter):
    """
    Some engine builds instantiate their own `Fluid` and expect .Rs/.Bo/.Bg/.mu_g/.mu_o.
    If missing, attach wrappers that forward to our adapter. Safe no-op if import fails.
    """
    try:
        from core.blackoil_pvt1 import Fluid as EngineFluid  # optional; may not exist in all builds
        patched = []
        if not hasattr(EngineFluid, "Rs"):
            EngineFluid.Rs   = lambda self, p: adapter.Rs(p)
            patched.append("Rs")
        if not hasattr(EngineFluid, "Bo"):
            EngineFluid.Bo   = lambda self, p: adapter.Bo(p)
            patched.append("Bo")
        if not hasattr(EngineFluid, "Bg"):
            EngineFluid.Bg   = lambda self, p: adapter.Bg(p)
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
    def g(val, label, suffix, color, vmax):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(val),
            number={'suffix': f" {suffix}", 'font': {'size': 44, 'color': '#0b2545'}},
            title={'text': f"<b>{label}</b>", 'font': {'size': 22, 'color': '#0b2545'}},
            gauge={'shape': 'angular',
                   'axis': {'range': [0, vmax], 'tickwidth': 1.2, 'tickcolor': '#0b2545'},
                   'bar': {'color': color, 'thickness': 0.28},
                   'bgcolor': 'white', 'borderwidth': 1, 'bordercolor': '#cfe0ff'},
            steps=[{'range': [0, 0.6 * vmax], 'color': 'rgba(0,0,0,0.04)'},
                   {'range': [0.6 * vmax, 0.85 * vmax], 'color': 'rgba(0,0,0,0.07)'}],
            threshold={'line': {'color': 'green' if color == '#d62728' else 'red', 'width': 4},
                       'thickness': 0.9, 'value': float(val)}
        ))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="#ffffff")
        return fig
    gmax = max(1.0, np.ceil(EUR_g_BCF / 5.0) * 5.0)
    omax = max(0.5, np.ceil(EUR_o_MMBO / 0.5) * 0.5)
    return g(EUR_g_BCF, "EUR Gas", "BCF", "#d62728", gmax), g(EUR_o_MMBO, "EUR Oil", "MMBO", "#2ca02c", omax)

def semi_log_layout(title, xaxis="Day (log scale)", yaxis="Rate"):
    return dict(
        title=f"<b>{title}</b>", template="plotly_white",
        xaxis=dict(type="log", title=xaxis, showgrid=True, gridcolor="rgba(0,0,0,0.15)"),
        yaxis=dict(title=yaxis, showgrid=True, gridcolor="rgba(0,0,0,0.15)"),
        legend=dict(orientation="h")
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

# --- Engine wrapper ---
def run_simulation_engine(state):
    t0 = time.time()
    inputs = {k: state.get(k) for k in defaults.keys()}
    inputs['engine_type'] = state.get('engine_type')
    inputs.update({
        'grid': {k: state.get(k) for k in ['nx', 'ny', 'nz', 'dx', 'dy', 'dz']},
        'rock': {
            'kx_md': st.session_state.get('kx'),
            'ky_md': st.session_state.get('ky'),
            'phi':   st.session_state.get('phi'),
        },
        'relperm': {k: state.get(k) for k in ['krw_end','kro_end','nw','no','Swc','Sor']},
        'init':    {'p_init_psi': state.get('p_init_psi'), 'Sw_init': state.get('Swc')},
        'schedule': {'bhp_psi': state.get('pad_bhp_psi'), 'rate_mscfd': state.get('pad_rate_mscfd'), 'control': state.get('pad_ctrl', 'BHP')},
        'msw': {'laterals': state.get('n_laterals'), 'L_ft': state.get('L_ft'), 'stage_spacing_ft': state.get('stage_spacing_ft'), 'hf_ft': state.get('hf_ft')},
    })
    inputs['pvt'] = _pvt_from_state(state)

    try:
        engine_results = simulate(inputs)
    except Exception as e:
        st.error(f"Simulation error: {e}")
        return None

    def _pick(d, *keys):
        for k in keys:
            v = d.get(k)
            if v is not None:
                return v
        return None

    t  = _pick(engine_results, 't',  't_days', 'time_days')
    qg = _pick(engine_results, 'qg', 'qg_Mscfd', 'qg_mscfd')
    qo = _pick(engine_results, 'qo', 'qo_STBpd', 'qo_stbpd')

    if t is None or qg is None or qo is None:
        st.error("Engine missing required arrays (t/qg/qo).")
        return None

    EUR_g_BCF  = np.trapz(qg, t) / 1e6
    EUR_o_MMBO = np.trapz(qo, t) / 1e6

    engine_results['t'] = t
    engine_results['qg'] = qg
    engine_results['qo'] = qo
    engine_results['runtime_s'] = time.time() - t0
    engine_results['EUR_g_BCF'] = EUR_g_BCF
    engine_results['EUR_o_MMBO'] = EUR_o_MMBO
    return engine_results
# ------------------------ SIDEBAR AND MAIN APP LAYOUT ------------------------
with st.sidebar:
    st.markdown("## Simulation Setup")
    st.markdown("### Engine & Presets")

    # Unique UI key; mirror into canonical key used in code
    engine_type_ui = st.selectbox(
        "Engine Type",
        ["Analytical Model (Fast Proxy)", "3D Three-Phase Implicit (Phase 1b)"],
        key="engine_type_ui",
        help="Select the core calculation engine. The implicit model is in development, while the analytical model is a stable, fast approximation.",
    )
    st.session_state["engine_type"] = engine_type_ui

   model_choice = st.selectbox("Model Type", VALID_MODEL_TYPES, key="sim_mode")
st.session_state.fluid_model = "black_oil" if "Black Oil" in model_choice else "unconventional"

    play = st.selectbox("Shale Play Preset", PLAY_LIST, index=0, key="play_sel")
    if st.button("Apply Preset", use_container_width=True):
        payload = defaults.copy()
        payload.update(PLAY_PRESETS[st.session_state.play_sel])
        if st.session_state.fluid_model == "black_oil":
            payload.update(dict(
                Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00, mug_pb_cp=0.020, a_g=0.15,
                p_init_psi=max(3500.0, float(payload.get("p_init_psi", 5200.0))),
                pad_ctrl="BHP",
                pad_bhp_psi=min(float(payload.get("p_init_psi", 5200.0)) - 500.0, 3000.0),
            ))
        st.session_state.sim, st.session_state.apply_preset_payload = None, payload
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

state = {k: st.session_state[k] for k in defaults.keys() if k in st.session_state}

# --- Tab list ---
tab_names = [
    "Setup Preview", "Generate 3D property volumes", "PVT (Black-Oil)", "MSW Wellbore", "RTA", "Results",
    "3D Viewer", "Slice Viewer", "QA / Material Balance", "Economics", "EUR vs Lateral Length", "Field Match (CSV)",
    "Uncertainty & Monte Carlo", "Well Placement Optimization", "User’s Manual", "Solver & Profiling", "DFN Viewer"
]

st.write(
    '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} '
    '.stRadio > label {display:none;} '
    'div.row-widget.stRadio > div > div {border: 1px solid #ccc; padding: 6px 12px; border-radius: 4px; margin: 2px; background-color: #f0f2f6;} '
    'div.row-widget.stRadio > div > div[aria-checked="true"] {background-color: #e57373; color: white; border-color: #d32f2f;}</style>',
    unsafe_allow_html=True
)
selected_tab = st.radio("Navigation", tab_names, label_visibility="collapsed")

# ------------------------ TAB CONTENT DEFINITIONS ------------------------
if selected_tab == "Setup Preview":
    st.header("Setup Preview")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Grid & Rock Summary")
        grid_data = {
            "Parameter": ["Grid Dimensions (nx, ny, nz)", "Cell Size (dx, dy, dz) (ft)", "Total Volume (MM-ft³)",
                          "Facies Style", "Permeability Anisotropy (kx/ky)"],
            "Value": [f"{state['nx']} x {state['ny']} x {state['nz']}",
                      f"{state['dx']} x {state['dy']} x {state['dz']}",
                      f"{state['nx']*state['ny']*state['nz']*state['dx']*state['dy']*state['dz']/1e6:.1f}",
                      state['facies_style'], f"{state['anis_kxky']:.2f}"]
        }
        st.table(pd.DataFrame(grid_data))
        with st.expander("Click for details"):
            st.markdown(
                "- **Grid Dimensions**: The number of cells in the X, Y, and Z directions.\n"
                "- **Cell Size**: The physical size of each grid cell in feet.\n"
                "- **Total Volume**: The total bulk volume of the reservoir model.\n"
                "- **Facies Style**: The method used to generate geological heterogeneity.\n"
                "- **Anisotropy**: The ratio of permeability in the X-direction (kx) to the Y-direction (ky)."
            )

        st.markdown("#### Well & Frac Summary")
        well_data = {
            "Parameter": ["Laterals", "Lateral Length (ft)", "Frac Half-length (ft)", "Frac Height (ft)", "Stages", "Clusters/Stage"],
            "Value": [state['n_laterals'], state['L_ft'], state['xf_ft'], state['hf_ft'],
                      int(state['L_ft'] / state['stage_spacing_ft']), state['clusters_per_stage']]
        }
        st.table(pd.DataFrame(well_data))
        with st.expander("Click for details"):
            st.markdown(
                "- **Laterals**: The number of horizontal wells in the pad.\n"
                "- **Lateral Length**: The length of each horizontal wellbore.\n"
                "- **Frac Half-length (xf)**: The distance a hydraulic fracture extends from the wellbore.\n"
                "- **Frac Height (hf)**: The vertical extent of the hydraulic fractures.\n"
                "- **Stages**: The number of separate hydraulic fracturing treatments.\n"
                "- **Clusters/Stage**: The number of perforation clusters within each stage."
            )
    with c2:
        st.markdown("#### Top-Down Schematic")
        fig = go.Figure()
        nx, ny, dx, dy = state['nx'], state['ny'], state['dx'], state['dy']
        L_ft, xf_ft, ss_ft, n_lats = state['L_ft'], state['xf_ft'], state['stage_spacing_ft'], state['n_laterals']
        fig.add_shape(type="rect", x0=0, y0=0, x1=nx*dx, y1=ny*dy, line=dict(color="RoyalBlue"), fillcolor="lightskyblue", opacity=0.3)
        lat_rows_y = [ny*dy/3, 2*ny*dy/3] if n_lats >= 2 else [ny*dy/2]
        n_stages = max(1, int(L_ft / max(ss_ft, 1.0)))
        for i, y_lat in enumerate(lat_rows_y):
            fig.add_trace(go.Scatter(x=[0, L_ft], y=[y_lat, y_lat], mode='lines', line=dict(color='black', width=3),
                                     name='Lateral', showlegend=(i == 0)))
            for j in range(n_stages):
                x_stage = (j + 0.5) * ss_ft
                if x_stage > L_ft:
                    continue
                fig.add_trace(go.Scatter(x=[x_stage, x_stage], y=[y_lat - xf_ft, y_lat + xf_ft], mode='lines',
                                         line=dict(color='red', width=2), name='Frac', showlegend=(i == 0 and j == 0)))
        fig.update_layout(title="<b>Well and Fracture Geometry</b>", xaxis_title="X (ft)", yaxis_title="Y (ft)", yaxis_range=[-0.1*ny*dy, 1.1*ny*dy])
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        with st.expander("Click for details"):
            st.markdown(
                "This plot provides a bird's-eye view of the simulation model.\n"
                "- The **light blue rectangle** is the reservoir boundary.\n"
                "- The **black line(s)** are the horizontal well laterals.\n"
                "- The **red lines** are the hydraulic fractures."
            )

    st.markdown("---")
    st.markdown("### Production Forecast Preview (Analytical Model)")
    preview = _get_sim_preview()
    p_c1, p_c2 = st.columns(2)
    with p_c1:
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=preview['t'], y=preview['qg'], name="Gas Rate", line=dict(color="#d62728")))
        fig_g.update_layout(**semi_log_layout("Gas Production Preview", yaxis="Gas Rate (Mscf/d)"))
        st.plotly_chart(fig_g, use_container_width=True, theme="streamlit")
    with p_c2:
        fig_o = go.Figure()
        fig_o.add_trace(go.Scatter(x=preview['t'], y=preview['qo'], name="Oil Rate", line=dict(color="#2ca02c")))
        fig_o.update_layout(**semi_log_layout("Oil Production Preview", yaxis="Oil Rate (STB/d)"))
        st.plotly_chart(fig_o, use_container_width=True, theme="streamlit")
    with st.expander("Click for details"):
        st.markdown("These charts show a rapid forecast based on a simplified analytical model (Arps decline curve). They are for quick iteration before running the full 3D simulation.")

elif selected_tab == "Generate 3D property volumes":
    st.header("Generate 3D Property Volumes (kx, ky, ϕ)")
    if st.button("Re-generate rock property volumes"):
        st.session_state.kx, st.session_state.ky, st.session_state.phi = None, None, None
        _safe_rerun()

    if st.session_state.get('kx') is None:
        rng = np.random.default_rng(int(st.session_state.rng_seed))
        nz, ny, nx = int(state["nz"]), int(state["ny"]), int(state["nx"])
        kx_mid = 0.05 + state["k_stdev"] * rng.standard_normal((ny, nx))
        ky_mid = (0.05 / state["anis_kxky"]) + state["k_stdev"] * rng.standard_normal((ny, nx))
        phi_mid = 0.10 + state["phi_stdev"] * rng.standard_normal((ny, nx))
        kz_scale = np.linspace(0.95, 1.05, nz)[:, None, None]
        st.session_state.kx = np.clip(kx_mid[None, ...] * kz_scale, 1e-4, None)
        st.session_state.ky = np.clip(ky_mid[None, ...] * kz_scale, 1e-4, None)
        st.session_state.phi = np.clip(phi_mid[None, ...] * kz_scale, 0.01, 0.35)

    kx_display = get_k_slice(st.session_state.kx, state['nz'] // 2)
    ky_display = get_k_slice(st.session_state.ky, state['nz'] // 2)
    phi_display = get_k_slice(st.session_state.phi, state['nz'] // 2)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.imshow(kx_display, origin="lower", color_continuous_scale="Viridis", labels=dict(color="mD"), title="<b>kx — mid-layer (mD)</b>"),
            use_container_width=True, theme="streamlit"
        )
        with st.expander("Click for details"):
            st.markdown("**Permeability (k)** is a measure of a rock's ability to transmit fluids. This map shows kx for the middle layer.\n- **High values (yellow)** are 'sweet spots'.\n- **Low values (purple)** are tighter rock.")
    with c2:
        st.plotly_chart(
            px.imshow(ky_display, origin="lower", color_continuous_scale="Cividis", labels=dict(color="mD"), title="<b>ky — mid-layer (mD)</b>"),
            use_container_width=True, theme="streamlit"
        )
        with st.expander("Click for details"):
            st.markdown("This map shows permeability in the y-direction (ky). Comparing this to the kx map allows you to visually assess **anisotropy**.")
    st.plotly_chart(
        px.imshow(phi_display, origin="lower", color_continuous_scale="Magma", labels=dict(color="ϕ"), title="<b>Porosity ϕ — mid-layer (fraction)</b>"),
        use_container_width=True, theme="streamlit"
    )
    with st.expander("Click for details"):
        st.markdown("**Porosity (ϕ)** is the void space in the rock where fluids are stored.\n- **High values (yellow/white)** indicate more storage capacity.")

elif selected_tab == "PVT (Black-Oil)":
    st.header("PVT (Black-Oil) Analysis")
    P = np.linspace(max(1000, state["p_min_bhp_psi"]), max(2000, state["p_init_psi"] + 1000), 120)
    Rs = Rs_of_p(P, state["pb_psi"], state["Rs_pb_scf_stb"])
    Bo = Bo_of_p(P, state["pb_psi"], state["Bo_pb_rb_stb"])
    Bg = Bg_of_p(P)
    mug = mu_g_of_p(P, state["pb_psi"], state["mug_pb_cp"])

    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=P, y=Rs, line=dict(color="firebrick", width=3)))
    f1.add_vline(x=state["pb_psi"], line_dash="dash", line_width=2, annotation_text="Bubble Point")
    f1.update_layout(template="plotly_white", title="<b>Solution GOR Rs vs Pressure</b>", xaxis_title="Pressure (psi)", yaxis_title="Rs (scf/STB)")
    st.plotly_chart(f1, use_container_width=True, theme="streamlit")
    with st.expander("Click for details"):
        st.markdown("The **Solution Gas-Oil Ratio (Rs)** is the amount of gas dissolved in oil. As pressure drops below the Bubble Point, gas comes out of solution.")

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=P, y=Bo, line=dict(color="seagreen", width=3)))
    f2.add_vline(x=state["pb_psi"], line_dash="dash", line_width=2, annotation_text="Bubble Point")
    f2.update_layout(template="plotly_white", title="<b>Oil FVF Bo vs Pressure</b>", xaxis_title="Pressure (psi)", yaxis_title="Bo (rb/STB)")
    st.plotly_chart(f2, use_container_width=True, theme="streamlit")
    with st.expander("Click for details"):
        st.markdown("The **Oil Formation Volume Factor (Bo)** is the volume of reservoir oil needed to produce one barrel at the surface. As gas comes out of solution, the oil shrinks, and Bo decreases.")

    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=P, y=Bg, line=dict(color="steelblue", width=3)))
    f3.add_vline(x=state["pb_psi"], line_dash="dash", line_width=2)
    f3.update_layout(template="plotly_white", title="<b>Gas FVF Bg vs Pressure</b>", xaxis_title="Pressure (psi)", yaxis_title="Bg (rb/scf)")
    st.plotly_chart(f3, use_container_width=True, theme="streamlit")
    with st.expander("Click for details"):
        st.markdown("The **Gas Formation Volume Factor (Bg)** is the volume of reservoir gas needed to produce one standard cubic foot at the surface. Gas expands as pressure drops, so Bg increases.")

    f4 = go.Figure()
    f4.add_trace(go.Scatter(x=P, y=mug, line=dict(color="mediumpurple", width=3)))
    f4.add_vline(x=state["pb_psi"], line_dash="dash", line_width=2)
    f4.update_layout(template="plotly_white", title="<b>Gas viscosity μg vs Pressure</b>", xaxis_title="Pressure (psi)", yaxis_title="μg (cP)")
    st.plotly_chart(f4, use_container_width=True, theme="streamlit")
    with st.expander("Click for details"):
        st.markdown("**Gas Viscosity (μg)** is the measure of gas's resistance to flow and is a key input for flow calculations.")

elif selected_tab == "MSW Wellbore":
    st.header("MSW Wellbore Physics — Heel–Toe & Limited-Entry")
    try:
        L_ft, ss_ft, n_clusters = float(state['L_ft']), float(state['stage_spacing_ft']), int(state['clusters_per_stage'])
        n_stages = max(1, int(L_ft / ss_ft))
        well_id_ft, f_fric, dP_le = float(state['wellbore_ID_ft']), float(state['f_fric']), float(state['dP_LE_psi'])
        p_bhp, p_res = float(state['pad_bhp_psi']), float(state['p_init_psi'])

        preview = _get_sim_preview()
        q_oil_total_stbd = preview['qo'][0]
        rho_o_lb_ft3 = 50.0

        q_dist = np.ones(n_stages) / n_stages
        for _ in range(5):
            q_per_stage_bpd = q_dist * q_oil_total_stbd
            p_wellbore_at_stage = np.zeros(n_stages)
            p_current = p_bhp
            flow_rate_bpd = q_oil_total_stbd
            for i in range(n_stages):
                p_wellbore_at_stage[i] = p_current
                q_ft3_s = flow_rate_bpd * 5.615 / (24*3600)
                area_ft2 = np.pi * (well_id_ft/2)**2
                v_fps = q_ft3_s / area_ft2
                dp_psi_segment = (2 * f_fric * rho_o_lb_ft3 * v_fps**2 * ss_ft / well_id_ft) / 144.0
                p_current += dp_psi_segment
                flow_rate_bpd -= q_per_stage_bpd[i]
            drawdown = p_res - p_wellbore_at_stage - dP_le
            q_new_dist_unnorm = np.sqrt(np.maximum(0, drawdown))
            if np.sum(q_new_dist_unnorm) > 1e-9:
                q_dist = q_new_dist_unnorm / np.sum(q_new_dist_unnorm)

        c1_msw, c2_msw = st.columns(2)
        with c1_msw:
            fig_p = go.Figure(go.Scatter(x=np.arange(n_stages)*ss_ft, y=p_wellbore_at_stage, mode='lines+markers', name='Wellbore Pressure'))
            fig_p.update_layout(title="<b>Wellbore Pressure Profile</b>", xaxis_title="Position along Lateral (ft, 0=Heel)", yaxis_title="Pressure (psi)", template="plotly_white")
            st.plotly_chart(fig_p, use_container_width=True, theme="streamlit")
            with st.expander("Click for details"):
                st.markdown("This plot shows the pressure inside the horizontal wellbore. Due to friction, pressure is lowest at the heel and increases towards the toe, driving the \"heel-toe effect.\" ")
        with c2_msw:
            fig_q = go.Figure(go.Bar(x=np.arange(n_stages)*ss_ft, y=q_dist * 100, name='Flow Contribution'))
            fig_q.update_layout(title="<b>Flow Contribution per Stage</b>", xaxis_title="Position along Lateral (ft, 0=Heel)", yaxis_title="Contribution (%)", template="plotly_white")
            st.plotly_chart(fig_q, use_container_width=True, theme="streamlit")
            with st.expander("Click for details"):
                st.markdown("This chart shows the production percentage from each stage. Stages near the heel often produce more due to higher drawdown. The **Δp limited-entry** parameter can be used to mitigate this.")
    except Exception as e:
        st.warning(f"Could not compute wellbore hydraulics. Error: {e}")

elif selected_tab == "RTA":
    st.header("RTA — Quick Diagnostics")
    sim_data = st.session_state.sim if st.session_state.sim is not None else _get_sim_preview()
    t, qg = sim_data["t"], sim_data["qg"]

    rate_y_mode_rta = st.radio("Rate y-axis", ["Linear", "Log"], index=0, horizontal=True, key="rta_rate_y_mode_unique")
    y_type_rta = "log" if rate_y_mode_rta == "Log" else "linear"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=qg, line=dict(color="firebrick", width=3), name="Gas"))
    fig.update_layout(**semi_log_layout("R1. Gas rate (q) vs time", yaxis="q (Mscf/d)"))
    fig.update_yaxes(type=y_type_rta)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    with st.expander("Click for details"):
        st.markdown("This is a standard **Rate vs. Time** plot on a semi-log scale, showing the production decline.")

    t_safe = np.maximum(t, 1e-9)
    qg_safe = np.maximum(qg, 1e-9)
    logt, logq = np.log(t_safe), np.log(qg_safe)
    slope = np.gradient(logq, logt)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=slope, line=dict(color="teal", width=3), name="dlogq/dlogt"))
    fig2.update_layout(**semi_log_layout("R2. Log-log derivative", yaxis="Slope"))
    st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
    with st.expander("Click for details"):
        st.markdown(
            "This is the **log-log derivative** plot, a core RTA diagnostic tool. The slope indicates the dominant flow regime:\n"
            "- **Slope ≈ 0.5**: Indicates **linear flow** from fractures.\n"
            "- **Slope → 0**: Suggests **boundary-dominated flow**."
        )
# ==== BEGIN RESULTS BLOCK ====
elif selected_tab == "Results":
    st.header("Simulation Results")

    run_clicked = st.button("Run simulation", type="primary", use_container_width=True)
    if run_clicked:
        with st.spinner("Running full 3D simulation..."):
            sim_out = run_simulation_engine(state)
        if sim_out is None:
            st.session_state.sim = None
            st.error("Simulation failed. Showing results from fast preview solver.")
        else:
            st.session_state.sim = sim_out

    sim_data = st.session_state.get("sim")

    if sim_data is None:
        st.info("Click **Run simulation** to compute and display the full 3D results.")
    else:
        st.success(f"Simulation complete in {sim_data.get('runtime_s', 0):.2f} seconds.")

        # --- EUR gauges ---
        st.markdown("### EUR (30-year forecast)")
        g1, g2 = st.columns(2)
        eur_g = float(sim_data.get('EUR_g_BCF', 0.0))
        eur_o = float(sim_data.get('EUR_o_MMBO', 0.0))
        eur_g_fig, eur_o_fig = eur_gauges(eur_g, eur_o)
        with g1:
            st.plotly_chart(eur_g_fig, use_container_width=True)
        with g2:
            st.plotly_chart(eur_o_fig, use_container_width=True)
        with st.expander("Click for details"):
            st.markdown(
                "**Estimated Ultimate Recovery (EUR)** is the total hydrocarbons expected to be recovered over the well's life.\n"
                "- **BCF**: Billion Cubic Feet (gas)\n"
                "- **MMBO**: Million Stock-Tank Barrels of Oil"
            )

        # --- Production profiles ---
        st.markdown("### Production Profiles")
        rate_y_mode = st.radio("Rate y-axis", ["Linear", "Log"], index=0, horizontal=True, key="res_rate_y_mode")
        y_type = "log" if rate_y_mode == "Log" else "linear"

        fig_rate = go.Figure()
        fig_rate.add_trace(go.Scatter(x=sim_data['t'], y=sim_data['qg'], name="Gas Rate",
                                      line=dict(color="#d62728"), yaxis="y1"))
        fig_rate.add_trace(go.Scatter(x=sim_data['t'], y=sim_data['qo'], name="Oil Rate",
                                      line=dict(color="#2ca02c"), yaxis="y2"))
        layout_config = semi_log_layout("Gas & Oil Production Rate", yaxis="Gas Rate (Mscf/d)")
        layout_config.update(
            yaxis=dict(title="Gas Rate (Mscf/d)", side="left", type=y_type, color="#d62728",
                       showgrid=True, gridcolor="rgba(0,0,0,0.15)"),
            yaxis2=dict(title="Oil Rate (STB/d)", side="right", overlaying="y", type=y_type,
                        color="#2ca02c", showgrid=False),
        )
        fig_rate.update_layout(layout_config)
        st.plotly_chart(fig_rate, use_container_width=True, theme="streamlit")
        with st.expander("Click for details"):
            st.markdown("This plot shows simulated **production rates** for gas (red) and oil (green) over time.")

        # --- GOR & Cumulative ---
        c1_res, c2_res = st.columns(2)

        with c1_res:
            gor = np.divide(
                sim_data['qg'] * 1000, sim_data['qo'],
                out=np.full_like(sim_data['qg'], np.nan, dtype=float),
                where=sim_data['qo'] > 1e-3
            )
            fig_gor = go.Figure(go.Scatter(x=sim_data['t'], y=gor, name="GOR", line=dict(color="orange")))
            gor_layout = semi_log_layout("Gas-Oil Ratio (GOR)", yaxis="GOR (scf/STB)")
            gor_layout['xaxis']['type'] = 'linear'
            gor_layout['xaxis']['title'] = 'Day'
            gor_layout['yaxis']['type'] = 'linear'
            fig_gor.update_layout(gor_layout)
            st.plotly_chart(fig_gor, use_container_width=True, theme="streamlit")
            with st.expander("Click for details"):
                st.markdown("**GOR** = produced gas / produced oil. Its trend is a fluid-type diagnostic.")

        with c2_res:
            cum_g = cumulative_trapezoid(sim_data['qg'], sim_data['t'], initial=0) / 1e6   # BCF
            cum_o = cumulative_trapezoid(sim_data['qo'], sim_data['t'], initial=0) / 1e6   # MMSTB
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(x=sim_data['t'], y=cum_g, name="Cumulative Gas",
                                         line=dict(color="#d62728"), yaxis="y1"))
            fig_cum.add_trace(go.Scatter(x=sim_data['t'], y=cum_o, name="Cumulative Oil",
                                         line=dict(color="#2ca02c"), yaxis="y2"))
            cum_layout = semi_log_layout("Cumulative Production", yaxis="Cumulative Gas (BCF)")
            cum_layout['xaxis']['type'] = 'linear'
            cum_layout['xaxis']['title'] = 'Day'
            cum_layout.update(
                yaxis=dict(title="Cumulative Gas (BCF)", showgrid=True, gridcolor="rgba(0,0,0,0.15)"),
                yaxis2=dict(title="Cumulative Oil (MMSTB)", overlaying="y", side="right", showgrid=False),
            )
            fig_cum.update_layout(cum_layout)
            st.plotly_chart(fig_cum, use_container_width=True, theme="streamlit")
            with st.expander("Click for details"):
                st.markdown("Final points on these curves correspond to the EURs.")
# ==== END RESULTS BLOCK ====

# ==== BEGIN 3D VIEWER BLOCK ====
elif selected_tab == "3D Viewer":
    st.header("3D Viewer")
    sim_data = st.session_state.get("sim")

    if sim_data is None and st.session_state.get('kx') is None:
        st.warning("Please generate rock properties on Tab 2 or run a simulation on Tab 5 to enable the 3D viewer.")
    else:
        prop_list = ['Permeability (kx)', 'Porosity (ϕ)']
        if sim_data:
            prop_list.extend(['Pressure (psi)', 'Pressure Change (ΔP)', 'Original Oil In Place (OOIP)'])

        prop_3d = st.selectbox("Select property to view:", prop_list)
        c1_3d, c2_3d = st.columns(2)
        with c1_3d:
            st.session_state.vol_downsample = st.slider("Downsample factor", 1, 10, st.session_state.vol_downsample, 1, key="vol_ds")
        with c2_3d:
            st.session_state.iso_value_rel = st.slider("Isosurface value (relative)", 0.05, 0.95, st.session_state.iso_value_rel, 0.05, key="iso_val_rel")

        data_3d, colorscale, colorbar_title = (None, None, None)
        if 'kx' in prop_3d:
            data_3d, colorscale, colorbar_title = st.session_state.get('kx'), 'Viridis', 'kx (mD)'
        elif 'ϕ' in prop_3d:
            data_3d, colorscale, colorbar_title = st.session_state.get('phi'), 'Magma', 'Porosity (ϕ)'
        elif 'Pressure (psi)' in prop_3d and sim_data:
            data_3d, colorscale, colorbar_title = sim_data.get('press_matrix'), 'jet', 'Pressure (psi)'
        elif 'Pressure Change' in prop_3d and sim_data:
            p_final = sim_data.get('press_matrix'); p_init = sim_data.get('p_init_3d')
            if p_final is not None and p_init is not None:
                data_3d, colorscale, colorbar_title = p_init - p_final, 'inferno', 'ΔP (psi)'
        elif 'OOIP' in prop_3d and sim_data:
            data_3d, colorscale, colorbar_title = sim_data.get('ooip_3d'), 'plasma', 'OOIP (STB/cell)'

        if data_3d is not None:
            with st.spinner("Generating 3D plot..."):
                data_3d_ds = downsample_3d(data_3d, st.session_state.vol_downsample)
                v_min, v_max = np.min(data_3d_ds), np.max(data_3d_ds)
                isoval = v_min + (v_max - v_min) * st.session_state.iso_value_rel if v_max > v_min else v_min
                nz, ny, nx = data_3d_ds.shape
                Z, Y, X = np.mgrid[0:nz*state['dz']:nz, 0:ny*state['dy']:ny, 0:nx*state['dx']:nx]
                fig3d = go.Figure()
                fig3d.add_trace(go.Isosurface(
                    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                    value=data_3d_ds.flatten(), isomin=isoval, isomax=v_max, surface_count=1,
                    caps=dict(x_show=False, y_show=False),
                    colorscale=colorscale, colorbar=dict(title=colorbar_title)
                ))
                if sim_data:
                    n_lats = int(state.get('n_laterals', 1))
                    L_ft = float(state.get('L_ft', 10000.0))
                    lat_rows_y = [ny*state['dy']/3, 2*ny*state['dy']/3] if n_lats >= 2 else [ny*state['dy']/2]
                    lat_z = nz * state['dz'] / 2
                    for i, lat_y in enumerate(lat_rows_y):
                        fig3d.add_trace(go.Scatter3d(
                            x=[0, L_ft], y=[lat_y, lat_y], z=[lat_z, lat_z],
                            mode='lines', line=dict(color='white', width=10),
                            name='Well Lateral' if i == 0 else '', showlegend=(i == 0)
                        ))
                fig3d.update_layout(
                    title=f"<b>3D Isosurface for {prop_3d}</b>",
                    scene=dict(xaxis_title='X (ft)', yaxis_title='Y (ft)', zaxis_title='Z (ft)', aspectmode='data'),
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                st.plotly_chart(fig3d, use_container_width=True, theme="streamlit")
                with st.expander("Click for details on the visualized property"):
                    if 'Pressure Change' in prop_3d:
                        st.markdown("This shows the **Pressure Drawdown (Initial - Final Pressure)**. The resulting 'plume' is a powerful visualization of the Stimulated Reservoir Volume (SRV).")
                    elif 'OOIP' in prop_3d:
                        st.markdown("This shows the **Original Oil In Place** per grid cell, highlighting the 'sweet spots' in the reservoir.")
                    else:
                        st.markdown("This visualization shows the 3D distribution of the selected reservoir property. An **isosurface** is a 3D contour that connects points of equal value.")
        else:
            st.warning(f"Data for '{prop_3d}' could not be generated or found. Please run a simulation.")
# ==== END 3D VIEWER BLOCK ====

elif selected_tab == "Slice Viewer":
    st.header("Slice Viewer")
    sim_data = st.session_state.get("sim")
    if sim_data is None and st.session_state.get('kx') is None:
        st.warning("Please generate rock properties on Tab 2 or run a simulation on Tab 5 to enable the slice viewer.")
    else:
        slice_prop_list = ['Permeability (kx)', 'Permeability (ky)', 'Porosity (ϕ)']
        if sim_data and sim_data.get('press_matrix') is not None:
            slice_prop_list.append('Pressure (psi)')
        c1_sl, c2_sl = st.columns(2)
        with c1_sl:
            prop_slice = st.selectbox("Select property to view:", slice_prop_list, key="slice_prop_select")
        with c2_sl:
            plane_slice = st.selectbox("Select plane:", ["k-plane (z, top-down)", "j-plane (y, side-view)", "i-plane (x, end-view)"], key="slice_plane_select")

        data_slice_3d = (
            st.session_state.get('kx') if 'kx' in prop_slice else
            st.session_state.get('ky') if 'ky' in prop_slice else
            st.session_state.get('phi') if 'ϕ' in prop_slice else
            sim_data.get('press_matrix')
        )
        if data_slice_3d is not None:
            nz, ny, nx = data_slice_3d.shape
            if "k-plane" in plane_slice:
                slice_idx = st.slider("k-index (z-layer)", 0, nz - 1, nz // 2)
                data_2d = data_slice_3d[slice_idx, :, :]
                labels = dict(x="i-index", y="j-index")
                title_plane = f"k = {slice_idx}"
            elif "j-plane" in plane_slice:
                slice_idx = st.slider("j-index (y-layer)", 0, ny - 1, ny // 2)
                data_2d = data_slice_3d[:, slice_idx, :]
                labels = dict(x="i-index", y="k-index")
                title_plane = f"j = {slice_idx}"
            else:
                slice_idx = st.slider("i-index (x-layer)", 0, nx - 1, nx // 2)
                data_2d = data_slice_3d[:, :, slice_idx]
                labels = dict(x="j-index", y="k-index")
                title_plane = f"i = {slice_idx}"

            fig_slice = px.imshow(data_2d, origin="lower", aspect='equal', labels=labels, color_continuous_scale='viridis')
            fig_slice.update_layout(title=f"<b>{prop_slice} @ {title_plane}</b>")
            st.plotly_chart(fig_slice, use_container_width=True, theme="streamlit")
            with st.expander("Click for details"):
                st.markdown(
                    "This tool allows you to inspect 2D cross-sections of the 3D data volumes.\n"
                    "- **k-plane**: A horizontal, top-down view.\n"
                    "- **j-plane**: A vertical slice parallel to the well laterals.\n"
                    "- **i-plane**: A vertical slice perpendicular to the well laterals."
                )
        else:
            st.warning(f"Data for '{prop_slice}' not found.")
elif selected_tab == "QA / Material Balance":
    st.header("QA / Material Balance")
    sim_data = st.session_state.get("sim")
    if sim_data is None:
        st.warning("Run a simulation on the 'Results' tab to view QA plots.")
    elif sim_data.get('pm_mid_psi') is None:
        st.info("The selected solver did not return the necessary pressure evolution data for this tab.")
    else:
        if 't' in sim_data and 'qg' in sim_data and 'qo' in sim_data and len(sim_data['t']) > 1:
            cum_g_mmscf = cumulative_trapezoid(sim_data['qg'], sim_data['t'], initial=0)
            p_avg_series = np.array([np.mean(p_slice) for p_slice in sim_data.get('pm_mid_psi', [])])

            if len(p_avg_series) == len(sim_data['t']):
                st.markdown("### Gas Material Balance")
                z_factors = z_factor_approx(p_avg_series, p_init_psi=state['p_init_psi'])
                p_over_z = p_avg_series / z_factors
                fit_start_index = len(cum_g_mmscf) // 4
                slope, intercept, _, _, _ = stats.linregress(cum_g_mmscf[fit_start_index:], p_over_z[fit_start_index:])
                giip_bcf = (-intercept / slope) / 1000.0 if slope != 0 else 0
                sim_eur_g_bcf = sim_data['EUR_g_BCF']
                c1, c2 = st.columns(2)
                c1.metric("Simulator Gas EUR", f"{sim_eur_g_bcf:.2f} BCF")
                c2.metric("Material Balance GIIP (from P/Z)", f"{giip_bcf:.2f} BCF", delta=f"{(giip_bcf-sim_eur_g_bcf)/sim_eur_g_bcf:.1%} vs Sim")

                fig_pz_gas = go.Figure()
                fig_pz_gas.add_trace(go.Scatter(x=cum_g_mmscf, y=p_over_z, mode='markers', name='P/Z Data Points'))
                x_fit = np.array([0, giip_bcf * 1000]); y_fit = slope * x_fit + intercept
                fig_pz_gas.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Linear Extrapolation', line=dict(dash='dash')))
                fig_pz_gas.update_layout(title="<b>P/Z vs. Cumulative Gas Production</b>", xaxis_title="Gp - Cumulative Gas Production (MMscf)", yaxis_title="P/Z", template="plotly_white", xaxis_range=[0, giip_bcf * 1100])
                st.plotly_chart(fig_pz_gas, use_container_width=True, theme="streamlit")
                with st.expander("Click for details"):
                    st.markdown(
                        "This is a classic **P/Z vs. Gp** material balance plot for gas reservoirs. It serves as a powerful validation tool.\n"
                        "- **Theory**: For a volumetric gas reservoir, a plot of (Pressure/Z-factor) vs. Cumulative Gas Production (Gp) should form a straight line.\n"
                        "- **Interpretation**: The data points (blue dots) are calculated from the simulator's output. A linear regression (dashed line) is fitted to these points.\n"
                        "- **GIIP**: Extrapolating the line to the x-axis (where P/Z = 0) gives an independent estimate of the Gas Initially In Place (GIIP).\n"
                        "- **Validation**: The GIIP from this plot is compared to the simulator's EUR. A close match provides high confidence that the simulation is conserving mass and behaving physically correctly."
                    )

                st.markdown("---")
                st.markdown("### Oil Material Balance")
                Np = cumulative_trapezoid(sim_data['qo'], sim_data['t'], initial=0)
                Gp = cumulative_trapezoid(sim_data['qg'] * 1000, sim_data['t'], initial=0)
                Rp = np.divide(Gp, Np, out=np.zeros_like(Gp), where=Np > 1e-3)
                Bo = Bo_of_p(p_avg_series, state['pb_psi'], state['Bo_pb_rb_stb'])
                Rs = Rs_of_p(p_avg_series, state['pb_psi'], state['Rs_pb_scf_stb'])
                Bg = Bg_of_p(p_avg_series)

                p_init = state['p_init_psi']
                Boi = Bo_of_p(p_init, state['pb_psi'], state['Bo_pb_rb_stb'])
                Rsi = Rs_of_p(p_init, state['pb_psi'], state['Rs_pb_scf_stb'])
                F = Np * (Bo + (Rp - Rs) * Bg)
                Et = (Bo - Boi) + (Rsi - Rs) * Bg
                fit_start_index_oil = len(F) // 4
                slope_oil, _, _, _, _ = stats.linregress(Et[fit_start_index_oil:], F[fit_start_index_oil:])
                ooip_mmstb = slope_oil / 1e6 if slope_oil > 0 else 0
                sim_eur_o_mmstb = sim_data.get('EUR_o_MMBO', 0)
                rec_factor = (sim_eur_o_mmstb / ooip_mmstb) * 100 if ooip_mmstb > 0 else 0

                c1, c2, c3 = st.columns(3)
                c1.metric("Simulator Oil EUR", f"{sim_eur_o_mmstb:.2f} MMSTB")
                c2.metric("Material Balance OOIP (from F vs Et)", f"{ooip_mmstb:.2f} MMSTB")
                c3.metric("Implied Recovery Factor", f"{rec_factor:.1f}%")

                fig_mbe_oil = go.Figure()
                fig_mbe_oil.add_trace(go.Scatter(x=Et, y=F, mode='markers', name='F vs Et Data Points'))
                x_fit_oil = np.array([0, np.max(Et)]); y_fit_oil = slope_oil * x_fit_oil
                fig_mbe_oil.add_trace(go.Scatter(x=x_fit_oil, y=y_fit_oil, mode='lines', name=f'Slope (OOIP) = {ooip_mmstb:.2f} MMSTB', line=dict(dash='dash')))
                fig_mbe_oil.update_layout(title="<b>F vs. Et (Havlena-Odeh Plot)</b>", xaxis_title="Et - Total Expansion (rb/STB)", yaxis_title="F - Underground Withdrawal (rb)", template="plotly_white")
                st.plotly_chart(fig_mbe_oil, use_container_width=True, theme="streamlit")
                with st.expander("Click for details"):
                    st.markdown(
                        "This is a **Havlena-Odeh** material balance plot for solution-gas drive reservoirs.\n"
                        "- The slope gives an independent **OOIP** estimate.\n"
                        "- Compare to simulator EUR to sanity-check recovery."
                    )
            else:
                st.warning("Could not create plots. Pressure and time data have mismatched lengths.")
        else:
            st.warning("Simulation data is missing required arrays ('t', 'qg', 'qo') for this analysis.")

elif selected_tab == "Economics":
    st.header("Economics Analysis")
    sim_data = st.session_state.get("sim")
    if not sim_data or 't' not in sim_data:
        st.warning("Please run a simulation on the 'Results' tab to perform an economic analysis.")
    else:
        st.markdown("#### Economic Inputs")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            oil_price = st.number_input("Oil Price ($/bbl)", 0.0, 200.0, 75.0, 1.0)
        with c2:
            gas_price = st.number_input("Gas Price ($/Mcf)", 0.0, 20.0, 3.50, 0.10)
        with c3:
            discount_rate = st.number_input("Discount Rate (%)", 0.0, 25.0, 10.0, 0.5) / 100.0
        with c4:
            capex = st.number_input("CAPEX ($MM)", 0.0, 50.0, 10.0, 0.5) * 1e6

        days = sim_data['t']; qo = sim_data['qo']; qg = sim_data['qg']
        df = pd.DataFrame({'days': days, 'oil': qo, 'gas_mcfd': qg})
        df['year'] = (df['days'] / 365.25).astype(int)

        yearly_prod = df.groupby('year')[['oil', 'gas_mcfd']].mean()
        yearly_prod['yearly_oil'] = yearly_prod['oil'] * 365.25
        yearly_prod['yearly_gas_mcf'] = yearly_prod['gas_mcfd'] * 365.25

        yearly_prod['revenue'] = (yearly_prod['yearly_oil'] * oil_price) + (yearly_prod['yearly_gas_mcf'] * gas_price)
        cash_flow = [-capex] + yearly_prod['revenue'].tolist()

        npv = npf.npv(discount_rate, cash_flow)
        irr = npf.irr(cash_flow) * 100 if npv > 0 else 0
        cumulative_cash_flow = np.cumsum(cash_flow)
        payout_year = "N/A"
        if np.any(cumulative_cash_flow > 0):
            payout_year = np.argmax(cumulative_cash_flow > 0)

        st.markdown("---"); st.markdown("#### Key Financial Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Net Present Value (NPV)", f"${npv/1e6:.2f} MM")
        m2.metric("Internal Rate of Return (IRR)", f"{irr:.1f}%")
        m3.metric("Payout (Year)", f"{payout_year}")

        st.markdown("---"); st.markdown("#### Cash Flow Analysis")
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(x=list(range(len(cumulative_cash_flow))), y=cumulative_cash_flow, name='Cumulative Cash Flow'))
        fig_cf.update_layout(title="<b>Cumulative Cash Flow Over Time</b>", xaxis_title="Year", yaxis_title="Cumulative Cash Flow ($)", template="plotly_white")
        st.plotly_chart(fig_cf, use_container_width=True, theme="streamlit")

elif selected_tab == "EUR vs Lateral Length":
    st.header("Sensitivity: EUR vs Lateral Length")
    c1, c2, c3 = st.columns(3)
    with c1:
        L_min = st.number_input("Min Lateral Length (ft)", 1000, 20000, 5000, 500)
    with c2:
        L_max = st.number_input("Max Lateral Length (ft)", 1000, 20000, 15000, 500)
    with c3:
        L_steps = st.number_input("Number of steps", 2, 20, 11, 1)

    if st.button("Run Sensitivity Analysis", key="run_sens_L"):
        lengths = np.linspace(L_min, L_max, L_steps)
        eur_g_list, eur_o_list = [], []
        bar = st.progress(0, text="Running sensitivities...")
        base_state, rng_sens = state.copy(), np.random.default_rng(st.session_state.rng_seed)
        for i, length in enumerate(lengths):
            temp_state = {**base_state, 'L_ft': float(length)}
            result = fallback_fast_solver(temp_state, rng_sens)
            eur_g_list.append(float(result['EUR_g_BCF']))
            eur_o_list.append(float(result['EUR_o_MMBO']))
            bar.progress((i + 1) / L_steps, text=f"Running for L = {int(length)} ft...")
        st.session_state.sensitivity_results = {'L_ft': lengths, 'EUR_g': eur_g_list, 'EUR_o': eur_o_list}
        bar.empty()

    if 'sensitivity_results' in st.session_state:
        res = st.session_state.sensitivity_results
        c1_eur, c2_eur = st.columns(2)
        with c1_eur:
            fig_g_eur = go.Figure(go.Scatter(x=res['L_ft'], y=res['EUR_g'], mode='lines+markers'))
            fig_g_eur.update_layout(title="<b>Gas EUR vs. Lateral Length</b>", xaxis_title="Lateral Length (ft)", yaxis_title="EUR (BCF)", template="plotly_white")
            st.plotly_chart(fig_g_eur, use_container_width=True, theme="streamlit")
        with c2_eur:
            fig_o_eur = go.Figure(go.Scatter(x=res['L_ft'], y=res['EUR_o'], mode='lines+markers', marker_color='green'))
            fig_o_eur.update_layout(title="<b>Oil EUR vs. Lateral Length</b>", xaxis_title="Lateral Length (ft)", yaxis_title="EUR (MMSTB)", template="plotly_white")
            st.plotly_chart(fig_o_eur, use_container_width=True, theme="streamlit")

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
        fig_match.add_trace(go.Scatter(x=sim_data['t'], y=sim_data['qg'], mode='lines',
                                       name='Simulated Gas', line=dict(color="#d62728")))
        fig_match.add_trace(go.Scatter(x=sim_data['t'], y=sim_data['qo'], mode='lines',
                                       name='Simulated Oil', line=dict(color="#2ca02c"), yaxis="y2"))

        if {'Day', 'Gas_Rate_Mscfd'}.issubset(field_data.columns):
            fig_match.add_trace(go.Scatter(x=field_data['Day'], y=field_data['Gas_Rate_Mscfd'], mode='markers',
                                           name='Field Gas', marker=dict(color="#d62728", symbol='cross')))
        if {'Day', 'Oil_Rate_STBpd'}.issubset(field_data.columns):
            fig_match.add_trace(go.Scatter(x=field_data['Day'], y=field_data['Oil_Rate_STBpd'], mode='markers',
                                           name='Field Oil', marker=dict(color="#2ca02c", symbol='cross'), yaxis="y2"))

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
            st.plotly_chart(fig.update_layout(**semi_log_layout("Gas Rate Probabilistic Forecast", yaxis="Gas Rate (Mscf/d)")),
                            use_container_width=True, theme="streamlit")

            st.plotly_chart(
                px.histogram(x=mc['eur_g'], nbins=30, labels={'x': 'Gas EUR (BCF)'}).update_layout(
                    title="<b>Distribution of Gas EUR</b>", template="plotly_white"
                ),
                use_container_width=True, theme="streamlit"
            )

        with c2:
            fig = go.Figure([
                go.Scatter(x=mc['t'], y=p90_o, fill=None, mode='lines', line_color='lightgreen', name='P10'),
                go.Scatter(x=mc['t'], y=p10_o, fill='tonexty', mode='lines', line_color='lightgreen', name='P90'),
                go.Scatter(x=mc['t'], y=p50_o, mode='lines', line_color='green', name='P50'),
            ])
            st.plotly_chart(fig.update_layout(**semi_log_layout("Oil Rate Probabilistic Forecast", yaxis="Oil Rate (STB/d)")),
                            use_container_width=True, theme="streamlit")

            st.plotly_chart(
                px.histogram(x=mc['eur_o'], nbins=30, labels={'x': 'Oil EUR (MMSTB)'},
                             color_discrete_sequence=['green']).update_layout(
                    title="<b>Distribution of Oil EUR</b>", template="plotly_white"
                ),
                use_container_width=True, theme="streamlit"
            )

        with st.expander("Click for details"):
            st.markdown(
                "Monte Carlo quantifies forecast uncertainty. The P50 is the most likely case; "
                "shaded area shows P10–P90 envelope. Histograms summarize EUR distributions."
            )

# ---------------- Well Placement Optimization ----------------
elif selected_tab == "Well Placement Optimization":
    st.header("Well Placement Optimization")

    st.markdown("#### 1. General Parameters")
    c1_opt, c2_opt, c3_opt = st.columns(3)
    with c1_opt:
        objective = st.selectbox(
            "Objective Property",
            ["Maximize Oil EUR", "Maximize Gas EUR"],
            key="opt_objective"
        )
    with c2_opt:
        iterations = st.number_input(
            "Number of optimization steps",
            min_value=5, max_value=1000, value=100, step=10
        )
    with c3_opt:
        st.selectbox(
            "Forbidden Zone",
            ["Numerical Faults"],
            help="The optimizer will avoid placing wells near the fault defined in the sidebar."
        )

    st.markdown("#### 2. Well Parameters")
    c1_well, c2_well = st.columns(2)
    with c1_well:
        num_wells = st.number_input(
            "Number of wells to place",
            min_value=1, max_value=1, value=1,
            disabled=True,
            help="Currently supports optimizing a single well location."
        )
    with c2_well:
        st.text_input("Well name prefix", "OptiWell", disabled=True)

    # Button outside the column blocks
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
                icon="⚠️"
            )
            st.stop()

        y_max = base_state['ny'] * base_state['dy']
        progress_bar = st.progress(0, text="Starting optimization...")

        # ---- Main loop (FOR) ----
        for i in range(int(iterations)):
            # Propose a random heel location and check feasibility
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
                "Step": i + 1,
                "x_ft": float(x_heel_ft),
                "y_ft": float(y_heel_ft),
                "Score": float(score),
            })

            progress_bar.progress(
                (i + 1) / int(iterations),
                text=f"Step {i+1}/{int(iterations)} | Score: {score:.3f}"
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
            z=phi_map, dx=state['dx'], dy=state['dy'],
            colorscale='viridis', colorbar=dict(title='Porosity')
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
                x=fault_x, y=fault_y, mode='lines',
                line=dict(color='white', width=4, dash='dash'), name='Fault'
            ))
        fig_opt.update_layout(
            title="<b>Well Placement Optimization Map</b>",
            xaxis_title="X position (ft)", yaxis_title="Y position (ft)",
            template="plotly_white", height=600
        )
        st.plotly_chart(fig_opt, use_container_width=True, theme="streamlit")

        with st.expander("Click for details"):
            st.markdown(
                "Map shows tested locations (colored by score), the best location (cyan star), "
                "and the fault (white dashed) if enabled."
            )

# ---------------- User’s Manual ----------------
elif selected_tab == "User’s Manual":
    st.header("User’s Manual")
    st.markdown("---")
    st.markdown("""
    ### 1. Introduction
    Welcome to the **Full 3D Unconventional & Black-Oil Reservoir Simulator**. This tool helps
    you explore reservoir behavior and forecast production for MSW wells.
    """)
    st.markdown("---")
    st.markdown("""
    ### 2. Quick Start
    1) Pick a **Preset** in the sidebar and click **Apply Preset**  
    2) Optional: **Generate 3D property volumes** (Tab 2)  
    3) Run **Results → Run simulation**  
    4) Review plots; iterate parameters as needed
    """)
    st.markdown("---")
    st.markdown("""
    ### 3. History Matching Workflow
    Use **Field Match (CSV)** to load historical data, adjust parameters in the sidebar, and rerun until
    simulated rates align with measured points.
    """)

# ---------------- Solver & Profiling ----------------
elif selected_tab == "Solver & Profiling":
    st.header("Solver & Profiling")
    st.info("This tab shows numerical solver settings and performance of the last run.")
    st.markdown("### Current Numerical Solver Settings")
    solver_settings = {
        "Parameter": [
            "Newton Tolerance", "Max Newton Iterations", "Threads",
            "Use OpenMP", "Use MKL", "Use PyAMG", "Use cuSPARSE"
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

# ---------------- DFN Viewer ----------------
elif selected_tab == "DFN Viewer":
    st.header("DFN Viewer — 3D line segments")
    segs = st.session_state.get('dfn_segments')
    if segs is None or len(segs) == 0:
        st.info("No DFN loaded. Upload a CSV or use 'Generate DFN from stages' in the sidebar.")
    else:
        figd = go.Figure()
        for i, seg in enumerate(segs):
            figd.add_trace(go.Scatter3d(
                x=[seg[0], seg[3]], y=[seg[1], seg[4]], z=[seg[2], seg[5]],
                mode="lines", line=dict(width=4, color="red"),
                name="DFN" if i == 0 else None, showlegend=(i == 0)
            ))
        figd.update_layout(
            template="plotly_white",
            scene=dict(xaxis_title="x (ft)", yaxis_title="y (ft)", zaxis_title="z (ft)"),
            height=640, margin=dict(l=0, r=0, t=40, b=0),
            title="<b>DFN Segments</b>",
        )
        st.plotly_chart(figd, use_container_width=True, theme="streamlit")
        with st.expander("Click for details"):
            st.markdown(
                """
                This plot shows a 3D visualization of the Discrete Fracture Network (DFN) segments loaded into the simulator.
                - Each **red line** represents an individual natural fracture defined in the input file.
                - Use this for QC to verify locations/orientations inside the reservoir model.
                """
            )
