"""App entrypoint and UI wiring."""
from __future__ import annotations  # optional on Py 3.11+, but fine to keep

# ---- typing & aliases
from typing import Dict, Tuple, Union
Bounds = Dict[str, Union[Tuple[float, float], float]]  # optional if used

# ---- stdlib
import time
import warnings

# ---- third-party (import only what you use)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import cumulative_trapezoid as _ctr
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution

# ---- local modules (required if you call them)
from core.full3d import simulate
from engines.fast import fallback_fast_solver

# ---- utils hot-reload (optional)
try:
    import utils
    from importlib import reload as _reload
    _reload(utils)
except Exception:
    utils = None  # no trailing period

# ---- constants (optional, keep if used)
GAS_RED     = "#D62728"
OIL_GREEN   = "#2CA02C"
WATER_BLUE  = "#1F77B4"
PROP_ORANGE = "#FF7F0E"

# ---- utilities (column 0)
def safe_power(x, p):
    """Robust power that won't emit warnings or create complex numbers."""
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        non_integer_exp = not float(p).is_integer()
    except Exception:
        non_integer_exp = False
    if non_integer_exp:
        x = np.maximum(x, 0.0)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        y = np.power(x, p)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

# ======================================================================
# All page/UI code lives inside this function (prevents dangling-elif)
# ======================================================================
def render_users_manual():
    st.markdown(r"""
### 1. Introduction
Welcome to the **Full 3D Unconventional & Black-Oil Reservoir Simulator**. This application is designed for petroleum engineers to model, forecast, and optimize production from multi-stage fractured horizontal wells.

### 2. Quick Start Guide
1) Select a Play in the sidebar (Preset dropdown).
2) Apply Preset to load typical reservoir, fluid, and completion parameters.
3) Generate Geology: open **Generate 3D property volumes** and click the button to create 3D permeability/porosity grids.
4) Run Simulation: go to **Results** and click **Run simulation**.
5) Analyze: review EUR gauges, rate–time plots, and cumulative production charts.
6) Iterate: tweak parameters (e.g., frac half-length `xf_ft` or pad BHP `pad_bhp_psi`) and re-run.

### 3. Key Tabs Explained

#### Results
Primary dashboard for outputs (EURs, rate–time, cumulative). Simulation runs only when you click **Run simulation** on this tab.

#### Economics
Financial model based on the Results profile.
- Inputs: CAPEX, price decks, OPEX, fiscal terms.
- Metrics: **NPV**, **IRR**, **Payout Period**.
- Outputs: yearly & cumulative cash flows plus a detailed table.

#### Field Match & Automated Match
History matching against real data.
- Field Match (CSV): upload with columns `Day` and one of `Oil_Rate_STBpd` or `Gas_Rate_Mscfd`. Adjust parameters and re-run to align curves.
- Automated Match (genetic algorithm):
  1) Upload data  
  2) Select parameters (e.g., `xf_ft`, `k_stdev`)  
  3) Set bounds  
  4) Run Automated Match to minimize RMSE

#### 3D & Slice Viewers
- **3D Viewer:** interactive isosurfaces (perm/poro/pressure).
- **Slice Viewer:** 2D cross-sections in X/Y/Z.

### 4. Input Validation
- Automated Match warns if any min bound > max bound.
- Results sanity checks enforce realistic EURs; physically inconsistent results are flagged or withheld.
    """)  # end users manual

def render_app() -> None:
    st.set_page_config(page_title="3D Unconventional & Black-Oil Simulator", layout="wide")

    selected_tab = st.sidebar.radio(
        "Pages",
        ["Overview", "Inputs", "Simulation", "Results", "Solver & Profiling"],
        index=0,
    )

    def _require_inputs() -> bool:
        if "inputs" not in st.session_state:
            st.info("No inputs found. Go to **Inputs** to configure parameters.")
            return False
        return True

    def _require_results() -> bool:
        if "results" not in st.session_state:
            st.info("No results yet. Run a simulation on the **Simulation** page.")
            return False
        return True

    # ---- PAGES ----
    if selected_tab == "Overview":
        st.title("Full 3D Unconventional & Black-Oil Reservoir Simulator")
        render_users_manual()

    elif selected_tab == "Inputs":
        st.header("Model Inputs")
        with st.form("inputs_form"):
            nx = st.number_input("Cells in X", 10, 512, 64)
            ny = st.number_input("Cells in Y", 10, 512, 64)
            nz = st.number_input("Cells in Z", 1, 64, 10)
            submitted = st.form_submit_button("Save Inputs")
            if submitted:
                st.session_state.inputs = {"grid": {"nx": int(nx), "ny": int(ny), "nz": int(nz)}}
                st.success("Inputs saved. Proceed to **Simulation**.")

    elif selected_tab == "Simulation":
        st.header("Run Simulation")
        if not _require_inputs():
            st.stop()

        if st.button("Run Simulation", type="primary"):
            inputs = st.session_state.inputs
            with st.spinner("Running simulation..."):
                results = simulate(inputs)  # adjust to your real call
            st.session_state.results = results
            st.success("Simulation complete. See **Results** tab.")

    elif selected_tab == "Results":
        st.header("Results")
        if not _require_results():
            st.stop()

        results = st.session_state.results
        t = np.asarray(results.get("time_days", []))
        q = np.asarray(results.get("q_oil_stb_d", []))
        if t.size and q.size:
            st.plotly_chart(
                px.line(x=t, y=q, labels={"x": "Days", "y": "Oil Rate (STB/D)"}),
                use_container_width=True
            )

    elif selected_tab == "Solver & Profiling":
        st.header("Solver & Profiling")
        steps = st.slider("Benchmark steps", 10, 200, 50, 10)
        if st.button("Run tiny benchmark"):
            if not _require_inputs():
                st.stop()
            inputs = st.session_state.inputs
            short_inputs = dict(inputs)
            timings: Dict[str, Union[float, str]] = {}
            start = time.perf_counter()
            try:
                _ = fallback_fast_solver(short_inputs)
                timings["engines.fast.fallback_fast_solver"] = time.perf_counter() - start
            except Exception as e:
                timings["engines.fast.fallback_fast_solver"] = f"error: {e}"
            st.json(timings)
if __name__ == "__main__":
    render_app()


# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT # ---------------------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT # Brand colors (define once, globally)
# DUP_AFTER_ENTRYPOINT # ---------------------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT GAS_RED   = "#D62728"
# DUP_AFTER_ENTRYPOINT OIL_GREEN = "#2CA02C"
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT # ---------------------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT # Utilities
# DUP_AFTER_ENTRYPOINT # ---------------------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT def safe_power(x, p):
# DUP_AFTER_ENTRYPOINT     """Robust power that won't emit warnings or create complex numbers."""
# DUP_AFTER_ENTRYPOINT     x = np.asarray(x, dtype=float)
# DUP_AFTER_ENTRYPOINT     # replace NaN/Inf with 0
# DUP_AFTER_ENTRYPOINT     x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # negative base + non-integer exponent -> clamp to 0 to avoid complex domain
# DUP_AFTER_ENTRYPOINT     try:
# DUP_AFTER_ENTRYPOINT         non_integer_exp = not float(p).is_integer()
# DUP_AFTER_ENTRYPOINT     except Exception:
# DUP_AFTER_ENTRYPOINT         non_integer_exp = False
# DUP_AFTER_ENTRYPOINT     if non_integer_exp:
# DUP_AFTER_ENTRYPOINT         x = np.maximum(x, 0.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
# DUP_AFTER_ENTRYPOINT         y = np.power(x, p)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # clean any residual nan/inf
# DUP_AFTER_ENTRYPOINT     return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ---------------------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT     # Navigation (Tabs / Sections) — place after imports & constants
# DUP_AFTER_ENTRYPOINT     # ---------------------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # 1) Define nav items exactly as shown in your left menu
# DUP_AFTER_ENTRYPOINT     NAV_ITEMS = [
# DUP_AFTER_ENTRYPOINT         "Setup Preview",
# DUP_AFTER_ENTRYPOINT         "Generate 3D property volumes",
# DUP_AFTER_ENTRYPOINT         "PVT (Black-Oil)",
# DUP_AFTER_ENTRYPOINT         "MSW Wellbore",
# DUP_AFTER_ENTRYPOINT         "RTA",
# DUP_AFTER_ENTRYPOINT         "Results",
# DUP_AFTER_ENTRYPOINT         "3D Viewer",
# DUP_AFTER_ENTRYPOINT         "Slice Viewer",
# DUP_AFTER_ENTRYPOINT         "QA / Material Balance",
# DUP_AFTER_ENTRYPOINT         "Economics",
# DUP_AFTER_ENTRYPOINT         "EUR vs Lateral Length",
# DUP_AFTER_ENTRYPOINT         "Field Match (CSV)",
# DUP_AFTER_ENTRYPOINT         "Automated Match",
# DUP_AFTER_ENTRYPOINT         "Uncertainty & Monte Carlo",
# DUP_AFTER_ENTRYPOINT         "Well Placement Optimization",
# DUP_AFTER_ENTRYPOINT         "User's Manual",
# DUP_AFTER_ENTRYPOINT         "Solver & Profiling",
# DUP_AFTER_ENTRYPOINT         "DFN Viewer",
# DUP_AFTER_ENTRYPOINT     ]
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # 2) Helpers: resolve renderer functions if you already have them defined elsewhere.
# DUP_AFTER_ENTRYPOINT     #    If not present yet, show a friendly placeholder instead of crashing.
# DUP_AFTER_ENTRYPOINT     def _resolve(name: str, fallback: str):
# DUP_AFTER_ENTRYPOINT         fn = globals().get(name)
# DUP_AFTER_ENTRYPOINT         if callable(fn):
# DUP_AFTER_ENTRYPOINT             return fn
# DUP_AFTER_ENTRYPOINT         return lambda: st.info(fallback)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # 3) Map nav labels -> renderer functions (rename targets if you already have them)
# DUP_AFTER_ENTRYPOINT     PAGES = {
# DUP_AFTER_ENTRYPOINT         "Setup Preview": render_setup_preview,
# DUP_AFTER_ENTRYPOINT         "Generate 3D property volumes": render_generate_volumes,
# DUP_AFTER_ENTRYPOINT         "PVT (Black-Oil)": render_pvt_black_oil,
# DUP_AFTER_ENTRYPOINT         "MSW Wellbore": render_msw_wellbore,
# DUP_AFTER_ENTRYPOINT         "RTA": render_rta,
# DUP_AFTER_ENTRYPOINT         "Results": render_results_panel,
# DUP_AFTER_ENTRYPOINT         "3D Viewer": render_3d_viewer,
# DUP_AFTER_ENTRYPOINT         "Slice Viewer": render_slice_viewer,
# DUP_AFTER_ENTRYPOINT         "QA / Material Balance": render_qa_material_balance,
# DUP_AFTER_ENTRYPOINT         "Economics": render_economics,
# DUP_AFTER_ENTRYPOINT         "EUR vs Lateral Length": render_eur_vs_lateral,
# DUP_AFTER_ENTRYPOINT         "Field Match (CSV)": render_field_match_csv,
# DUP_AFTER_ENTRYPOINT         "Automated Match": render_automated_match,
# DUP_AFTER_ENTRYPOINT         "Uncertainty & Monte Carlo": render_uncertainty_monte_carlo,
# DUP_AFTER_ENTRYPOINT         "Well Placement Optimization": render_well_placement_optimization,
# DUP_AFTER_ENTRYPOINT         "User's Manual": render_users_manual,   # ← added
# DUP_AFTER_ENTRYPOINT         "Solver & Profiling": render_solver_profiling,
# DUP_AFTER_ENTRYPOINT         "DFN Viewer": render_dfn_viewer,
# DUP_AFTER_ENTRYPOINT     }
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # 4) Radio + dispatcher. Keep this near the top-level (not inside another tab),
# DUP_AFTER_ENTRYPOINT     #    and do NOT render any other panels below it (that would cause fall-through).
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ---- LEFT MENU (only copy) ----
# DUP_AFTER_ENTRYPOINT     with st.sidebar:
# DUP_AFTER_ENTRYPOINT         selected = st.radio("Navigation", NAV_ITEMS, index=0, key="nav_main", label_visibility="collapsed")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # Legacy compatibility for any old code that still references selected_tab
# DUP_AFTER_ENTRYPOINT     selected_tab = selected
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ---- DISPATCH (only copy) ----
# DUP_AFTER_ENTRYPOINT     page_fn = PAGES.get(selected, lambda: st.info("Setup Preview"))
# DUP_AFTER_ENTRYPOINT     page_fn()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # Legacy compatibility (if any old code still references selected_tab)
# DUP_AFTER_ENTRYPOINT     selected_tab = selected
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ---- DISPATCH (only copy) ----
# DUP_AFTER_ENTRYPOINT     page_fn = PAGES.get(selected, lambda: st.info("Setup Preview"))
# DUP_AFTER_ENTRYPOINT     page_fn()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     selected_tab = selected  # <— add this line right after the block
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ---- DISPATCH (only copy!) ----
# DUP_AFTER_ENTRYPOINT     # No undefined-function references and no second call
# DUP_AFTER_ENTRYPOINT     page_fn = PAGES.get(selected, lambda: st.info("Setup Preview"))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     page_fn()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # --- Minimal page renderers to keep routing clean ---
# DUP_AFTER_ENTRYPOINT     def render_setup_preview():
# DUP_AFTER_ENTRYPOINT         st.header("Setup Preview")
# DUP_AFTER_ENTRYPOINT         # TODO: move your existing setup preview section into this function
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_generate_volumes():
# DUP_AFTER_ENTRYPOINT         st.header("Generate 3D property volumes")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_pvt_black_oil():
# DUP_AFTER_ENTRYPOINT         st.header("PVT (Black-Oil)")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_msw_wellbore():
# DUP_AFTER_ENTRYPOINT         st.header("MSW Wellbore")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_rta():
# DUP_AFTER_ENTRYPOINT         st.header("RTA")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_results_panel():
# DUP_AFTER_ENTRYPOINT         st.header("Simulation Results")
# DUP_AFTER_ENTRYPOINT         # TODO: move your current "Simulation Results" code into this function
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_3d_viewer():
# DUP_AFTER_ENTRYPOINT         st.header("3D Viewer")
# DUP_AFTER_ENTRYPOINT         # TODO: place 3D visualization here; for now keeps the tab switching clean
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_slice_viewer():
# DUP_AFTER_ENTRYPOINT         st.header("Slice Viewer")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_qa_material_balance():
# DUP_AFTER_ENTRYPOINT         st.header("QA / Material Balance")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_economics():
# DUP_AFTER_ENTRYPOINT         st.header("Economics")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_eur_vs_lateral():
# DUP_AFTER_ENTRYPOINT         st.header("EUR vs Lateral Length")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_field_match_csv():
# DUP_AFTER_ENTRYPOINT         st.header("Field Match (CSV)")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_automated_match():
# DUP_AFTER_ENTRYPOINT         st.header("Automated Match")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_uncertainty_monte_carlo():
# DUP_AFTER_ENTRYPOINT         st.header("Uncertainty & Monte Carlo")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_well_placement_optimization():
# DUP_AFTER_ENTRYPOINT         st.header("Well Placement Optimization")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_users_manual():
# DUP_AFTER_ENTRYPOINT         st.header("User's Manual")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_solver_profiling():
# DUP_AFTER_ENTRYPOINT         st.header("Solver & Profiling")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def render_dfn_viewer():
# DUP_AFTER_ENTRYPOINT         st.header("DFN Viewer")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ---- DISPATCH ----
# DUP_AFTER_ENTRYPOINT     # Safe default that doesn't reference an undefined function
# DUP_AFTER_ENTRYPOINT     page_fn = PAGES.get(selected, lambda: render_setup_preview() if 'render_setup_preview' in globals() else st.info("Setup Preview"))
# DUP_AFTER_ENTRYPOINT     page_fn()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # Call the selected page renderer
# DUP_AFTER_ENTRYPOINT     PAGES[selected]()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ---------------------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT     # Optional safety nets (helpers only)
# DUP_AFTER_ENTRYPOINT     #   - harmless if your project already defines these elsewhere
# DUP_AFTER_ENTRYPOINT     #   - keeps Cloud hot-reload / module order from biting you
# DUP_AFTER_ENTRYPOINT     # ---------------------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT     if "gauge_max" not in globals():
# DUP_AFTER_ENTRYPOINT         def gauge_max(value, typical_hi, floor=0.1, safety=0.15):
# DUP_AFTER_ENTRYPOINT             """Reasonable gauge max: cover typical_hi and current value with margin."""
# DUP_AFTER_ENTRYPOINT             if value is None or (isinstance(value, (int, float)) and _np.isnan(value)) or value <= 0:
# DUP_AFTER_ENTRYPOINT                 return max(floor, typical_hi)
# DUP_AFTER_ENTRYPOINT             return max(floor, typical_hi * (1.0 + safety), float(value) * 1.25)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     if "_recovery_to_date_pct" not in globals():
# DUP_AFTER_ENTRYPOINT         def _recovery_to_date_pct(
# DUP_AFTER_ENTRYPOINT             cum_oil_stb: float,
# DUP_AFTER_ENTRYPOINT             eur_oil_mmbo: float,
# DUP_AFTER_ENTRYPOINT             cum_gas_mscf: float,
# DUP_AFTER_ENTRYPOINT             eur_gas_bcf: float,
# DUP_AFTER_ENTRYPOINT         ) -> tuple[float, float]:
# DUP_AFTER_ENTRYPOINT             """Return (oil_RF_pct, gas_RF_pct) as 0–100, clipped."""
# DUP_AFTER_ENTRYPOINT             oil_rf = 0.0
# DUP_AFTER_ENTRYPOINT             gas_rf = 0.0
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             if eur_oil_mmbo and eur_oil_mmbo > 0:
# DUP_AFTER_ENTRYPOINT                 oil_rf = 100.0 * (float(cum_oil_stb) / (float(eur_oil_mmbo) * 1_000_000.0))
# DUP_AFTER_ENTRYPOINT                 oil_rf = max(0.0, min(100.0, oil_rf))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             if eur_gas_bcf and eur_gas_bcf > 0:
# DUP_AFTER_ENTRYPOINT                 gas_rf = 100.0 * (float(cum_gas_mscf) / (float(eur_gas_bcf) * 1_000_000.0))
# DUP_AFTER_ENTRYPOINT                 gas_rf = max(0.0, min(100.0, gas_rf))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             return oil_rf, gas_rf
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     MIDLAND_BOUNDS = {
# DUP_AFTER_ENTRYPOINT         "oil_mmbo": (0.3, 1.5),   # typical sanity window
# DUP_AFTER_ENTRYPOINT         "gas_bcf":  (0.3, 3.0),
# DUP_AFTER_ENTRYPOINT     }
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def _cum_trapz_days(t_days, y_per_day):
# DUP_AFTER_ENTRYPOINT         if y_per_day is None:
# DUP_AFTER_ENTRYPOINT             return None
# DUP_AFTER_ENTRYPOINT         t = _np.asarray(t_days, float)
# DUP_AFTER_ENTRYPOINT         y = _np.nan_to_num(_np.asarray(y_per_day, float), nan=0.0)
# DUP_AFTER_ENTRYPOINT         return _ctr(y, t, initial=0.0)  # returns same length as t
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def _apply_economic_cutoffs(t, y, *, cutoff_days=None, min_rate=0.0):
# DUP_AFTER_ENTRYPOINT         if y is None:
# DUP_AFTER_ENTRYPOINT             return _np.asarray(t, float), None
# DUP_AFTER_ENTRYPOINT         t = _np.asarray(t, float)
# DUP_AFTER_ENTRYPOINT         y = _np.asarray(y, float)
# DUP_AFTER_ENTRYPOINT         mask = _np.ones_like(t, dtype=bool)
# DUP_AFTER_ENTRYPOINT         if cutoff_days is not None and cutoff_days > 0:
# DUP_AFTER_ENTRYPOINT             mask &= (t <= float(cutoff_days))
# DUP_AFTER_ENTRYPOINT         if min_rate and _np.any(y < float(min_rate)):
# DUP_AFTER_ENTRYPOINT             below = y < float(min_rate)
# DUP_AFTER_ENTRYPOINT             first = _np.argmax(below) if _np.any(below) else None
# DUP_AFTER_ENTRYPOINT             if first is not None and below[first]:
# DUP_AFTER_ENTRYPOINT                 mask[first:] = False
# DUP_AFTER_ENTRYPOINT         return t[mask], y[mask]
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def _apply_economic_cutoffs_ui(t, q, kind: str):
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         Use EUR options from the UI to trim the series BEFORE integration.
# DUP_AFTER_ENTRYPOINT         kind: "gas", "oil", or "water".
# DUP_AFTER_ENTRYPOINT         Returns (t_trim, q_trim). If q is None, returns (t, None).
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         import numpy as np
# DUP_AFTER_ENTRYPOINT         import streamlit as st
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         if q is None:
# DUP_AFTER_ENTRYPOINT             return np.asarray(t, float), None
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Coerce arrays (guard NaN/Inf to be safe for masking)
# DUP_AFTER_ENTRYPOINT         t = np.asarray(t, float)
# DUP_AFTER_ENTRYPOINT         q = np.asarray(q, float)
# DUP_AFTER_ENTRYPOINT         t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
# DUP_AFTER_ENTRYPOINT         q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         cutoff_days = float(st.session_state.get("eur_cutoff_days", 30.0 * 365.25))
# DUP_AFTER_ENTRYPOINT         if kind == "gas":
# DUP_AFTER_ENTRYPOINT             min_rate = float(st.session_state.get("eur_min_rate_gas_mscfd", 100.0))
# DUP_AFTER_ENTRYPOINT         elif kind == "oil":
# DUP_AFTER_ENTRYPOINT             min_rate = float(st.session_state.get("eur_min_rate_oil_stbd", 30.0))
# DUP_AFTER_ENTRYPOINT         else:
# DUP_AFTER_ENTRYPOINT             # Water default: no economic floor
# DUP_AFTER_ENTRYPOINT             min_rate = 0.0
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Keep samples within horizon AND above floor
# DUP_AFTER_ENTRYPOINT         mask = (t <= cutoff_days) & (q >= min_rate)
# DUP_AFTER_ENTRYPOINT         if not np.any(mask):
# DUP_AFTER_ENTRYPOINT             # No valid samples → return safe tiny series so plots/EUR don't crash
# DUP_AFTER_ENTRYPOINT             return np.array([0.0]), np.array([0.0])
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         return t[mask], q[mask]
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def _compute_eurs_and_cums(t, qg=None, qo=None, qw=None):
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         Compute cumulative volumes and EURs from rate vectors.
# DUP_AFTER_ENTRYPOINT         t  : days (1D)
# DUP_AFTER_ENTRYPOINT         qg : gas rate, Mscf/d
# DUP_AFTER_ENTRYPOINT         qo : oil rate, STB/d
# DUP_AFTER_ENTRYPOINT         qw : water rate, STB/d
# DUP_AFTER_ENTRYPOINT         Returns dict with cum arrays and EURs (gas in BCF, oil/water in MMbbl).
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         import numpy as np
# DUP_AFTER_ENTRYPOINT         from scipy.integrate import cumulative_trapezoid
# DUP_AFTER_ENTRYPOINT         import plotly.graph_objects as go
# DUP_AFTER_ENTRYPOINT         from plotly.subplots import make_subplots
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         out = {}
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Coerce & guard
# DUP_AFTER_ENTRYPOINT         t = np.asarray(t, float)
# DUP_AFTER_ENTRYPOINT         if t.size == 0:
# DUP_AFTER_ENTRYPOINT             return out
# DUP_AFTER_ENTRYPOINT         t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
# DUP_AFTER_ENTRYPOINT         t = np.maximum(t, 0.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         def _clean(y):
# DUP_AFTER_ENTRYPOINT             if y is None:
# DUP_AFTER_ENTRYPOINT                 return None
# DUP_AFTER_ENTRYPOINT             y = np.asarray(y, float)
# DUP_AFTER_ENTRYPOINT             y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
# DUP_AFTER_ENTRYPOINT             y = np.maximum(y, 0.0)
# DUP_AFTER_ENTRYPOINT             return y
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         qg = _clean(qg)
# DUP_AFTER_ENTRYPOINT         qo = _clean(qo)
# DUP_AFTER_ENTRYPOINT         qw = _clean(qw)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Integrate to cumulative (area under rate vs time)
# DUP_AFTER_ENTRYPOINT         if qg is not None:
# DUP_AFTER_ENTRYPOINT             cumg_mscf = cumulative_trapezoid(qg, t, initial=0.0)   # Mscf
# DUP_AFTER_ENTRYPOINT             out["cum_g_BCF"] = cumg_mscf / 1.0e6                   # BCF
# DUP_AFTER_ENTRYPOINT             out["eur_gas_BCF"] = float(out["cum_g_BCF"][-1])
# DUP_AFTER_ENTRYPOINT         if qo is not None:
# DUP_AFTER_ENTRYPOINT             cumo_stb = cumulative_trapezoid(qo, t, initial=0.0)    # STB
# DUP_AFTER_ENTRYPOINT             out["cum_o_MMBO"] = cumo_stb / 1.0e6                   # MMbbl
# DUP_AFTER_ENTRYPOINT             out["eur_oil_MMBO"] = float(out["cum_o_MMBO"][-1])
# DUP_AFTER_ENTRYPOINT         if qw is not None:
# DUP_AFTER_ENTRYPOINT             cumw_stb = cumulative_trapezoid(qw, t, initial=0.0)    # STB
# DUP_AFTER_ENTRYPOINT             out["cum_w_MMBL"] = cumw_stb / 1.0e6                   # MMbbl
# DUP_AFTER_ENTRYPOINT             out["eur_water_MMBL"] = float(out["cum_w_MMBL"][-1])
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Implied EUR GOR (scf/STB) if both exist
# DUP_AFTER_ENTRYPOINT         if ("eur_gas_BCF" in out) and ("eur_oil_MMBO" in out) and out["eur_oil_MMBO"] > 0:
# DUP_AFTER_ENTRYPOINT             gas_scf = out["eur_gas_BCF"] * 1.0e9    # BCF -> scf
# DUP_AFTER_ENTRYPOINT             oil_stb = out["eur_oil_MMBO"] * 1.0e6   # MMbbl -> STB
# DUP_AFTER_ENTRYPOINT             out["eur_gor_scfstb"] = float(gas_scf / oil_stb)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         return out
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def _apply_play_bounds_to_results(sim_like: dict, play_name: str, engine_name: str):
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         Enforce play-specific EUR sanity ranges.
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         - For the **Analytical** engine: we *soft-clamp* EURs for UI realism
# DUP_AFTER_ENTRYPOINT           (scale the cumulative curves so gauges/plots stay believable), but we do
# DUP_AFTER_ENTRYPOINT           NOT block results. We still surface a warning message.
# DUP_AFTER_ENTRYPOINT         - For the **full 3D** engine: we do not soft-clamp here; we only mark the
# DUP_AFTER_ENTRYPOINT           result invalid so the UI gate can block publishing.
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         Parameters
# DUP_AFTER_ENTRYPOINT         ----------
# DUP_AFTER_ENTRYPOINT         sim_like : dict
# DUP_AFTER_ENTRYPOINT             Simulation result dictionary to be annotated/adjusted.
# DUP_AFTER_ENTRYPOINT         play_name : str
# DUP_AFTER_ENTRYPOINT             Name of the shale play used to determine bounds.
# DUP_AFTER_ENTRYPOINT         engine_name : str
# DUP_AFTER_ENTRYPOINT             Selected engine ("Analytical", "Full 3D", etc.) to decide behavior.
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         Returns
# DUP_AFTER_ENTRYPOINT         -------
# DUP_AFTER_ENTRYPOINT         dict
# DUP_AFTER_ENTRYPOINT             The same dict with possible cumulative rescaling (Analytical only) and
# DUP_AFTER_ENTRYPOINT             these added fields:
# DUP_AFTER_ENTRYPOINT                 - 'eur_valid' : bool
# DUP_AFTER_ENTRYPOINT                 - 'eur_validation_msg' : str
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         import numpy as np
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         bounds = _sanity_bounds_for_play(play_name)
# DUP_AFTER_ENTRYPOINT         eur_valid = True
# DUP_AFTER_ENTRYPOINT         msgs = []
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         eur_g = sim_like.get("eur_gas_BCF")
# DUP_AFTER_ENTRYPOINT         eur_o = sim_like.get("eur_oil_MMBO")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # ---------------------- Analytical-only soft clamping ----------------------
# DUP_AFTER_ENTRYPOINT         if "analytical" in (engine_name or "").lower():
# DUP_AFTER_ENTRYPOINT             # Gas bounds
# DUP_AFTER_ENTRYPOINT             if eur_g is not None:
# DUP_AFTER_ENTRYPOINT                 lo, hi = bounds["gas_bcf"]
# DUP_AFTER_ENTRYPOINT                 if eur_g < lo or eur_g > hi:
# DUP_AFTER_ENTRYPOINT                     eur_valid = False
# DUP_AFTER_ENTRYPOINT                     clamp = min(max(eur_g, lo), hi)
# DUP_AFTER_ENTRYPOINT                     msgs.append(
# DUP_AFTER_ENTRYPOINT                         f"Gas EUR {eur_g:.2f} BCF clamped to [{lo:.1f}, {hi:.1f}] → {clamp:.2f} BCF."
# DUP_AFTER_ENTRYPOINT                     )
# DUP_AFTER_ENTRYPOINT                     if "cum_g_BCF" in sim_like and eur_g and eur_g > 0:
# DUP_AFTER_ENTRYPOINT                         scale = clamp / eur_g
# DUP_AFTER_ENTRYPOINT                         sim_like["cum_g_BCF"] = np.asarray(sim_like["cum_g_BCF"], float) * scale
# DUP_AFTER_ENTRYPOINT                     sim_like["eur_gas_BCF"] = clamp
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             # Oil bounds
# DUP_AFTER_ENTRYPOINT             if eur_o is not None:
# DUP_AFTER_ENTRYPOINT                 lo, hi = bounds["oil_mmbo"]
# DUP_AFTER_ENTRYPOINT                 if eur_o < lo or eur_o > hi:
# DUP_AFTER_ENTRYPOINT                     eur_valid = False
# DUP_AFTER_ENTRYPOINT                     clamp = min(max(eur_o, lo), hi)
# DUP_AFTER_ENTRYPOINT                     msgs.append(
# DUP_AFTER_ENTRYPOINT                         f"Oil EUR {eur_o:.2f} MMBO clamped to [{lo:.1f}, {hi:.1f}] → {clamp:.2f} MMBO."
# DUP_AFTER_ENTRYPOINT                     )
# DUP_AFTER_ENTRYPOINT                     if "cum_o_MMBO" in sim_like and eur_o and eur_o > 0:
# DUP_AFTER_ENTRYPOINT                         scale = clamp / eur_o
# DUP_AFTER_ENTRYPOINT                         sim_like["cum_o_MMBO"] = np.asarray(sim_like["cum_o_MMBO"], float) * scale
# DUP_AFTER_ENTRYPOINT                     sim_like["eur_oil_MMBO"] = clamp
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             # Optional: enforce a max EUR GOR
# DUP_AFTER_ENTRYPOINT             if ("eur_gor_scfstb" in sim_like) and ("eur_oil_MMBO" in sim_like):
# DUP_AFTER_ENTRYPOINT                 gor = float(sim_like["eur_gor_scfstb"])
# DUP_AFTER_ENTRYPOINT                 max_gor = bounds.get("max_eur_gor_scfstb", None)
# DUP_AFTER_ENTRYPOINT                 if max_gor and gor > max_gor and sim_like.get("eur_oil_MMBO", 0) > 0:
# DUP_AFTER_ENTRYPOINT                     eur_valid = False
# DUP_AFTER_ENTRYPOINT                     target_gas_scf = max_gor * (sim_like["eur_oil_MMBO"] * 1.0e6)
# DUP_AFTER_ENTRYPOINT                     target_gas_bcf = target_gas_scf / 1.0e9
# DUP_AFTER_ENTRYPOINT                     msgs.append(
# DUP_AFTER_ENTRYPOINT                         f"EUR GOR {gor:,.0f} > {max_gor:,.0f}; gas clamped to {target_gas_bcf:.2f} BCF."
# DUP_AFTER_ENTRYPOINT                     )
# DUP_AFTER_ENTRYPOINT                     if ("eur_gas_BCF" in sim_like) and ("cum_g_BCF" in sim_like) and sim_like["eur_gas_BCF"] > 0:
# DUP_AFTER_ENTRYPOINT                         scale = target_gas_bcf / sim_like["eur_gas_BCF"]
# DUP_AFTER_ENTRYPOINT                         sim_like["cum_g_BCF"] = np.asarray(sim_like["cum_g_BCF"], float) * scale
# DUP_AFTER_ENTRYPOINT                         sim_like["eur_gas_BCF"] = target_gas_bcf
# DUP_AFTER_ENTRYPOINT                     sim_like["eur_gor_scfstb"] = max_gor
# DUP_AFTER_ENTRYPOINT         # --------------------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # --- Final validity flags & message (helper-only policy) ---
# DUP_AFTER_ENTRYPOINT         is_analytical = "analytical" in (engine_name or "").lower()
# DUP_AFTER_ENTRYPOINT         had_issues = bool(msgs)  # were any clamps / violations detected?
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         if is_analytical:
# DUP_AFTER_ENTRYPOINT             # For the Analytical proxy: never block, but KEEP the message so we can warn.
# DUP_AFTER_ENTRYPOINT             sim_like["eur_valid"] = True
# DUP_AFTER_ENTRYPOINT             sim_like["eur_validation_msg"] = "OK" if not had_issues else " | ".join(msgs)
# DUP_AFTER_ENTRYPOINT         else:
# DUP_AFTER_ENTRYPOINT             # For the full 3D engine: allow blocking when invalid.
# DUP_AFTER_ENTRYPOINT             sim_like["eur_valid"] = eur_valid
# DUP_AFTER_ENTRYPOINT             sim_like["eur_validation_msg"] = "OK" if eur_valid else " | ".join(msgs)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         return sim_like
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def validate_midland_eur(EUR_o_MMBO, EUR_g_BCF, *, pb_psi=None, Rs_pb=None):
# DUP_AFTER_ENTRYPOINT         lo_o, hi_o = MIDLAND_BOUNDS["oil_mmbo"]
# DUP_AFTER_ENTRYPOINT         lo_g, hi_g = MIDLAND_BOUNDS["gas_bcf"]
# DUP_AFTER_ENTRYPOINT         msgs, ok = [], True
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         if EUR_o_MMBO < lo_o or EUR_o_MMBO > hi_o:
# DUP_AFTER_ENTRYPOINT             ok = False
# DUP_AFTER_ENTRYPOINT             msgs.append(f"Oil EUR {EUR_o_MMBO:.2f} MMBO outside Midland sanity [{lo_o}, {hi_o}] MMBO.")
# DUP_AFTER_ENTRYPOINT         if EUR_g_BCF < lo_g or EUR_g_BCF > hi_g:
# DUP_AFTER_ENTRYPOINT             ok = False
# DUP_AFTER_ENTRYPOINT             msgs.append(f"Gas EUR {EUR_g_BCF:.2f} BCF outside Midland sanity [{lo_g}, {hi_g}] BCF.")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Tolerance-aware PVT/GOR consistency (~3×Rs at pb)
# DUP_AFTER_ENTRYPOINT         if EUR_o_MMBO > 0 and Rs_pb not in (None, 0):
# DUP_AFTER_ENTRYPOINT             implied_GOR = (EUR_g_BCF * 1e9) / (EUR_o_MMBO * 1e6)  # scf/STB
# DUP_AFTER_ENTRYPOINT             limit = 3.0 * float(Rs_pb)
# DUP_AFTER_ENTRYPOINT             tol = 1e-6
# DUP_AFTER_ENTRYPOINT             if implied_GOR > (limit + tol) and (pb_psi or 0) > 1.0:
# DUP_AFTER_ENTRYPOINT                 ok = False
# DUP_AFTER_ENTRYPOINT                 msgs.append(
# DUP_AFTER_ENTRYPOINT                     f"Implied EUR GOR {implied_GOR:,.0f} scf/STB inconsistent with Rs(pb)≈{Rs_pb:,.0f} "
# DUP_AFTER_ENTRYPOINT                     f"(>{limit:,.0f})."
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         return ok, " ".join(msgs) if msgs else "OK"
# DUP_AFTER_ENTRYPOINT     # ----------------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ======================================================================
# DUP_AFTER_ENTRYPOINT     # Robust Arps + Gauge helpers
# DUP_AFTER_ENTRYPOINT     # ======================================================================
# DUP_AFTER_ENTRYPOINT     import numpy as _np
# DUP_AFTER_ENTRYPOINT     try:
# DUP_AFTER_ENTRYPOINT         from scipy.integrate import cumulative_trapezoid as _ctr
# DUP_AFTER_ENTRYPOINT     except Exception:
# DUP_AFTER_ENTRYPOINT         _ctr = None  # numeric cumulative optional
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def arps_rate(qi: float, Di: float, b: float, t) -> _np.ndarray:
# DUP_AFTER_ENTRYPOINT         """Robust Arps rate with exponential fallback and safe power."""
# DUP_AFTER_ENTRYPOINT         t  = _np.asarray(t, float)
# DUP_AFTER_ENTRYPOINT         t  = _np.maximum(t, 0.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         qi = float(qi)
# DUP_AFTER_ENTRYPOINT         Di = float(Di)
# DUP_AFTER_ENTRYPOINT         b  = float(b)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Exponential decline (b ~ 0)
# DUP_AFTER_ENTRYPOINT         if abs(b) < 1e-12:
# DUP_AFTER_ENTRYPOINT             return qi * _np.exp(-Di * t)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Hyperbolic: q = qi * (1 + b*Di*t)^(-1/b)
# DUP_AFTER_ENTRYPOINT         base = 1.0 + b * Di * t
# DUP_AFTER_ENTRYPOINT         # guard against NaN/Inf and non-positive base
# DUP_AFTER_ENTRYPOINT         base = _np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
# DUP_AFTER_ENTRYPOINT         base = _np.maximum(base, 1e-12)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         exponent = -1.0 / b
# DUP_AFTER_ENTRYPOINT         return qi * safe_power(base, exponent)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def arps_cum(qi: float, Di: float, b: float, t) -> _np.ndarray:
# DUP_AFTER_ENTRYPOINT         """Robust Arps cumulative (analytic), exponential fallback for b≈0."""
# DUP_AFTER_ENTRYPOINT         t  = _np.asarray(t, float)
# DUP_AFTER_ENTRYPOINT         t  = _np.maximum(t, 0.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         qi = float(qi)
# DUP_AFTER_ENTRYPOINT         Di = float(Di)
# DUP_AFTER_ENTRYPOINT         b  = float(b)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Exponential cumulative: Np(t) = (qi/Di) * (1 - e^{-Di t})
# DUP_AFTER_ENTRYPOINT         if abs(b) < 1e-12:
# DUP_AFTER_ENTRYPOINT             Di_safe = max(Di, 1e-16)
# DUP_AFTER_ENTRYPOINT             return (qi / Di_safe) * (1.0 - _np.exp(-Di * t))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Hyperbolic cumulative: Np = (qi / ((1-b)*Di)) * [1 - (1 + b*Di*t)^{(1-b)/b}]
# DUP_AFTER_ENTRYPOINT         base = 1.0 + b * Di * t
# DUP_AFTER_ENTRYPOINT         base = _np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
# DUP_AFTER_ENTRYPOINT         base = _np.maximum(base, 1e-12)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         one_minus_b = 1.0 - b
# DUP_AFTER_ENTRYPOINT         denom = max(one_minus_b * Di, 1e-16)
# DUP_AFTER_ENTRYPOINT         exponent = (one_minus_b / b)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         return (qi / denom) * (1.0 - safe_power(base, exponent))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # Gauge helper with subtitle + unit suffix
# DUP_AFTER_ENTRYPOINT     def _render_gauge_v2(
# DUP_AFTER_ENTRYPOINT         title: str,
# DUP_AFTER_ENTRYPOINT         value: float,
# DUP_AFTER_ENTRYPOINT         minmax=(0.0, 1.0),
# DUP_AFTER_ENTRYPOINT         fmt: str = "{:,.2f}",
# DUP_AFTER_ENTRYPOINT         unit_suffix: str = "",
# DUP_AFTER_ENTRYPOINT         **kwargs,
# DUP_AFTER_ENTRYPOINT     ):
# DUP_AFTER_ENTRYPOINT         import math
# DUP_AFTER_ENTRYPOINT         import plotly.graph_objects as go
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Prefer the utils implementation if it exists and accepts our args.
# DUP_AFTER_ENTRYPOINT         if utils and hasattr(utils, "_render_gauge"):
# DUP_AFTER_ENTRYPOINT             # Try calling utils._render_gauge; if it rejects unit_suffix, fall back.
# DUP_AFTER_ENTRYPOINT             try:
# DUP_AFTER_ENTRYPOINT                 return utils._render_gauge(
# DUP_AFTER_ENTRYPOINT                     title=title, value=value, minmax=minmax, fmt=fmt, unit_suffix=unit_suffix
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT             except TypeError:
# DUP_AFTER_ENTRYPOINT                 # Call without unit_suffix, then append suffix if possible
# DUP_AFTER_ENTRYPOINT                 fig = utils._render_gauge(title=title, value=value, minmax=minmax, fmt=fmt)
# DUP_AFTER_ENTRYPOINT                 try:
# DUP_AFTER_ENTRYPOINT                     fig.update_traces(number={"suffix": f" {unit_suffix}" if unit_suffix else ""})
# DUP_AFTER_ENTRYPOINT                 except Exception:
# DUP_AFTER_ENTRYPOINT                     pass
# DUP_AFTER_ENTRYPOINT                 return fig
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Fallback: local implementation
# DUP_AFTER_ENTRYPOINT         try:
# DUP_AFTER_ENTRYPOINT             vmin, vmax = (minmax if isinstance(minmax, (list, tuple)) and len(minmax) == 2 else (0.0, 1.0))
# DUP_AFTER_ENTRYPOINT         except Exception:
# DUP_AFTER_ENTRYPOINT             vmin, vmax = 0.0, 1.0
# DUP_AFTER_ENTRYPOINT         if vmax <= vmin:
# DUP_AFTER_ENTRYPOINT             vmax = vmin + 1.0
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         x = float(value) if value is not None and not isinstance(value, str) else 0.0
# DUP_AFTER_ENTRYPOINT         if math.isnan(x) or math.isinf(x):
# DUP_AFTER_ENTRYPOINT             x = 0.0
# DUP_AFTER_ENTRYPOINT         x = max(vmin, min(vmax, x))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         vf = fmt.replace("{", "").replace("}", "").replace(":", "")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         fig = go.Figure(
# DUP_AFTER_ENTRYPOINT             go.Indicator(
# DUP_AFTER_ENTRYPOINT                 mode="gauge+number",
# DUP_AFTER_ENTRYPOINT                 value=x,
# DUP_AFTER_ENTRYPOINT                 title={"text": title},
# DUP_AFTER_ENTRYPOINT                 number={"valueformat": vf, "suffix": f" {unit_suffix}" if unit_suffix else ""},
# DUP_AFTER_ENTRYPOINT                 gauge={"axis": {"range": [vmin, vmax]}},
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT         )
# DUP_AFTER_ENTRYPOINT         fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
# DUP_AFTER_ENTRYPOINT         return fig
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def fmt_qty(v: float, unit: str) -> str:
# DUP_AFTER_ENTRYPOINT         """Small formatter for quantities with units."""
# DUP_AFTER_ENTRYPOINT         try:
# DUP_AFTER_ENTRYPOINT             x = float(v)
# DUP_AFTER_ENTRYPOINT         except Exception:
# DUP_AFTER_ENTRYPOINT             return f"{v}"
# DUP_AFTER_ENTRYPOINT         unit_up = (unit or "").upper()
# DUP_AFTER_ENTRYPOINT         if unit_up == "BCF":
# DUP_AFTER_ENTRYPOINT             return f"{x:,.2f} BCF"
# DUP_AFTER_ENTRYPOINT         if unit_up == "MMBO":
# DUP_AFTER_ENTRYPOINT             return f"{x:,.2f} MMBO"
# DUP_AFTER_ENTRYPOINT         return f"{x:,.2f} {unit}"
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ---------------------- Plot Style Pack (Gas=RED, Oil=GREEN) ----------------------
# DUP_AFTER_ENTRYPOINT     COLOR_GAS = COLOR_GAS if "COLOR_GAS" in globals() else "#1f77b4"
# DUP_AFTER_ENTRYPOINT     COLOR_OIL = COLOR_OIL if "COLOR_OIL" in globals() else "#ff7f0e"
# DUP_AFTER_ENTRYPOINT     COLOR_WATER = COLOR_WATER if "COLOR_WATER" in globals() else "#2ca02c"
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # Clean global template
# DUP_AFTER_ENTRYPOINT     pio.templates.default = "plotly_white"
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def _style_fig(fig, title, xlab, ylab_left, ylab_right=None):
# DUP_AFTER_ENTRYPOINT         fig.update_layout(
# DUP_AFTER_ENTRYPOINT             title=dict(text=f"<b>{title}</b>", x=0, xanchor="left"),
# DUP_AFTER_ENTRYPOINT             font=dict(size=14),
# DUP_AFTER_ENTRYPOINT             margin=dict(l=60, r=90, t=60, b=60),
# DUP_AFTER_ENTRYPOINT             legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
# DUP_AFTER_ENTRYPOINT         )
# DUP_AFTER_ENTRYPOINT         fig.update_xaxes(title=xlab, showline=True, linewidth=1, mirror=True)
# DUP_AFTER_ENTRYPOINT         fig.update_yaxes(title=ylab_left, showline=True, linewidth=1, mirror=True, secondary_y=False)
# DUP_AFTER_ENTRYPOINT         if ylab_right:
# DUP_AFTER_ENTRYPOINT             fig.update_yaxes(title=ylab_right, secondary_y=True, showgrid=False)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def rate_chart(t, qg=None, qo=None, qw=None):
# DUP_AFTER_ENTRYPOINT         """Dual-axis rate chart: Gas left (red), Liquids right (green/blue)."""
# DUP_AFTER_ENTRYPOINT         fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
# DUP_AFTER_ENTRYPOINT         if qg is not None:
# DUP_AFTER_ENTRYPOINT             fig.add_trace(
# DUP_AFTER_ENTRYPOINT                 go.Scatter(x=t, y=qg, name="Gas (Mscf/d)", line=dict(width=2, color=COLOR_GAS)),
# DUP_AFTER_ENTRYPOINT                 secondary_y=False,
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT         if qo is not None:
# DUP_AFTER_ENTRYPOINT             fig.add_trace(
# DUP_AFTER_ENTRYPOINT                 go.Scatter(x=t, y=qo, name="Oil (STB/d)", line=dict(width=2, color=COLOR_OIL)),
# DUP_AFTER_ENTRYPOINT                 secondary_y=True,
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT         if qw is not None:
# DUP_AFTER_ENTRYPOINT             fig.add_trace(
# DUP_AFTER_ENTRYPOINT                 go.Scatter(x=t, y=qw, name="Water (STB/d)", line=dict(width=2, color=COLOR_WATER)),
# DUP_AFTER_ENTRYPOINT                 secondary_y=True,
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT         _style_fig(
# DUP_AFTER_ENTRYPOINT             fig, "Production Rate vs. Time", "Time (days)", "Gas Rate (Mscf/d)", "Liquid Rate (STB/d)"
# DUP_AFTER_ENTRYPOINT         )
# DUP_AFTER_ENTRYPOINT         return fig
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # High-resolution export button for all charts
# DUP_AFTER_ENTRYPOINT     PLOT_CONFIG = {
# DUP_AFTER_ENTRYPOINT         "displaylogo": False,
# DUP_AFTER_ENTRYPOINT         "toImageButtonOptions": {
# DUP_AFTER_ENTRYPOINT             "format": "png",
# DUP_AFTER_ENTRYPOINT             "filename": "plot",
# DUP_AFTER_ENTRYPOINT             "height": 720,
# DUP_AFTER_ENTRYPOINT             "width": 1280,
# DUP_AFTER_ENTRYPOINT             "scale": 3,
# DUP_AFTER_ENTRYPOINT         },
# DUP_AFTER_ENTRYPOINT     }
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ----------------------------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT     # ------------------------ Utils ------------------------
# DUP_AFTER_ENTRYPOINT     def _setdefault(k, v):
# DUP_AFTER_ENTRYPOINT         if k not in st.session_state:
# DUP_AFTER_ENTRYPOINT             st.session_state[k] = v
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def _on_play_change():
# DUP_AFTER_ENTRYPOINT         # Clear prior results so the UI cannot show stale EURs
# DUP_AFTER_ENTRYPOINT         st.session_state.sim = None
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def _safe_rerun():
# DUP_AFTER_ENTRYPOINT         if hasattr(st, "rerun"):
# DUP_AFTER_ENTRYPOINT             st.rerun()
# DUP_AFTER_ENTRYPOINT         elif hasattr(st, "experimental_rerun"):
# DUP_AFTER_ENTRYPOINT             st.experimental_rerun()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def _sim_signature_from_state():
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         Lightweight signature of knobs that materially change physics/EUR policy.
# DUP_AFTER_ENTRYPOINT         Keep this at module scope so both the engine and Results tab can use it.
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         s = st.session_state
# DUP_AFTER_ENTRYPOINT         play = s.get("play_sel", "")
# DUP_AFTER_ENTRYPOINT         engine = s.get("engine_type", "")
# DUP_AFTER_ENTRYPOINT         ctrl  = s.get("pad_ctrl", "BHP")
# DUP_AFTER_ENTRYPOINT         bhp   = float(s.get("pad_bhp_psi", 0.0))
# DUP_AFTER_ENTRYPOINT         r_m   = float(s.get("pad_rate_mscfd", 0.0))
# DUP_AFTER_ENTRYPOINT         r_o   = float(s.get("pad_rate_stbd", 0.0))
# DUP_AFTER_ENTRYPOINT         pb    = float(s.get("pb_psi", 0.0))
# DUP_AFTER_ENTRYPOINT         rs    = float(s.get("Rs_pb_scf_stb", 0.0))
# DUP_AFTER_ENTRYPOINT         bo    = float(s.get("Bo_pb_rb_stb", 1.0))
# DUP_AFTER_ENTRYPOINT         pinit = float(s.get("p_init_psi", 0.0))
# DUP_AFTER_ENTRYPOINT         key = (play, engine, ctrl, bhp, r_m, r_o, pb, rs, bo, pinit)
# DUP_AFTER_ENTRYPOINT         return hash(key)
# DUP_AFTER_ENTRYPOINT     
# DUP_AFTER_ENTRYPOINT     def is_heel_location_valid(x_heel_ft, y_heel_ft, state):
# DUP_AFTER_ENTRYPOINT         """Simple feasibility check for well placement (stay inside model and avoid fault strip)."""
# DUP_AFTER_ENTRYPOINT         x_max = state['nx'] * state['dx'] - state['L_ft']
# DUP_AFTER_ENTRYPOINT         y_max = state['ny'] * state['dy']
# DUP_AFTER_ENTRYPOINT         if not (0 <= x_heel_ft <= x_max and 0 <= y_heel_ft <= y_max):
# DUP_AFTER_ENTRYPOINT             return False
# DUP_AFTER_ENTRYPOINT         if state.get('use_fault'):
# DUP_AFTER_ENTRYPOINT             plane = state.get('fault_plane', 'i-plane (vertical)')
# DUP_AFTER_ENTRYPOINT             if 'i-plane' in plane:
# DUP_AFTER_ENTRYPOINT                 fault_x = state['fault_index'] * state['dx']
# DUP_AFTER_ENTRYPOINT                 return abs(x_heel_ft - fault_x) > 2 * state['dx']
# DUP_AFTER_ENTRYPOINT             else:
# DUP_AFTER_ENTRYPOINT                 fault_y = state['fault_index'] * state['dy']
# DUP_AFTER_ENTRYPOINT                 return abs(y_heel_ft - fault_y) > 2 * state['dy']
# DUP_AFTER_ENTRYPOINT         return True
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # near the top of app.py (once)
# DUP_AFTER_ENTRYPOINT     st.set_page_config(page_title="Reservoir Simulator", layout="wide")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ------------------------ Defaults ------------------------
# DUP_AFTER_ENTRYPOINT     _setdefault("apply_preset_payload", None)
# DUP_AFTER_ENTRYPOINT     _setdefault("sim", None)
# DUP_AFTER_ENTRYPOINT     _setdefault("rng_seed", 1234)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # --- Engine & Model type options ---
# DUP_AFTER_ENTRYPOINT     ENGINE_TYPES = [
# DUP_AFTER_ENTRYPOINT         "Analytical Model (Fast Proxy)",
# DUP_AFTER_ENTRYPOINT         "3D Three-Phase Implicit (Phase 1a)",
# DUP_AFTER_ENTRYPOINT         "3D Three-Phase Implicit (Phase 1b)",
# DUP_AFTER_ENTRYPOINT     ]
# DUP_AFTER_ENTRYPOINT     # Model Type options (must match the sidebar selectbox exactly)
# DUP_AFTER_ENTRYPOINT     VALID_MODEL_TYPES = ["Unconventional Reservoir", "Black Oil Reservoir"]
# DUP_AFTER_ENTRYPOINT     _setdefault("sim_mode", VALID_MODEL_TYPES[0])  # Default to the first allowed value
# DUP_AFTER_ENTRYPOINT     _setdefault("sim_mode", VALID_MODEL_TYPES[0])
# DUP_AFTER_ENTRYPOINT     _setdefault("dfn_segments", None)
# DUP_AFTER_ENTRYPOINT     _setdefault("use_dfn_sink", True)
# DUP_AFTER_ENTRYPOINT     _setdefault("use_auto_dfn", True)
# DUP_AFTER_ENTRYPOINT     _setdefault("vol_downsample", 2)
# DUP_AFTER_ENTRYPOINT     _setdefault("iso_value_rel", 0.5)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     defaults = dict(
# DUP_AFTER_ENTRYPOINT         nx=300,
# DUP_AFTER_ENTRYPOINT         ny=60,
# DUP_AFTER_ENTRYPOINT         nz=12,
# DUP_AFTER_ENTRYPOINT         dx=40.0,
# DUP_AFTER_ENTRYPOINT         dy=40.0,
# DUP_AFTER_ENTRYPOINT         dz=15.0,
# DUP_AFTER_ENTRYPOINT         k_stdev=0.02,
# DUP_AFTER_ENTRYPOINT         phi_stdev=0.02,
# DUP_AFTER_ENTRYPOINT         anis_kxky=1.0,
# DUP_AFTER_ENTRYPOINT         facies_style="Continuous (Gaussian)",
# DUP_AFTER_ENTRYPOINT         use_fault=False,
# DUP_AFTER_ENTRYPOINT         fault_plane="i-plane (vertical)",
# DUP_AFTER_ENTRYPOINT         fault_index=60,
# DUP_AFTER_ENTRYPOINT         fault_tm=0.10,
# DUP_AFTER_ENTRYPOINT         n_laterals=2,
# DUP_AFTER_ENTRYPOINT         L_ft=10000.0,
# DUP_AFTER_ENTRYPOINT         stage_spacing_ft=250.0,
# DUP_AFTER_ENTRYPOINT         clusters_per_stage=3,
# DUP_AFTER_ENTRYPOINT         dP_LE_psi=200.0,
# DUP_AFTER_ENTRYPOINT         f_fric=0.02,
# DUP_AFTER_ENTRYPOINT         wellbore_ID_ft=0.30,
# DUP_AFTER_ENTRYPOINT         xf_ft=300.0,
# DUP_AFTER_ENTRYPOINT         hf_ft=180.0,
# DUP_AFTER_ENTRYPOINT         pad_interf=0.20,
# DUP_AFTER_ENTRYPOINT         pad_ctrl="BHP",
# DUP_AFTER_ENTRYPOINT         pad_bhp_psi=2500.0,
# DUP_AFTER_ENTRYPOINT         pad_rate_mscfd=100000.0,
# DUP_AFTER_ENTRYPOINT         outer_bc="Infinite-acting",
# DUP_AFTER_ENTRYPOINT         p_outer_psi=7950.0,
# DUP_AFTER_ENTRYPOINT         pb_psi=5200.0,
# DUP_AFTER_ENTRYPOINT         Rs_pb_scf_stb=650.0,
# DUP_AFTER_ENTRYPOINT         Bo_pb_rb_stb=1.35,
# DUP_AFTER_ENTRYPOINT         muo_pb_cp=1.20,
# DUP_AFTER_ENTRYPOINT         mug_pb_cp=0.020,
# DUP_AFTER_ENTRYPOINT         a_g=0.15,
# DUP_AFTER_ENTRYPOINT         z_g=0.90,
# DUP_AFTER_ENTRYPOINT         p_init_psi=5800.0,
# DUP_AFTER_ENTRYPOINT         p_min_bhp_psi=2500.0,
# DUP_AFTER_ENTRYPOINT         ct_1_over_psi=0.000015,
# DUP_AFTER_ENTRYPOINT         include_RsP=True,
# DUP_AFTER_ENTRYPOINT         krw_end=0.6,
# DUP_AFTER_ENTRYPOINT         kro_end=0.8,
# DUP_AFTER_ENTRYPOINT         nw=2.0,
# DUP_AFTER_ENTRYPOINT         no=2.0,
# DUP_AFTER_ENTRYPOINT         Swc=0.15,
# DUP_AFTER_ENTRYPOINT         Sor=0.25,
# DUP_AFTER_ENTRYPOINT         pc_slope_psi=0.0,
# DUP_AFTER_ENTRYPOINT         ct_o_1psi=8e-6,
# DUP_AFTER_ENTRYPOINT         ct_g_1psi=3e-6,
# DUP_AFTER_ENTRYPOINT         ct_w_1psi=3e-6,
# DUP_AFTER_ENTRYPOINT         newton_tol=1e-6,
# DUP_AFTER_ENTRYPOINT         trans_tol=1e-7,
# DUP_AFTER_ENTRYPOINT         max_newton=12,
# DUP_AFTER_ENTRYPOINT         max_lin=200,
# DUP_AFTER_ENTRYPOINT         threads=0,
# DUP_AFTER_ENTRYPOINT         use_omp=False,
# DUP_AFTER_ENTRYPOINT         use_mkl=False,
# DUP_AFTER_ENTRYPOINT         use_pyamg=False,
# DUP_AFTER_ENTRYPOINT         use_cusparse=False,
# DUP_AFTER_ENTRYPOINT         dfn_radius_ft=60.0,
# DUP_AFTER_ENTRYPOINT         dfn_strength_psi=500.0,
# DUP_AFTER_ENTRYPOINT         engine_type="Analytical Model (Fast Proxy)"  # Set stable engine as default
# DUP_AFTER_ENTRYPOINT     )
# DUP_AFTER_ENTRYPOINT     for k, v in defaults.items():
# DUP_AFTER_ENTRYPOINT         _setdefault(k, v)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     if st.session_state.apply_preset_payload is not None:
# DUP_AFTER_ENTRYPOINT         for k, v in st.session_state.apply_preset_payload.items():
# DUP_AFTER_ENTRYPOINT             st.session_state[k] = v
# DUP_AFTER_ENTRYPOINT         st.session_state.apply_preset_payload = None
# DUP_AFTER_ENTRYPOINT         _safe_rerun()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ------------------------ PRESETS (US + Canada Shale Plays) ------------------------
# DUP_AFTER_ENTRYPOINT     # Typical, rounded values for quick-start modeling. Tune as needed per asset.
# DUP_AFTER_ENTRYPOINT     PLAY_PRESETS = {
# DUP_AFTER_ENTRYPOINT         # --- UNITED STATES ---
# DUP_AFTER_ENTRYPOINT         "Permian – Midland (Oil)": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=10000.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=250.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=300.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=180.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=650.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=5200.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.35,
# DUP_AFTER_ENTRYPOINT             p_init_psi=5800.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT         "Permian – Delaware (Oil/Gas)": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=10000.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=225.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=320.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=200.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=700.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=5400.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.36,
# DUP_AFTER_ENTRYPOINT             p_init_psi=6000.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT         "Eagle Ford (Oil Window)": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=9000.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=225.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=270.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=150.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=700.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=5400.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.34,
# DUP_AFTER_ENTRYPOINT             p_init_psi=5600.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT         "Eagle Ford (Condensate)": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=9000.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=225.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=300.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=160.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=900.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=5600.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.30,
# DUP_AFTER_ENTRYPOINT             p_init_psi=5800.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT         "Bakken / Three Forks (Oil)": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=10000.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=240.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=280.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=160.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=350.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=4300.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.20,
# DUP_AFTER_ENTRYPOINT             p_init_psi=4700.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT         "Haynesville (Dry Gas)": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=10000.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=200.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=350.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=180.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=0.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=1.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.00,
# DUP_AFTER_ENTRYPOINT             p_init_psi=7000.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT         "Marcellus (Dry Gas)": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=9000.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=225.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=300.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=150.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=0.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=1.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.00,
# DUP_AFTER_ENTRYPOINT             p_init_psi=5200.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT         "Utica (Liquids-Rich)": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=10000.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=225.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=320.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=180.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=400.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=5000.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.22,
# DUP_AFTER_ENTRYPOINT             p_init_psi=5500.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT         "Barnett (Gas)": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=6500.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=200.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=250.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=120.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=0.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=1.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.00,
# DUP_AFTER_ENTRYPOINT             p_init_psi=4200.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT         "Niobrara / DJ (Oil)": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=9000.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=225.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=280.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=140.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=250.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=3800.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.18,
# DUP_AFTER_ENTRYPOINT             p_init_psi=4200.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT         "Anadarko – Woodford": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=10000.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=225.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=300.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=160.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=300.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=4600.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.20,
# DUP_AFTER_ENTRYPOINT             p_init_psi=5000.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT         "Granite Wash": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=8000.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=225.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=280.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=150.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=200.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=4200.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.15,
# DUP_AFTER_ENTRYPOINT             p_init_psi=4600.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT         "Fayetteville (Gas)": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=6000.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=200.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=240.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=120.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=0.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=1.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.00,
# DUP_AFTER_ENTRYPOINT             p_init_psi=3500.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT         "Tuscaloosa Marine (Oil)": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=10000.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=250.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=300.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=160.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=450.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=5000.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.25,
# DUP_AFTER_ENTRYPOINT             p_init_psi=5400.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT         # --- CANADA ---
# DUP_AFTER_ENTRYPOINT         "Montney (Condensate-Rich)": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=10000.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=225.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=330.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=180.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=600.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=5200.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.28,
# DUP_AFTER_ENTRYPOINT             p_init_psi=5600.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT         "Duvernay (Liquids)": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=9500.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=225.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=320.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=180.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=700.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=5400.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.32,
# DUP_AFTER_ENTRYPOINT             p_init_psi=5800.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT         "Horn River (Dry Gas)": dict(
# DUP_AFTER_ENTRYPOINT             L_ft=9000.0,
# DUP_AFTER_ENTRYPOINT             stage_spacing_ft=225.0,
# DUP_AFTER_ENTRYPOINT             xf_ft=320.0,
# DUP_AFTER_ENTRYPOINT             hf_ft=170.0,
# DUP_AFTER_ENTRYPOINT             Rs_pb_scf_stb=0.0,
# DUP_AFTER_ENTRYPOINT             pb_psi=1.0,
# DUP_AFTER_ENTRYPOINT             Bo_pb_rb_stb=1.00,
# DUP_AFTER_ENTRYPOINT             p_init_psi=6500.0,
# DUP_AFTER_ENTRYPOINT         ),
# DUP_AFTER_ENTRYPOINT     }
# DUP_AFTER_ENTRYPOINT     PLAY_LIST = list(PLAY_PRESETS.keys())
# DUP_AFTER_ENTRYPOINT     # Create a single 'state' dictionary from session_state for cleaner access
# DUP_AFTER_ENTRYPOINT     # This makes the variable available globally for all tabs to use.
# DUP_AFTER_ENTRYPOINT     state = {k: st.session_state.get(k, v) for k, v in defaults.items()}
# DUP_AFTER_ENTRYPOINT     #### Part 2: Core Logic, Simulation Engine, and Sidebar UI ####
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ------------------------ HELPER FUNCTIONS ------------------------
# DUP_AFTER_ENTRYPOINT     def Rs_of_p(p, pb, Rs_pb):
# DUP_AFTER_ENTRYPOINT         p = np.asarray(p, float)
# DUP_AFTER_ENTRYPOINT         return np.where(p <= pb, Rs_pb, Rs_pb + 0.00012 * (p - pb) ** 1.1)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def Bo_of_p(p, pb, Bo_pb):
# DUP_AFTER_ENTRYPOINT         p = np.asarray(p, float)
# DUP_AFTER_ENTRYPOINT         slope = -1.0e-5
# DUP_AFTER_ENTRYPOINT         return np.where(p <= pb, Bo_pb, Bo_pb + slope * (p - pb))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def Bg_of_p(p):
# DUP_AFTER_ENTRYPOINT         p = np.asarray(p, float)
# DUP_AFTER_ENTRYPOINT         return 1.2e-5 + (7.0e-6 - 1.2e-5) * (p - p.min()) / (p.max() - p.min() + 1e-12)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def mu_g_of_p(p, pb, mug_pb):
# DUP_AFTER_ENTRYPOINT         p = np.asarray(p, float)
# DUP_AFTER_ENTRYPOINT         peak = mug_pb * 1.03
# DUP_AFTER_ENTRYPOINT         left = mug_pb - 0.0006
# DUP_AFTER_ENTRYPOINT         right = mug_pb - 0.0008
# DUP_AFTER_ENTRYPOINT         mu = np.where(
# DUP_AFTER_ENTRYPOINT             p < pb,
# DUP_AFTER_ENTRYPOINT             left + (peak - left) * (p - p.min()) / (pb - p.min() + 1e-9),
# DUP_AFTER_ENTRYPOINT             peak + (right - peak) * (p - pb) / (p.max() - pb + 1e-9),
# DUP_AFTER_ENTRYPOINT         )
# DUP_AFTER_ENTRYPOINT         return np.clip(mu, 0.001, None)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def z_factor_approx(p_psi, p_init_psi=5800.0):
# DUP_AFTER_ENTRYPOINT         p_norm = p_psi / p_init_psi
# DUP_AFTER_ENTRYPOINT         return 0.95 - 0.2 * (1 - p_norm) + 0.4 * (1 - p_norm) ** 2
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # --- PVT adapter: callables named exactly as the engine expects ---
# DUP_AFTER_ENTRYPOINT     class _PVTAdapter(dict):
# DUP_AFTER_ENTRYPOINT         """Adapter that holds PVT callables and parameters; supports attribute & dict access."""
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         def __init__(self, **kwargs):
# DUP_AFTER_ENTRYPOINT             super().__init__(**kwargs)
# DUP_AFTER_ENTRYPOINT             self.__dict__.update(kwargs)  # allows pvt.Rs(...) and pvt['Rs'](...)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def _build_pvt_payload_from_state(state):
# DUP_AFTER_ENTRYPOINT         """Build a PVT payload the engine can use directly (with callables)."""
# DUP_AFTER_ENTRYPOINT         pb = float(state.get('pb_psi', 1.0))
# DUP_AFTER_ENTRYPOINT         Rs_pb = float(state.get('Rs_pb_scf_stb', 0.0))
# DUP_AFTER_ENTRYPOINT         Bo_pb = float(state.get('Bo_pb_rb_stb', 1.0))
# DUP_AFTER_ENTRYPOINT         mug_pb = float(state.get('mug_pb_cp', 0.020))
# DUP_AFTER_ENTRYPOINT         muo_pb = float(state.get('muo_pb_cp', 1.20))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         def Rs(p):
# DUP_AFTER_ENTRYPOINT             return Rs_of_p(p, pb, Rs_pb)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         def Bo(p):
# DUP_AFTER_ENTRYPOINT             return Bo_of_p(p, pb, Bo_pb)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         def Bg(p):
# DUP_AFTER_ENTRYPOINT             return Bg_of_p(p)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         def mu_g(p):
# DUP_AFTER_ENTRYPOINT             return mu_g_of_p(p, pb, mug_pb)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         def mu_o(p):
# DUP_AFTER_ENTRYPOINT             return np.full_like(np.asarray(p, float), muo_pb, dtype=float)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         return _PVTAdapter(
# DUP_AFTER_ENTRYPOINT             Rs=Rs,
# DUP_AFTER_ENTRYPOINT             Bo=Bo,
# DUP_AFTER_ENTRYPOINT             Bg=Bg,
# DUP_AFTER_ENTRYPOINT             mu_g=mu_g,
# DUP_AFTER_ENTRYPOINT             mu_o=mu_o,
# DUP_AFTER_ENTRYPOINT             ct_o_1psi=state.get('ct_o_1psi', 8e-6),
# DUP_AFTER_ENTRYPOINT             ct_g_1psi=state.get('ct_g_1psi', 3e-6),
# DUP_AFTER_ENTRYPOINT             ct_w_1psi=state.get('ct_w_1psi', 3e-6),
# DUP_AFTER_ENTRYPOINT             include_RsP=bool(state.get('include_RsP', True)),
# DUP_AFTER_ENTRYPOINT             pb_psi=pb,
# DUP_AFTER_ENTRYPOINT         )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # --- Defensive monkey-patch: if engine's Fluid class lacks methods, inject thin wrappers ---
# DUP_AFTER_ENTRYPOINT     def _monkeypatch_engine_fluid_if_needed(adapter):
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         Some engine builds instantiate their own Fluid and expect .Rs/.Bo/.Bg/.mu_g/.mu_o.
# DUP_AFTER_ENTRYPOINT         If missing, attach wrappers that forward to our adapter. Safe no-op if import fails.
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         try:
# DUP_AFTER_ENTRYPOINT             from core.blackoil_pvt1 import Fluid as EngineFluid  # optional; may not exist in all builds
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             patched = []
# DUP_AFTER_ENTRYPOINT             if not hasattr(EngineFluid, "Rs"):
# DUP_AFTER_ENTRYPOINT                 EngineFluid.Rs = lambda self, p: adapter.Rs(p)
# DUP_AFTER_ENTRYPOINT                 patched.append("Rs")
# DUP_AFTER_ENTRYPOINT             if not hasattr(EngineFluid, "Bo"):
# DUP_AFTER_ENTRYPOINT                 EngineFluid.Bo = lambda self, p: adapter.Bo(p)
# DUP_AFTER_ENTRYPOINT                 patched.append("Bo")
# DUP_AFTER_ENTRYPOINT             if not hasattr(EngineFluid, "Bg"):
# DUP_AFTER_ENTRYPOINT                 EngineFluid.Bg = lambda self, p: adapter.Bg(p)
# DUP_AFTER_ENTRYPOINT                 patched.append("Bg")
# DUP_AFTER_ENTRYPOINT             if not hasattr(EngineFluid, "mu_g"):
# DUP_AFTER_ENTRYPOINT                 EngineFluid.mu_g = lambda self, p: adapter.mu_g(p)
# DUP_AFTER_ENTRYPOINT                 patched.append("mu_g")
# DUP_AFTER_ENTRYPOINT             if not hasattr(EngineFluid, "mu_o"):
# DUP_AFTER_ENTRYPOINT                 EngineFluid.mu_o = lambda self, p: adapter.mu_o(p)
# DUP_AFTER_ENTRYPOINT                 patched.append("mu_o")
# DUP_AFTER_ENTRYPOINT             if patched:
# DUP_AFTER_ENTRYPOINT                 print(f"[PVT patch] Injected Fluid methods: {patched}")
# DUP_AFTER_ENTRYPOINT         except Exception:
# DUP_AFTER_ENTRYPOINT             pass  # safety net
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # --- Public helper used by run_simulation_engine(...) ---
# DUP_AFTER_ENTRYPOINT     def _pvt_from_state(state):
# DUP_AFTER_ENTRYPOINT         adapter = _build_pvt_payload_from_state(state)
# DUP_AFTER_ENTRYPOINT         _monkeypatch_engine_fluid_if_needed(adapter)
# DUP_AFTER_ENTRYPOINT         return adapter
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def eur_gauges(EUR_g_BCF, EUR_o_MMBO):
# DUP_AFTER_ENTRYPOINT         import plotly.graph_objects as go
# DUP_AFTER_ENTRYPOINT         import numpy as np
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def g(val, label, suffix, color, vmax):
# DUP_AFTER_ENTRYPOINT         # --- sanitize inputs ---
# DUP_AFTER_ENTRYPOINT         try:
# DUP_AFTER_ENTRYPOINT             vmax = float(vmax)
# DUP_AFTER_ENTRYPOINT         except Exception:
# DUP_AFTER_ENTRYPOINT             vmax = 1.0
# DUP_AFTER_ENTRYPOINT         if vmax <= 0:
# DUP_AFTER_ENTRYPOINT             vmax = 1.0
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         try:
# DUP_AFTER_ENTRYPOINT             x = float(val)
# DUP_AFTER_ENTRYPOINT         except Exception:
# DUP_AFTER_ENTRYPOINT             x = 0.0
# DUP_AFTER_ENTRYPOINT         if x != x or x == float("inf") or x == float("-inf"):
# DUP_AFTER_ENTRYPOINT             x = 0.0
# DUP_AFTER_ENTRYPOINT         # clamp into [0, vmax] so the gauge never overflows
# DUP_AFTER_ENTRYPOINT         x = max(0.0, min(vmax, x))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         fig = go.Figure(
# DUP_AFTER_ENTRYPOINT             go.Indicator(
# DUP_AFTER_ENTRYPOINT                 mode="gauge+number",
# DUP_AFTER_ENTRYPOINT                 value=x,
# DUP_AFTER_ENTRYPOINT                 number={"suffix": f" {suffix}" if suffix else "", "font": {"size": 44, "color": "#0b2545"}},
# DUP_AFTER_ENTRYPOINT                 title={"text": f"<b>{label}</b>", "font": {"size": 22, "color": "#0b2545"}},
# DUP_AFTER_ENTRYPOINT                 gauge={
# DUP_AFTER_ENTRYPOINT                     "shape": "angular",
# DUP_AFTER_ENTRYPOINT                     "axis": {"range": [0, vmax], "tickwidth": 1.2, "tickcolor": "#0b2545"},
# DUP_AFTER_ENTRYPOINT                     "bar": {"color": color, "thickness": 0.28},
# DUP_AFTER_ENTRYPOINT                     "bgcolor": "white",
# DUP_AFTER_ENTRYPOINT                     "borderwidth": 1,
# DUP_AFTER_ENTRYPOINT                     "bordercolor": "#cfe0ff",
# DUP_AFTER_ENTRYPOINT                     "steps": [
# DUP_AFTER_ENTRYPOINT                         {"range": [0, 0.6 * vmax], "color": "rgba(0,0,0,0.04)"},
# DUP_AFTER_ENTRYPOINT                         {"range": [0.6 * vmax, 0.85 * vmax], "color": "rgba(0,0,0,0.07)"},
# DUP_AFTER_ENTRYPOINT                     ],
# DUP_AFTER_ENTRYPOINT                     "threshold": {
# DUP_AFTER_ENTRYPOINT                         "line": {"color": ("#2CA02C" if str(color).lower() == "#d62728" else "#D62728"), "width": 4},
# DUP_AFTER_ENTRYPOINT                         "thickness": 0.9,
# DUP_AFTER_ENTRYPOINT                         "value": x,
# DUP_AFTER_ENTRYPOINT                     },
# DUP_AFTER_ENTRYPOINT                 },
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT         )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # compact layout so two gauges fit side-by-side
# DUP_AFTER_ENTRYPOINT         fig.update_layout(height=320, margin=dict(l=6, r=6, t=36, b=6), paper_bgcolor="#ffffff")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # shrink inner-axis tick labels (optional; remove if you want bigger ticks)
# DUP_AFTER_ENTRYPOINT         fig.update_traces(gauge={"axis": {"tickfont": {"size": 10}}})
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         return fig
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         gmax = max(1.0, np.ceil(EUR_g_BCF / 5.0) * 5.0)
# DUP_AFTER_ENTRYPOINT         omax = max(0.5, np.ceil(EUR_o_MMBO / 0.5) * 0.5)
# DUP_AFTER_ENTRYPOINT         return g(EUR_g_BCF, "EUR Gas", "BCF", "#d62728", gmax), g(
# DUP_AFTER_ENTRYPOINT             EUR_o_MMBO, "EUR Oil", "MMBO", "#2ca02c", omax
# DUP_AFTER_ENTRYPOINT         )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def semi_log_layout(title, xaxis="Day (log scale)", yaxis="Rate"):
# DUP_AFTER_ENTRYPOINT         return dict(
# DUP_AFTER_ENTRYPOINT             title=f"<b>{title}</b>",
# DUP_AFTER_ENTRYPOINT             template="plotly_white",
# DUP_AFTER_ENTRYPOINT             xaxis=dict(type="log", title=xaxis, showgrid=True, gridcolor="rgba(0,0,0,0.15)"),
# DUP_AFTER_ENTRYPOINT             yaxis=dict(title=yaxis, showgrid=True, gridcolor="rgba(0,0,0,0.15)"),
# DUP_AFTER_ENTRYPOINT             legend=dict(orientation="h"),
# DUP_AFTER_ENTRYPOINT         )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def ensure_3d(arr2d_or_3d):
# DUP_AFTER_ENTRYPOINT         a = np.asarray(arr2d_or_3d)
# DUP_AFTER_ENTRYPOINT         return a[None, ...] if a.ndim == 2 else a
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def get_k_slice(A, k):
# DUP_AFTER_ENTRYPOINT         A3 = ensure_3d(A)
# DUP_AFTER_ENTRYPOINT         nz = A3.shape[0]
# DUP_AFTER_ENTRYPOINT         k = int(np.clip(k, 0, nz - 1))
# DUP_AFTER_ENTRYPOINT         return A3[k, :, :]
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def downsample_3d(A, ds):
# DUP_AFTER_ENTRYPOINT         A3 = ensure_3d(A)
# DUP_AFTER_ENTRYPOINT         ds = max(1, int(ds))
# DUP_AFTER_ENTRYPOINT         return A3[::ds, ::ds, ::ds]
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def parse_dfn_csv(uploaded_file):
# DUP_AFTER_ENTRYPOINT         df = pd.read_csv(uploaded_file)
# DUP_AFTER_ENTRYPOINT         req = ["x0", "y0", "z0", "x1", "y1", "z1"]
# DUP_AFTER_ENTRYPOINT         for c in req:
# DUP_AFTER_ENTRYPOINT             if c not in df.columns:
# DUP_AFTER_ENTRYPOINT                 raise ValueError("DFN CSV must include columns: x0,y0,z0,x1,y1,z1[,k_mult,aperture_ft]")
# DUP_AFTER_ENTRYPOINT         arr = df[req].to_numpy(float)
# DUP_AFTER_ENTRYPOINT         if "k_mult" in df.columns or "aperture_ft" in df.columns:
# DUP_AFTER_ENTRYPOINT             k_mult = df["k_mult"].to_numpy(float) if "k_mult" in df.columns else np.ones(len(df))
# DUP_AFTER_ENTRYPOINT             ap = df["aperture_ft"].to_numpy(float) if "aperture_ft" in df.columns else np.full(len(df), np.nan)
# DUP_AFTER_ENTRYPOINT             arr = np.column_stack([arr, k_mult, ap])
# DUP_AFTER_ENTRYPOINT         return arr
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def gen_auto_dfn_from_stages(nx, ny, nz, dx, dy, dz, L_ft, stage_spacing_ft, n_lats, hf_ft):
# DUP_AFTER_ENTRYPOINT         n_stages = max(1, int(L_ft / max(stage_spacing_ft, 1.0)))
# DUP_AFTER_ENTRYPOINT         Lcells = int(L_ft / max(dx, 1.0))
# DUP_AFTER_ENTRYPOINT         xs = np.linspace(5, max(6, Lcells - 5), n_stages) * dx
# DUP_AFTER_ENTRYPOINT         lat_rows = [ny // 3, 2 * ny // 3] if n_lats >= 2 else [ny // 2]
# DUP_AFTER_ENTRYPOINT         segs = []
# DUP_AFTER_ENTRYPOINT         half_h = hf_ft / 2.0
# DUP_AFTER_ENTRYPOINT         for jr in lat_rows:
# DUP_AFTER_ENTRYPOINT             y_ft = jr * dy
# DUP_AFTER_ENTRYPOINT             for xcell in xs:
# DUP_AFTER_ENTRYPOINT                 x_ft = xcell
# DUP_AFTER_ENTRYPOINT                 z0 = max(0.0, (nz * dz) / 2.0 - half_h)
# DUP_AFTER_ENTRYPOINT                 z1 = min(nz * dz, (nz * dz) / 2.0 + half_h)
# DUP_AFTER_ENTRYPOINT                 segs.append([x_ft, y_ft, z0, x_ft, y_ft, z1])
# DUP_AFTER_ENTRYPOINT         return np.array(segs, float) if segs else None
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def _get_sim_preview():
# DUP_AFTER_ENTRYPOINT         tmp = {k: st.session_state[k] for k in list(defaults.keys()) if k in st.session_state}
# DUP_AFTER_ENTRYPOINT         rng_preview = np.random.default_rng(int(st.session_state.get("rng_seed", 1234)) + 999)
# DUP_AFTER_ENTRYPOINT         try:
# DUP_AFTER_ENTRYPOINT             # --- SAFETY NET FOR PREVIEW ---
# DUP_AFTER_ENTRYPOINT             # We explicitly watch for the classic Arps power failure:
# DUP_AFTER_ENTRYPOINT             # RuntimeWarning: invalid value encountered in power
# DUP_AFTER_ENTRYPOINT             # That happens if (1 + b*D*t) becomes negative and is raised to 1/b.
# DUP_AFTER_ENTRYPOINT             with warnings.catch_warnings(record=True) as w:
# DUP_AFTER_ENTRYPOINT                 warnings.simplefilter("always", RuntimeWarning)
# DUP_AFTER_ENTRYPOINT                 result = fallback_fast_solver(tmp, rng_preview)
# DUP_AFTER_ENTRYPOINT                 # If the solver emitted a "power" invalid warning, attempt one sanitized retry.
# DUP_AFTER_ENTRYPOINT                 bad_power = any(("invalid value encountered in power" in str(x.message)) for x in w)
# DUP_AFTER_ENTRYPOINT                 if bad_power or _looks_nan_like(result):
# DUP_AFTER_ENTRYPOINT                     tmp2 = _sanitize_decline_params(tmp.copy())
# DUP_AFTER_ENTRYPOINT                     result = fallback_fast_solver(tmp2, rng_preview)
# DUP_AFTER_ENTRYPOINT                 # Ensure no NaNs leak into charts
# DUP_AFTER_ENTRYPOINT                 result = _nan_guard_result(result)
# DUP_AFTER_ENTRYPOINT                 return result
# DUP_AFTER_ENTRYPOINT         except Exception as e:
# DUP_AFTER_ENTRYPOINT             st.error("ERROR IN PREVIEW SOLVER (_get_sim_preview):")
# DUP_AFTER_ENTRYPOINT             st.exception(e)
# DUP_AFTER_ENTRYPOINT             # Return a dummy structure to prevent crashing the UI layout
# DUP_AFTER_ENTRYPOINT             return {'t': [0], 'qg': [0], 'qo': [0], 'EUR_g_BCF': 0, 'EUR_o_MMBO': 0}
# DUP_AFTER_ENTRYPOINT     # ------------------------ Arps/decline safety helpers (ANALYTICAL ONLY) ------------------------
# DUP_AFTER_ENTRYPOINT     def _sanitize_decline_params(state_like: dict) -> dict:
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         Human note: Some builds feed different names for hyperbolic parameters.
# DUP_AFTER_ENTRYPOINT         We defensively scan keys for anything that LOOKS like a hyperbolic 'b' or a decline rate,
# DUP_AFTER_ENTRYPOINT         clamp 'b' into a safe (0,1) interval, and force any negative declines positive.
# DUP_AFTER_ENTRYPOINT         This keeps (1 + b*D*t) from ever going negative during power().
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         SAFE_B_MIN, SAFE_B_MAX = 1.0e-6, 0.95
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         def _clip_b(x):
# DUP_AFTER_ENTRYPOINT             try:
# DUP_AFTER_ENTRYPOINT                 xv = float(x)
# DUP_AFTER_ENTRYPOINT                 return min(max(xv, SAFE_B_MIN), SAFE_B_MAX)
# DUP_AFTER_ENTRYPOINT             except Exception:
# DUP_AFTER_ENTRYPOINT                 return x
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         def _abs_decline(x):
# DUP_AFTER_ENTRYPOINT             try:
# DUP_AFTER_ENTRYPOINT                 return abs(float(x))
# DUP_AFTER_ENTRYPOINT             except Exception:
# DUP_AFTER_ENTRYPOINT                 return x
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         for k in list(state_like.keys()):
# DUP_AFTER_ENTRYPOINT             lk = k.lower()
# DUP_AFTER_ENTRYPOINT             # Common patterns we've seen across fast proxies
# DUP_AFTER_ENTRYPOINT             if lk in ("b", "b_oil", "b_gas", "b_liq", "b_decline", "b_hyp", "bhyp", "bexp"):
# DUP_AFTER_ENTRYPOINT                 state_like[k] = _clip_b(state_like[k])
# DUP_AFTER_ENTRYPOINT             # Decline rates frequently show up as D, Di, D1, decline_*, etc.
# DUP_AFTER_ENTRYPOINT             if lk in ("d", "di", "d1", "decline", "decline_rate") or ("decline" in lk):
# DUP_AFTER_ENTRYPOINT                 state_like[k] = _abs_decline(state_like[k])
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Optional: mark that we sanitized to help debug later
# DUP_AFTER_ENTRYPOINT         state_like["__analytical_sanitized__"] = True
# DUP_AFTER_ENTRYPOINT         return state_like
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def _looks_nan_like(result: dict) -> bool:
# DUP_AFTER_ENTRYPOINT         """Return True if any primary arrays contain NaNs or infs."""
# DUP_AFTER_ENTRYPOINT         if not isinstance(result, dict):
# DUP_AFTER_ENTRYPOINT             return True
# DUP_AFTER_ENTRYPOINT         for key in ("t", "qg", "qo"):
# DUP_AFTER_ENTRYPOINT             if key in result and result[key] is not None:
# DUP_AFTER_ENTRYPOINT                 arr = np.asarray(result[key], float)
# DUP_AFTER_ENTRYPOINT                 if not np.all(np.isfinite(arr)):
# DUP_AFTER_ENTRYPOINT                     return True
# DUP_AFTER_ENTRYPOINT         return False
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def _nan_guard_result(result: dict) -> dict:
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         Replace NaNs/infs in rate vectors so the UI can draw safely.
# DUP_AFTER_ENTRYPOINT         We do NOT change EURs here; the engine will re-compute them later in 'Results'.
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         if not isinstance(result, dict):
# DUP_AFTER_ENTRYPOINT             return result
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         out = dict(result)
# DUP_AFTER_ENTRYPOINT         for key in ("t", "qg", "qo", "qw"):
# DUP_AFTER_ENTRYPOINT             if key in out and out[key] is not None:
# DUP_AFTER_ENTRYPOINT                 arr = np.asarray(out[key], float)
# DUP_AFTER_ENTRYPOINT                 arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
# DUP_AFTER_ENTRYPOINT                 out[key] = arr
# DUP_AFTER_ENTRYPOINT         return out
# DUP_AFTER_ENTRYPOINT     # -----------------------------------------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def generate_property_volumes(state):
# DUP_AFTER_ENTRYPOINT         """Generates kx, ky, and phi volumes based on sidebar settings and stores them in session_state."""
# DUP_AFTER_ENTRYPOINT         rng = np.random.default_rng(int(st.session_state.rng_seed))
# DUP_AFTER_ENTRYPOINT         nz, ny, nx = int(state["nz"]), int(state["ny"]), int(state["nx"])
# DUP_AFTER_ENTRYPOINT         # Use the facies style from the state to generate the base 2D maps
# DUP_AFTER_ENTRYPOINT         style = state.get("facies_style", "Continuous (Gaussian)")
# DUP_AFTER_ENTRYPOINT         if "Continuous" in style:
# DUP_AFTER_ENTRYPOINT             kx_mid = 0.05 + state["k_stdev"] * rng.standard_normal((ny, nx))
# DUP_AFTER_ENTRYPOINT             ky_mid = (0.05 / state["anis_kxky"]) + state["k_stdev"] * rng.standard_normal((ny, nx))
# DUP_AFTER_ENTRYPOINT             phi_mid = 0.10 + state["phi_stdev"] * rng.standard_normal((ny, nx))
# DUP_AFTER_ENTRYPOINT         elif "Speckled" in style:
# DUP_AFTER_ENTRYPOINT             # High variance using log-normal distribution for more contrast
# DUP_AFTER_ENTRYPOINT             kx_mid = np.exp(rng.normal(np.log(0.05), 1.5 + state["k_stdev"]*5, (ny, nx)))
# DUP_AFTER_ENTRYPOINT             ky_mid = kx_mid / state["anis_kxky"]
# DUP_AFTER_ENTRYPOINT             phi_mid = np.exp(rng.normal(np.log(0.10), 0.8 + state["phi_stdev"]*3, (ny, nx)))
# DUP_AFTER_ENTRYPOINT         elif "Layered" in style:
# DUP_AFTER_ENTRYPOINT             # Vertical bands (variation primarily in y-direction)
# DUP_AFTER_ENTRYPOINT             base_profile_k = 0.05 + state["k_stdev"] * rng.standard_normal(ny)
# DUP_AFTER_ENTRYPOINT             kx_mid = np.tile(base_profile_k[:, None], (1, nx))
# DUP_AFTER_ENTRYPOINT             ky_mid = kx_mid / state["anis_kxky"]
# DUP_AFTER_ENTRYPOINT             base_profile_phi = 0.10 + state["phi_stdev"] * rng.standard_normal(ny)
# DUP_AFTER_ENTRYPOINT             phi_mid = np.tile(base_profile_phi[:, None], (1, nx))
# DUP_AFTER_ENTRYPOINT         # Apply a slight vertical trend and store in session_state
# DUP_AFTER_ENTRYPOINT         kz_scale = np.linspace(0.95, 1.05, nz)[:, None, None]
# DUP_AFTER_ENTRYPOINT         st.session_state.kx = np.clip(kx_mid[None, ...] * kz_scale, 1e-4, 5.0)
# DUP_AFTER_ENTRYPOINT         st.session_state.ky = np.clip(ky_mid[None, ...] * kz_scale, 1e-4, 5.0)
# DUP_AFTER_ENTRYPOINT         st.session_state.phi = np.clip(phi_mid[None, ...] * kz_scale, 0.01, 0.35)
# DUP_AFTER_ENTRYPOINT         st.success("Successfully generated 3D property volumes!")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def _sanity_bounds_for_play(play_name: str) -> Bounds:
# DUP_AFTER_ENTRYPOINT     
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         Return per-play sanity envelopes for EUR Gas (BCF), EUR Oil (MMBO),
# DUP_AFTER_ENTRYPOINT         and a soft cap on implied EUR GOR (scf/STB). Envelopes are conservative
# DUP_AFTER_ENTRYPOINT         (meant to catch only outliers) and are used for Results-tab warnings.
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         If a play name isn't recognized, safe defaults are returned.
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         s = (play_name or "").lower()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # -------- Global fallback (conservative, oil-window-ish) --------
# DUP_AFTER_ENTRYPOINT         defaults = dict(
# DUP_AFTER_ENTRYPOINT             gas_bcf=(0.3, 5.0),
# DUP_AFTER_ENTRYPOINT             oil_mmbo=(0.2, 2.5),
# DUP_AFTER_ENTRYPOINT             max_eur_gor_scfstb=2200.0,
# DUP_AFTER_ENTRYPOINT         )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # -------- Play-specific envelopes --------
# DUP_AFTER_ENTRYPOINT         # Permian – Midland (Oil)
# DUP_AFTER_ENTRYPOINT         if "permian" in s and "midland" in s:
# DUP_AFTER_ENTRYPOINT             # Covers ~0.8–4.6 BCF gas and ~0.6–2.2 MMBO oil seen in your runs
# DUP_AFTER_ENTRYPOINT             return dict(gas_bcf=(0.8, 4.6), oil_mmbo=(0.6, 2.2), max_eur_gor_scfstb=2200.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Permian – Delaware (Oil/Gas)
# DUP_AFTER_ENTRYPOINT         if "permian" in s and "delaware" in s:
# DUP_AFTER_ENTRYPOINT             return dict(gas_bcf=(1.0, 5.0), oil_mmbo=(0.6, 2.4), max_eur_gor_scfstb=2600.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Eagle Ford – Condensate
# DUP_AFTER_ENTRYPOINT         if "eagle" in s and "ford" in s and "condensate" in s:
# DUP_AFTER_ENTRYPOINT             return dict(gas_bcf=(1.5, 5.0), oil_mmbo=(0.4, 2.5), max_eur_gor_scfstb=3000.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Eagle Ford – Oil Window
# DUP_AFTER_ENTRYPOINT         if "eagle" in s and "ford" in s:
# DUP_AFTER_ENTRYPOINT             # Matches ~3.5–3.7 BCF gas and ~1.6–1.7 MMBO oil
# DUP_AFTER_ENTRYPOINT             return dict(gas_bcf=(0.8, 4.8), oil_mmbo=(0.3, 2.2), max_eur_gor_scfstb=2300.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Bakken / Three Forks (Oil)
# DUP_AFTER_ENTRYPOINT         if "bakken" in s or "three forks" in s:
# DUP_AFTER_ENTRYPOINT             return dict(
# DUP_AFTER_ENTRYPOINT                 gas_bcf=(0.6, 4.6),
# DUP_AFTER_ENTRYPOINT                 oil_mmbo=(0.8, 2.2),
# DUP_AFTER_ENTRYPOINT                 # Raised cap so GOR~2,200 scf/STB does not warn
# DUP_AFTER_ENTRYPOINT                 max_eur_gor_scfstb=2300.0,
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Niobrara / DJ (Oil)
# DUP_AFTER_ENTRYPOINT         if "niobrara" in s or " dj" in s:
# DUP_AFTER_ENTRYPOINT             return dict(gas_bcf=(0.3, 2.5), oil_mmbo=(0.3, 1.8), max_eur_gor_scfstb=1800.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Anadarko – Woodford
# DUP_AFTER_ENTRYPOINT         if "anadarko" in s or "woodford" in s:
# DUP_AFTER_ENTRYPOINT             return dict(gas_bcf=(0.5, 4.0), oil_mmbo=(0.2, 1.5), max_eur_gor_scfstb=3500.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Granite Wash (liquids-rich gas)
# DUP_AFTER_ENTRYPOINT         if "granite wash" in s:
# DUP_AFTER_ENTRYPOINT             return dict(gas_bcf=(0.5, 5.0), oil_mmbo=(0.1, 1.0), max_eur_gor_scfstb=4000.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Tuscaloosa Marine (Oil)
# DUP_AFTER_ENTRYPOINT         if "tuscaloosa" in s:
# DUP_AFTER_ENTRYPOINT             return dict(gas_bcf=(0.3, 3.0), oil_mmbo=(0.3, 2.2), max_eur_gor_scfstb=2200.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Montney (Condensate-Rich)
# DUP_AFTER_ENTRYPOINT         if "montney" in s:
# DUP_AFTER_ENTRYPOINT             return dict(gas_bcf=(0.8, 6.0), oil_mmbo=(0.2, 2.0), max_eur_gor_scfstb=4000.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Duvernay (Liquids)
# DUP_AFTER_ENTRYPOINT         if "duvernay" in s:
# DUP_AFTER_ENTRYPOINT             return dict(gas_bcf=(0.5, 5.0), oil_mmbo=(0.3, 2.0), max_eur_gor_scfstb=3000.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Haynesville (Dry Gas) — widened min gas so your 2.8–4.4 BCF runs don't warn
# DUP_AFTER_ENTRYPOINT         if "haynesville" in s:
# DUP_AFTER_ENTRYPOINT             return dict(gas_bcf=(2.5, 20.0), oil_mmbo=(0.0, 0.3), max_eur_gor_scfstb=10000.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Marcellus (Dry Gas)
# DUP_AFTER_ENTRYPOINT         if "marcellus" in s:
# DUP_AFTER_ENTRYPOINT             return dict(gas_bcf=(2.0, 15.0), oil_mmbo=(0.0, 0.3), max_eur_gor_scfstb=10000.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Barnett (Gas)
# DUP_AFTER_ENTRYPOINT         if "barnett" in s:
# DUP_AFTER_ENTRYPOINT             return dict(gas_bcf=(0.5, 6.0), oil_mmbo=(0.0, 0.3), max_eur_gor_scfstb=8000.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Fayetteville (Gas)
# DUP_AFTER_ENTRYPOINT         if "fayetteville" in s:
# DUP_AFTER_ENTRYPOINT             return dict(gas_bcf=(0.5, 5.0), oil_mmbo=(0.0, 0.3), max_eur_gor_scfstb=8000.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Horn River (Dry Gas)
# DUP_AFTER_ENTRYPOINT         if "horn river" in s:
# DUP_AFTER_ENTRYPOINT             return dict(gas_bcf=(3.0, 15.0), oil_mmbo=(0.0, 0.3), max_eur_gor_scfstb=12000.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Unknown / not listed → safe defaults
# DUP_AFTER_ENTRYPOINT         return defaults
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def run_simulation_engine(state):
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         Run either the Analytical proxy or the full 3D simulator, then:
# DUP_AFTER_ENTRYPOINT           - guard against NaNs/Infs,
# DUP_AFTER_ENTRYPOINT           - compute authoritative cumulative volumes & EURs (correct units),
# DUP_AFTER_ENTRYPOINT           - soft-clamp Analytical results to play bounds for UI realism,
# DUP_AFTER_ENTRYPOINT           - return a single dict 'sim' that downstream tabs use.
# DUP_AFTER_ENTRYPOINT         """
# DUP_AFTER_ENTRYPOINT         import warnings
# DUP_AFTER_ENTRYPOINT         import time
# DUP_AFTER_ENTRYPOINT         import numpy as np
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Local imports are safe here even if you already import them at top-of-file.
# DUP_AFTER_ENTRYPOINT         # If you prefer, you can delete these two lines if they're already imported globally.
# DUP_AFTER_ENTRYPOINT         from engines.fast import fallback_fast_solver
# DUP_AFTER_ENTRYPOINT         from core.full3d import simulate
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         t0 = time.time()
# DUP_AFTER_ENTRYPOINT         chosen_engine = st.session_state.get("engine_type", "")
# DUP_AFTER_ENTRYPOINT         out = None
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         try:
# DUP_AFTER_ENTRYPOINT             if "Analytical" in chosen_engine:
# DUP_AFTER_ENTRYPOINT                 # ---------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT                 # B) Ensure the proxy receives the widget values (pad_ctrl, BHP)
# DUP_AFTER_ENTRYPOINT                 # ---------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT                 state = dict(state)  # work on a copy so we don't mutate caller
# DUP_AFTER_ENTRYPOINT                 state["pad_ctrl"] = str(st.session_state.get("pad_ctrl", "BHP"))
# DUP_AFTER_ENTRYPOINT                 state["pad_bhp_psi"] = float(st.session_state.get("pad_bhp_psi", 2500.0))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                 rng = np.random.default_rng(int(st.session_state.get("rng_seed", 1234)))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                 # ---------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT                 # A) DEBUG: Confirm what the proxy actually receives
# DUP_AFTER_ENTRYPOINT                 # ---------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT                 st.caption(
# DUP_AFTER_ENTRYPOINT                     "DEBUG (Analytical inputs) → "
# DUP_AFTER_ENTRYPOINT                     f"pad_ctrl={state.get('pad_ctrl')}  "
# DUP_AFTER_ENTRYPOINT                     f"pad_bhp_psi={state.get('pad_bhp_psi')}  "
# DUP_AFTER_ENTRYPOINT                     f"pb_psi={state.get('pb_psi')}"
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                 # --- CRASH-PROOF ANALYTICAL CALL PATH ---
# DUP_AFTER_ENTRYPOINT                 with warnings.catch_warnings(record=True) as w:
# DUP_AFTER_ENTRYPOINT                     warnings.simplefilter("always", RuntimeWarning)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                     # Call the fast proxy
# DUP_AFTER_ENTRYPOINT                     out = fallback_fast_solver(state, rng)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                     # Detect classic hyperbolic-power warnings + NaN-ish results; retry sanitized
# DUP_AFTER_ENTRYPOINT                     bad_power = any("invalid value encountered in power" in str(x.message) for x in w)
# DUP_AFTER_ENTRYPOINT                     if bad_power or _looks_nan_like(out):
# DUP_AFTER_ENTRYPOINT                         st.info("Analytical model hit an unstable power term; retrying with safe parameters …")
# DUP_AFTER_ENTRYPOINT                         safe_state = _sanitize_decline_params(state.copy())
# DUP_AFTER_ENTRYPOINT                         out = fallback_fast_solver(safe_state, rng)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                     # Guard the result dict against stray NaNs/Infs
# DUP_AFTER_ENTRYPOINT                     out = _nan_guard_result(out)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             else:
# DUP_AFTER_ENTRYPOINT                 # ---------------------------
# DUP_AFTER_ENTRYPOINT                 # Full 3D implicit simulator
# DUP_AFTER_ENTRYPOINT                 # ---------------------------
# DUP_AFTER_ENTRYPOINT                 inputs = {
# DUP_AFTER_ENTRYPOINT                     "engine": "implicit",
# DUP_AFTER_ENTRYPOINT                     "nx": int(state.get("nx", 20)),
# DUP_AFTER_ENTRYPOINT                     "ny": int(state.get("ny", 20)),
# DUP_AFTER_ENTRYPOINT                     "nz": int(state.get("nz", 5)),
# DUP_AFTER_ENTRYPOINT                     "dx": float(state.get("dx_ft", state.get("dx", 100.0))),
# DUP_AFTER_ENTRYPOINT                     "dy": float(state.get("dy_ft", state.get("dy", 100.0))),
# DUP_AFTER_ENTRYPOINT                     "dz": float(state.get("dz_ft", state.get("dz", 50.0))),
# DUP_AFTER_ENTRYPOINT                     "phi": st.session_state.get("phi"),
# DUP_AFTER_ENTRYPOINT                     "kx_md": st.session_state.get("kx"),
# DUP_AFTER_ENTRYPOINT                     "ky_md": st.session_state.get("ky"),
# DUP_AFTER_ENTRYPOINT                     "p_init_psi": float(state.get("p_init_psi", 5000.0)),
# DUP_AFTER_ENTRYPOINT                     "nw": float(state.get("nw", 2.0)),
# DUP_AFTER_ENTRYPOINT                     "no": float(state.get("no", 2.0)),
# DUP_AFTER_ENTRYPOINT                     "krw_end": float(state.get("krw_end", 0.6)),
# DUP_AFTER_ENTRYPOINT                     "kro_end": float(state.get("kro_end", 0.8)),
# DUP_AFTER_ENTRYPOINT                     "pb_psi": float(state.get("pb_psi", 3000.0)),
# DUP_AFTER_ENTRYPOINT                     "Bo_pb_rb_stb": float(state.get("Bo_pb_rb_stb", 1.2)),
# DUP_AFTER_ENTRYPOINT                     "Rs_pb_scf_stb": float(state.get("Rs_pb_scf_stb", 600.0)),
# DUP_AFTER_ENTRYPOINT                     "mu_o_cp": float(state.get("muo_pb_cp", 1.2)),
# DUP_AFTER_ENTRYPOINT                     "mu_g_cp": float(state.get("mug_pb_cp", 0.02)),
# DUP_AFTER_ENTRYPOINT                     "control": str(state.get("pad_ctrl", "BHP")),
# DUP_AFTER_ENTRYPOINT                     "bhp_psi": float(state.get("pad_bhp_psi", 2500.0)),
# DUP_AFTER_ENTRYPOINT                     "rate_mscfd": float(state.get("pad_rate_mscfd", 0.0)),
# DUP_AFTER_ENTRYPOINT                     "rate_stbd": float(state.get("pad_rate_stbd", 0.0)),
# DUP_AFTER_ENTRYPOINT                     "dt_days": 30.0,
# DUP_AFTER_ENTRYPOINT                     "t_end_days": 30 * 365.25,
# DUP_AFTER_ENTRYPOINT                     "use_gravity": bool(state.get("use_gravity", True)),
# DUP_AFTER_ENTRYPOINT                     "kvkh": 1.0 / float(state.get("anis_kxky", 1.0)),
# DUP_AFTER_ENTRYPOINT                     "geo_alpha": float(state.get("geo_alpha", 0.0)),
# DUP_AFTER_ENTRYPOINT                 }
# DUP_AFTER_ENTRYPOINT                 out = simulate(inputs)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         except Exception as e:
# DUP_AFTER_ENTRYPOINT             st.error(f"FATAL SIMULATOR CRASH in '{chosen_engine}':")
# DUP_AFTER_ENTRYPOINT             st.exception(e)
# DUP_AFTER_ENTRYPOINT             return None
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # If the engine returned nothing, bail gracefully
# DUP_AFTER_ENTRYPOINT         if not isinstance(out, dict):
# DUP_AFTER_ENTRYPOINT             st.error("Engine did not return a result dictionary.")
# DUP_AFTER_ENTRYPOINT             return None
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # ----------------------------------------
# DUP_AFTER_ENTRYPOINT         # Pull arrays and do the final NaN/Inf guard
# DUP_AFTER_ENTRYPOINT         # ----------------------------------------
# DUP_AFTER_ENTRYPOINT         t = out.get("t")
# DUP_AFTER_ENTRYPOINT         qg = out.get("qg")
# DUP_AFTER_ENTRYPOINT         qo = out.get("qo")
# DUP_AFTER_ENTRYPOINT         qw = out.get("qw")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         t = np.nan_to_num(np.asarray(t, float), nan=0.0, posinf=0.0, neginf=0.0) if t is not None else np.array([], dtype=float)
# DUP_AFTER_ENTRYPOINT         qg = None if qg is None else np.nan_to_num(np.asarray(qg, float), nan=0.0, posinf=0.0, neginf=0.0)
# DUP_AFTER_ENTRYPOINT         qo = None if qo is None else np.nan_to_num(np.asarray(qo, float), nan=0.0, posinf=0.0, neginf=0.0)
# DUP_AFTER_ENTRYPOINT         qw = None if qw is None else np.nan_to_num(np.asarray(qw, float), nan=0.0, posinf=0.0, neginf=0.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # ------------------------------------------------------
# DUP_AFTER_ENTRYPOINT         # Authoritative cumulatives & EURs (unit-correct, robust)
# DUP_AFTER_ENTRYPOINT         # ------------------------------------------------------
# DUP_AFTER_ENTRYPOINT         sim = dict(out)  # start with engine output
# DUP_AFTER_ENTRYPOINT         sim.update(_compute_eurs_and_cums(t, qg=qg, qo=qo, qw=qw))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # ------------------------------------------------------
# DUP_AFTER_ENTRYPOINT         # Apply play bounds (Analytical: soft UI clamp only)
# DUP_AFTER_ENTRYPOINT         # ------------------------------------------------------
# DUP_AFTER_ENTRYPOINT         current_play = st.session_state.get("play_name", st.session_state.get("shale_play", ""))
# DUP_AFTER_ENTRYPOINT         engine_name = chosen_engine
# DUP_AFTER_ENTRYPOINT         sim = _apply_play_bounds_to_results(sim, current_play, engine_name)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Bookkeeping
# DUP_AFTER_ENTRYPOINT         sim["_sim_signature"] = _sim_signature_from_state()
# DUP_AFTER_ENTRYPOINT         sim["runtime_s"] = float(time.time() - t0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         return sim
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT   
# DUP_AFTER_ENTRYPOINT         # --- Build sim dict and compute EURs/cumulatives ---
# DUP_AFTER_ENTRYPOINT         sim = dict(out) if isinstance(out, dict) else {}
# DUP_AFTER_ENTRYPOINT         sim["t"], sim["qg"], sim["qo"], sim["qw"] = t, qg, qo, qw
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # 1) Numerically integrate to cum arrays + EURs
# DUP_AFTER_ENTRYPOINT         sim.update(_compute_eurs_and_cums(t, qg=qg, qo=qo, qw=qw))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # 2) Apply play-specific soft bounds (Analytical only)
# DUP_AFTER_ENTRYPOINT         play_name   = st.session_state.get("play_name", st.session_state.get("shale_play", ""))
# DUP_AFTER_ENTRYPOINT         engine_name = chosen_engine
# DUP_AFTER_ENTRYPOINT         sim = _apply_play_bounds_to_results(sim, play_name, engine_name)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         return sim
# DUP_AFTER_ENTRYPOINT     
# DUP_AFTER_ENTRYPOINT         # ... continue with EUR calc and the rest of your function ...
# DUP_AFTER_ENTRYPOINT         
# DUP_AFTER_ENTRYPOINT         eur_cutoff_days = float(st.session_state.get("eur_cutoff_days", 30.0 * 365.25))
# DUP_AFTER_ENTRYPOINT         min_gas_rate_mscfd = float(st.session_state.get("eur_min_rate_gas_mscfd", 100.0))
# DUP_AFTER_ENTRYPOINT         min_oil_rate_stbd = float(st.session_state.get("eur_min_rate_oil_stbd", 30.0))
# DUP_AFTER_ENTRYPOINT     
# DUP_AFTER_ENTRYPOINT         tg, qg2 = _apply_economic_cutoffs(t, qg, cutoff_days=eur_cutoff_days, min_rate=min_gas_rate_mscfd)
# DUP_AFTER_ENTRYPOINT         to, qo2 = _apply_economic_cutoffs(t, qo, cutoff_days=eur_cutoff_days, min_rate=min_oil_rate_stbd)
# DUP_AFTER_ENTRYPOINT         tw, qw2 = _apply_economic_cutoffs(t, out.get("qw"), cutoff_days=eur_cutoff_days, min_rate=0.0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         cum_g_Mscf, cum_o_STB, cum_w_STB = _cum_trapz_days(tg, qg2), _cum_trapz_days(to, qo2), _cum_trapz_days(tw, qw2)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         EUR_g_BCF  = float(cum_g_Mscf[-1]/1e6) if cum_g_Mscf is not None and len(cum_g_Mscf) > 0 else 0.0
# DUP_AFTER_ENTRYPOINT         EUR_o_MMBO = float(cum_o_STB[-1]/1e6)  if cum_o_STB is not None and len(cum_o_STB) > 0 else 0.0
# DUP_AFTER_ENTRYPOINT         EUR_w_MMBL = float(cum_w_STB[-1]/1e6)  if cum_w_STB is not None and len(cum_w_STB) > 0 else 0.0
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         final = {
# DUP_AFTER_ENTRYPOINT             "t": t, "qg": qg, "qo": qo, "qw": out.get("qw"),
# DUP_AFTER_ENTRYPOINT             "cum_g_BCF": (cum_g_Mscf / 1e6) if cum_g_Mscf is not None else None,
# DUP_AFTER_ENTRYPOINT             "cum_o_MMBO": (cum_o_STB / 1e6) if cum_o_STB is not None else None,
# DUP_AFTER_ENTRYPOINT             "cum_w_MMBL": (cum_w_STB / 1e6) if cum_w_STB is not None else None,
# DUP_AFTER_ENTRYPOINT             "EUR_g_BCF": EUR_g_BCF, "EUR_o_MMBO": EUR_o_MMBO, "EUR_w_MMBL": EUR_w_MMBL,
# DUP_AFTER_ENTRYPOINT             "runtime_s": time.time() - t0,
# DUP_AFTER_ENTRYPOINT         }
# DUP_AFTER_ENTRYPOINT     
# DUP_AFTER_ENTRYPOINT         for k in ("p_avg_psi", "pm_mid_psi", "p_initial", "p_final"):
# DUP_AFTER_ENTRYPOINT             if k in out:
# DUP_AFTER_ENTRYPOINT                 final[k] = out[k]
# DUP_AFTER_ENTRYPOINT     
# DUP_AFTER_ENTRYPOINT         if "p_avg_psi" not in final or final["p_avg_psi"] is None:
# DUP_AFTER_ENTRYPOINT             p_initial_grid = final.get("p_initial")
# DUP_AFTER_ENTRYPOINT             p_final_grid = final.get("p_final")
# DUP_AFTER_ENTRYPOINT             if p_initial_grid is not None and p_final_grid is not None and t is not None and len(t) > 1:
# DUP_AFTER_ENTRYPOINT                 p_avg_initial = np.mean(p_initial_grid)
# DUP_AFTER_ENTRYPOINT                 p_avg_final = np.mean(p_final_grid)
# DUP_AFTER_ENTRYPOINT                 p_avg_series = np.linspace(p_avg_initial, p_avg_final, num=len(t))
# DUP_AFTER_ENTRYPOINT                 final["p_avg_psi"] = p_avg_series
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         final["_sim_signature"] = _sim_signature_from_state()
# DUP_AFTER_ENTRYPOINT         return final    
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ------------------------ Engine & Presets (SIDEBAR) ------------------------
# DUP_AFTER_ENTRYPOINT     with st.sidebar:
# DUP_AFTER_ENTRYPOINT         st.markdown("## Simulation Setup")
# DUP_AFTER_ENTRYPOINT     
# DUP_AFTER_ENTRYPOINT         # --- All controls are now correctly inside the sidebar ---
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         with st.expander("Engine & Presets", expanded=True):
# DUP_AFTER_ENTRYPOINT             engine_type_ui = st.selectbox(
# DUP_AFTER_ENTRYPOINT                 "Engine Type",
# DUP_AFTER_ENTRYPOINT                 ENGINE_TYPES,
# DUP_AFTER_ENTRYPOINT                 key="engine_type_ui",
# DUP_AFTER_ENTRYPOINT                 help="Choose the calculation engine. Phase 1a/1b are the developing implicit engines; the analytical model is a fast proxy.",
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             st.session_state["engine_type"] = engine_type_ui
# DUP_AFTER_ENTRYPOINT             model_choice = st.selectbox("Model Type", VALID_MODEL_TYPES, key="sim_mode")
# DUP_AFTER_ENTRYPOINT             st.session_state.fluid_model = (
# DUP_AFTER_ENTRYPOINT                 "black_oil" if "Black Oil" in model_choice else "unconventional"
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         with st.expander("Shale Play Preset", expanded=True):
# DUP_AFTER_ENTRYPOINT             _current_play = st.session_state.get("play_sel", PLAY_LIST[0])
# DUP_AFTER_ENTRYPOINT             try:
# DUP_AFTER_ENTRYPOINT                 _default_idx = PLAY_LIST.index(_current_play)
# DUP_AFTER_ENTRYPOINT             except ValueError:
# DUP_AFTER_ENTRYPOINT                 _default_idx = 0
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             play = st.selectbox(
# DUP_AFTER_ENTRYPOINT                 "Select a Play",
# DUP_AFTER_ENTRYPOINT                 PLAY_LIST,
# DUP_AFTER_ENTRYPOINT                 index=_default_idx,
# DUP_AFTER_ENTRYPOINT                 key="play_sel",
# DUP_AFTER_ENTRYPOINT                 label_visibility="visible", # Use a visible label in the sidebar
# DUP_AFTER_ENTRYPOINT                 on_change=_on_play_change,
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             apply_clicked = st.button("Apply Preset", use_container_width=True, type="primary")
# DUP_AFTER_ENTRYPOINT             if apply_clicked:
# DUP_AFTER_ENTRYPOINT                 payload = defaults.copy()
# DUP_AFTER_ENTRYPOINT                 payload.update(PLAY_PRESETS[st.session_state.play_sel])
# DUP_AFTER_ENTRYPOINT                 if st.session_state.fluid_model == "black_oil":
# DUP_AFTER_ENTRYPOINT                     payload.update(
# DUP_AFTER_ENTRYPOINT                         dict(
# DUP_AFTER_ENTRYPOINT                             Rs_pb_scf_stb=0.0, pb_psi=1.0, Bo_pb_rb_stb=1.00,
# DUP_AFTER_ENTRYPOINT                             p_init_psi=max(3500.0, float(payload.get("p_init_psi", 5200.0))),
# DUP_AFTER_ENTRYPOINT                         )
# DUP_AFTER_ENTRYPOINT                     )
# DUP_AFTER_ENTRYPOINT                 st.session_state.sim = None
# DUP_AFTER_ENTRYPOINT                 st.session_state.apply_preset_payload = payload
# DUP_AFTER_ENTRYPOINT                 _safe_rerun()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         with st.expander("Grid & Heterogeneity", expanded=False):
# DUP_AFTER_ENTRYPOINT             st.markdown("#### Grid (ft)")
# DUP_AFTER_ENTRYPOINT             c1, c2, c3 = st.columns(3)
# DUP_AFTER_ENTRYPOINT             with c1: st.number_input("nx", 1, 500, key="nx")
# DUP_AFTER_ENTRYPOINT             with c2: st.number_input("ny", 1, 500, key="ny")
# DUP_AFTER_ENTRYPOINT             with c3: st.number_input("nz", 1, 200, key="nz")
# DUP_AFTER_ENTRYPOINT             c1, c2, c3 = st.columns(3)
# DUP_AFTER_ENTRYPOINT             with c1: st.number_input("dx", step=1.0, key="dx")
# DUP_AFTER_ENTRYPOINT             with c2: st.number_input("dy", step=1.0, key="dy")
# DUP_AFTER_ENTRYPOINT             with c3: st.number_input("dz", step=1.0, key="dz")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             st.markdown("#### Heterogeneity & Anisotropy")
# DUP_AFTER_ENTRYPOINT             st.selectbox("Facies style", ["Continuous (Gaussian)", "Speckled (high-variance)", "Layered (vertical bands)"], key="facies_style")
# DUP_AFTER_ENTRYPOINT             st.slider("k stdev", 0.0, 0.20, float(st.session_state.k_stdev), 0.01, key="k_stdev", help="Standard deviation for permeability field generation.")
# DUP_AFTER_ENTRYPOINT             st.slider("ϕ stdev", 0.0, 0.20, float(st.session_state.phi_stdev), 0.01, key="phi_stdev", help="Standard deviation for porosity field generation.")
# DUP_AFTER_ENTRYPOINT             st.slider("Anisotropy kx/ky", 0.5, 3.0, float(st.session_state.anis_kxky), 0.05, key="anis_kxky")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         with st.expander("Faults", expanded=False):
# DUP_AFTER_ENTRYPOINT             st.checkbox("Enable fault TMULT", value=bool(st.session_state.use_fault), key="use_fault")
# DUP_AFTER_ENTRYPOINT             fault_plane_choice = st.selectbox("Fault plane", ["i-plane (vertical)", "j-plane (vertical)"], index=0, key="fault_plane")
# DUP_AFTER_ENTRYPOINT             max_idx = int(st.session_state.nx) - 2 if 'i-plane' in fault_plane_choice else int(st.session_state.ny) - 2
# DUP_AFTER_ENTRYPOINT             if st.session_state.fault_index > max_idx: st.session_state.fault_index = max_idx
# DUP_AFTER_ENTRYPOINT             st.number_input("Plane index", 1, max(1, max_idx), key="fault_index")
# DUP_AFTER_ENTRYPOINT             st.number_input("Transmissibility multiplier", value=float(st.session_state.fault_tm), step=0.01, key="fault_tm")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         with st.expander("Pad / Wellbore & Frac", expanded=False):
# DUP_AFTER_ENTRYPOINT             st.number_input("Laterals", 1, 6, int(st.session_state.n_laterals), 1, key="n_laterals")
# DUP_AFTER_ENTRYPOINT             st.number_input("Lateral length (ft)", value=float(st.session_state.L_ft), step=50.0, key="L_ft")
# DUP_AFTER_ENTRYPOINT             st.number_input("Stage spacing (ft)", value=float(st.session_state.stage_spacing_ft), step=5.0, key="stage_spacing_ft")
# DUP_AFTER_ENTRYPOINT             st.number_input("Clusters per stage", 1, 12, int(st.session_state.clusters_per_stage), 1, key="clusters_per_stage")
# DUP_AFTER_ENTRYPOINT             st.number_input("Δp limited-entry (psi)", value=float(st.session_state.dP_LE_psi), step=5.0, key="dP_LE_psi")
# DUP_AFTER_ENTRYPOINT             st.number_input("Wellbore friction factor", value=float(st.session_state.f_fric), format="%.3f", step=0.005, key="f_fric")
# DUP_AFTER_ENTRYPOINT             st.number_input("Wellbore ID (ft)", value=float(st.session_state.wellbore_ID_ft), step=0.01, key="wellbore_ID_ft")
# DUP_AFTER_ENTRYPOINT             st.number_input("Frac half-length xf (ft)", value=float(st.session_state.xf_ft), step=5.0, key="xf_ft")
# DUP_AFTER_ENTRYPOINT             st.number_input("Frac height hf (ft)", value=float(st.session_state.hf_ft), step=5.0, key="hf_ft")
# DUP_AFTER_ENTRYPOINT             st.slider("Pad interference coeff.", 0.00, 0.80, float(st.session_state.pad_interf), 0.01, key="pad_interf")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         with st.expander("Controls & Boundary", expanded=False):
# DUP_AFTER_ENTRYPOINT             st.selectbox("Pad control", ["BHP", "RATE"], key="pad_ctrl")
# DUP_AFTER_ENTRYPOINT             st.number_input("Pad BHP (psi)", value=float(st.session_state.pad_bhp_psi), step=10.0, key="pad_bhp_psi")
# DUP_AFTER_ENTRYPOINT             st.number_input("Pad RATE (Mscf/d)", value=float(st.session_state.pad_rate_mscfd), step=1000.0, key="pad_rate_mscfd")
# DUP_AFTER_ENTRYPOINT             st.selectbox("Outer boundary", ["Infinite-acting", "Constant-p"], key="outer_bc")
# DUP_AFTER_ENTRYPOINT             st.number_input("Boundary pressure (psi)", value=float(st.session_state.p_outer_psi), step=10.0, key="p_outer_psi")
# DUP_AFTER_ENTRYPOINT     
# DUP_AFTER_ENTRYPOINT         with st.expander("DFN (Discrete Fracture Network)", expanded=False):
# DUP_AFTER_ENTRYPOINT             st.checkbox("Use DFN-driven sink in solver", value=bool(st.session_state.use_dfn_sink), key="use_dfn_sink")
# DUP_AFTER_ENTRYPOINT             st.checkbox("Auto-generate DFN from stages", value=bool(st.session_state.use_auto_dfn), key="use_auto_dfn")
# DUP_AFTER_ENTRYPOINT             st.number_input("DFN influence radius (ft)", value=float(st.session_state.dfn_radius_ft), step=5.0, key="dfn_radius_ft")
# DUP_AFTER_ENTRYPOINT             st.number_input("DFN sink strength (psi)", value=float(st.session_state.dfn_strength_psi), step=10.0, key="dfn_strength_psi")
# DUP_AFTER_ENTRYPOINT             dfn_up = st.file_uploader("Upload DFN CSV", type=["csv"], key="dfn_csv")
# DUP_AFTER_ENTRYPOINT             c1, c2 = st.columns(2)
# DUP_AFTER_ENTRYPOINT             with c1:
# DUP_AFTER_ENTRYPOINT                 if st.button("Load DFN"):
# DUP_AFTER_ENTRYPOINT                     if dfn_up:
# DUP_AFTER_ENTRYPOINT                         st.session_state.dfn_segments = parse_dfn_csv(dfn_up)
# DUP_AFTER_ENTRYPOINT                         st.success(f"Loaded {len(st.session_state.dfn_segments)} segments")
# DUP_AFTER_ENTRYPOINT                     else: st.warning("Please choose a CSV")
# DUP_AFTER_ENTRYPOINT             with c2:
# DUP_AFTER_ENTRYPOINT                 if st.button("Generate DFN"):
# DUP_AFTER_ENTRYPOINT                     segs = gen_auto_dfn_from_stages(
# DUP_AFTER_ENTRYPOINT                         int(st.session_state.nx), int(st.session_state.ny), int(st.session_state.nz),
# DUP_AFTER_ENTRYPOINT                         float(st.session_state.dx), float(st.session_state.dy), float(st.session_state.dz),
# DUP_AFTER_ENTRYPOINT                         float(st.session_state.L_ft), float(st.session_state.stage_spacing_ft),
# DUP_AFTER_ENTRYPOINT                         int(st.session_state.n_laterals), float(st.session_state.hf_ft),
# DUP_AFTER_ENTRYPOINT                     )
# DUP_AFTER_ENTRYPOINT                     st.session_state.dfn_segments = segs
# DUP_AFTER_ENTRYPOINT                     st.success(f"Generated {0 if segs is None else len(segs)} segments")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         with st.expander("Solver & Profiling", expanded=False):
# DUP_AFTER_ENTRYPOINT             st.number_input("Newton tolerance", value=float(st.session_state.newton_tol), format="%.1e", key="newton_tol")
# DUP_AFTER_ENTRYPOINT             st.number_input("Transmissibility tolerance", value=float(st.session_state.trans_tol), format="%.1e", key="trans_tol")
# DUP_AFTER_ENTRYPOINT             st.number_input("Max Newton iterations", value=int(st.session_state.max_newton), step=1, key="max_newton")
# DUP_AFTER_ENTRYPOINT             st.number_input("Max linear solver iterations", value=int(st.session_state.max_lin), step=10, key="max_lin")
# DUP_AFTER_ENTRYPOINT             st.number_input("Threads (0 for auto)", value=int(st.session_state.threads), step=1, key="threads")
# DUP_AFTER_ENTRYPOINT             st.checkbox("Use OpenMP", value=bool(st.session_state.use_omp), key="use_omp")
# DUP_AFTER_ENTRYPOINT             st.checkbox("Use Intel MKL", value=bool(st.session_state.use_mkl), key="use_mkl")
# DUP_AFTER_ENTRYPOINT             st.checkbox("Use PyAMG solver", value=bool(st.session_state.use_pyamg), key="use_pyamg")
# DUP_AFTER_ENTRYPOINT             st.checkbox("Use NVIDIA cuSPARSE", value=bool(st.session_state.use_cusparse), key="use_cusparse")
# DUP_AFTER_ENTRYPOINT         
# DUP_AFTER_ENTRYPOINT         st.markdown("---")
# DUP_AFTER_ENTRYPOINT         st.markdown("##### Developed by:")
# DUP_AFTER_ENTRYPOINT         st.markdown("##### Omar Nur, Petroleum Engineer")
# DUP_AFTER_ENTRYPOINT         st.markdown("---")
# DUP_AFTER_ENTRYPOINT     #### Part 3: Main Application UI - Primary Workflow Tabs ####
# DUP_AFTER_ENTRYPOINT     # --- Tab list ---
# DUP_AFTER_ENTRYPOINT     tab_names = [
# DUP_AFTER_ENTRYPOINT         "Setup Preview",
# DUP_AFTER_ENTRYPOINT         "Generate 3D property volumes",
# DUP_AFTER_ENTRYPOINT         "PVT (Black-Oil)",
# DUP_AFTER_ENTRYPOINT         "MSW Wellbore",
# DUP_AFTER_ENTRYPOINT         "RTA",
# DUP_AFTER_ENTRYPOINT         "Results",
# DUP_AFTER_ENTRYPOINT         "3D Viewer",
# DUP_AFTER_ENTRYPOINT         "Slice Viewer",
# DUP_AFTER_ENTRYPOINT         "QA / Material Balance",
# DUP_AFTER_ENTRYPOINT         "Economics",
# DUP_AFTER_ENTRYPOINT         "EUR vs Lateral Length",
# DUP_AFTER_ENTRYPOINT         "Field Match (CSV)",
# DUP_AFTER_ENTRYPOINT         "Automated Match", # <-- NEW TAB
# DUP_AFTER_ENTRYPOINT         "Uncertainty & Monte Carlo",
# DUP_AFTER_ENTRYPOINT         "Well Placement Optimization",
# DUP_AFTER_ENTRYPOINT         "User's Manual",
# DUP_AFTER_ENTRYPOINT         "Solver & Profiling",
# DUP_AFTER_ENTRYPOINT         "DFN Viewer",
# DUP_AFTER_ENTRYPOINT     ]
# DUP_AFTER_ENTRYPOINT     st.write(
# DUP_AFTER_ENTRYPOINT         '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} '
# DUP_AFTER_ENTRYPOINT         '.stRadio > label {display:none;} '
# DUP_AFTER_ENTRYPOINT         'div.row-widget.stRadio > div > div {border: 1px solid #ccc; padding: 6px 12px; border-radius: 4px; margin: 2px; background-color: #f0f2f6;} '
# DUP_AFTER_ENTRYPOINT         'div.row-widget.stRadio > div > div[aria-checked="true"] {background-color: #e57373; color: white; border-color: #d32f2f;}</style>',
# DUP_AFTER_ENTRYPOINT         unsafe_allow_html=True,
# DUP_AFTER_ENTRYPOINT     )
# DUP_AFTER_ENTRYPOINT     """  # START: disable legacy nav block
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ------------------------ TAB CONTENT DEFINITIONS ------------------------
# DUP_AFTER_ENTRYPOINT     if selected_tab == "Setup Preview":
# DUP_AFTER_ENTRYPOINT         st.header("Setup Preview")
# DUP_AFTER_ENTRYPOINT         c1, c2 = st.columns([1, 1])
# DUP_AFTER_ENTRYPOINT         # ----- LEFT COLUMN -----
# DUP_AFTER_ENTRYPOINT         with c1:
# DUP_AFTER_ENTRYPOINT             st.markdown("#### Grid & Rock Summary")
# DUP_AFTER_ENTRYPOINT             grid_data = {
# DUP_AFTER_ENTRYPOINT                 "Parameter": [
# DUP_AFTER_ENTRYPOINT                     "Grid Dimensions (nx, ny, nz)",
# DUP_AFTER_ENTRYPOINT                     "Cell Size (dx, dy, dz) (ft)",
# DUP_AFTER_ENTRYPOINT                     "Total Volume (MM-ft³)",
# DUP_AFTER_ENTRYPOINT                     "Facies Style",
# DUP_AFTER_ENTRYPOINT                     "Permeability Anisotropy (kx/ky)",
# DUP_AFTER_ENTRYPOINT                 ],
# DUP_AFTER_ENTRYPOINT                 "Value": [
# DUP_AFTER_ENTRYPOINT                     f"{state['nx']} x {state['ny']} x {state['nz']}",
# DUP_AFTER_ENTRYPOINT                     f"{state['dx']} x {state['dy']} x {state['dz']}",
# DUP_AFTER_ENTRYPOINT                     f"{state['nx']*state['ny']*state['nz']*state['dx']*state['dy']*state['dz']/1e6:.1f}",
# DUP_AFTER_ENTRYPOINT                     state['facies_style'],
# DUP_AFTER_ENTRYPOINT                     f"{state['anis_kxky']:.2f}",
# DUP_AFTER_ENTRYPOINT                 ],
# DUP_AFTER_ENTRYPOINT             }
# DUP_AFTER_ENTRYPOINT             st.table(pd.DataFrame(grid_data))
# DUP_AFTER_ENTRYPOINT             with st.expander("Click for details"):
# DUP_AFTER_ENTRYPOINT                 st.markdown(
# DUP_AFTER_ENTRYPOINT                     "- **Grid Dimensions**: The number of cells in the X, Y, and Z directions.\n"
# DUP_AFTER_ENTRYPOINT                     "- **Cell Size**: The physical size of each grid cell in feet.\n"
# DUP_AFTER_ENTRYPOINT                     "- **Total Volume**: The total bulk volume of the reservoir model.\n"
# DUP_AFTER_ENTRYPOINT                     "- **Facies Style**: The method used to generate geological heterogeneity.\n"
# DUP_AFTER_ENTRYPOINT                     "- **Anisotropy**: The ratio of permeability in X (kx) to Y (ky)."
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT             with st.expander("Preset sanity check (debug)"):
# DUP_AFTER_ENTRYPOINT                 st.write({
# DUP_AFTER_ENTRYPOINT                     "Play selected": st.session_state.get("play_sel"),
# DUP_AFTER_ENTRYPOINT                     "Model Type (sim_mode)": st.session_state.get("sim_mode"),
# DUP_AFTER_ENTRYPOINT                     "fluid_model": st.session_state.get("fluid_model"),
# DUP_AFTER_ENTRYPOINT                     "Engine Type": st.session_state.get("engine_type"),
# DUP_AFTER_ENTRYPOINT                     "L_ft": state.get("L_ft"),
# DUP_AFTER_ENTRYPOINT                     "stage_spacing_ft": state.get("stage_spacing_ft"),
# DUP_AFTER_ENTRYPOINT                     "xf_ft": state.get("xf_ft"),
# DUP_AFTER_ENTRYPOINT                     "hf_ft": state.get("hf_ft"),
# DUP_AFTER_ENTRYPOINT                     "pb_psi": state.get("pb_psi"),
# DUP_AFTER_ENTRYPOINT                     "Rs_pb_scf_stb": state.get("Rs_pb_scf_stb"),
# DUP_AFTER_ENTRYPOINT                     "Bo_pb_rb_stb": state.get("Bo_pb_rb_stb"),
# DUP_AFTER_ENTRYPOINT                     "p_init_psi": state.get("p_init_psi"),
# DUP_AFTER_ENTRYPOINT                 })
# DUP_AFTER_ENTRYPOINT             st.markdown("#### Well & Frac Summary")
# DUP_AFTER_ENTRYPOINT             well_data = {
# DUP_AFTER_ENTRYPOINT                 "Parameter": [
# DUP_AFTER_ENTRYPOINT                     "Laterals",
# DUP_AFTER_ENTRYPOINT                     "Lateral Length (ft)",
# DUP_AFTER_ENTRYPOINT                     "Frac Half-length (ft)",
# DUP_AFTER_ENTRYPOINT                     "Frac Height (ft)",
# DUP_AFTER_ENTRYPOINT                     "Stages",
# DUP_AFTER_ENTRYPOINT                     "Clusters/Stage",
# DUP_AFTER_ENTRYPOINT                 ],
# DUP_AFTER_ENTRYPOINT                 "Value": [
# DUP_AFTER_ENTRYPOINT                     state['n_laterals'],
# DUP_AFTER_ENTRYPOINT                     state['L_ft'],
# DUP_AFTER_ENTRYPOINT                     state['xf_ft'],
# DUP_AFTER_ENTRYPOINT                     state['hf_ft'],
# DUP_AFTER_ENTRYPOINT                     int(state['L_ft'] / state['stage_spacing_ft']),
# DUP_AFTER_ENTRYPOINT                     state['clusters_per_stage'],
# DUP_AFTER_ENTRYPOINT                 ],
# DUP_AFTER_ENTRYPOINT             }
# DUP_AFTER_ENTRYPOINT             st.table(pd.DataFrame(well_data))
# DUP_AFTER_ENTRYPOINT             with st.expander("Click for details"):
# DUP_AFTER_ENTRYPOINT                 st.markdown(
# DUP_AFTER_ENTRYPOINT                     "- **Laterals**: Number of horizontal wells in the pad.\n"
# DUP_AFTER_ENTRYPOINT                     "- **Lateral Length**: Length of each horizontal wellbore.\n"
# DUP_AFTER_ENTRYPOINT                     "- **Frac Half-length (xf)**: Distance a hydraulic fracture extends from the wellbore.\n"
# DUP_AFTER_ENTRYPOINT                     "- **Frac Height (hf)**: Vertical extent of the hydraulic fractures.\n"
# DUP_AFTER_ENTRYPOINT                     "- **Stages**: Number of fracturing treatments.\n"
# DUP_AFTER_ENTRYPOINT                     "- **Clusters/Stage**: Perforation clusters within each stage."
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT         # ----- RIGHT COLUMN -----
# DUP_AFTER_ENTRYPOINT         with c2:
# DUP_AFTER_ENTRYPOINT             st.markdown("#### Top-Down Schematic")
# DUP_AFTER_ENTRYPOINT             fig = go.Figure()
# DUP_AFTER_ENTRYPOINT             nx, ny, dx, dy = state['nx'], state['ny'], state['dx'], state['dy']
# DUP_AFTER_ENTRYPOINT             L_ft, xf_ft, ss_ft, n_lats = state['L_ft'], state['xf_ft'], state['stage_spacing_ft'], state['n_laterals']
# DUP_AFTER_ENTRYPOINT             fig.add_shape(
# DUP_AFTER_ENTRYPOINT                 type="rect",
# DUP_AFTER_ENTRYPOINT                 x0=0,
# DUP_AFTER_ENTRYPOINT                 y0=0,
# DUP_AFTER_ENTRYPOINT                 x1=nx * dx,
# DUP_AFTER_ENTRYPOINT                 y1=ny * dy,
# DUP_AFTER_ENTRYPOINT                 line=dict(color="RoyalBlue"),
# DUP_AFTER_ENTRYPOINT                 fillcolor="lightskyblue",
# DUP_AFTER_ENTRYPOINT                 opacity=0.3,
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             lat_rows_y = [ny * dy / 3, 2 * ny * dy / 3] if n_lats >= 2 else [ny * dy / 2]
# DUP_AFTER_ENTRYPOINT             n_stages = max(1, int(L_ft / max(ss_ft, 1.0)))
# DUP_AFTER_ENTRYPOINT             for i, y_lat in enumerate(lat_rows_y):
# DUP_AFTER_ENTRYPOINT                 fig.add_trace(go.Scatter(
# DUP_AFTER_ENTRYPOINT                     x=[0, L_ft],
# DUP_AFTER_ENTRYPOINT                     y=[y_lat, y_lat],
# DUP_AFTER_ENTRYPOINT                     mode='lines',
# DUP_AFTER_ENTRYPOINT                     line=dict(color='black', width=3),
# DUP_AFTER_ENTRYPOINT                     name='Lateral',
# DUP_AFTER_ENTRYPOINT                     showlegend=(i == 0),
# DUP_AFTER_ENTRYPOINT                 ))
# DUP_AFTER_ENTRYPOINT                 for j in range(n_stages):
# DUP_AFTER_ENTRYPOINT                     x_stage = (j + 0.5) * ss_ft
# DUP_AFTER_ENTRYPOINT                     if x_stage > L_ft:
# DUP_AFTER_ENTRYPOINT                         continue
# DUP_AFTER_ENTRYPOINT                     fig.add_trace(go.Scatter(
# DUP_AFTER_ENTRYPOINT                         x=[x_stage, x_stage],
# DUP_AFTER_ENTRYPOINT                         y=[y_lat - xf_ft, y_lat + xf_ft],
# DUP_AFTER_ENTRYPOINT                         mode='lines',
# DUP_AFTER_ENTRYPOINT                         line=dict(color='red', width=2),
# DUP_AFTER_ENTRYPOINT                         name='Frac',
# DUP_AFTER_ENTRYPOINT                         showlegend=(i == 0 and j == 0),
# DUP_AFTER_ENTRYPOINT                     ))
# DUP_AFTER_ENTRYPOINT             fig.update_layout(
# DUP_AFTER_ENTRYPOINT                 title="<b>Well and Fracture Geometry</b>",
# DUP_AFTER_ENTRYPOINT                 xaxis_title="X (ft)",
# DUP_AFTER_ENTRYPOINT                 yaxis_title="Y (ft)",
# DUP_AFTER_ENTRYPOINT                 yaxis_range=[-0.1 * ny * dy, 1.1 * ny * dy],
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             fig.update_yaxes(scaleanchor="x", scaleratio=1)
# DUP_AFTER_ENTRYPOINT             st.plotly_chart(fig, use_container_width=True, theme="streamlit")
# DUP_AFTER_ENTRYPOINT             with st.expander("Click for details"):
# DUP_AFTER_ENTRYPOINT                 st.markdown(
# DUP_AFTER_ENTRYPOINT                     "Bird's-eye view of the simulation model:\n"
# DUP_AFTER_ENTRYPOINT                     "- **Light blue** = reservoir boundary\n"
# DUP_AFTER_ENTRYPOINT                     "- **Black** = horizontal well laterals\n"
# DUP_AFTER_ENTRYPOINT                     "- **Red** = hydraulic fractures"
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT         st.markdown("---")
# DUP_AFTER_ENTRYPOINT         st.markdown("### Production Forecast Preview (Analytical Model)")
# DUP_AFTER_ENTRYPOINT         preview = _get_sim_preview()
# DUP_AFTER_ENTRYPOINT         p_c1, p_c2 = st.columns(2)
# DUP_AFTER_ENTRYPOINT         with p_c1:
# DUP_AFTER_ENTRYPOINT             fig_g = go.Figure(go.Scatter(x=preview['t'], y=preview['qg'], name="Gas Rate", line=dict(color="#d62728")))
# DUP_AFTER_ENTRYPOINT             fig_g.update_layout(**semi_log_layout("Gas Production Preview", yaxis="Gas Rate (Mscf/d)"))
# DUP_AFTER_ENTRYPOINT             st.plotly_chart(fig_g, use_container_width=True, theme="streamlit")
# DUP_AFTER_ENTRYPOINT         with p_c2:
# DUP_AFTER_ENTRYPOINT             fig_o = go.Figure(go.Scatter(x=preview['t'], y=preview['qo'], name="Oil Rate", line=dict(color="#2ca02c")))
# DUP_AFTER_ENTRYPOINT             fig_o.update_layout(**semi_log_layout("Oil Production Preview", yaxis="Oil Rate (STB/d)"))
# DUP_AFTER_ENTRYPOINT             st.plotly_chart(fig_o, use_container_width=True, theme="streamlit")
# DUP_AFTER_ENTRYPOINT         with st.expander("Click for details"):
# DUP_AFTER_ENTRYPOINT             st.markdown("These charts use a simplified analytical model for quick iteration.")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "Control Panel":
# DUP_AFTER_ENTRYPOINT         st.header("Control Panel")
# DUP_AFTER_ENTRYPOINT         c1, c2 = st.columns(2)
# DUP_AFTER_ENTRYPOINT         with c1:
# DUP_AFTER_ENTRYPOINT             st.selectbox(
# DUP_AFTER_ENTRYPOINT                 "Well control",
# DUP_AFTER_ENTRYPOINT                 ["BHP", "RATE_GAS_MSCFD", "RATE_OIL_STBD", "RATE_LIQ_STBD"],
# DUP_AFTER_ENTRYPOINT                 index=["BHP", "RATE_GAS_MSCFD", "RATE_OIL_STBD", "RATE_LIQ_STBD"].index(st.session_state.get("control", "BHP")),
# DUP_AFTER_ENTRYPOINT                 key="control",
# DUP_AFTER_ENTRYPOINT                 help="Choose BHP control or a rate target.",
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             if st.session_state.control == "BHP":
# DUP_AFTER_ENTRYPOINT                 st.number_input("BHP (psi)", 500.0, 15000.0, float(st.session_state.get("bhp_psi", 2500.0)), 50.0, key="pad_bhp_psi")
# DUP_AFTER_ENTRYPOINT             elif st.session_state.control == "RATE_GAS_MSCFD":
# DUP_AFTER_ENTRYPOINT                 st.number_input("Gas rate (Mscf/d)", 0.0, 500000.0, float(st.session_state.get("rate_mscfd", 5000.0)), 100.0, key="pad_rate_mscfd")
# DUP_AFTER_ENTRYPOINT             elif st.session_state.control == "RATE_OIL_STBD":
# DUP_AFTER_ENTRYPOINT                 st.number_input("Oil rate (STB/d)", 0.0, 20000.0, float(st.session_state.get("rate_stbd", 800.0)), 10.0, key="pad_rate_stbd")
# DUP_AFTER_ENTRYPOINT             elif st.session_state.control == "RATE_LIQ_STBD":
# DUP_AFTER_ENTRYPOINT                 st.number_input("Liquid rate (STB/d)", 0.0, 40000.0, float(st.session_state.get("rate_stbd", 1200.0)), 10.0, key="pad_rate_stbd")
# DUP_AFTER_ENTRYPOINT         with c2:
# DUP_AFTER_ENTRYPOINT             st.checkbox("Use gravity", bool(st.session_state.get("use_gravity", True)), key="use_gravity")
# DUP_AFTER_ENTRYPOINT             st.number_input("kv/kh", 0.01, 1.0, float(st.session_state.get("kvkh", 0.10)), 0.01, "%.2f", key="kvkh")
# DUP_AFTER_ENTRYPOINT             st.number_input("Geomech α (1/psi)", 0.0, 1e-3, float(st.session_state.get("geo_alpha", 0.0)), 1e-5, "%.5f", key="geo_alpha")
# DUP_AFTER_ENTRYPOINT         st.markdown("#### Control Summary")
# DUP_AFTER_ENTRYPOINT         summary = {
# DUP_AFTER_ENTRYPOINT             "Control": st.session_state.get("pad_ctrl"),
# DUP_AFTER_ENTRYPOINT             "BHP (psi)": st.session_state.get("pad_bhp_psi"),
# DUP_AFTER_ENTRYPOINT             "Gas rate (Mscf/d)": st.session_state.get("pad_rate_mscfd"),
# DUP_AFTER_ENTRYPOINT             "Oil/Liq rate (STB/d)": st.session_state.get("pad_rate_stbd"),
# DUP_AFTER_ENTRYPOINT             "Use gravity": st.session_state.get("use_gravity"),
# DUP_AFTER_ENTRYPOINT             "kv/kh": st.session_state.get("kvkh"),
# DUP_AFTER_ENTRYPOINT             "Geomech α (1/psi)": st.session_state.get("geo_alpha"),
# DUP_AFTER_ENTRYPOINT         }
# DUP_AFTER_ENTRYPOINT         st.write(summary)
# DUP_AFTER_ENTRYPOINT     
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "Generate 3D property volumes":
# DUP_AFTER_ENTRYPOINT         st.header("Generate 3D Property Volumes (kx, ky, ϕ)")
# DUP_AFTER_ENTRYPOINT         st.info("Use this tab to (re)generate φ/k grids based on sidebar parameters.")
# DUP_AFTER_ENTRYPOINT     
# DUP_AFTER_ENTRYPOINT         if st.button("Generate New Property Volumes", use_container_width=True, type="primary"):
# DUP_AFTER_ENTRYPOINT             generate_property_volumes(state)
# DUP_AFTER_ENTRYPOINT         
# DUP_AFTER_ENTRYPOINT         st.markdown("---")
# DUP_AFTER_ENTRYPOINT     
# DUP_AFTER_ENTRYPOINT         if st.session_state.get('kx') is not None:
# DUP_AFTER_ENTRYPOINT             st.markdown("### Mid-Layer Property Maps")
# DUP_AFTER_ENTRYPOINT             kx_display = get_k_slice(st.session_state.kx, state['nz'] // 2)
# DUP_AFTER_ENTRYPOINT             ky_display = get_k_slice(st.session_state.ky, state['nz'] // 2)
# DUP_AFTER_ENTRYPOINT             phi_display = get_k_slice(st.session_state.phi, state['nz'] // 2)
# DUP_AFTER_ENTRYPOINT         
# DUP_AFTER_ENTRYPOINT             c1, c2 = st.columns(2)
# DUP_AFTER_ENTRYPOINT             with c1:
# DUP_AFTER_ENTRYPOINT                 st.plotly_chart(px.imshow(kx_display, origin="lower", color_continuous_scale="Viridis", labels=dict(color="mD"), title="<b>kx — mid-layer (mD)</b>"), use_container_width=True)
# DUP_AFTER_ENTRYPOINT             with c2:
# DUP_AFTER_ENTRYPOINT                 st.plotly_chart(px.imshow(ky_display, origin="lower", color_continuous_scale="Cividis", labels=dict(color="mD"), title="<b>ky — mid-layer (mD)</b>"), use_container_width=True)
# DUP_AFTER_ENTRYPOINT             
# DUP_AFTER_ENTRYPOINT             st.plotly_chart(px.imshow(phi_display, origin="lower", color_continuous_scale="Magma", labels=dict(color="ϕ"), title="<b>Porosity ϕ — mid-layer (fraction)</b>"), use_container_width=True)
# DUP_AFTER_ENTRYPOINT         else:
# DUP_AFTER_ENTRYPOINT             st.info("Click the button above to generate initial property volumes.")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "PVT (Black-Oil)":
# DUP_AFTER_ENTRYPOINT         st.header("PVT (Black-Oil) Analysis")
# DUP_AFTER_ENTRYPOINT         P = np.linspace(max(1000, state["p_min_bhp_psi"]), max(2000, state["p_init_psi"] + 1000), 120)
# DUP_AFTER_ENTRYPOINT         Rs, Bo, Bg, mug = (Rs_of_p(P, state["pb_psi"], state["Rs_pb_scf_stb"]), Bo_of_p(P, state["pb_psi"], state["Bo_pb_rb_stb"]), Bg_of_p(P), mu_g_of_p(P, state["pb_psi"], state["mug_pb_cp"]))
# DUP_AFTER_ENTRYPOINT         f1 = go.Figure(go.Scatter(x=P, y=Rs, line=dict(color="firebrick", width=3)))
# DUP_AFTER_ENTRYPOINT         f1.add_vline(x=state["pb_psi"], line_dash="dash", line_width=2, annotation_text="Bubble Point")
# DUP_AFTER_ENTRYPOINT         f1.update_layout(template="plotly_white", title="<b>Solution GOR Rs vs Pressure</b>", xaxis_title="Pressure (psi)", yaxis_title="Rs (scf/STB)")
# DUP_AFTER_ENTRYPOINT         st.plotly_chart(f1, use_container_width=True)
# DUP_AFTER_ENTRYPOINT         f2 = go.Figure(go.Scatter(x=P, y=Bo, line=dict(color="seagreen", width=3)))
# DUP_AFTER_ENTRYPOINT         f2.add_vline(x=state["pb_psi"], line_dash="dash", line_width=2, annotation_text="Bubble Point")
# DUP_AFTER_ENTRYPOINT         f2.update_layout(template="plotly_white", title="<b>Oil FVF Bo vs Pressure</b>", xaxis_title="Pressure (psi)", yaxis_title="Bo (rb/STB)")
# DUP_AFTER_ENTRYPOINT         st.plotly_chart(f2, use_container_width=True)
# DUP_AFTER_ENTRYPOINT         f3 = go.Figure(go.Scatter(x=P, y=Bg, line=dict(color="steelblue", width=3)))
# DUP_AFTER_ENTRYPOINT         f3.update_layout(template="plotly_white", title="<b>Gas FVF Bg vs Pressure</b>", xaxis_title="Pressure (psi)", yaxis_title="Bg (rb/scf)")
# DUP_AFTER_ENTRYPOINT         st.plotly_chart(f3, use_container_width=True)
# DUP_AFTER_ENTRYPOINT         f4 = go.Figure(go.Scatter(x=P, y=mug, line=dict(color="mediumpurple", width=3)))
# DUP_AFTER_ENTRYPOINT         f4.update_layout(template="plotly_white", title="<b>Gas viscosity μg vs Pressure</b>", xaxis_title="Pressure (psi)", yaxis_title="μg (cP)")
# DUP_AFTER_ENTRYPOINT         st.plotly_chart(f4, use_container_width=True)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "MSW Wellbore":
# DUP_AFTER_ENTRYPOINT         st.header("MSW Wellbore Physics — Heel–Toe & Limited-Entry")
# DUP_AFTER_ENTRYPOINT         try:
# DUP_AFTER_ENTRYPOINT             L_ft, ss_ft = float(state['L_ft']), float(state['stage_spacing_ft'])
# DUP_AFTER_ENTRYPOINT             n_stages = max(1, int(L_ft / ss_ft))
# DUP_AFTER_ENTRYPOINT             well_id_ft, f_fric, dP_le = float(state['wellbore_ID_ft']), float(state['f_fric']), float(state['dP_LE_psi'])
# DUP_AFTER_ENTRYPOINT             p_bhp, p_res = float(state['pad_bhp_psi']), float(state['p_init_psi'])
# DUP_AFTER_ENTRYPOINT             q_oil_total_stbd = _get_sim_preview()['qo'][0]
# DUP_AFTER_ENTRYPOINT             q_dist = np.ones(n_stages) / n_stages
# DUP_AFTER_ENTRYPOINT             for _ in range(5):
# DUP_AFTER_ENTRYPOINT                 q_per_stage_bpd, p_wellbore_at_stage = q_dist * q_oil_total_stbd, np.zeros(n_stages)
# DUP_AFTER_ENTRYPOINT                 p_current, flow_rate_bpd = p_bhp, q_oil_total_stbd
# DUP_AFTER_ENTRYPOINT                 for i in range(n_stages):
# DUP_AFTER_ENTRYPOINT                     p_wellbore_at_stage[i] = p_current
# DUP_AFTER_ENTRYPOINT                     v_fps = (flow_rate_bpd * 5.615 / (24*3600)) / (np.pi * (well_id_ft/2)**2)
# DUP_AFTER_ENTRYPOINT                     p_current += (2 * f_fric * 50.0 * v_fps**2 * ss_ft / well_id_ft) / 144.0
# DUP_AFTER_ENTRYPOINT                     flow_rate_bpd -= q_per_stage_bpd[i]
# DUP_AFTER_ENTRYPOINT                 drawdown = p_res - p_wellbore_at_stage - dP_le
# DUP_AFTER_ENTRYPOINT                 q_new_dist_unnorm = np.sqrt(np.maximum(0, drawdown))
# DUP_AFTER_ENTRYPOINT                 if np.sum(q_new_dist_unnorm) > 1e-9:
# DUP_AFTER_ENTRYPOINT                     q_dist = q_new_dist_unnorm / np.sum(q_new_dist_unnorm)
# DUP_AFTER_ENTRYPOINT             c1_msw, c2_msw = st.columns(2)
# DUP_AFTER_ENTRYPOINT             with c1_msw:
# DUP_AFTER_ENTRYPOINT                 fig_p = go.Figure(go.Scatter(x=np.arange(n_stages)*ss_ft, y=p_wellbore_at_stage, mode='lines+markers'))
# DUP_AFTER_ENTRYPOINT                 fig_p.update_layout(title="<b>Wellbore Pressure Profile</b>", xaxis_title="Dist. from Heel (ft)", yaxis_title="Pressure (psi)", template="plotly_white")
# DUP_AFTER_ENTRYPOINT                 st.plotly_chart(fig_p, use_container_width=True)
# DUP_AFTER_ENTRYPOINT             with c2_msw:
# DUP_AFTER_ENTRYPOINT                 fig_q = go.Figure(go.Bar(x=np.arange(n_stages)*ss_ft, y=q_dist * 100))
# DUP_AFTER_ENTRYPOINT                 fig_q.update_layout(title="<b>Flow Contribution per Stage</b>", xaxis_title="Dist. from Heel (ft)", yaxis_title="Contribution (%)", template="plotly_white")
# DUP_AFTER_ENTRYPOINT                 st.plotly_chart(fig_q, use_container_width=True)
# DUP_AFTER_ENTRYPOINT         except Exception as e:
# DUP_AFTER_ENTRYPOINT             st.warning(f"Could not compute wellbore hydraulics. Error: {e}")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "RTA":
# DUP_AFTER_ENTRYPOINT         st.header("RTA — Quick Diagnostics")
# DUP_AFTER_ENTRYPOINT         sim_data = st.session_state.sim if st.session_state.sim is not None else _get_sim_preview()
# DUP_AFTER_ENTRYPOINT         t, qg = sim_data["t"], sim_data["qg"]
# DUP_AFTER_ENTRYPOINT         y_type_rta = "log" if st.radio("Rate y-axis", ["Linear", "Log"], horizontal=True) == "Log" else "linear"
# DUP_AFTER_ENTRYPOINT         fig = go.Figure(go.Scatter(x=t, y=qg, line=dict(color="firebrick", width=3), name="Gas"))
# DUP_AFTER_ENTRYPOINT         fig.update_layout(**semi_log_layout("R1. Gas rate (q) vs time", yaxis="q (Mscf/d)"))
# DUP_AFTER_ENTRYPOINT         fig.update_yaxes(type=y_type_rta)
# DUP_AFTER_ENTRYPOINT         st.plotly_chart(fig, use_container_width=True)
# DUP_AFTER_ENTRYPOINT         slope = np.gradient(np.log(np.maximum(qg, 1e-9)), np.log(np.maximum(t, 1e-9)))
# DUP_AFTER_ENTRYPOINT         fig2 = go.Figure(go.Scatter(x=t, y=slope, line=dict(color="teal", width=3), name="dlogq/dlogt"))
# DUP_AFTER_ENTRYPOINT         fig2.update_layout(**semi_log_layout("R2. Log-log derivative", yaxis="Slope"))
# DUP_AFTER_ENTRYPOINT         st.plotly_chart(fig2, use_container_width=True)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ======== Page view selection (left sidebar) ========
# DUP_AFTER_ENTRYPOINT     selected_tab = st.sidebar.radio(
# DUP_AFTER_ENTRYPOINT         "View",
# DUP_AFTER_ENTRYPOINT         ["Results", "3D Viewer", "Slice Viewer", "Debug"],
# DUP_AFTER_ENTRYPOINT         index=0
# DUP_AFTER_ENTRYPOINT     )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ----------------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT     # EUR helpers: Recovery-to-date % and gauge renderer with subtitle
# DUP_AFTER_ENTRYPOINT     # ----------------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT     def _recovery_to_date_pct(
# DUP_AFTER_ENTRYPOINT         cum_oil_stb: float,
# DUP_AFTER_ENTRYPOINT         eur_oil_mmbo: float,
# DUP_AFTER_ENTRYPOINT         cum_gas_mscf: float,
# DUP_AFTER_ENTRYPOINT         eur_gas_bcf: float,
# DUP_AFTER_ENTRYPOINT     ) -> tuple[float, float]:
# DUP_AFTER_ENTRYPOINT         # Return (oil_RF_pct, gas_RF_pct) as 0-100, clipped to [0, 100].
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         oil_rf = 0.0
# DUP_AFTER_ENTRYPOINT         gas_rf = 0.0
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Oil RF% = cum oil (STB) / EUR oil (STB)
# DUP_AFTER_ENTRYPOINT         if eur_oil_mmbo and eur_oil_mmbo > 0:
# DUP_AFTER_ENTRYPOINT             eur_oil_stb = float(eur_oil_mmbo) * 1_000_000.0  # MMBO → STB
# DUP_AFTER_ENTRYPOINT             oil_rf = 100.0 * (float(cum_oil_stb) / eur_oil_stb)
# DUP_AFTER_ENTRYPOINT             oil_rf = max(0.0, min(100.0, oil_rf))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Gas RF% = cum gas (Mscf) / EUR gas (Mscf)
# DUP_AFTER_ENTRYPOINT         if eur_gas_bcf and eur_gas_bcf > 0:
# DUP_AFTER_ENTRYPOINT             eur_gas_mscf = float(eur_gas_bcf) * 1_000_000.0  # BCF → Mscf
# DUP_AFTER_ENTRYPOINT             gas_rf = 100.0 * (float(cum_gas_mscf) / eur_gas_mscf)
# DUP_AFTER_ENTRYPOINT             gas_rf = max(0.0, min(100.0, gas_rf))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         return oil_rf, gas_rf
# DUP_AFTER_ENTRYPOINT     # >>> REPLACE EVERYTHING from your current def _render_gauge_v2(...) down to its return with THIS EXACT BLOCK:
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def _render_gauge_v2(
# DUP_AFTER_ENTRYPOINT         title: str,
# DUP_AFTER_ENTRYPOINT         value: float,
# DUP_AFTER_ENTRYPOINT         minmax=(0.0, 1.0),
# DUP_AFTER_ENTRYPOINT         fmt: str = "{:,.2f}",
# DUP_AFTER_ENTRYPOINT         unit_suffix: str = "",
# DUP_AFTER_ENTRYPOINT         subtitle: str | None = None,
# DUP_AFTER_ENTRYPOINT     ):
# DUP_AFTER_ENTRYPOINT         # Build a Plotly gauge+number figure with an optional subtitle (small text under the title).
# DUP_AFTER_ENTRYPOINT         # Requires: go (plotly.graph_objects as go) and your gauge_max(...) helper.
# DUP_AFTER_ENTRYPOINT         import plotly.graph_objects as go
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Normalize inputs and compute a reasonable vmax
# DUP_AFTER_ENTRYPOINT         lo, hi = (minmax if isinstance(minmax, (list, tuple)) and len(minmax) == 2 else (0.0, 1.0))
# DUP_AFTER_ENTRYPOINT         vmax = gauge_max(value, hi, floor=max(lo, 0.1), safety=0.15)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         sub_html = f"<br><span style='font-size:12px;color:#666'>{subtitle}</span>" if subtitle else ""
# DUP_AFTER_ENTRYPOINT         number_fmt = fmt.replace("{", "").replace("}", "").replace(":", "")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         fig = go.Figure(
# DUP_AFTER_ENTRYPOINT             go.Indicator(
# DUP_AFTER_ENTRYPOINT                 mode="gauge+number",
# DUP_AFTER_ENTRYPOINT                 value=float(value) if value is not None else 0.0,
# DUP_AFTER_ENTRYPOINT                 title={"text": f"{title}{sub_html}"},
# DUP_AFTER_ENTRYPOINT                 number={"valueformat": number_fmt, "suffix": f" {unit_suffix}" if unit_suffix else ""},
# DUP_AFTER_ENTRYPOINT                 gauge={"axis": {"range": [max(lo, 0.0), max(vmax, max(lo, 0.0) + 1e-12)]}},
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT         )
# DUP_AFTER_ENTRYPOINT         fig.update_layout(height=320, margin=dict(l=6, r=6, t=36, b=6), paper_bgcolor="#ffffff")
# DUP_AFTER_ENTRYPOINT         return fig
# DUP_AFTER_ENTRYPOINT     # <<< END REPLACEMENT
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ---- Brand colors for gauges (global) ----
# DUP_AFTER_ENTRYPOINT     GAS_RED   = "#D62728"  # Plotly red for gas
# DUP_AFTER_ENTRYPOINT     OIL_GREEN = "#2CA02C"  # Plotly green for oil
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ========= Tab switcher (TOP-LEVEL; column 0) =========
# DUP_AFTER_ENTRYPOINT     # ============================= RESULTS TAB =============================
# DUP_AFTER_ENTRYPOINT     if selected_tab == "Results":
# DUP_AFTER_ENTRYPOINT         st.header("Simulation Results")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # --- EUR options (UI passthrough; engine has resource-aware defaults) ---
# DUP_AFTER_ENTRYPOINT         with st.expander("EUR options", expanded=False):
# DUP_AFTER_ENTRYPOINT             st.number_input(
# DUP_AFTER_ENTRYPOINT                 "Cutoff horizon (days)",
# DUP_AFTER_ENTRYPOINT                 min_value=0.0,
# DUP_AFTER_ENTRYPOINT                 value=float(st.session_state.get("eur_cutoff_days", 30.0 * 365.25)),
# DUP_AFTER_ENTRYPOINT                 step=30.0,
# DUP_AFTER_ENTRYPOINT                 key="eur_cutoff_days",
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             st.number_input(
# DUP_AFTER_ENTRYPOINT                 "Min gas rate (Mscf/d)",
# DUP_AFTER_ENTRYPOINT                 min_value=0.0,
# DUP_AFTER_ENTRYPOINT                 value=float(st.session_state.get("eur_min_rate_gas_mscfd", 100.0)),
# DUP_AFTER_ENTRYPOINT                 step=10.0,
# DUP_AFTER_ENTRYPOINT                 key="eur_min_rate_gas_mscfd",
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             st.number_input(
# DUP_AFTER_ENTRYPOINT                 "Min oil rate (STB/d)",
# DUP_AFTER_ENTRYPOINT                 min_value=0.0,
# DUP_AFTER_ENTRYPOINT                 value=float(st.session_state.get("eur_min_rate_oil_stbd", 30.0)),
# DUP_AFTER_ENTRYPOINT                 step=5.0,
# DUP_AFTER_ENTRYPOINT                 key="eur_min_rate_oil_stbd",
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # --- Run button (always clear stale results before new run) ---
# DUP_AFTER_ENTRYPOINT         run_clicked = st.button("Run simulation", type="primary", use_container_width=True)
# DUP_AFTER_ENTRYPOINT         if run_clicked:
# DUP_AFTER_ENTRYPOINT             st.session_state.sim = None  # clear prior run
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             if "kx" not in st.session_state:
# DUP_AFTER_ENTRYPOINT                 st.info("Rock properties not found. Generating them first...")
# DUP_AFTER_ENTRYPOINT                 generate_property_volumes(state)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             with st.spinner("Running full 3D simulation..."):
# DUP_AFTER_ENTRYPOINT                 sim_out = run_simulation_engine(state)
# DUP_AFTER_ENTRYPOINT                 if sim_out is not None:
# DUP_AFTER_ENTRYPOINT                     st.session_state.sim = sim_out
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # ---- fetch sim & guard against stale signatures ----
# DUP_AFTER_ENTRYPOINT         sim = st.session_state.get("sim")
# DUP_AFTER_ENTRYPOINT         cur_sig = _sim_signature_from_state()
# DUP_AFTER_ENTRYPOINT         prev_sig = sim.get("_sim_signature") if isinstance(sim, dict) else None
# DUP_AFTER_ENTRYPOINT         if (sim is not None) and (prev_sig is not None) and (cur_sig != prev_sig):
# DUP_AFTER_ENTRYPOINT             st.session_state.sim = None
# DUP_AFTER_ENTRYPOINT             sim = None
# DUP_AFTER_ENTRYPOINT             st.info("Play/engine/physics changed. Please click **Run simulation** to refresh results.")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # If nothing to show yet, stop early
# DUP_AFTER_ENTRYPOINT         if not isinstance(sim, dict) or not sim:
# DUP_AFTER_ENTRYPOINT             st.info("Click **Run simulation** to compute and display the results.")
# DUP_AFTER_ENTRYPOINT             st.stop()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         if sim.get("runtime_s") is not None:
# DUP_AFTER_ENTRYPOINT             st.success(f"Simulation complete in {sim.get('runtime_s', 0):.2f} seconds.")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # --- Sanity gate: block publishing if EURs are out-of-bounds ---
# DUP_AFTER_ENTRYPOINT         eur_g = float(sim.get("eur_gas_BCF", sim.get("EUR_g_BCF", 0.0)))
# DUP_AFTER_ENTRYPOINT         eur_o = float(sim.get("eur_o_MMBO",  sim.get("EUR_o_MMBO", 0.0)))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         play_name = st.session_state.get("play_sel", "")
# DUP_AFTER_ENTRYPOINT         b = _sanity_bounds_for_play(play_name)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         implied_eur_gor = (1000.0 * eur_g / eur_o) if eur_o > 1e-12 else np.inf
# DUP_AFTER_ENTRYPOINT         gor_cap = float(b.get("max_eur_gor_scfstb", 2000.0))
# DUP_AFTER_ENTRYPOINT         tol = 1e-6
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         issues = []
# DUP_AFTER_ENTRYPOINT         chosen_engine = st.session_state.get("engine_type", "")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Check Gas EUR
# DUP_AFTER_ENTRYPOINT         if not (b["gas_bcf"][0] <= eur_g <= b["gas_bcf"][1]):
# DUP_AFTER_ENTRYPOINT             issues.append(f"Gas EUR {eur_g:.2f} BCF outside sanity {b['gas_bcf']} BCF")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Check Oil EUR
# DUP_AFTER_ENTRYPOINT         if eur_o < b["oil_mmbo"][0] or eur_o > b["oil_mmbo"][1]:
# DUP_AFTER_ENTRYPOINT             issues.append(f"Oil EUR {eur_o:.2f} MMBO outside sanity {b['oil_mmbo']} MMBO")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Strict GOR check only for Analytical
# DUP_AFTER_ENTRYPOINT         if "Analytical" in chosen_engine and implied_eur_gor > (gor_cap + tol):
# DUP_AFTER_ENTRYPOINT             issues.append(f"Implied EUR GOR {implied_eur_gor:,.0f} scf/STB exceeds {gor_cap:,.0f}")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         if issues:
# DUP_AFTER_ENTRYPOINT             hint = (
# DUP_AFTER_ENTRYPOINT                 " Tip: Try increasing the 'Pad BHP (psi)' in the sidebar to be closer to the 'pb_psi' "
# DUP_AFTER_ENTRYPOINT                 "to reduce gas production."
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             if "Analytical" in chosen_engine:
# DUP_AFTER_ENTRYPOINT                 # During proxy debugging, warn but do not block publishing
# DUP_AFTER_ENTRYPOINT                 st.warning(
# DUP_AFTER_ENTRYPOINT                     "Sanity checks flagged issues (Analytical engine), but results are shown for debugging.\n\n"
# DUP_AFTER_ENTRYPOINT                     "Details:\n- " + "\n- ".join(issues) + hint
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT             else:
# DUP_AFTER_ENTRYPOINT                 st.error(
# DUP_AFTER_ENTRYPOINT                     "Production results failed sanity checks and were not published.\n\n"
# DUP_AFTER_ENTRYPOINT                     "Details:\n- " + "\n- ".join(issues) + hint
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT                 st.stop()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # ---- Validation gate (engine-side) ----
# DUP_AFTER_ENTRYPOINT     eur_valid = bool(sim.get("eur_valid", True))
# DUP_AFTER_ENTRYPOINT     eur_msg = sim.get("eur_validation_msg", "OK")
# DUP_AFTER_ENTRYPOINT     if not eur_valid:
# DUP_AFTER_ENTRYPOINT         st.error(
# DUP_AFTER_ENTRYPOINT             "Production results failed sanity checks and were not published.\n\n"
# DUP_AFTER_ENTRYPOINT             f"Details: {eur_msg}\n\n"
# DUP_AFTER_ENTRYPOINT             "Please review PVT, controls, and units; then re-run.",
# DUP_AFTER_ENTRYPOINT             icon="🚫",
# DUP_AFTER_ENTRYPOINT         )
# DUP_AFTER_ENTRYPOINT         st.stop()
# DUP_AFTER_ENTRYPOINT     # ... inside: if selected_tab == "Results": 
# DUP_AFTER_ENTRYPOINT     # after your sanity checks/validation and BEFORE the gauges
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # Safety net in case hot-reload skipped the global color block
# DUP_AFTER_ENTRYPOINT     try:
# DUP_AFTER_ENTRYPOINT         OIL_GREEN
# DUP_AFTER_ENTRYPOINT         GAS_RED
# DUP_AFTER_ENTRYPOINT     except NameError:
# DUP_AFTER_ENTRYPOINT         GAS_RED   = "#D62728"
# DUP_AFTER_ENTRYPOINT         OIL_GREEN = "#2CA02C"
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ---------- Recovery to date & Gauges (Oil first, then Gas) ----------
# DUP_AFTER_ENTRYPOINT     # (now render _render_gauge(...))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ----------------------------------------------------------------------
# DUP_AFTER_ENTRYPOINT     # ---------- Recovery to date & Gauges (Oil first, then Gas) ----------
# DUP_AFTER_ENTRYPOINT     # Pull cumulative-to-date (use last sample if arrays exist)
# DUP_AFTER_ENTRYPOINT     _cum_o = sim.get("cum_o_MMBO")
# DUP_AFTER_ENTRYPOINT     _cum_g = sim.get("cum_g_BCF")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     if isinstance(_cum_o, (list, tuple, np.ndarray)) and len(_cum_o) > 0:
# DUP_AFTER_ENTRYPOINT         cum_oil_stb = float(_cum_o[-1]) * 1_000_000.0  # MMBO → STB
# DUP_AFTER_ENTRYPOINT     else:
# DUP_AFTER_ENTRYPOINT         cum_oil_stb = 0.0
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     if isinstance(_cum_g, (list, tuple, np.ndarray)) and len(_cum_g) > 0:
# DUP_AFTER_ENTRYPOINT         cum_gas_mscf = float(_cum_g[-1]) * 1_000_000.0  # BCF → Mscf
# DUP_AFTER_ENTRYPOINT     else:
# DUP_AFTER_ENTRYPOINT         cum_gas_mscf = 0.0
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     oil_rf_pct, gas_rf_pct = _recovery_to_date_pct(
# DUP_AFTER_ENTRYPOINT         cum_oil_stb=cum_oil_stb,
# DUP_AFTER_ENTRYPOINT         eur_oil_mmbo=float(eur_o or 0.0),
# DUP_AFTER_ENTRYPOINT         cum_gas_mscf=cum_gas_mscf,
# DUP_AFTER_ENTRYPOINT         eur_gas_bcf=float(eur_g or 0.0),
# DUP_AFTER_ENTRYPOINT     )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     with st.container():
# DUP_AFTER_ENTRYPOINT         c1, c2 = st.columns([1, 1], gap="small")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # --- OIL ---
# DUP_AFTER_ENTRYPOINT         with c1:
# DUP_AFTER_ENTRYPOINT             v_o = float(eur_o) if eur_o is not None else 0.0
# DUP_AFTER_ENTRYPOINT             oil_fig = _render_gauge_v2(
# DUP_AFTER_ENTRYPOINT                 title="EUR Oil",
# DUP_AFTER_ENTRYPOINT                 value=v_o,
# DUP_AFTER_ENTRYPOINT                 minmax=b["oil_mmbo"],   # expects (lo, hi)
# DUP_AFTER_ENTRYPOINT                 unit_suffix="MMBO",
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             oil_fig.update_traces(
# DUP_AFTER_ENTRYPOINT                 number={"font": {"size": 46}},
# DUP_AFTER_ENTRYPOINT                 gauge={
# DUP_AFTER_ENTRYPOINT                     "bar": {"color": "#2CA02C", "thickness": 0.30},
# DUP_AFTER_ENTRYPOINT                     "axis": {"tickfont": {"size": 10}},
# DUP_AFTER_ENTRYPOINT                 },
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             st.plotly_chart(oil_fig, use_container_width=True, theme=None, key="eur_gauge_oil")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # --- GAS ---
# DUP_AFTER_ENTRYPOINT         with c2:
# DUP_AFTER_ENTRYPOINT             v_g = float(eur_g) if eur_g is not None else 0.0
# DUP_AFTER_ENTRYPOINT             gas_fig = _render_gauge_v2(
# DUP_AFTER_ENTRYPOINT                 title="EUR Gas",
# DUP_AFTER_ENTRYPOINT                 value=v_g,
# DUP_AFTER_ENTRYPOINT                 minmax=b["gas_bcf"],
# DUP_AFTER_ENTRYPOINT                 unit_suffix="BCF",
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             gas_fig.update_traces(
# DUP_AFTER_ENTRYPOINT                 number={"font": {"size": 46}},
# DUP_AFTER_ENTRYPOINT                 gauge={
# DUP_AFTER_ENTRYPOINT                     "bar": {"color": "#D62728", "thickness": 0.30},
# DUP_AFTER_ENTRYPOINT                     "axis": {"tickfont": {"size": 10}},
# DUP_AFTER_ENTRYPOINT                 },
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             st.plotly_chart(gas_fig, use_container_width=True, theme=None, key="eur_gauge_gas")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ---------- Expected ranges (play sanity envelope) ----------
# DUP_AFTER_ENTRYPOINT     oil_rng = b["oil_mmbo"]
# DUP_AFTER_ENTRYPOINT     gas_rng = b["gas_bcf"]
# DUP_AFTER_ENTRYPOINT     gor_cap = float(b.get("max_eur_gor_scfstb", np.inf))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     def _in_range(val: float, rng: tuple[float, float]) -> bool:
# DUP_AFTER_ENTRYPOINT         try:
# DUP_AFTER_ENTRYPOINT             v = float(val)
# DUP_AFTER_ENTRYPOINT         except Exception:
# DUP_AFTER_ENTRYPOINT             return False
# DUP_AFTER_ENTRYPOINT         return (rng[0] - 1e-9) <= v <= (rng[1] + 1e-9)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     oil_ok = _in_range(eur_o, oil_rng)
# DUP_AFTER_ENTRYPOINT     gas_ok = _in_range(eur_g, gas_rng)
# DUP_AFTER_ENTRYPOINT     gor_ok = (implied_eur_gor <= gor_cap) if np.isfinite(implied_eur_gor) else False
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     status = (lambda ok: "✅ OK" if ok else "⚠️ Check")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     pad_ctrl = str(st.session_state.get("pad_ctrl", ""))
# DUP_AFTER_ENTRYPOINT     bhp = st.session_state.get("pad_bhp_psi", None)
# DUP_AFTER_ENTRYPOINT     pb  = st.session_state.get("pb_psi", None)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     with st.container():
# DUP_AFTER_ENTRYPOINT         st.markdown("#### Expected ranges (play sanity envelope)")
# DUP_AFTER_ENTRYPOINT         c3, c4, c5 = st.columns(3)
# DUP_AFTER_ENTRYPOINT         with c3:
# DUP_AFTER_ENTRYPOINT             st.markdown(
# DUP_AFTER_ENTRYPOINT                 f"**Oil EUR (MMBO)**  \n"
# DUP_AFTER_ENTRYPOINT                 f"Observed: **{eur_o:.2f}**  \n"
# DUP_AFTER_ENTRYPOINT                 f"Envelope: {oil_rng[0]:.2f}–{oil_rng[1]:.2f}  \n"
# DUP_AFTER_ENTRYPOINT                 f"{status(oil_ok)}"
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT         with c4:
# DUP_AFTER_ENTRYPOINT             st.markdown(
# DUP_AFTER_ENTRYPOINT                 f"**Gas EUR (BCF)**  \n"
# DUP_AFTER_ENTRYPOINT                 f"Observed: **{eur_g:.2f}**  \n"
# DUP_AFTER_ENTRYPOINT                 f"Envelope: {gas_rng[0]:.2f}–{gas_rng[1]:.2f}  \n"
# DUP_AFTER_ENTRYPOINT                 f"{status(gas_ok)}"
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT         with c5:
# DUP_AFTER_ENTRYPOINT             _cap_str = "∞" if not np.isfinite(gor_cap) else f"{gor_cap:,.0f}"
# DUP_AFTER_ENTRYPOINT             st.markdown(
# DUP_AFTER_ENTRYPOINT                 f"**Implied EUR GOR (scf/STB)**  \n"
# DUP_AFTER_ENTRYPOINT                 f"Observed: **{implied_eur_gor:,.0f}**  \n"
# DUP_AFTER_ENTRYPOINT                 f"Cap: {_cap_str}  \n"
# DUP_AFTER_ENTRYPOINT                 f"{status(gor_ok)}"
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT         _ctx = []
# DUP_AFTER_ENTRYPOINT         if pad_ctrl: _ctx.append(f"Control: {pad_ctrl}")
# DUP_AFTER_ENTRYPOINT         if isinstance(bhp, (int, float)): _ctx.append(f"BHP: {float(bhp):.0f} psi")
# DUP_AFTER_ENTRYPOINT         if isinstance(pb,  (int, float)): _ctx.append(f"pb: {float(pb):.0f} psi")
# DUP_AFTER_ENTRYPOINT         if _ctx: st.caption(" · ".join(_ctx))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Optional small operating context line (helps debug BHP vs pb)
# DUP_AFTER_ENTRYPOINT         _ctx = []
# DUP_AFTER_ENTRYPOINT         if pad_ctrl:
# DUP_AFTER_ENTRYPOINT             _ctx.append(f"Control: {pad_ctrl}")
# DUP_AFTER_ENTRYPOINT         if isinstance(bhp, (int, float)):
# DUP_AFTER_ENTRYPOINT             _ctx.append(f"BHP: {float(bhp):.0f} psi")
# DUP_AFTER_ENTRYPOINT         if isinstance(pb, (int, float)):
# DUP_AFTER_ENTRYPOINT             _ctx.append(f"pb: {float(pb):.0f} psi")
# DUP_AFTER_ENTRYPOINT         if _ctx:
# DUP_AFTER_ENTRYPOINT             st.caption(" · ".join(_ctx))
# DUP_AFTER_ENTRYPOINT     
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # ===================== BHP Sensitivity (Analytical only) =====================
# DUP_AFTER_ENTRYPOINT         with st.expander("BHP sensitivity (Analytical proxy)", expanded=False):
# DUP_AFTER_ENTRYPOINT             colA, colB, colC, colD = st.columns(4)
# DUP_AFTER_ENTRYPOINT             with colA:
# DUP_AFTER_ENTRYPOINT                 bhp_start = st.number_input(
# DUP_AFTER_ENTRYPOINT                     "Start BHP (psi)", 3000.0, 9000.0,
# DUP_AFTER_ENTRYPOINT                     float(st.session_state.get("pad_bhp_psi", 5200.0)),
# DUP_AFTER_ENTRYPOINT                     step=50.0, key="bhp_sens_start"
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT             with colB:
# DUP_AFTER_ENTRYPOINT                 bhp_end = st.number_input(
# DUP_AFTER_ENTRYPOINT                     "End BHP (psi)", 3000.0, 9000.0,
# DUP_AFTER_ENTRYPOINT                     4400.0, step=50.0, key="bhp_sens_end"
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT             with colC:
# DUP_AFTER_ENTRYPOINT                 bhp_step = st.number_input(
# DUP_AFTER_ENTRYPOINT                     "Step (psi)", 10.0, 1000.0, 200.0,
# DUP_AFTER_ENTRYPOINT                     step=10.0, key="bhp_sens_step"
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT             with colD:
# DUP_AFTER_ENTRYPOINT                 run_btn = st.button("Run sweep", type="primary", key="bhp_sens_go")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             if run_btn:
# DUP_AFTER_ENTRYPOINT                 import numpy as _np
# DUP_AFTER_ENTRYPOINT                 import pandas as _pd
# DUP_AFTER_ENTRYPOINT                 import plotly.graph_objects as _go
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                 if bhp_step <= 0 or bhp_end == bhp_start:
# DUP_AFTER_ENTRYPOINT                     st.warning("Choose a non-zero step and different start/end values.")
# DUP_AFTER_ENTRYPOINT                     st.stop()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                 # Build the BHP list (inclusive of end where possible)
# DUP_AFTER_ENTRYPOINT                 if bhp_end > bhp_start:
# DUP_AFTER_ENTRYPOINT                     bhps = _np.arange(bhp_start, bhp_end + 0.5 * bhp_step, bhp_step)
# DUP_AFTER_ENTRYPOINT                 else:
# DUP_AFTER_ENTRYPOINT                     bhps = _np.arange(bhp_start, bhp_end - 0.5 * bhp_step, -abs(bhp_step))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                 rows = []
# DUP_AFTER_ENTRYPOINT                 rng_seed = int(st.session_state.get("rng_seed", 1234))
# DUP_AFTER_ENTRYPOINT                 rng = _np.random.default_rng(rng_seed)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                 for bhp in bhps:
# DUP_AFTER_ENTRYPOINT                     # Make an isolated copy of state and set BHP control for this run
# DUP_AFTER_ENTRYPOINT                     _state = dict(st.session_state.get("state_for_solver", {}))
# DUP_AFTER_ENTRYPOINT                     _state["pad_ctrl"] = "BHP"
# DUP_AFTER_ENTRYPOINT                     _state["pad_bhp_psi"] = float(bhp)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                     # Ensure pb/p_res are present if your proxy uses them
# DUP_AFTER_ENTRYPOINT                     if "pb_psi" not in _state:
# DUP_AFTER_ENTRYPOINT                         _state["pb_psi"] = float(st.session_state.get("pb_psi", 5200.0))
# DUP_AFTER_ENTRYPOINT                     if "p_res_psi" not in _state:
# DUP_AFTER_ENTRYPOINT                         _state["p_res_psi"] = float(st.session_state.get("p_res_psi", 5800.0))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                     try:
# DUP_AFTER_ENTRYPOINT                         out = fallback_fast_solver(_state, rng=rng)
# DUP_AFTER_ENTRYPOINT                     except Exception as e:
# DUP_AFTER_ENTRYPOINT                         st.error(f"Analytical run failed at BHP={bhp:.0f} psi: {e}")
# DUP_AFTER_ENTRYPOINT                         continue
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                     # Normalize keys and compute GOR safely
# DUP_AFTER_ENTRYPOINT                     eur_g_s = float(out.get("EUR_g_BCF", out.get("eur_gas_BCF", 0.0)))
# DUP_AFTER_ENTRYPOINT                     eur_o_s = float(out.get("EUR_o_MMBO", out.get("eur_oil_MMBO", 0.0)))
# DUP_AFTER_ENTRYPOINT                     gor_s   = (1000.0 * eur_g_s / eur_o_s) if eur_o_s > 1e-12 else _np.inf
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                     rows.append(dict(
# DUP_AFTER_ENTRYPOINT                         BHP_psi=float(bhp),
# DUP_AFTER_ENTRYPOINT                         EUR_g_BCF=eur_g_s,
# DUP_AFTER_ENTRYPOINT                         EUR_o_MMBO=eur_o_s,
# DUP_AFTER_ENTRYPOINT                         EUR_GOR_scf_stb=gor_s
# DUP_AFTER_ENTRYPOINT                     ))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                 if not rows:
# DUP_AFTER_ENTRYPOINT                     st.warning("No results returned.")
# DUP_AFTER_ENTRYPOINT                     st.stop()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                 df = _pd.DataFrame(rows).sort_values("BHP_psi", ascending=False).reset_index(drop=True)
# DUP_AFTER_ENTRYPOINT                 st.dataframe(df, use_container_width=True)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                 # --- Plot EURs vs BHP ---
# DUP_AFTER_ENTRYPOINT                 fig_eur = _go.Figure()
# DUP_AFTER_ENTRYPOINT                 fig_eur.add_trace(_go.Scatter(
# DUP_AFTER_ENTRYPOINT                     x=df["BHP_psi"], y=df["EUR_g_BCF"],
# DUP_AFTER_ENTRYPOINT                     name="Gas EUR (BCF)", mode="lines+markers",
# DUP_AFTER_ENTRYPOINT                     line=dict(width=3, color=GAS_RED)
# DUP_AFTER_ENTRYPOINT                 ))
# DUP_AFTER_ENTRYPOINT                 fig_eur.add_trace(_go.Scatter(
# DUP_AFTER_ENTRYPOINT                     x=df["BHP_psi"], y=df["EUR_o_MMBO"],
# DUP_AFTER_ENTRYPOINT                     name="Oil EUR (MMBO)", mode="lines+markers",
# DUP_AFTER_ENTRYPOINT                     line=dict(width=3, color=OIL_GREEN), yaxis="y2"
# DUP_AFTER_ENTRYPOINT                 ))
# DUP_AFTER_ENTRYPOINT                 fig_eur.update_layout(
# DUP_AFTER_ENTRYPOINT                     template="plotly_white",
# DUP_AFTER_ENTRYPOINT                     title="<b>EUR vs BHP</b>",
# DUP_AFTER_ENTRYPOINT                     height=420,
# DUP_AFTER_ENTRYPOINT                     margin=dict(l=10, r=10, t=50, b=10),
# DUP_AFTER_ENTRYPOINT                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
# DUP_AFTER_ENTRYPOINT                     yaxis=dict(title="Gas EUR (BCF)"),
# DUP_AFTER_ENTRYPOINT                     yaxis2=dict(title="Oil EUR (MMBO)", overlaying="y", side="right"),
# DUP_AFTER_ENTRYPOINT                     xaxis=dict(title="BHP (psi)")
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT                 st.plotly_chart(fig_eur, use_container_width=True, theme=None, key="bhp_sens_eur_plot")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                 # --- Plot GOR vs BHP ---
# DUP_AFTER_ENTRYPOINT                 fig_gor = _go.Figure(_go.Scatter(
# DUP_AFTER_ENTRYPOINT                     x=df["BHP_psi"], y=df["EUR_GOR_scf_stb"],
# DUP_AFTER_ENTRYPOINT                     name="EUR GOR (scf/STB)", mode="lines+markers"
# DUP_AFTER_ENTRYPOINT                 ))
# DUP_AFTER_ENTRYPOINT                 # Optional: draw your GOR cap line if defined for the play
# DUP_AFTER_ENTRYPOINT                 _bounds = _sanity_bounds_for_play(st.session_state.get("play_sel", ""))
# DUP_AFTER_ENTRYPOINT                 _gor_cap = float(_bounds.get("max_eur_gor_scfstb", 2000.0))
# DUP_AFTER_ENTRYPOINT                 fig_gor.add_hline(y=_gor_cap, line=dict(color="rgba(0,0,0,0.3)", dash="dot"))
# DUP_AFTER_ENTRYPOINT                 fig_gor.update_layout(
# DUP_AFTER_ENTRYPOINT                     template="plotly_white",
# DUP_AFTER_ENTRYPOINT                     title="<b>EUR GOR vs BHP</b>",
# DUP_AFTER_ENTRYPOINT                     height=400,
# DUP_AFTER_ENTRYPOINT                     margin=dict(l=10, r=10, t=50, b=10),
# DUP_AFTER_ENTRYPOINT                     yaxis=dict(title="scf/STB"),
# DUP_AFTER_ENTRYPOINT                     xaxis=dict(title="BHP (psi)"),
# DUP_AFTER_ENTRYPOINT                     showlegend=False
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT                 st.plotly_chart(fig_gor, use_container_width=True, theme=None, key="bhp_sens_gor_plot")
# DUP_AFTER_ENTRYPOINT         # =================== end BHP Sensitivity block ===================
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # ======== Semi-log plots (Rate & Cumulative) ========
# DUP_AFTER_ENTRYPOINT         t  = sim.get("t")
# DUP_AFTER_ENTRYPOINT         qg = sim.get("qg")
# DUP_AFTER_ENTRYPOINT         qo = sim.get("qo")
# DUP_AFTER_ENTRYPOINT         qw = sim.get("qw")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # --- Semi-log Rate vs Time (with decade lines & cycles) ---
# DUP_AFTER_ENTRYPOINT         if t is not None and (qg is not None or qo is not None or qw is not None):
# DUP_AFTER_ENTRYPOINT             t_arr = np.asarray(t, float)
# DUP_AFTER_ENTRYPOINT             t_min = float(np.nanmin(t_arr[t_arr > 0])) if np.any(t_arr > 0) else 1.0
# DUP_AFTER_ENTRYPOINT             t_max = float(np.nanmax(t_arr)) if t_arr.size else 10.0
# DUP_AFTER_ENTRYPOINT             n_cycles = max(0.0, np.log10(max(t_max / max(t_min, 1e-12), 1.0)))
# DUP_AFTER_ENTRYPOINT             decade_ticks = [x for x in [1, 10, 100, 1000, 10000, 100000]
# DUP_AFTER_ENTRYPOINT                             if x >= max(1, t_min/1.0001) and x <= t_max*1.0001]
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             fig_rate = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
# DUP_AFTER_ENTRYPOINT             if qg is not None:
# DUP_AFTER_ENTRYPOINT                 fig_rate.add_trace(
# DUP_AFTER_ENTRYPOINT                     go.Scatter(x=t, y=qg, name="Gas (Mscf/d)",
# DUP_AFTER_ENTRYPOINT                                line=dict(width=2, color=GAS_RED)),
# DUP_AFTER_ENTRYPOINT                     secondary_y=False,
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT             if qo is not None:
# DUP_AFTER_ENTRYPOINT                 fig_rate.add_trace(
# DUP_AFTER_ENTRYPOINT                     go.Scatter(x=t, y=qo, name="Oil (STB/d)",
# DUP_AFTER_ENTRYPOINT                                line=dict(width=2, color=OIL_GREEN)),
# DUP_AFTER_ENTRYPOINT                     secondary_y=True,
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT             if qw is not None:
# DUP_AFTER_ENTRYPOINT                 fig_rate.add_trace(
# DUP_AFTER_ENTRYPOINT                     go.Scatter(x=t, y=qw, name="Water (STB/d)",
# DUP_AFTER_ENTRYPOINT                                line=dict(width=1.8, dash="dot", color="#1f77b4")),
# DUP_AFTER_ENTRYPOINT                     secondary_y=True,
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             vshapes = [
# DUP_AFTER_ENTRYPOINT                 dict(type="line", x0=dt, x1=dt, yref="paper", y0=0.0, y1=1.0,
# DUP_AFTER_ENTRYPOINT                      line=dict(width=1, color="rgba(0,0,0,0.10)", dash="dot"))
# DUP_AFTER_ENTRYPOINT                 for dt in decade_ticks
# DUP_AFTER_ENTRYPOINT             ]
# DUP_AFTER_ENTRYPOINT             fig_rate.update_layout(
# DUP_AFTER_ENTRYPOINT                 template="plotly_white",
# DUP_AFTER_ENTRYPOINT                 title_text="<b>Production Rate vs. Time</b>",
# DUP_AFTER_ENTRYPOINT                 height=460,
# DUP_AFTER_ENTRYPOINT                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
# DUP_AFTER_ENTRYPOINT                 font=dict(size=13),
# DUP_AFTER_ENTRYPOINT                 margin=dict(l=10, r=10, t=50, b=10),
# DUP_AFTER_ENTRYPOINT                 shapes=vshapes,
# DUP_AFTER_ENTRYPOINT                 annotations=[
# DUP_AFTER_ENTRYPOINT                     dict(
# DUP_AFTER_ENTRYPOINT                         xref="paper", yref="paper", x=0.01, y=1.08, showarrow=False,
# DUP_AFTER_ENTRYPOINT                         text=f"Log cycles (x-axis): {n_cycles:.2f}",
# DUP_AFTER_ENTRYPOINT                         font=dict(size=12, color="#444")
# DUP_AFTER_ENTRYPOINT                     )
# DUP_AFTER_ENTRYPOINT                 ],
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             fig_rate.update_xaxes(
# DUP_AFTER_ENTRYPOINT                 type="log", dtick=1, tickvals=decade_ticks, title="Time (days)",
# DUP_AFTER_ENTRYPOINT                 showgrid=True, gridcolor="rgba(0,0,0,0.12)"
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             fig_rate.update_yaxes(
# DUP_AFTER_ENTRYPOINT                 title_text="Gas rate (Mscf/d)", secondary_y=False,
# DUP_AFTER_ENTRYPOINT                 showgrid=True, gridcolor="rgba(0,0,0,0.15)"
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             fig_rate.update_yaxes(
# DUP_AFTER_ENTRYPOINT                 title_text="Liquid rates (STB/d)", secondary_y=True, showgrid=False
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             st.plotly_chart(fig_rate, use_container_width=True, theme=None, key="rate_semilog_chart")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             with st.expander("How to read this plot"):
# DUP_AFTER_ENTRYPOINT                 st.markdown(
# DUP_AFTER_ENTRYPOINT                     "- **Semi-log X** emphasizes early-time behavior and decline trends.\n"
# DUP_AFTER_ENTRYPOINT                     "- **Vertical dotted lines** mark decade boundaries on time (1, 10, 100, … days).\n"
# DUP_AFTER_ENTRYPOINT                     "- **Cycles** = number of log decades spanned on the x-axis.\n"
# DUP_AFTER_ENTRYPOINT                     "- Gas is on the **left axis**; liquids (oil/water) on the **right axis**.\n"
# DUP_AFTER_ENTRYPOINT                     "- Look for slope changes that may indicate **boundary effects** or **flow regime transitions**."
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT         else:
# DUP_AFTER_ENTRYPOINT             st.warning("Rate series not available.")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # --- Semi-log Cumulative vs Time (with decade lines & cycles) ---
# DUP_AFTER_ENTRYPOINT         cum_g = sim.get("cum_g_BCF")
# DUP_AFTER_ENTRYPOINT         cum_o = sim.get("cum_o_MMBO")
# DUP_AFTER_ENTRYPOINT         cum_w = sim.get("cum_w_MMBL")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         if t is not None and (cum_g is not None or cum_o is not None or cum_w is not None):
# DUP_AFTER_ENTRYPOINT             t_arr = np.asarray(t, float)
# DUP_AFTER_ENTRYPOINT             t_min = float(np.nanmin(t_arr[t_arr > 0])) if np.any(t_arr > 0) else 1.0
# DUP_AFTER_ENTRYPOINT             t_max = float(np.nanmax(t_arr)) if t_arr.size else 10.0
# DUP_AFTER_ENTRYPOINT             n_cycles = max(0.0, np.log10(max(t_max / max(t_min, 1e-12), 1.0)))
# DUP_AFTER_ENTRYPOINT             decade_ticks = [x for x in [1, 10, 100, 1000, 10000, 100000]
# DUP_AFTER_ENTRYPOINT                             if x >= max(1, t_min/1.0001) and x <= t_max*1.0001]
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             fig_cum = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
# DUP_AFTER_ENTRYPOINT             if cum_g is not None:
# DUP_AFTER_ENTRYPOINT                 fig_cum.add_trace(
# DUP_AFTER_ENTRYPOINT                     go.Scatter(x=t, y=cum_g, name="Cum Gas (BCF)",
# DUP_AFTER_ENTRYPOINT                                line=dict(width=3, color=GAS_RED)),
# DUP_AFTER_ENTRYPOINT                     secondary_y=False
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT             if cum_o is not None:
# DUP_AFTER_ENTRYPOINT                 fig_cum.add_trace(
# DUP_AFTER_ENTRYPOINT                     go.Scatter(x=t, y=cum_o, name="Cum Oil (MMbbl)",
# DUP_AFTER_ENTRYPOINT                                line=dict(width=3, color=OIL_GREEN)),
# DUP_AFTER_ENTRYPOINT                     secondary_y=True
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT             if cum_w is not None:
# DUP_AFTER_ENTRYPOINT                 fig_cum.add_trace(
# DUP_AFTER_ENTRYPOINT                     go.Scatter(x=t, y=cum_w, name="Cum Water (MMbbl)",
# DUP_AFTER_ENTRYPOINT                                line=dict(width=2, dash="dot", color="#1f77b4")),
# DUP_AFTER_ENTRYPOINT                     secondary_y=True
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             vshapes = [
# DUP_AFTER_ENTRYPOINT                 dict(type="line", x0=dt, x1=dt, yref="paper", y0=0.0, y1=1.0,
# DUP_AFTER_ENTRYPOINT                      line=dict(width=1, color="rgba(0,0,0,0.10)", dash="dot"))
# DUP_AFTER_ENTRYPOINT                 for dt in decade_ticks
# DUP_AFTER_ENTRYPOINT             ]
# DUP_AFTER_ENTRYPOINT             fig_cum.update_layout(
# DUP_AFTER_ENTRYPOINT                 template="plotly_white",
# DUP_AFTER_ENTRYPOINT                 title_text="<b>Cumulative Production</b>",
# DUP_AFTER_ENTRYPOINT                 height=460,
# DUP_AFTER_ENTRYPOINT                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
# DUP_AFTER_ENTRYPOINT                 font=dict(size=13),
# DUP_AFTER_ENTRYPOINT                 margin=dict(l=10, r=10, t=50, b=10),
# DUP_AFTER_ENTRYPOINT                 shapes=vshapes,
# DUP_AFTER_ENTRYPOINT                 annotations=[
# DUP_AFTER_ENTRYPOINT                     dict(
# DUP_AFTER_ENTRYPOINT                         xref="paper", yref="paper", x=0.01, y=1.08, showarrow=False,
# DUP_AFTER_ENTRYPOINT                         text=f"Log cycles (x-axis): {n_cycles:.2f}",
# DUP_AFTER_ENTRYPOINT                         font=dict(size=12, color="#444")
# DUP_AFTER_ENTRYPOINT                     )
# DUP_AFTER_ENTRYPOINT                 ],
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             fig_cum.update_xaxes(
# DUP_AFTER_ENTRYPOINT                 type="log", dtick=1, tickvals=decade_ticks, title="Time (days)",
# DUP_AFTER_ENTRYPOINT                 showgrid=True, gridcolor="rgba(0,0,0,0.12)"
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             fig_cum.update_yaxes(
# DUP_AFTER_ENTRYPOINT                 title_text="Gas (BCF)", secondary_y=False,
# DUP_AFTER_ENTRYPOINT                 showgrid=True, gridcolor="rgba(0,0,0,0.15)"
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             fig_cum.update_yaxes(
# DUP_AFTER_ENTRYPOINT                 title_text="Liquids (MMbbl)", secondary_y=True, showgrid=False
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             st.plotly_chart(fig_cum, use_container_width=True, theme=None, key="cum_semilog_chart")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             with st.expander("How to read this plot"):
# DUP_AFTER_ENTRYPOINT                 st.markdown(
# DUP_AFTER_ENTRYPOINT                     "- **Semi-log X** shows cumulative growth vs. decades of time.\n"
# DUP_AFTER_ENTRYPOINT                     "- **Cum Gas (left)** and **Cum Oil/Water (right)** track recoveries directly tied to EUR.\n"
# DUP_AFTER_ENTRYPOINT                     "- Expect smooth, monotonic curves; kinks often reflect **operating changes** or **model boundaries**."
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT         else:
# DUP_AFTER_ENTRYPOINT             st.warning("Cumulative series not available.")
# DUP_AFTER_ENTRYPOINT     # =========================== END RESULTS TAB ==========================
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     # ======== 3D Viewer tab ========
# DUP_AFTER_ENTRYPOINT     if selected_tab == "3D Viewer":
# DUP_AFTER_ENTRYPOINT         st.subheader("3D Viewer")
# DUP_AFTER_ENTRYPOINT         st.info("Render your 3D grid / fractures / saturation maps here.")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         sim = st.session_state.get("sim") or {}
# DUP_AFTER_ENTRYPOINT         kx_vol = st.session_state.get("kx")    # expected (nz, ny, nx)
# DUP_AFTER_ENTRYPOINT         phi_vol = st.session_state.get("phi")  # expected (nz, ny, nx)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # If nothing at all is available, bail early
# DUP_AFTER_ENTRYPOINT         if kx_vol is None and phi_vol is None and not sim:
# DUP_AFTER_ENTRYPOINT             st.warning("Please generate rock properties or run a simulation to enable the 3D viewer.")
# DUP_AFTER_ENTRYPOINT             st.stop()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Build the property list only from fields that actually exist
# DUP_AFTER_ENTRYPOINT         menu = []
# DUP_AFTER_ENTRYPOINT         if kx_vol is not None:
# DUP_AFTER_ENTRYPOINT             menu.append("Permeability (kx)")
# DUP_AFTER_ENTRYPOINT         if phi_vol is not None:
# DUP_AFTER_ENTRYPOINT             menu.append("Porosity (ϕ)")
# DUP_AFTER_ENTRYPOINT         if sim.get("press_matrix") is not None:
# DUP_AFTER_ENTRYPOINT             menu.append("Pressure (psi)")
# DUP_AFTER_ENTRYPOINT         if sim.get("press_matrix") is not None and sim.get("p_init_3d") is not None:
# DUP_AFTER_ENTRYPOINT             menu.append("Pressure Change (ΔP)")
# DUP_AFTER_ENTRYPOINT         if sim.get("ooip_3d") is not None:
# DUP_AFTER_ENTRYPOINT             menu.append("Original Oil In Place (OOIP)")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         if not menu:
# DUP_AFTER_ENTRYPOINT             st.info("No 3D properties are available yet. Run a simulation to populate pressure/OOIP.")
# DUP_AFTER_ENTRYPOINT             st.stop()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         prop_3d = st.selectbox("Select property to view:", menu, index=0)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         c1, c2 = st.columns(2)
# DUP_AFTER_ENTRYPOINT         with c1:
# DUP_AFTER_ENTRYPOINT             ds = st.slider(
# DUP_AFTER_ENTRYPOINT                 "Downsample factor",
# DUP_AFTER_ENTRYPOINT                 min_value=1,
# DUP_AFTER_ENTRYPOINT                 max_value=10,
# DUP_AFTER_ENTRYPOINT                 value=int(st.session_state.get("vol_downsample", 2)),
# DUP_AFTER_ENTRYPOINT                 step=1,
# DUP_AFTER_ENTRYPOINT                 key="vol_ds",
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT         with c2:
# DUP_AFTER_ENTRYPOINT             iso_rel = st.slider(
# DUP_AFTER_ENTRYPOINT                 "Isosurface value (relative)",
# DUP_AFTER_ENTRYPOINT                 min_value=0.05,
# DUP_AFTER_ENTRYPOINT                 max_value=0.95,
# DUP_AFTER_ENTRYPOINT                 value=float(st.session_state.get("iso_value_rel", 0.85)),
# DUP_AFTER_ENTRYPOINT                 step=0.05,
# DUP_AFTER_ENTRYPOINT                 key="iso_val_rel",
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Resolve grid spacing (accept *_ft or raw)
# DUP_AFTER_ENTRYPOINT         # Prefer `state` if it exists; fall back to session
# DUP_AFTER_ENTRYPOINT         state_src = locals().get("state") or st.session_state
# DUP_AFTER_ENTRYPOINT         dx = float(state_src.get("dx_ft", state_src.get("dx", 1.0)))
# DUP_AFTER_ENTRYPOINT         dy = float(state_src.get("dy_ft", state_src.get("dy", 1.0)))
# DUP_AFTER_ENTRYPOINT         dz = float(state_src.get("dz_ft", state_src.get("dz", 1.0)))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Select data and styling
# DUP_AFTER_ENTRYPOINT         data_3d = None
# DUP_AFTER_ENTRYPOINT         colorscale = "Viridis"
# DUP_AFTER_ENTRYPOINT         colorbar_title = ""
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         if prop_3d.startswith("Permeability"):
# DUP_AFTER_ENTRYPOINT             data_3d = kx_vol
# DUP_AFTER_ENTRYPOINT             colorscale = "Viridis"
# DUP_AFTER_ENTRYPOINT             colorbar_title = "kx (mD)"
# DUP_AFTER_ENTRYPOINT         elif prop_3d.startswith("Porosity"):
# DUP_AFTER_ENTRYPOINT             data_3d = phi_vol
# DUP_AFTER_ENTRYPOINT             colorscale = "Magma"
# DUP_AFTER_ENTRYPOINT             colorbar_title = "Porosity (ϕ)"
# DUP_AFTER_ENTRYPOINT         elif prop_3d.startswith("Pressure (psi)"):
# DUP_AFTER_ENTRYPOINT             data_3d = sim.get("press_matrix")  # (nz, ny, nx)
# DUP_AFTER_ENTRYPOINT             colorscale = "Jet"
# DUP_AFTER_ENTRYPOINT             colorbar_title = "Pressure (psi)"
# DUP_AFTER_ENTRYPOINT         elif prop_3d.startswith("Pressure Change"):
# DUP_AFTER_ENTRYPOINT             p_final = sim.get("press_matrix")
# DUP_AFTER_ENTRYPOINT             p_init = sim.get("p_init_3d")
# DUP_AFTER_ENTRYPOINT             if p_final is not None and p_init is not None:
# DUP_AFTER_ENTRYPOINT                 data_3d = (np.asarray(p_init) - np.asarray(p_final))  # ΔP = Pin − Pfinal
# DUP_AFTER_ENTRYPOINT                 colorscale = "Inferno"
# DUP_AFTER_ENTRYPOINT                 colorbar_title = "ΔP (psi)"
# DUP_AFTER_ENTRYPOINT         elif prop_3d.startswith("Original Oil"):
# DUP_AFTER_ENTRYPOINT             data_3d = sim.get("ooip_3d")
# DUP_AFTER_ENTRYPOINT             colorscale = "Plasma"
# DUP_AFTER_ENTRYPOINT             colorbar_title = "OOIP (STB/cell)"
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Validate selection
# DUP_AFTER_ENTRYPOINT         if data_3d is None:
# DUP_AFTER_ENTRYPOINT             st.warning(f"Data for '{prop_3d}' not found. Please run a simulation.")
# DUP_AFTER_ENTRYPOINT             st.stop()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         data_3d = np.asarray(data_3d)
# DUP_AFTER_ENTRYPOINT         if data_3d.ndim != 3:
# DUP_AFTER_ENTRYPOINT             st.warning("3D data is not in the expected (nz, ny, nx) shape.")
# DUP_AFTER_ENTRYPOINT             st.stop()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Downsample (use your helper if available)
# DUP_AFTER_ENTRYPOINT         try:
# DUP_AFTER_ENTRYPOINT             data_ds = downsample_3d(data_3d, ds)  # noqa: F821  (if helper exists)
# DUP_AFTER_ENTRYPOINT         except Exception:
# DUP_AFTER_ENTRYPOINT             # Simple stride fallback
# DUP_AFTER_ENTRYPOINT             data_ds = data_3d[::ds, ::ds, ::ds]
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         vmin, vmax = float(np.nanmin(data_ds)), float(np.nanmax(data_ds))
# DUP_AFTER_ENTRYPOINT         isoval = vmin + (vmax - vmin) * iso_rel
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # Build coordinates consistent with (nz, ny, nx)
# DUP_AFTER_ENTRYPOINT         nz, ny, nx = data_ds.shape
# DUP_AFTER_ENTRYPOINT         z = np.arange(nz) * dz * ds
# DUP_AFTER_ENTRYPOINT         y = np.arange(ny) * dy * ds
# DUP_AFTER_ENTRYPOINT         x = np.arange(nx) * dx * ds
# DUP_AFTER_ENTRYPOINT         Z, Y, X = np.meshgrid(z, y, x, indexing="ij")  # shapes (nz, ny, nx)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         with st.spinner("Generating 3D plot..."):
# DUP_AFTER_ENTRYPOINT             fig3d = go.Figure(
# DUP_AFTER_ENTRYPOINT                 go.Isosurface(
# DUP_AFTER_ENTRYPOINT                     x=X.ravel(),
# DUP_AFTER_ENTRYPOINT                     y=Y.ravel(),
# DUP_AFTER_ENTRYPOINT                     z=Z.ravel(),
# DUP_AFTER_ENTRYPOINT                     value=data_ds.ravel(),
# DUP_AFTER_ENTRYPOINT                     isomin=isoval,
# DUP_AFTER_ENTRYPOINT                     isomax=vmax,
# DUP_AFTER_ENTRYPOINT                     surface_count=1,
# DUP_AFTER_ENTRYPOINT                     caps=dict(x_show=False, y_show=False, z_show=False),
# DUP_AFTER_ENTRYPOINT                     colorscale=colorscale,
# DUP_AFTER_ENTRYPOINT                     colorbar=dict(title=colorbar_title),
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             # Optional horizontal well overlay (best-effort)
# DUP_AFTER_ENTRYPOINT             try:
# DUP_AFTER_ENTRYPOINT                 L_ft = float(state_src.get("L_ft", nx * dx))
# DUP_AFTER_ENTRYPOINT                 n_lat = int(state_src.get("n_laterals", 1))
# DUP_AFTER_ENTRYPOINT                 y_span = ny * dy * ds
# DUP_AFTER_ENTRYPOINT                 y_positions = ([y_span / 3.0, 2 * y_span / 3.0] if n_lat >= 2 else [y_span / 2.0])
# DUP_AFTER_ENTRYPOINT                 z_mid = (nz * dz * ds) / 2.0
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                 for i, y_pos in enumerate(y_positions):
# DUP_AFTER_ENTRYPOINT                     fig3d.add_trace(
# DUP_AFTER_ENTRYPOINT                         go.Scatter3d(
# DUP_AFTER_ENTRYPOINT                             x=[0.0, L_ft],
# DUP_AFTER_ENTRYPOINT                             y=[y_pos, y_pos],
# DUP_AFTER_ENTRYPOINT                             z=[z_mid, z_mid],
# DUP_AFTER_ENTRYPOINT                             mode="lines",
# DUP_AFTER_ENTRYPOINT                             line=dict(width=8),
# DUP_AFTER_ENTRYPOINT                             name=("Well" if i == 0 else ""),
# DUP_AFTER_ENTRYPOINT                             showlegend=(i == 0),
# DUP_AFTER_ENTRYPOINT                         )
# DUP_AFTER_ENTRYPOINT                     )
# DUP_AFTER_ENTRYPOINT             except Exception:
# DUP_AFTER_ENTRYPOINT                 pass
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             fig3d.update_layout(
# DUP_AFTER_ENTRYPOINT                 title=f"<b>3D Isosurface: {prop_3d}</b>",
# DUP_AFTER_ENTRYPOINT                 scene=dict(
# DUP_AFTER_ENTRYPOINT                     xaxis_title="X (ft)",
# DUP_AFTER_ENTRYPOINT                     yaxis_title="Y (ft)",
# DUP_AFTER_ENTRYPOINT                     zaxis_title="Z (ft)",
# DUP_AFTER_ENTRYPOINT                     aspectmode="data",
# DUP_AFTER_ENTRYPOINT                 ),
# DUP_AFTER_ENTRYPOINT                 margin=dict(l=0, r=0, b=0, t=40),
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             st.plotly_chart(fig3d, use_container_width=True)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "Slice Viewer":
# DUP_AFTER_ENTRYPOINT         # ======== Slice Viewer tab ========
# DUP_AFTER_ENTRYPOINT         st.header("Slice Viewer")
# DUP_AFTER_ENTRYPOINT         sim_data = st.session_state.get("sim")
# DUP_AFTER_ENTRYPOINT         if sim_data is None and st.session_state.get('kx') is None:
# DUP_AFTER_ENTRYPOINT             st.warning("Please generate rock properties or run a simulation to enable the slice viewer.")
# DUP_AFTER_ENTRYPOINT         else:
# DUP_AFTER_ENTRYPOINT             prop_list = ['Permeability (kx)', 'Permeability (ky)', 'Porosity (ϕ)']
# DUP_AFTER_ENTRYPOINT             if sim_data and sim_data.get('press_matrix') is not None:
# DUP_AFTER_ENTRYPOINT                 prop_list.append('Pressure (psi)')
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             c1, c2 = st.columns(2)
# DUP_AFTER_ENTRYPOINT             with c1:
# DUP_AFTER_ENTRYPOINT                 prop_slice = st.selectbox("Select property:", prop_list)
# DUP_AFTER_ENTRYPOINT             with c2:
# DUP_AFTER_ENTRYPOINT                 plane_slice = st.selectbox("Select plane:", ["k-plane (z, top-down)", "j-plane (y, side-view)", "i-plane (x, end-view)"])
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             data_3d = (
# DUP_AFTER_ENTRYPOINT                 st.session_state.get('kx') if 'kx' in prop_slice
# DUP_AFTER_ENTRYPOINT                 else st.session_state.get('ky') if 'ky' in prop_slice
# DUP_AFTER_ENTRYPOINT                 else st.session_state.get('phi') if 'ϕ' in prop_slice
# DUP_AFTER_ENTRYPOINT                 else sim_data.get('press_matrix')
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             if data_3d is not None:
# DUP_AFTER_ENTRYPOINT                 data_3d = np.asarray(data_3d)
# DUP_AFTER_ENTRYPOINT                 if data_3d.ndim != 3:
# DUP_AFTER_ENTRYPOINT                     st.warning("3D data is not in the expected (nz, ny, nx) shape.")
# DUP_AFTER_ENTRYPOINT                     st.stop()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                 nz, ny, nx = data_3d.shape
# DUP_AFTER_ENTRYPOINT                 if "k-plane" in plane_slice:
# DUP_AFTER_ENTRYPOINT                     idx, axis_name = st.slider("k-index (z-layer)", 0, nz - 1, nz // 2), "k"
# DUP_AFTER_ENTRYPOINT                     data_2d, labels = data_3d[idx, :, :], dict(x="i-index", y="j-index")
# DUP_AFTER_ENTRYPOINT                 elif "j-plane" in plane_slice:
# DUP_AFTER_ENTRYPOINT                     idx, axis_name = st.slider("j-index (y-layer)", 0, ny - 1, ny // 2), "j"
# DUP_AFTER_ENTRYPOINT                     data_2d, labels = data_3d[:, idx, :], dict(x="i-index", y="k-index")
# DUP_AFTER_ENTRYPOINT                 else:
# DUP_AFTER_ENTRYPOINT                     idx, axis_name = st.slider("i-index (x-layer)", 0, nx - 1, nx // 2), "i"
# DUP_AFTER_ENTRYPOINT                     data_2d, labels = data_3d[:, :, idx], dict(x="j-index", y="k-index")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                 fig = px.imshow(data_2d, origin="lower", aspect='equal', labels=labels, color_continuous_scale='viridis')
# DUP_AFTER_ENTRYPOINT                 fig.update_layout(title=f"<b>{prop_slice} @ {axis_name} = {idx}</b>")
# DUP_AFTER_ENTRYPOINT                 st.plotly_chart(fig, use_container_width=True)
# DUP_AFTER_ENTRYPOINT             else:
# DUP_AFTER_ENTRYPOINT                 st.warning(f"Data for '{prop_slice}' not found.")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "Debug":
# DUP_AFTER_ENTRYPOINT         # ======== Debug tab ========
# DUP_AFTER_ENTRYPOINT         st.subheader("Debug")
# DUP_AFTER_ENTRYPOINT         st.json(st.session_state.get("sim") or {})
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "QA / Material Balance":
# DUP_AFTER_ENTRYPOINT         st.header("QA / Material Balance")
# DUP_AFTER_ENTRYPOINT         sim = st.session_state.get("sim")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         if sim is None:
# DUP_AFTER_ENTRYPOINT             st.warning("Run a simulation on the 'Results' tab to view QA plots.")
# DUP_AFTER_ENTRYPOINT             st.stop()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         pavg = sim.get("p_avg_psi") or sim.get("pm_mid_psi")
# DUP_AFTER_ENTRYPOINT         if pavg is None:
# DUP_AFTER_ENTRYPOINT             st.info("Average reservoir pressure time series was not returned by the solver. Cannot generate Material Balance plots.")
# DUP_AFTER_ENTRYPOINT             st.stop()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         if "t" in sim and len(sim["t"]) == len(pavg):
# DUP_AFTER_ENTRYPOINT             fig_p = go.Figure(go.Scatter(x=sim["t"], y=pavg, name="p̄ reservoir (psi)"))
# DUP_AFTER_ENTRYPOINT             fig_p.update_layout(template="plotly_white", title_text="<b>Average Reservoir Pressure</b>", xaxis_title="Time (days)", yaxis_title="Pressure (psi)")
# DUP_AFTER_ENTRYPOINT             st.plotly_chart(fig_p, use_container_width=True, theme=None)
# DUP_AFTER_ENTRYPOINT     
# DUP_AFTER_ENTRYPOINT         if not all(k in sim for k in ("t", "qg", "qo")) or len(sim["t"]) < 2:
# DUP_AFTER_ENTRYPOINT             st.warning("Simulation data is missing required rate arrays ('qg', 'qo') for this analysis.")
# DUP_AFTER_ENTRYPOINT             st.stop()
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         t = np.asarray(sim["t"], float)
# DUP_AFTER_ENTRYPOINT         qg = np.asarray(sim["qg"], float)
# DUP_AFTER_ENTRYPOINT         qo = np.asarray(sim["qo"], float)
# DUP_AFTER_ENTRYPOINT     
# DUP_AFTER_ENTRYPOINT         st.markdown("### Gas Material Balance")
# DUP_AFTER_ENTRYPOINT         # ... (rest of the code is identical and correct)
# DUP_AFTER_ENTRYPOINT         Gp_MMscf = cumulative_trapezoid(qg, t, initial=0.0) / 1e3
# DUP_AFTER_ENTRYPOINT         z_factors = z_factor_approx(np.asarray(pavg), p_init_psi=state["p_init_psi"])
# DUP_AFTER_ENTRYPOINT         p_over_z = np.asarray(pavg) / np.maximum(z_factors, 1e-12)
# DUP_AFTER_ENTRYPOINT         fit_start = max(1, len(Gp_MMscf) // 4)
# DUP_AFTER_ENTRYPOINT         if len(Gp_MMscf[fit_start:]) > 1:
# DUP_AFTER_ENTRYPOINT             slope, intercept, _, _, _ = stats.linregress(Gp_MMscf[fit_start:], p_over_z[fit_start:])
# DUP_AFTER_ENTRYPOINT             giip_bcf = max(0.0, -intercept / slope / 1000.0) if slope != 0 else 0.0
# DUP_AFTER_ENTRYPOINT             sim_eur_g_bcf = sim.get("EUR_g_BCF", np.trapz(qg, t) / 1e6)
# DUP_AFTER_ENTRYPOINT             c1, c2 = st.columns(2)
# DUP_AFTER_ENTRYPOINT             c1.metric("Simulator Gas EUR", f"{sim_eur_g_bcf:.2f} BCF")
# DUP_AFTER_ENTRYPOINT             c2.metric("Material Balance GIIP (from P/Z)", f"{giip_bcf:.2f} BCF", delta=(f"{(giip_bcf - sim_eur_g_bcf)/sim_eur_g_bcf:.1%} vs Sim" if sim_eur_g_bcf > 0 else None))
# DUP_AFTER_ENTRYPOINT             fig_pz_gas = go.Figure()
# DUP_AFTER_ENTRYPOINT             fig_pz_gas.add_trace(go.Scatter(x=Gp_MMscf, y=p_over_z, mode="markers", name="P/Z Data"))
# DUP_AFTER_ENTRYPOINT             x_fit = np.array([0.0, giip_bcf * 1000.0])
# DUP_AFTER_ENTRYPOINT             y_fit = slope * x_fit + intercept
# DUP_AFTER_ENTRYPOINT             fig_pz_gas.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines", name="Linear Extrapolation", line=dict(dash="dash")))
# DUP_AFTER_ENTRYPOINT             fig_pz_gas.update_layout(title="<b>P/Z vs. Cumulative Gas Production</b>", xaxis_title="Gp - Cumulative Gas Production (MMscf)", yaxis_title="P/Z", template="plotly_white")
# DUP_AFTER_ENTRYPOINT             st.plotly_chart(fig_pz_gas, use_container_width=True)
# DUP_AFTER_ENTRYPOINT         else:
# DUP_AFTER_ENTRYPOINT             st.info("Not enough data points for Gas Material Balance plot.")
# DUP_AFTER_ENTRYPOINT     
# DUP_AFTER_ENTRYPOINT         st.markdown("---")
# DUP_AFTER_ENTRYPOINT         st.markdown("### Oil Material Balance")
# DUP_AFTER_ENTRYPOINT         Np_STB = cumulative_trapezoid(qo, t, initial=0.0)
# DUP_AFTER_ENTRYPOINT         Gp_scf = cumulative_trapezoid(qg * 1_000.0, t, initial=0.0)
# DUP_AFTER_ENTRYPOINT         Rp = np.divide(Gp_scf, Np_STB, out=np.zeros_like(Gp_scf), where=Np_STB > 1e-3)
# DUP_AFTER_ENTRYPOINT         Bo = Bo_of_p(pavg, state["pb_psi"], state["Bo_pb_rb_stb"])
# DUP_AFTER_ENTRYPOINT         Rs = Rs_of_p(pavg, state["pb_psi"], state["Rs_pb_scf_stb"])
# DUP_AFTER_ENTRYPOINT         Bg = Bg_of_p(pavg)
# DUP_AFTER_ENTRYPOINT         p_init = state["p_init_psi"]
# DUP_AFTER_ENTRYPOINT         Boi = Bo_of_p(p_init, state["pb_psi"], state["Bo_pb_rb_stb"])
# DUP_AFTER_ENTRYPOINT         Rsi = Rs_of_p(p_init, state["pb_psi"], state["Rs_pb_scf_stb"])
# DUP_AFTER_ENTRYPOINT         F = Np_STB * (Bo + (Rp - Rs) * Bg)
# DUP_AFTER_ENTRYPOINT         Et = (Bo - Boi) + (Rsi - Rs) * Bg
# DUP_AFTER_ENTRYPOINT         fit_start_oil = max(1, len(F) // 4)
# DUP_AFTER_ENTRYPOINT         if len(F[fit_start_oil:]) > 1:
# DUP_AFTER_ENTRYPOINT             slope_oil, _, _, _, _ = stats.linregress(Et[fit_start_oil:], F[fit_start_oil:])
# DUP_AFTER_ENTRYPOINT             ooip_mmstb = max(0.0, slope_oil / 1e6)
# DUP_AFTER_ENTRYPOINT             sim_eur_o_mmstb = sim.get("EUR_o_MMBO", np.trapz(qo, t) / 1e6)
# DUP_AFTER_ENTRYPOINT             rec_factor = (sim_eur_o_mmstb / ooip_mmstb * 100.0) if ooip_mmstb > 0 else 0.0
# DUP_AFTER_ENTRYPOINT             c1, c2, c3 = st.columns(3)
# DUP_AFTER_ENTRYPOINT             c1.metric("Simulator Oil EUR", f"{sim_eur_o_mmstb:.2f} MMSTB")
# DUP_AFTER_ENTRYPOINT             c2.metric("Material Balance OOIP (F vs Et)", f"{ooip_mmstb:.2f} MMSTB")
# DUP_AFTER_ENTRYPOINT             c3.metric("Implied Recovery Factor", f"{rec_factor:.1f}%")
# DUP_AFTER_ENTRYPOINT             fig_mbe_oil = go.Figure()
# DUP_AFTER_ENTRYPOINT             fig_mbe_oil.add_trace(go.Scatter(x=Et, y=F, mode="markers", name="F vs Et Data"))
# DUP_AFTER_ENTRYPOINT             x_fit_oil = np.array([0.0, np.nanmax(Et)])
# DUP_AFTER_ENTRYPOINT             y_fit_oil = slope_oil * x_fit_oil
# DUP_AFTER_ENTRYPOINT             fig_mbe_oil.add_trace(go.Scatter(x=x_fit_oil, y=y_fit_oil, mode="lines", name=f"Slope (OOIP) = {ooip_mmstb:.2f} MMSTB", line=dict(dash="dash")))
# DUP_AFTER_ENTRYPOINT             fig_mbe_oil.update_layout(title="<b>F vs. Et (Havlena–Odeh)</b>", xaxis_title="Et - Total Expansion (rb/STB)", yaxis_title="F - Underground Withdrawal (rb)", template="plotly_white")
# DUP_AFTER_ENTRYPOINT             st.plotly_chart(fig_mbe_oil, use_container_width=True)
# DUP_AFTER_ENTRYPOINT         else:
# DUP_AFTER_ENTRYPOINT             st.info("Not enough data points for Oil Material Balance plot.")
# DUP_AFTER_ENTRYPOINT         
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "Economics":
# DUP_AFTER_ENTRYPOINT         st.header("Financial Model")
# DUP_AFTER_ENTRYPOINT         if st.session_state.get("sim") is None:
# DUP_AFTER_ENTRYPOINT             st.info("Run a simulation on the 'Results' tab first to populate the financial model.")
# DUP_AFTER_ENTRYPOINT         else:
# DUP_AFTER_ENTRYPOINT             sim = st.session_state["sim"]
# DUP_AFTER_ENTRYPOINT             t = np.asarray(sim.get("t", []), float)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             # Handle potentially missing simulation outputs by creating zero-arrays
# DUP_AFTER_ENTRYPOINT             qo_raw = sim.get("qo")
# DUP_AFTER_ENTRYPOINT             qg_raw = sim.get("qg")
# DUP_AFTER_ENTRYPOINT             qw_raw = sim.get("qw")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             qo = np.nan_to_num(np.asarray(qo_raw), nan=0.0) if qo_raw is not None else np.zeros_like(t)
# DUP_AFTER_ENTRYPOINT             qg = np.nan_to_num(np.asarray(qg_raw), nan=0.0) if qg_raw is not None else np.zeros_like(t)
# DUP_AFTER_ENTRYPOINT             qw = np.nan_to_num(np.asarray(qw_raw), nan=0.0) if qw_raw is not None else np.zeros_like(t)
# DUP_AFTER_ENTRYPOINT         
# DUP_AFTER_ENTRYPOINT             st.subheader("Economic Assumptions")
# DUP_AFTER_ENTRYPOINT             c1, c2, c3, c4 = st.columns(4)
# DUP_AFTER_ENTRYPOINT             with c1: capex = st.number_input("CAPEX ($MM)", 1.0, 100.0, 15.0, 0.5, key="econ_capex") * 1e6
# DUP_AFTER_ENTRYPOINT             with c2: oil_price = st.number_input("Oil price ($/bbl)", 0.0, 500.0, 75.0, 1.0, key="econ_oil_price")
# DUP_AFTER_ENTRYPOINT             with c3: gas_price = st.number_input("Gas price ($/Mcf)", 0.0, 50.0, 2.50, 0.1, key="econ_gas_price")
# DUP_AFTER_ENTRYPOINT             with c4: disc_rate = st.number_input("Discount rate (fraction)", 0.0, 1.0, 0.10, 0.01, key="econ_disc")
# DUP_AFTER_ENTRYPOINT             c1, c2, c3, c4 = st.columns(4)
# DUP_AFTER_ENTRYPOINT             with c1: royalty = st.number_input("Royalty (fraction)", 0.0, 0.99, 0.20, 0.01, key="econ_royalty")
# DUP_AFTER_ENTRYPOINT             with c2: tax = st.number_input("Severance tax (fraction)", 0.0, 0.99, 0.045, 0.005, key="econ_tax")
# DUP_AFTER_ENTRYPOINT             with c3: opex_bpd = st.number_input("OPEX ($/bbl liquids)", 0.0, 200.0, 6.0, 0.5, key="econ_opex")
# DUP_AFTER_ENTRYPOINT             with c4: wd_cost = st.number_input("Water disposal ($/bbl)", 0.0, 50.0, 1.5, 0.1, key="econ_wd")
# DUP_AFTER_ENTRYPOINT         
# DUP_AFTER_ENTRYPOINT             # --- Robust Yearly Cash Flow Calculation ---
# DUP_AFTER_ENTRYPOINT             df_yearly = pd.DataFrame()
# DUP_AFTER_ENTRYPOINT             if len(t) > 1:
# DUP_AFTER_ENTRYPOINT                 df = pd.DataFrame({'days': t, 'oil_stb_d': qo, 'gas_mscf_d': qg, 'water_stb_d': qw})
# DUP_AFTER_ENTRYPOINT                 df['year'] = (df['days'] / 365.25).astype(int)
# DUP_AFTER_ENTRYPOINT             
# DUP_AFTER_ENTRYPOINT                 yearly_data = []
# DUP_AFTER_ENTRYPOINT                 for year, group in df.groupby('year'):
# DUP_AFTER_ENTRYPOINT                     days_in_year = group['days'].values
# DUP_AFTER_ENTRYPOINT                     if len(days_in_year) > 1:
# DUP_AFTER_ENTRYPOINT                         yearly_data.append({
# DUP_AFTER_ENTRYPOINT                             'year': year,
# DUP_AFTER_ENTRYPOINT                             'oil_stb': np.trapz(group['oil_stb_d'].values, days_in_year),
# DUP_AFTER_ENTRYPOINT                             'gas_mscf': np.trapz(group['gas_mscf_d'].values, days_in_year),
# DUP_AFTER_ENTRYPOINT                             'water_stb': np.trapz(group['water_stb_d'].values, days_in_year),
# DUP_AFTER_ENTRYPOINT                         })
# DUP_AFTER_ENTRYPOINT                 if yearly_data:
# DUP_AFTER_ENTRYPOINT                     df_yearly = pd.DataFrame(yearly_data)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             if not df_yearly.empty:
# DUP_AFTER_ENTRYPOINT                 df_yearly['Revenue'] = (df_yearly['oil_stb'] * oil_price) + (df_yearly['gas_mscf'] * gas_price)
# DUP_AFTER_ENTRYPOINT                 df_yearly['Royalty'] = df_yearly['Revenue'] * royalty
# DUP_AFTER_ENTRYPOINT                 df_yearly['Taxes'] = (df_yearly['Revenue'] - df_yearly['Royalty']) * tax
# DUP_AFTER_ENTRYPOINT                 df_yearly['OPEX'] = (df_yearly['oil_stb'] + df_yearly['water_stb']) * opex_bpd + (df_yearly['water_stb'] * wd_cost)
# DUP_AFTER_ENTRYPOINT                 df_yearly['Net Cash Flow'] = df_yearly['Revenue'] - df_yearly['Royalty'] - df_yearly['Taxes'] - df_yearly['OPEX']
# DUP_AFTER_ENTRYPOINT                 cash_flows = [-capex] + df_yearly['Net Cash Flow'].tolist()
# DUP_AFTER_ENTRYPOINT             else:
# DUP_AFTER_ENTRYPOINT                 cash_flows = [-capex]
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             # --- Financial Metrics ---
# DUP_AFTER_ENTRYPOINT             npv = npf.npv(disc_rate, cash_flows) if cash_flows else 0
# DUP_AFTER_ENTRYPOINT             try:
# DUP_AFTER_ENTRYPOINT                 irr = npf.irr(cash_flows) if len(cash_flows) > 1 else np.nan
# DUP_AFTER_ENTRYPOINT             except ValueError:
# DUP_AFTER_ENTRYPOINT                 irr = np.nan
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             # --- Payout Calculation & Final DataFrame for Display ---
# DUP_AFTER_ENTRYPOINT             display_df = pd.DataFrame({'year': range(-1, len(cash_flows)-1), 'Net Cash Flow': cash_flows})
# DUP_AFTER_ENTRYPOINT             display_df['Cumulative Cash Flow'] = display_df['Net Cash Flow'].cumsum()
# DUP_AFTER_ENTRYPOINT         
# DUP_AFTER_ENTRYPOINT             payout_period = np.nan
# DUP_AFTER_ENTRYPOINT             if (display_df['Cumulative Cash Flow'] > 0).any():
# DUP_AFTER_ENTRYPOINT                 first_positive_idx_loc = display_df['Cumulative Cash Flow'].gt(0).idxmax()
# DUP_AFTER_ENTRYPOINT                 if first_positive_idx_loc > 0 and display_df['Cumulative Cash Flow'].iloc[first_positive_idx_loc-1] < 0:
# DUP_AFTER_ENTRYPOINT                     last_neg_cum_flow = display_df['Cumulative Cash Flow'].iloc[first_positive_idx_loc - 1]
# DUP_AFTER_ENTRYPOINT                     current_year_ncf = display_df['Net Cash Flow'].iloc[first_positive_idx_loc]
# DUP_AFTER_ENTRYPOINT                     payout_period = (display_df['year'].iloc[first_positive_idx_loc-1]) + (-last_neg_cum_flow / current_year_ncf if current_year_ncf > 0 else np.inf)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             st.subheader("Key Financial Metrics")
# DUP_AFTER_ENTRYPOINT             m1, m2, m3 = st.columns(3)
# DUP_AFTER_ENTRYPOINT             m1.metric("NPV", f"${npv/1e6:,.2f} MM", help="Net Present Value at the specified discount rate.")
# DUP_AFTER_ENTRYPOINT             m2.metric("IRR", f"{irr:.1%}" if pd.notna(irr) and np.isfinite(irr) else "N/A", help="Internal Rate of Return.")
# DUP_AFTER_ENTRYPOINT             m3.metric("Payout Period (Years)", f"{payout_period:.2f}" if pd.notna(payout_period) else "N/A", help="Time until initial investment is recovered.")
# DUP_AFTER_ENTRYPOINT         
# DUP_AFTER_ENTRYPOINT             st.subheader("Cash Flow Details")
# DUP_AFTER_ENTRYPOINT             c1, c2 = st.columns(2)
# DUP_AFTER_ENTRYPOINT             with c1:
# DUP_AFTER_ENTRYPOINT                 fig_ncf = px.bar(display_df, x='year', y='Net Cash Flow', title="<b>Yearly Net Cash Flow</b>", labels={'year':'Year', 'Net Cash Flow':'Cash Flow ($)'})
# DUP_AFTER_ENTRYPOINT                 fig_ncf.update_layout(template='plotly_white', bargap=0.2)
# DUP_AFTER_ENTRYPOINT                 st.plotly_chart(fig_ncf, use_container_width=True)
# DUP_AFTER_ENTRYPOINT             with c2:
# DUP_AFTER_ENTRYPOINT                 fig_cum = px.line(display_df, x='year', y='Cumulative Cash Flow', title="<b>Cumulative Cash Flow</b>", markers=True, labels={'year':'Year', 'Cumulative Cash Flow':'Cash Flow ($)'})
# DUP_AFTER_ENTRYPOINT                 fig_cum.add_hline(y=0, line_dash="dash", line_color="red")
# DUP_AFTER_ENTRYPOINT                 fig_cum.update_layout(template='plotly_white')
# DUP_AFTER_ENTRYPOINT                 st.plotly_chart(fig_cum, use_container_width=True)
# DUP_AFTER_ENTRYPOINT         
# DUP_AFTER_ENTRYPOINT             # --- DEFINITIVE FIX FOR TABLE DISPLAY ---
# DUP_AFTER_ENTRYPOINT             st.markdown("##### Yearly Cash Flow Table")
# DUP_AFTER_ENTRYPOINT             # Start with the financial-only dataframe
# DUP_AFTER_ENTRYPOINT             final_table = display_df.copy()
# DUP_AFTER_ENTRYPOINT         
# DUP_AFTER_ENTRYPOINT             # If there is production data, merge it in.
# DUP_AFTER_ENTRYPOINT             if not df_yearly.empty:
# DUP_AFTER_ENTRYPOINT                 # We only need the production and revenue columns from df_yearly
# DUP_AFTER_ENTRYPOINT                 cols_to_merge = ['year', 'oil_stb', 'gas_mscf', 'water_stb', 'Revenue', 'Royalty', 'Taxes', 'OPEX']
# DUP_AFTER_ENTRYPOINT                 final_table = pd.merge(final_table, df_yearly[cols_to_merge], on='year', how='left')
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             # Ensure all required columns exist, filling with 0 if they don't
# DUP_AFTER_ENTRYPOINT             display_cols = ['year', 'oil_stb', 'gas_mscf', 'water_stb', 'Revenue', 'Royalty', 'Taxes', 'OPEX', 'Net Cash Flow', 'Cumulative Cash Flow']
# DUP_AFTER_ENTRYPOINT             for col in display_cols:
# DUP_AFTER_ENTRYPOINT                 if col not in final_table.columns:
# DUP_AFTER_ENTRYPOINT                     final_table[col] = 0
# DUP_AFTER_ENTRYPOINT         
# DUP_AFTER_ENTRYPOINT             # Reorder columns to the desired display order and fill any remaining NaNs
# DUP_AFTER_ENTRYPOINT             final_table = final_table[display_cols].fillna(0)
# DUP_AFTER_ENTRYPOINT         
# DUP_AFTER_ENTRYPOINT             st.dataframe(final_table.style.format({
# DUP_AFTER_ENTRYPOINT                 'oil_stb': '{:,.0f}', 'gas_mscf': '{:,.0f}', 'water_stb': '{:,.0f}',
# DUP_AFTER_ENTRYPOINT                 'Revenue': '${:,.0f}', 'Royalty': '${:,.0f}', 'Taxes': '${:,.0f}',
# DUP_AFTER_ENTRYPOINT                 'OPEX': '${:,.0f}', 'Net Cash Flow': '${:,.0f}', 'Cumulative Cash Flow': '${:,.0f}'
# DUP_AFTER_ENTRYPOINT             }), use_container_width=True)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "EUR vs Lateral Length":
# DUP_AFTER_ENTRYPOINT         st.header("EUR vs Lateral Length Sensitivity")
# DUP_AFTER_ENTRYPOINT         st.info("This feature is not yet implemented. It will allow you to run multiple simulations to see how EUR changes with lateral length.")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "Field Match (CSV)":
# DUP_AFTER_ENTRYPOINT         st.header("Field Match (CSV)")
# DUP_AFTER_ENTRYPOINT         c1, c2 = st.columns([3, 1])
# DUP_AFTER_ENTRYPOINT         with c1:
# DUP_AFTER_ENTRYPOINT             uploaded_file = st.file_uploader("Upload field production data (CSV)", type="csv")
# DUP_AFTER_ENTRYPOINT             if uploaded_file:
# DUP_AFTER_ENTRYPOINT                 try:
# DUP_AFTER_ENTRYPOINT                     st.session_state.field_data_match = pd.read_csv(uploaded_file)
# DUP_AFTER_ENTRYPOINT                 except Exception as e:
# DUP_AFTER_ENTRYPOINT                     st.error(f"Error reading CSV file: {e}")
# DUP_AFTER_ENTRYPOINT         with c2:
# DUP_AFTER_ENTRYPOINT             st.write("")
# DUP_AFTER_ENTRYPOINT             st.write("")
# DUP_AFTER_ENTRYPOINT             if st.button("Load Demo Data", use_container_width=True):
# DUP_AFTER_ENTRYPOINT                 rng = np.random.default_rng(123)
# DUP_AFTER_ENTRYPOINT                 days = np.arange(0, 731, 15)
# DUP_AFTER_ENTRYPOINT                 oil_rate = 950 * np.exp(-days / 400) + rng.uniform(-25, 25, size=days.shape)
# DUP_AFTER_ENTRYPOINT                 gas_rate = 8000 * np.exp(-days / 500) + rng.uniform(-200, 200, size=days.shape)
# DUP_AFTER_ENTRYPOINT                 oil_rate = np.clip(oil_rate, 0, None)
# DUP_AFTER_ENTRYPOINT                 gas_rate = np.clip(gas_rate, 0, None)
# DUP_AFTER_ENTRYPOINT                 demo_df = pd.DataFrame({"Day": days, "Gas_Rate_Mscfd": gas_rate, "Oil_Rate_STBpd": oil_rate})
# DUP_AFTER_ENTRYPOINT                 st.session_state.field_data_match = demo_df
# DUP_AFTER_ENTRYPOINT                 st.success("Demo production data loaded successfully!")
# DUP_AFTER_ENTRYPOINT         if 'field_data_match' in st.session_state:
# DUP_AFTER_ENTRYPOINT             st.markdown("---")
# DUP_AFTER_ENTRYPOINT             st.markdown("#### Loaded Production Data (first 5 rows)")
# DUP_AFTER_ENTRYPOINT             st.dataframe(st.session_state.field_data_match.head(), use_container_width=True)
# DUP_AFTER_ENTRYPOINT             if st.session_state.get("sim") is not None and st.session_state.get("field_data_match") is not None:
# DUP_AFTER_ENTRYPOINT                 sim_data = st.session_state.sim
# DUP_AFTER_ENTRYPOINT                 field_data = st.session_state.field_data_match
# DUP_AFTER_ENTRYPOINT                 fig_match = go.Figure()
# DUP_AFTER_ENTRYPOINT                 fig_match.add_trace(go.Scatter(x=sim_data['t'], y=sim_data['qg'], mode='lines', name='Simulated Gas', line=dict(color="#d62728")))
# DUP_AFTER_ENTRYPOINT                 fig_match.add_trace(go.Scatter(x=sim_data['t'], y=sim_data['qo'], mode='lines', name='Simulated Oil', line=dict(color="#2ca02c"), yaxis="y2"))
# DUP_AFTER_ENTRYPOINT                 if {'Day', 'Gas_Rate_Mscfd'}.issubset(field_data.columns):
# DUP_AFTER_ENTRYPOINT                     fig_match.add_trace(go.Scatter(x=field_data['Day'], y=field_data['Gas_Rate_Mscfd'], mode='markers', name='Field Gas', marker=dict(color="#d62728", symbol='cross')))
# DUP_AFTER_ENTRYPOINT                 if {'Day', 'Oil_Rate_STBpd'}.issubset(field_data.columns):
# DUP_AFTER_ENTRYPOINT                     fig_match.add_trace(go.Scatter(x=field_data['Day'], y=field_data['Oil_Rate_STBpd'], mode='markers', name='Field Oil', marker=dict(color="#2ca02c", symbol='cross'), yaxis="y2"))
# DUP_AFTER_ENTRYPOINT                 layout_config = semi_log_layout("Field Match: Simulation vs. Actual", yaxis="Gas Rate (Mscf/d)")
# DUP_AFTER_ENTRYPOINT                 layout_config.update(
# DUP_AFTER_ENTRYPOINT                     yaxis=dict(title="Gas Rate (Mscf/d)"),
# DUP_AFTER_ENTRYPOINT                     yaxis2=dict(title="Oil Rate (STB/d)", overlaying="y", side="right", showgrid=False),
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT                 fig_match.update_layout(layout_config)
# DUP_AFTER_ENTRYPOINT                 st.plotly_chart(fig_match, use_container_width=True, theme="streamlit")
# DUP_AFTER_ENTRYPOINT                 with st.expander("Click for details"):
# DUP_AFTER_ENTRYPOINT                     st.markdown(
# DUP_AFTER_ENTRYPOINT                         "This plot compares simulated production (solid lines) to historical data ('x' markers). "
# DUP_AFTER_ENTRYPOINT                         "Tune sidebar parameters and re-run until the match is reasonable; then use the calibrated model for forecasting."
# DUP_AFTER_ENTRYPOINT                     )
# DUP_AFTER_ENTRYPOINT             elif st.session_state.get("sim") is None and st.session_state.get("field_data_match") is not None:
# DUP_AFTER_ENTRYPOINT                 st.info("Demo/Field data loaded. Run a simulation on the 'Results' tab to view the comparison plot.")
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "Automated Match":
# DUP_AFTER_ENTRYPOINT         st.header("Automated History Matching")
# DUP_AFTER_ENTRYPOINT         st.info("This module uses a genetic algorithm (Differential Evolution) to automatically find the best parameters to match historical data.")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT         # --- 1. Load Data ---
# DUP_AFTER_ENTRYPOINT         with st.expander("1. Load Historical Data", expanded=True):
# DUP_AFTER_ENTRYPOINT             uploaded_file_match = st.file_uploader("Upload field production CSV", type="csv", key="auto_match_uploader")
# DUP_AFTER_ENTRYPOINT             if uploaded_file_match:
# DUP_AFTER_ENTRYPOINT                 try:
# DUP_AFTER_ENTRYPOINT                     st.session_state.field_data_auto_match = pd.read_csv(uploaded_file_match)
# DUP_AFTER_ENTRYPOINT                     st.success("File loaded successfully.")
# DUP_AFTER_ENTRYPOINT                 except Exception as e:
# DUP_AFTER_ENTRYPOINT                     st.error(f"Error reading CSV file: {e}")
# DUP_AFTER_ENTRYPOINT         
# DUP_AFTER_ENTRYPOINT             field_data = st.session_state.get("field_data_auto_match")
# DUP_AFTER_ENTRYPOINT             if field_data is not None:
# DUP_AFTER_ENTRYPOINT                 st.dataframe(field_data.head())
# DUP_AFTER_ENTRYPOINT                 # Validate columns
# DUP_AFTER_ENTRYPOINT                 if not ({'Day', 'Oil_Rate_STBpd'}.issubset(field_data.columns) or {'Day', 'Gas_Rate_Mscfd'}.issubset(field_data.columns)):
# DUP_AFTER_ENTRYPOINT                     st.error("CSV must contain 'Day' and at least one of 'Oil_Rate_STBpd' or 'Gas_Rate_Mscfd'.")
# DUP_AFTER_ENTRYPOINT                     field_data = None # Invalidate data
# DUP_AFTER_ENTRYPOINT     
# DUP_AFTER_ENTRYPOINT         if field_data is not None:
# DUP_AFTER_ENTRYPOINT             # --- 2. Select Parameters ---
# DUP_AFTER_ENTRYPOINT             with st.expander("2. Select Parameters and Define Bounds", expanded=True):
# DUP_AFTER_ENTRYPOINT                 param_options = {'xf_ft': (100.0, 500.0), 'hf_ft': (50.0, 300.0), 'k_stdev': (0.0, 0.2), 'pad_interf': (0.0, 0.8), 'p_init_psi': (3000.0, 8000.0)}
# DUP_AFTER_ENTRYPOINT                 selected_params = st.multiselect("Parameters to vary:", options=list(param_options.keys()), default=['xf_ft', 'k_stdev'])
# DUP_AFTER_ENTRYPOINT             
# DUP_AFTER_ENTRYPOINT                 bounds, valid_bounds = {}, True
# DUP_AFTER_ENTRYPOINT                 if selected_params:
# DUP_AFTER_ENTRYPOINT                     cols = st.columns(len(selected_params))
# DUP_AFTER_ENTRYPOINT                     for i, param in enumerate(selected_params):
# DUP_AFTER_ENTRYPOINT                         with cols[i]:
# DUP_AFTER_ENTRYPOINT                             st.markdown(f"**{param}**")
# DUP_AFTER_ENTRYPOINT                             min_val, max_val = st.slider("Range", param_options[param][0], param_options[param][1], (param_options[param][0], param_options[param][1]), key=f"range_{param}")
# DUP_AFTER_ENTRYPOINT                             if min_val >= max_val:
# DUP_AFTER_ENTRYPOINT                                 st.error("Min must be less than Max.")
# DUP_AFTER_ENTRYPOINT                                 valid_bounds = False
# DUP_AFTER_ENTRYPOINT                             bounds[param] = (min_val, max_val)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT             # --- 3. Configure and Run ---
# DUP_AFTER_ENTRYPOINT             with st.expander("3. Configure and Run Optimization", expanded=True):
# DUP_AFTER_ENTRYPOINT                 error_metric = st.selectbox("Error Metric to Minimize", ["RMSE (Oil)", "RMSE (Gas)", "RMSE (Combined)"])
# DUP_AFTER_ENTRYPOINT                 max_iter = st.slider("Max Iterations", 5, 50, 15)
# DUP_AFTER_ENTRYPOINT             
# DUP_AFTER_ENTRYPOINT                 run_auto_match = st.button("🚀 Run Automated Match", use_container_width=True, type="primary", disabled=not (valid_bounds and selected_params))
# DUP_AFTER_ENTRYPOINT             
# DUP_AFTER_ENTRYPOINT                 if run_auto_match:
# DUP_AFTER_ENTRYPOINT                     # Objective function for the optimizer
# DUP_AFTER_ENTRYPOINT                     def objective_function(params, param_names, base_state, field_data, error_metric):
# DUP_AFTER_ENTRYPOINT                         temp_state = base_state.copy()
# DUP_AFTER_ENTRYPOINT                         for name, value in zip(param_names, params):
# DUP_AFTER_ENTRYPOINT                             temp_state[name] = value
# DUP_AFTER_ENTRYPOINT                     
# DUP_AFTER_ENTRYPOINT                         sim_result = fallback_fast_solver(temp_state, np.random.default_rng())
# DUP_AFTER_ENTRYPOINT                     
# DUP_AFTER_ENTRYPOINT                         t_sim, qo_sim, qg_sim = sim_result['t'], sim_result['qo'], sim_result['qg']
# DUP_AFTER_ENTRYPOINT                         t_field = field_data['Day'].values
# DUP_AFTER_ENTRYPOINT                     
# DUP_AFTER_ENTRYPOINT                         f_qo = interp1d(t_sim, qo_sim, bounds_error=False, fill_value="extrapolate")
# DUP_AFTER_ENTRYPOINT                         f_qg = interp1d(t_sim, qg_sim, bounds_error=False, fill_value="extrapolate")
# DUP_AFTER_ENTRYPOINT                         qo_sim_interp, qg_sim_interp = f_qo(t_field), f_qg(t_field)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                         error_oil, error_gas = 0, 0
# DUP_AFTER_ENTRYPOINT                         if 'Oil_Rate_STBpd' in field_data.columns:
# DUP_AFTER_ENTRYPOINT                             qo_field = field_data['Oil_Rate_STBpd'].values
# DUP_AFTER_ENTRYPOINT                             error_oil = np.sqrt(np.mean((qo_sim_interp - qo_field)**2))
# DUP_AFTER_ENTRYPOINT                         if 'Gas_Rate_Mscfd' in field_data.columns:
# DUP_AFTER_ENTRYPOINT                             qg_field = field_data['Gas_Rate_Mscfd'].values
# DUP_AFTER_ENTRYPOINT                             error_gas = np.sqrt(np.mean((qg_sim_interp - qg_field)**2))
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                         if "Combined" in error_metric: return error_oil + error_gas
# DUP_AFTER_ENTRYPOINT                         elif "Oil" in error_metric: return error_oil
# DUP_AFTER_ENTRYPOINT                         else: return error_gas
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT                     with st.spinner("Running optimization... This may take several minutes."):
# DUP_AFTER_ENTRYPOINT                         param_names, bounds_list = list(bounds.keys()), [bounds[p] for p in bounds.keys()]
# DUP_AFTER_ENTRYPOINT                         result = differential_evolution(objective_function, bounds=bounds_list, args=(param_names, state.copy(), field_data, error_metric), maxiter=max_iter, disp=True)
# DUP_AFTER_ENTRYPOINT                         st.session_state.auto_match_result = result
# DUP_AFTER_ENTRYPOINT         
# DUP_AFTER_ENTRYPOINT             # --- 4. Display Results ---
# DUP_AFTER_ENTRYPOINT             if 'auto_match_result' in st.session_state:
# DUP_AFTER_ENTRYPOINT                 st.markdown("---")
# DUP_AFTER_ENTRYPOINT                 st.header("Optimization Results")
# DUP_AFTER_ENTRYPOINT                 result = st.session_state.auto_match_result
# DUP_AFTER_ENTRYPOINT             
# DUP_AFTER_ENTRYPOINT                 c1, c2 = st.columns([1,2])
# DUP_AFTER_ENTRYPOINT                 with c1:
# DUP_AFTER_ENTRYPOINT                     st.metric("Final Error (RMSE)", f"{result.fun:.2f}")
# DUP_AFTER_ENTRYPOINT                     st.markdown("##### Best-Fit Parameters:")
# DUP_AFTER_ENTRYPOINT                     best_params_df = pd.DataFrame({'Parameter': list(bounds.keys()), 'Value': result.x})
# DUP_AFTER_ENTRYPOINT                     st.table(best_params_df.style.format({'Value': '{:.2f}'}))
# DUP_AFTER_ENTRYPOINT             
# DUP_AFTER_ENTRYPOINT                 with c2:
# DUP_AFTER_ENTRYPOINT                     # Run one final simulation with the best parameters
# DUP_AFTER_ENTRYPOINT                     best_state = state.copy()
# DUP_AFTER_ENTRYPOINT                     for name, value in zip(list(bounds.keys()), result.x):
# DUP_AFTER_ENTRYPOINT                         best_state[name] = value
# DUP_AFTER_ENTRYPOINT                 
# DUP_AFTER_ENTRYPOINT                     final_sim = fallback_fast_solver(best_state, np.random.default_rng())
# DUP_AFTER_ENTRYPOINT                 
# DUP_AFTER_ENTRYPOINT                     fig_match = make_subplots(specs=[[{"secondary_y": True}]])
# DUP_AFTER_ENTRYPOINT                     if 'Gas_Rate_Mscfd' in field_data.columns:
# DUP_AFTER_ENTRYPOINT                         fig_match.add_trace(go.Scatter(x=field_data['Day'], y=field_data['Gas_Rate_Mscfd'], mode='markers', name='Field Gas', marker=dict(color=COLOR_GAS, symbol='cross')), secondary_y=False)
# DUP_AFTER_ENTRYPOINT                         fig_match.add_trace(go.Scatter(x=final_sim['t'], y=final_sim['qg'], mode='lines', name='Best Match Gas', line=dict(color=COLOR_GAS, width=3)), secondary_y=False)
# DUP_AFTER_ENTRYPOINT                     if 'Oil_Rate_STBpd' in field_data.columns:
# DUP_AFTER_ENTRYPOINT                         fig_match.add_trace(go.Scatter(x=field_data['Day'], y=field_data['Oil_Rate_STBpd'], mode='markers', name='Field Oil', marker=dict(color=COLOR_OIL, symbol='x')), secondary_y=True)
# DUP_AFTER_ENTRYPOINT                         fig_match.add_trace(go.Scatter(x=final_sim['t'], y=final_sim['qo'], mode='lines', name='Best Match Oil', line=dict(color=COLOR_OIL, width=3)), secondary_y=True)
# DUP_AFTER_ENTRYPOINT                 
# DUP_AFTER_ENTRYPOINT                     fig_match.update_layout(title="<b>Final History Match</b>", template="plotly_white", xaxis_title="Time (days)")
# DUP_AFTER_ENTRYPOINT                     st.plotly_chart(fig_match, use_container_width=True)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "Uncertainty & Monte Carlo":
# DUP_AFTER_ENTRYPOINT         st.header("Uncertainty & Monte Carlo")
# DUP_AFTER_ENTRYPOINT         p1, p2, p3 = st.columns(3)
# DUP_AFTER_ENTRYPOINT         with p1:
# DUP_AFTER_ENTRYPOINT             uc_k = st.checkbox("k stdev", True)
# DUP_AFTER_ENTRYPOINT             k_mean = st.slider("k_stdev Mean", 0.0, 0.2, state['k_stdev'], 0.01)
# DUP_AFTER_ENTRYPOINT             k_std = st.slider("k_stdev Stdev", 0.0, 0.1, 0.02, 0.005)
# DUP_AFTER_ENTRYPOINT         with p2:
# DUP_AFTER_ENTRYPOINT             uc_xf = st.checkbox("xf_ft", True)
# DUP_AFTER_ENTRYPOINT             xf_mean = st.slider("xf_ft Mean (ft)", 100.0, 500.0, state['xf_ft'], 10.0)
# DUP_AFTER_ENTRYPOINT             xf_std = st.slider("xf_ft Stdev (ft)", 0.0, 100.0, 30.0, 5.0)
# DUP_AFTER_ENTRYPOINT         with p3:
# DUP_AFTER_ENTRYPOINT             uc_int = st.checkbox("pad_interf", False)
# DUP_AFTER_ENTRYPOINT             int_min = st.slider("Interference Min", 0.0, 0.8, state['pad_interf'], 0.01)
# DUP_AFTER_ENTRYPOINT             int_max = st.slider("Interference Max", 0.0, 0.8, 0.5, 0.01)
# DUP_AFTER_ENTRYPOINT         num_runs = st.number_input("Number of Monte Carlo runs", 10, 500, 50, 10)
# DUP_AFTER_ENTRYPOINT         if st.button("Run Monte Carlo Simulation", key="run_mc"):
# DUP_AFTER_ENTRYPOINT             qg_runs, qo_runs, eur_g, eur_o = [], [], [], []
# DUP_AFTER_ENTRYPOINT             bar_mc = st.progress(0, text="Running Monte Carlo simulation...")
# DUP_AFTER_ENTRYPOINT             base_state = state.copy()
# DUP_AFTER_ENTRYPOINT             rng_mc = np.random.default_rng(st.session_state.rng_seed + 1)
# DUP_AFTER_ENTRYPOINT             for i in range(num_runs):
# DUP_AFTER_ENTRYPOINT                 temp_state = base_state.copy()
# DUP_AFTER_ENTRYPOINT                 if uc_k:
# DUP_AFTER_ENTRYPOINT                     temp_state['k_stdev'] = stats.truncnorm.rvs(
# DUP_AFTER_ENTRYPOINT                         (0 - k_mean) / k_std, (0.2 - k_mean) / k_std, loc=k_mean, scale=k_std, random_state=rng_mc
# DUP_AFTER_ENTRYPOINT                     )
# DUP_AFTER_ENTRYPOINT                 if uc_xf:
# DUP_AFTER_ENTRYPOINT                     temp_state['xf_ft'] = stats.truncnorm.rvs(
# DUP_AFTER_ENTRYPOINT                         (100 - xf_mean) / xf_std, (500 - xf_mean) / xf_std, loc=xf_mean, scale=xf_std, random_state=rng_mc
# DUP_AFTER_ENTRYPOINT                     )
# DUP_AFTER_ENTRYPOINT                 if uc_int:
# DUP_AFTER_ENTRYPOINT                     temp_state['pad_interf'] = stats.uniform.rvs(
# DUP_AFTER_ENTRYPOINT                         loc=int_min, scale=int_max - int_min, random_state=rng_mc
# DUP_AFTER_ENTRYPOINT                     )
# DUP_AFTER_ENTRYPOINT                 res = fallback_fast_solver(temp_state, rng_mc)
# DUP_AFTER_ENTRYPOINT                 qg_runs.append(res['qg'])
# DUP_AFTER_ENTRYPOINT                 qo_runs.append(res['qo'])
# DUP_AFTER_ENTRYPOINT                 eur_g.append(res['EUR_g_BCF'])
# DUP_AFTER_ENTRYPOINT                 eur_o.append(res['EUR_o_MMBO'])
# DUP_AFTER_ENTRYPOINT                 bar_mc.progress((i + 1) / num_runs, f"Run {i+1}/{num_runs}")
# DUP_AFTER_ENTRYPOINT             st.session_state.mc_results = {
# DUP_AFTER_ENTRYPOINT                 't': res['t'],
# DUP_AFTER_ENTRYPOINT                 'qg_runs': np.array(qg_runs),
# DUP_AFTER_ENTRYPOINT                 'qo_runs': np.array(qo_runs),
# DUP_AFTER_ENTRYPOINT                 'eur_g': np.array(eur_g),
# DUP_AFTER_ENTRYPOINT                 'eur_o': np.array(eur_o),
# DUP_AFTER_ENTRYPOINT             }
# DUP_AFTER_ENTRYPOINT             bar_mc.empty()
# DUP_AFTER_ENTRYPOINT         if 'mc_results' in st.session_state:
# DUP_AFTER_ENTRYPOINT             mc = st.session_state.mc_results
# DUP_AFTER_ENTRYPOINT             p10_g, p50_g, p90_g = np.percentile(mc['qg_runs'], [90, 50, 10], axis=0)
# DUP_AFTER_ENTRYPOINT             p10_o, p50_o, p90_o = np.percentile(mc['qo_runs'], [90, 50, 10], axis=0)
# DUP_AFTER_ENTRYPOINT             c1, c2 = st.columns(2)
# DUP_AFTER_ENTRYPOINT             with c1:
# DUP_AFTER_ENTRYPOINT                 fig = go.Figure([
# DUP_AFTER_ENTRYPOINT                     go.Scatter(x=mc['t'], y=p90_g, fill=None, mode='lines', line_color='lightgrey', name='P10'),
# DUP_AFTER_ENTRYPOINT                     go.Scatter(x=mc['t'], y=p10_g, fill='tonexty', mode='lines', line_color='lightgrey', name='P90'),
# DUP_AFTER_ENTRYPOINT                     go.Scatter(x=mc['t'], y=p50_g, mode='lines', line_color='red', name='P50'),
# DUP_AFTER_ENTRYPOINT                 ])
# DUP_AFTER_ENTRYPOINT                 st.plotly_chart(fig.update_layout(**semi_log_layout("Gas Rate Probabilistic Forecast", yaxis="Gas Rate (Mscf/d)")), use_container_width=True, theme="streamlit")
# DUP_AFTER_ENTRYPOINT                 st.plotly_chart(
# DUP_AFTER_ENTRYPOINT                     px.histogram(x=mc['eur_g'], nbins=30, labels={'x': 'Gas EUR (BCF)'}).update_layout(
# DUP_AFTER_ENTRYPOINT                         title="<b>Distribution of Gas EUR</b>", template="plotly_white"
# DUP_AFTER_ENTRYPOINT                     ), use_container_width=True, theme="streamlit"
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT             with c2:
# DUP_AFTER_ENTRYPOINT                 fig = go.Figure([
# DUP_AFTER_ENTRYPOINT                     go.Scatter(x=mc['t'], y=p90_o, fill=None, mode='lines', line_color='lightgreen', name='P10'),
# DUP_AFTER_ENTRYPOINT                     go.Scatter(x=mc['t'], y=p10_o, fill='tonexty', mode='lines', line_color='lightgreen', name='P90'),
# DUP_AFTER_ENTRYPOINT                     go.Scatter(x=mc['t'], y=p50_o, mode='lines', line_color='green', name='P50'),
# DUP_AFTER_ENTRYPOINT                 ])
# DUP_AFTER_ENTRYPOINT                 st.plotly_chart(fig.update_layout(**semi_log_layout("Oil Rate Probabilistic Forecast", yaxis="Oil Rate (STB/d)")), use_container_width=True, theme="streamlit")
# DUP_AFTER_ENTRYPOINT                 st.plotly_chart(
# DUP_AFTER_ENTRYPOINT                     px.histogram(x=mc['eur_o'], nbins=30, labels={'x': 'Oil EUR (MMSTB)'}, color_discrete_sequence=['green']).update_layout(
# DUP_AFTER_ENTRYPOINT                         title="<b>Distribution of Oil EUR</b>", template="plotly_white"
# DUP_AFTER_ENTRYPOINT                     ), use_container_width=True, theme="streamlit"
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "Well Placement Optimization":
# DUP_AFTER_ENTRYPOINT         st.header("Well Placement Optimization")
# DUP_AFTER_ENTRYPOINT         st.markdown("#### 1. General Parameters")
# DUP_AFTER_ENTRYPOINT         c1_opt, c2_opt, c3_opt = st.columns(3)
# DUP_AFTER_ENTRYPOINT         with c1_opt:
# DUP_AFTER_ENTRYPOINT             objective = st.selectbox(
# DUP_AFTER_ENTRYPOINT                 "Objective Property", ["Maximize Oil EUR", "Maximize Gas EUR"], key="opt_objective"
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT         with c2_opt:
# DUP_AFTER_ENTRYPOINT             iterations = st.number_input(
# DUP_AFTER_ENTRYPOINT                 "Number of optimization steps", min_value=5, max_value=1000, value=100, step=10
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT         with c3_opt:
# DUP_AFTER_ENTRYPOINT             st.selectbox(
# DUP_AFTER_ENTRYPOINT                 "Forbidden Zone", ["Numerical Faults"], help="The optimizer will avoid placing wells near the fault defined in the sidebar."
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT         st.markdown("#### 2. Well Parameters")
# DUP_AFTER_ENTRYPOINT         c1_well, c2_well = st.columns(2)
# DUP_AFTER_ENTRYPOINT         with c1_well:
# DUP_AFTER_ENTRYPOINT             num_wells = st.number_input(
# DUP_AFTER_ENTRYPOINT                 "Number of wells to place", min_value=1, max_value=1, value=1, disabled=True, help="Currently supports optimizing a single well location."
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT         with c2_well:
# DUP_AFTER_ENTRYPOINT             st.text_input("Well name prefix", "OptiWell", disabled=True)
# DUP_AFTER_ENTRYPOINT         launch_opt = st.button("🚀 Launch Optimization", use_container_width=True, type="primary")
# DUP_AFTER_ENTRYPOINT         if launch_opt:
# DUP_AFTER_ENTRYPOINT             opt_results = []
# DUP_AFTER_ENTRYPOINT             base_state = state.copy()
# DUP_AFTER_ENTRYPOINT             rng_opt = np.random.default_rng(int(st.session_state.rng_seed))
# DUP_AFTER_ENTRYPOINT             reservoir_x_dim = base_state['nx'] * base_state['dx']
# DUP_AFTER_ENTRYPOINT             x_max = reservoir_x_dim - base_state['L_ft']
# DUP_AFTER_ENTRYPOINT             if x_max < 0:
# DUP_AFTER_ENTRYPOINT                 st.error(
# DUP_AFTER_ENTRYPOINT                     "Optimization Cannot Run: The well is too long for the reservoir.\n\n"
# DUP_AFTER_ENTRYPOINT                     f"- Reservoir X-Dimension (nx * dx): **{reservoir_x_dim:.0f} ft**\n"
# DUP_AFTER_ENTRYPOINT                     f"- Well Lateral Length (L_ft): **{base_state['L_ft']:.0f} ft**\n\n"
# DUP_AFTER_ENTRYPOINT                     "Please decrease 'Lateral length (ft)' or increase 'nx'/'dx' in the sidebar.",
# DUP_AFTER_ENTRYPOINT                     icon="⚠️",
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT                 st.stop()
# DUP_AFTER_ENTRYPOINT             y_max = base_state['ny'] * base_state['dy']
# DUP_AFTER_ENTRYPOINT             progress_bar = st.progress(0, text="Starting optimization...")
# DUP_AFTER_ENTRYPOINT             for i in range(int(iterations)):
# DUP_AFTER_ENTRYPOINT                 is_valid = False
# DUP_AFTER_ENTRYPOINT                 guard = 0
# DUP_AFTER_ENTRYPOINT                 while (not is_valid) and (guard < 10000):
# DUP_AFTER_ENTRYPOINT                     x_heel_ft = rng_opt.uniform(0, x_max)
# DUP_AFTER_ENTRYPOINT                     y_heel_ft = rng_opt.uniform(50, y_max - 50)
# DUP_AFTER_ENTRYPOINT                     is_valid = is_heel_location_valid(x_heel_ft, y_heel_ft, base_state)
# DUP_AFTER_ENTRYPOINT                     guard += 1
# DUP_AFTER_ENTRYPOINT                 if not is_valid:
# DUP_AFTER_ENTRYPOINT                     st.error("Could not find a valid heel location. Check grid size, L_ft, and fault settings.")
# DUP_AFTER_ENTRYPOINT                     break
# DUP_AFTER_ENTRYPOINT                 temp_state = base_state.copy()
# DUP_AFTER_ENTRYPOINT                 x_norm = x_heel_ft / (base_state['nx'] * base_state['dx'])
# DUP_AFTER_ENTRYPOINT                 temp_state['pad_interf'] = 0.4 * x_norm
# DUP_AFTER_ENTRYPOINT                 result = fallback_fast_solver(temp_state, rng_opt)
# DUP_AFTER_ENTRYPOINT                 score = result['EUR_o_MMBO'] if "Oil" in objective else result['EUR_g_BCF']
# DUP_AFTER_ENTRYPOINT                 opt_results.append({
# DUP_AFTER_ENTRYPOINT                     "Step": i + 1, "x_ft": float(x_heel_ft), "y_ft": float(y_heel_ft), "Score": float(score),
# DUP_AFTER_ENTRYPOINT                 })
# DUP_AFTER_ENTRYPOINT                 progress_bar.progress(
# DUP_AFTER_ENTRYPOINT                     (i + 1) / int(iterations), text=f"Step {i+1}/{int(iterations)} | Score: {score:.3f}"
# DUP_AFTER_ENTRYPOINT                 )
# DUP_AFTER_ENTRYPOINT             st.session_state.opt_results = pd.DataFrame(opt_results)
# DUP_AFTER_ENTRYPOINT             progress_bar.empty()
# DUP_AFTER_ENTRYPOINT         if 'opt_results' in st.session_state and not st.session_state.opt_results.empty:
# DUP_AFTER_ENTRYPOINT             df_results = st.session_state.opt_results
# DUP_AFTER_ENTRYPOINT             best_run = df_results.loc[df_results['Score'].idxmax()]
# DUP_AFTER_ENTRYPOINT             st.markdown("---")
# DUP_AFTER_ENTRYPOINT             st.markdown("### Optimization Results")
# DUP_AFTER_ENTRYPOINT             c1_res, c2_res = st.columns(2)
# DUP_AFTER_ENTRYPOINT             with c1_res:
# DUP_AFTER_ENTRYPOINT                 st.markdown("##### Best Placement Found")
# DUP_AFTER_ENTRYPOINT                 score_unit = "MMBO" if "Oil" in st.session_state.get("opt_objective", "Maximize Oil EUR") else "BCF"
# DUP_AFTER_ENTRYPOINT                 st.metric(label=f"Best Score ({score_unit})", value=f"{best_run['Score']:.3f}")
# DUP_AFTER_ENTRYPOINT                 st.write(f"**Location (ft):** (x={best_run['x_ft']:.0f}, y={best_run['y_ft']:.0f})")
# DUP_AFTER_ENTRYPOINT                 st.write(f"Found at Step: {int(best_run['Step'])}")
# DUP_AFTER_ENTRYPOINT             with c2_res:
# DUP_AFTER_ENTRYPOINT                 st.markdown("##### Optimization Steps Log")
# DUP_AFTER_ENTRYPOINT                 st.dataframe(df_results.sort_values("Score", ascending=False).head(10), height=210)
# DUP_AFTER_ENTRYPOINT             fig_opt = go.Figure()
# DUP_AFTER_ENTRYPOINT             phi_map = get_k_slice(
# DUP_AFTER_ENTRYPOINT                 st.session_state.get('phi', np.zeros((state['nz'], state['ny'], state['nx']))),
# DUP_AFTER_ENTRYPOINT                 state['nz'] // 2
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             fig_opt.add_trace(go.Heatmap(
# DUP_AFTER_ENTRYPOINT                 z=phi_map, dx=state['dx'], dy=state['dy'], colorscale='viridis', colorbar=dict(title='Porosity')
# DUP_AFTER_ENTRYPOINT             ))
# DUP_AFTER_ENTRYPOINT             fig_opt.add_trace(go.Scatter(
# DUP_AFTER_ENTRYPOINT                 x=df_results['x_ft'], y=df_results['y_ft'], mode='markers',
# DUP_AFTER_ENTRYPOINT                 marker=dict(
# DUP_AFTER_ENTRYPOINT                     color=df_results['Score'], colorscale='Reds', showscale=True,
# DUP_AFTER_ENTRYPOINT                     colorbar=dict(title='Score'), size=8, opacity=0.7
# DUP_AFTER_ENTRYPOINT                 ),
# DUP_AFTER_ENTRYPOINT                 name='Tested Locations'
# DUP_AFTER_ENTRYPOINT             ))
# DUP_AFTER_ENTRYPOINT             fig_opt.add_trace(go.Scatter(
# DUP_AFTER_ENTRYPOINT                 x=[best_run['x_ft']], y=[best_run['y_ft']], mode='markers',
# DUP_AFTER_ENTRYPOINT                 marker=dict(color='cyan', size=16, symbol='star', line=dict(width=2, color='black')),
# DUP_AFTER_ENTRYPOINT                 name='Best Location'
# DUP_AFTER_ENTRYPOINT             ))
# DUP_AFTER_ENTRYPOINT             if state.get('use_fault'):
# DUP_AFTER_ENTRYPOINT                 fault_x = [state['fault_index'] * state['dx'], state['fault_index'] * state['dx']]
# DUP_AFTER_ENTRYPOINT                 fault_y = [0, state['ny'] * state['dy']]
# DUP_AFTER_ENTRYPOINT                 fig_opt.add_trace(go.Scatter(
# DUP_AFTER_ENTRYPOINT                     x=fault_x, y=fault_y, mode='lines', line=dict(color='white', width=4, dash='dash'), name='Fault'
# DUP_AFTER_ENTRYPOINT                 ))
# DUP_AFTER_ENTRYPOINT             fig_opt.update_layout(
# DUP_AFTER_ENTRYPOINT                 title="<b>Well Placement Optimization Map</b>",
# DUP_AFTER_ENTRYPOINT                 xaxis_title="X position (ft)",
# DUP_AFTER_ENTRYPOINT                 yaxis_title="Y position (ft)",
# DUP_AFTER_ENTRYPOINT                 template="plotly_white",
# DUP_AFTER_ENTRYPOINT                 height=600
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             st.plotly_chart(fig_opt, use_container_width=True, theme="streamlit")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "User's Manual":
# DUP_AFTER_ENTRYPOINT         st.header("User's Manual")
# DUP_AFTER_ENTRYPOINT         st.markdown("---")
# DUP_AFTER_ENTRYPOINT         st.markdown("""
# DUP_AFTER_ENTRYPOINT       
# DUP_AFTER_ENTRYPOINT def render_users_manual():
# DUP_AFTER_ENTRYPOINT     st.markdown(r"""
# DUP_AFTER_ENTRYPOINT ### 1. Introduction
# DUP_AFTER_ENTRYPOINT Welcome to the **Full 3D Unconventional & Black-Oil Reservoir Simulator**. This application is designed for petroleum engineers to model, forecast, and optimize production from multi-stage fractured horizontal wells.
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT ### 2. Quick Start Guide
# DUP_AFTER_ENTRYPOINT 1. Select a Play…
# DUP_AFTER_ENTRYPOINT 2. Apply Preset…
# DUP_AFTER_ENTRYPOINT 3. Generate Geology…
# DUP_AFTER_ENTRYPOINT 4. Run Simulation…
# DUP_AFTER_ENTRYPOINT 5. Analyze…
# DUP_AFTER_ENTRYPOINT 6. Iterate…
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT ### 3. Key Tabs Explained
# DUP_AFTER_ENTRYPOINT (put your long text here, using plain ASCII quotes " like these)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT ### 4. Input Validation
# DUP_AFTER_ENTRYPOINT - Automated Match warns if any min bound > max bound.
# DUP_AFTER_ENTRYPOINT - Results sanity checks enforce realistic EURs…
# DUP_AFTER_ENTRYPOINT     """)  # end users manual
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     #### Results
# DUP_AFTER_ENTRYPOINT     Primary dashboard for simulation outputs (EURs, rate-time, cumulative). Simulation runs only when you click **Run simulation** on this tab.
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     #### Economics
# DUP_AFTER_ENTRYPOINT     Financial model based on the Results profile.
# DUP_AFTER_ENTRYPOINT     - **Inputs:** CAPEX, price decks, OPEX, fiscal terms.
# DUP_AFTER_ENTRYPOINT     - **Metrics:** **NPV**, **IRR**, **Payout Period**.
# DUP_AFTER_ENTRYPOINT     - **Outputs:** Yearly & cumulative cash flows plus a detailed table.
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     #### Field Match & Automated Match
# DUP_AFTER_ENTRYPOINT     History matching against real data.
# DUP_AFTER_ENTRYPOINT     - **Field Match (CSV):** Upload CSV (must include `Day` and one of `Oil_Rate_STBpd` or `Gas_Rate_Mscfd`). Adjust parameters and re-run to align curves.
# DUP_AFTER_ENTRYPOINT     - **Automated Match:** Genetic algorithm:
# DUP_AFTER_ENTRYPOINT #       1) Upload data  2) Select parameters (e.g., `xf_ft`, `k_stdev`)  
# DUP_AFTER_ENTRYPOINT #       3) Set bounds  4) **Run Automated Match** to minimize RMSE.
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     #### 3D & Slice Viewers
# DUP_AFTER_ENTRYPOINT     - **3D Viewer:** Interactive isosurfaces (perm/poro/pressure).  
# DUP_AFTER_ENTRYPOINT     - **Slice Viewer:** 2D cross-sections in X/Y/Z for layer-by-layer inspection.
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT        ### 4. Input Validation
# DUP_AFTER_ENTRYPOINT     - **Automated Match:** warns if any min bound > max bound.  
# DUP_AFTER_ENTRYPOINT     - **Results:** sanity checks enforce realistic EURs for the selected play; physically inconsistent results are flagged or withheld.
# DUP_AFTER_ENTRYPOINT         """)  # end Overview markdown
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "Solver & Profiling":
# DUP_AFTER_ENTRYPOINT         st.header("Solver & Profiling")
# DUP_AFTER_ENTRYPOINT         st.info("This tab shows numerical solver settings and performance of the last run.")
# DUP_AFTER_ENTRYPOINT         st.markdown("### Current Numerical Solver Settings")
# DUP_AFTER_ENTRYPOINT         solver_settings = {
# DUP_AFTER_ENTRYPOINT             "Parameter": [
# DUP_AFTER_ENTRYPOINT                 "Newton Tolerance", "Max Newton Iterations", "Threads", "Use OpenMP",
# DUP_AFTER_ENTRYPOINT                 "Use MKL", "Use PyAMG", "Use cuSPARSE"
# DUP_AFTER_ENTRYPOINT             ],
# DUP_AFTER_ENTRYPOINT             "Value": [
# DUP_AFTER_ENTRYPOINT                 f"{state['newton_tol']:.1e}", state['max_newton'], "Auto" if state['threads'] == 0 else state['threads'],
# DUP_AFTER_ENTRYPOINT                 "✅" if state['use_omp'] else "❌", "✅" if state['use_mkl'] else "❌",
# DUP_AFTER_ENTRYPOINT                 "✅" if state['use_pyamg'] else "❌", "✅" if state['use_cusparse'] else "❌",
# DUP_AFTER_ENTRYPOINT             ],
# DUP_AFTER_ENTRYPOINT         }
# DUP_AFTER_ENTRYPOINT         st.table(pd.DataFrame(solver_settings))
# DUP_AFTER_ENTRYPOINT         st.markdown("### Profiling")
# DUP_AFTER_ENTRYPOINT         if st.session_state.get("sim") and 'runtime_s' in st.session_state.sim:
# DUP_AFTER_ENTRYPOINT             st.metric(label="Last Simulation Runtime", value=f"{st.session_state.sim['runtime_s']:.2f} seconds")
# DUP_AFTER_ENTRYPOINT         else:
# DUP_AFTER_ENTRYPOINT             st.info("Run a simulation on the 'Results' tab to see performance profiling.")
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     elif selected_tab == "DFN Viewer":
# DUP_AFTER_ENTRYPOINT         st.header("DFN Viewer — 3D line segments")
# DUP_AFTER_ENTRYPOINT         segs = st.session_state.get('dfn_segments')
# DUP_AFTER_ENTRYPOINT         if segs is None or len(segs) == 0:
# DUP_AFTER_ENTRYPOINT             st.info("No DFN loaded. Upload a CSV or use 'Generate DFN from stages' in the sidebar.")
# DUP_AFTER_ENTRYPOINT         else:
# DUP_AFTER_ENTRYPOINT             figd = go.Figure()
# DUP_AFTER_ENTRYPOINT             for i, seg in enumerate(segs):
# DUP_AFTER_ENTRYPOINT                 figd.add_trace(go.Scatter3d(
# DUP_AFTER_ENTRYPOINT                     x=[seg[0], seg[3]],
# DUP_AFTER_ENTRYPOINT                     y=[seg[1], seg[4]],
# DUP_AFTER_ENTRYPOINT                     z=[seg[2], seg[5]],
# DUP_AFTER_ENTRYPOINT                     mode="lines",
# DUP_AFTER_ENTRYPOINT                     line=dict(width=4, color="red"),
# DUP_AFTER_ENTRYPOINT                     name="DFN" if i == 0 else None,
# DUP_AFTER_ENTRYPOINT                     showlegend=(i == 0)
# DUP_AFTER_ENTRYPOINT                 ))
# DUP_AFTER_ENTRYPOINT             figd.update_layout(
# DUP_AFTER_ENTRYPOINT                 template="plotly_white",
# DUP_AFTER_ENTRYPOINT                 scene=dict(xaxis_title="x (ft)", yaxis_title="y (ft)", zaxis_title="z (ft)"),
# DUP_AFTER_ENTRYPOINT                 height=640,
# DUP_AFTER_ENTRYPOINT                 margin=dict(l=0, r=0, t=40, b=0),
# DUP_AFTER_ENTRYPOINT                 title="<b>DFN Segments</b>",
# DUP_AFTER_ENTRYPOINT             )
# DUP_AFTER_ENTRYPOINT             st.plotly_chart(figd, use_container_width=True, theme="streamlit")
# DUP_AFTER_ENTRYPOINT             with st.expander("Click for details"):
# DUP_AFTER_ENTRYPOINT                 st.markdown("""
# DUP_AFTER_ENTRYPOINT                 This plot shows a 3D visualization of the Discrete Fracture Network (DFN) segments loaded into the simulator.
# DUP_AFTER_ENTRYPOINT #                 - Each **red line** represents an individual natural fracture defined in the input file.
# DUP_AFTER_ENTRYPOINT #                 - Use this for QC to verify locations/orientations inside the reservoir model.
# DUP_AFTER_ENTRYPOINT                 """)
# DUP_AFTER_ENTRYPOINT # END: disable legacy nav block
# DUP_AFTER_ENTRYPOINT     def render_users_manual():
# DUP_AFTER_ENTRYPOINT         st.markdown(
# DUP_AFTER_ENTRYPOINT             """
# DUP_AFTER_ENTRYPOINT    st.markdown(r"""
# DUP_AFTER_ENTRYPOINT ### 1. Introduction
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT Welcome to the **Full 3D Unconventional & Black-Oil Reservoir Simulator**. This application is designed for petroleum engineers to model, forecast, and optimize production from multi-stage fractured horizontal wells.
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT """)
# DUP_AFTER_ENTRYPOINT 
# DUP_AFTER_ENTRYPOINT     ### 2. Quick Start Guide
# DUP_AFTER_ENTRYPOINT     1. **Select a Play:** In the sidebar, choose a shale play from the **Preset** dropdown (e.g., "Permian – Midland (Oil)").
# DUP_AFTER_ENTRYPOINT     2. **Apply Preset:** Click **Apply Preset**. This loads typical reservoir, fluid, and completion parameters into the sidebar.
# DUP_AFTER_ENTRYPOINT     3. **Generate Geology:** Open **Generate 3D property volumes** and click the large button to create 3D permeability/porosity grids.
# DUP_AFTER_ENTRYPOINT     4. **Run Simulation:** Go to **Results** and click **Run simulation**.
# DUP_AFTER_ENTRYPOINT     5. **Analyze:** Review EUR gauges, rate–time plots, and cumulative production charts.
# DUP_AFTER_ENTRYPOINT     6. **Iterate:** Adjust sidebar parameters (e.g., frac half-length `xf_ft` or pad BHP `pad_bhp_psi`) and re-run to see the impact.
# DUP_AFTER_ENTRYPOINT             """
# DUP_AFTER_ENTRYPOINT         )
