# core/full3d.py — UPDATED (Part 1/4)
# =============================================================================
# Full 3D Unconventional / Black-Oil Reservoir Simulator — Phase 2 (EDFM + basic geomech)
# =============================================================================
#
# Overview
# --------
# This module implements a compact, pragmatic 3-phase (oil/water/gas) black-oil
# simulator with:
#   • Pressure-dependent porosity φ(P) and water FVF Bw(P)
#   • Optional capillary pressures for water/oil and gas/oil systems
#   • Gravity effects in TPFA (two-point flux approximation)
#   • Boundary conditions: fixed pressure faces and simple “aquifer tanks”
#   • Wells: BHP control or “true rate” control via equivalent BHP unknowns
#   • EDFM: Embedded Discrete Fracture Model (matrix↔fracture + fracture↔fracture)
#   • Simple geomechanics via a permeability multiplier k(p)
#
# The code is structured for clarity: helpers → Jacobian assembly → PVT →
# scheduling/preparation → Newton driver → top-level simulate() API.
#
# Notes for integration:
# - This file is intentionally verbose with comments for clarity and future
#   maintenance. It is designed to be readable, hackable, and stable.
# - The “analytical” engine in simulate() calls an optional fast fallback
#   (engines/fast.py). If that import is not available, a simple local stub is
#   used so the module remains runnable.
#
# -----------------------------------------------------------------------------


from __future__ import annotations

import math
import numpy as np

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# EDFM leak utility (Phase-2 diagonal “leak” stabilization for matrix↔fracture)
# Keep only a single import in this updated file.
try:
    from utils.edfm_wf import apply_edfm_leak
except Exception:
    # If not available, define a no-op fall-back so the module remains runnable.
    def apply_edfm_leak(A, lam_o, lam_w, lam_g, Bo, Bw, Bg, options):
        # No-op: return the matrix as-is
        return A


# ---------------------------- constants/options ------------------------------

G_FT_S2 = 32.174  # ft/s^2 (kept for clarity; not used directly in grav psi/ft)
PSI_PER_FT = 1.0 / 144.0  # hydrostatic gradient ≈ rho(lb/ft^3)/144 psi/ft

DEFAULT_OPTIONS = {
    "use_gravity": True,
    "rho_o_lbft3": 53.0,
    "rho_w_lbft3": 62.4,
    "rho_g_lbft3": 0.06,
    "kvkh": 0.10,       # vertical anisotropy (kv/kh)
    "geo_alpha": 0.0,   # simple geomech k-multiplier (0 = off)
    "use_pc": False,    # capillary off by default
    "well_mode": "equivalent_bhp",  # or "true_rate"
    # --- EDFM inputs: you supply these externally; no heavy geometry here ---
    # options["frac_cells"] = [
    #   { "cell": <int 0..N-1>, "area_ft2": <float>, "normal": (nx,ny,nz),
    #     "aperture_ft": <float>, "k_md": <float optional>, "id": <int optional> }
    # ]
    # options["frac_links"] = [  # optional fracture⇄fracture couplings
    #   { "i": <fracell_i>, "j": <fracell_j>, "d_ft": <center distance>,
    #     "L_ft": <trace length>, "aperture_ft": <avg aperture>,
    #     "k_md": <avg k> }
    # ]
}


# ----------------------------- small helpers --------------------------------

def peaceman_wi_cartesian(kx, ky, dz, dx, dy, rw_ft=0.35, skin=0.0) -> float:
    """
    Peaceman well index (md*ft) for a vertical perf in one cell (anisotropic).
    """
    beta = np.sqrt(ky / np.maximum(kx, 1e-30))
    re = 0.28 * np.sqrt((dx**2 * beta + dy**2 / beta) / (beta + 1.0 / beta))
    kh = np.sqrt(kx * ky) * dz
    wi = 2.0 * np.pi * kh / (np.log(re / max(rw_ft, 1e-6)) + max(skin, 0.0))
    return float(wi)


def k_multiplier_from_pressure(p_cell, p_init, alpha=0.0, min_mult=0.2, max_mult=1.0):
    """
    Simple geomech k-multiplier: k_eff = k0 * exp(-alpha * (p_init - p)).
    Used to emulate stress-sensitive permeability in a first-order way.
    """
    dp = (p_init - np.asarray(p_cell))
    mult = np.exp(-alpha * dp)
    return np.clip(mult, min_mult, max_mult)


# --------------------------- indexing helpers (cells) ------------------------

def _cell_idx(i):    return 3 * i
def _cell_idx_sw(i): return 3 * i + 1
def _cell_idx_sg(i): return 3 * i + 2


# ------------------------------- EDFM helpers --------------------------------

def _proj_diag_perm_md(nx, ny, nz, kx_md, ky_md, kz_md):
    """
    Project diagonal permeability tensor diag(kx, ky, kz) along unit normal n.
    Returns k_n (md) = n^T K n = nx^2*kx + ny^2*ky + nz^2*kz
    """
    return (nx*nx) * kx_md + (ny*ny) * ky_md + (nz*nz) * kz_md


def _half_cell_distance_ft(dx, dy, dz, n):
    """
    Effective half-distance from matrix center to fracture plane along normal.
    We use 0.5*(dx*|nx| + dy*|ny| + dz*|nz|); later we add 0.5*aperture.
    """
    nx, ny, nz = map(abs, n)
    return 0.5 * (dx*nx + dy*ny + dz*nz)


def _safe_unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else np.array([1.0, 0.0, 0.0])


def _frac_initial_pressures(frac_cells, P_matrix):
    """
    Seed fracture pressures from their host matrix cell pressures.
    """
    if not frac_cells:
        return np.empty(0, float)
    Pf0 = np.empty(len(frac_cells), float)
    for idx, f in enumerate(frac_cells):
        Pf0[idx] = float(P_matrix[f["cell"]])
    return Pf0


# ---------------------------- EUR helpers (NEW) ------------------------------

def _apply_cutoffs_and_cap(t_days, qo_stbd, qg_mscfd,
                           oil_cutoff_stbd=30.0, gas_cutoff_mscfd=100.0,
                           cap_days=10950.0):
    """
    Zero-out rates below economic cutoffs and cap the horizon.
    Returns masked copies (qo_cut, qg_cut) and capped time array.
    """
    t = np.asarray(t_days, float)
    qo = np.asarray(qo_stbd, float).copy()
    qg = np.asarray(qg_mscfd, float).copy()

    # Cap the time series
    if t.size > 0 and t[-1] > cap_days:
        mask = t <= cap_days
        t, qo, qg = t[mask], qo[mask], qg[mask]

    # Apply economic cutoffs
    qo[qo < oil_cutoff_stbd] = 0.0
    qg[qg < gas_cutoff_mscfd] = 0.0
    return t, qo, qg


def _cum_trapz(t_days, y):
    """
    Monotone time cumulative integral with the trapezoid rule.
    """
    t = np.asarray(t_days, float)
    y = np.asarray(y, float)
    if t.size == 0:
        return np.asarray([], float)
    dt = np.diff(t)
    mid = 0.5 * (y[1:] + y[:-1])
    cum = np.concatenate([[0.0], np.cumsum(mid * dt)])
    return cum


def _compute_eur_and_cum(t_days, qo_stbd, qg_mscfd,
                         oil_cutoff_stbd=30.0, gas_cutoff_mscfd=100.0,
                         cap_days=10950.0):
    """
    Returns:
      t_cap, cum_o_stb, cum_g_mscf, eur_o_mmbo, eur_g_bcf

    Notes:
      - Oil EUR [MMBO] = cum_o_stb / 1e6
      - Gas EUR [BCF]  = cum_g_mscf / 1e3   (since 1 BCF = 1000 Mscf)
    """
    t_cap, qo_cut, qg_cut = _apply_cutoffs_and_cap(
        t_days, qo_stbd, qg_mscfd,
        oil_cutoff_stbd=oil_cutoff_stbd,
        gas_cutoff_mscfd=gas_cutoff_mscfd,
        cap_days=cap_days,
    )
    cum_o_stb = _cum_trapz(t_cap, qo_cut)            # STB
    cum_g_mscf = _cum_trapz(t_cap, qg_cut)           # Mscf
    eur_o_mmbo = float(cum_o_stb[-1] / 1e6) if cum_o_stb.size else 0.0
    eur_g_bcf  = float(cum_g_mscf[-1] / 1e3) if cum_g_mscf.size else 0.0
    return t_cap, cum_o_stb, cum_g_mscf, eur_o_mmbo, eur_g_bcf


def _validate_eur(eur_o_mmbo, eur_g_bcf,
                  basin="Midland", window="oil",
                  oil_bounds=(0.3, 1.5), gas_bounds=(0.3, 3.0)):
    """
    Simple guardrails. Returns dict with {valid, messages, bounds}.
    You can tweak bounds via inputs if needed.
    """
    msgs = []
    valid = True
    lo_o, hi_o = oil_bounds
    lo_g, hi_g = gas_bounds

    if eur_o_mmbo < lo_o or eur_o_mmbo > hi_o:
        valid = False
        msgs.append(
            f"Oil EUR {eur_o_mmbo:.2f} MMBO outside expected range "
            f"{lo_o}-{hi_o} MMBO for {basin} {window}-window."
        )
    if eur_g_bcf < lo_g or eur_g_bcf > hi_g:
        valid = False
        msgs.append(
            f"Gas EUR {eur_g_bcf:.2f} BCF outside expected range "
            f"{lo_g}-{hi_g} BCF for {basin} {window}-window."
        )

    return {
        "valid": valid,
        "messages": msgs,
        "oil_bounds_mmbo": list(oil_bounds),
        "gas_bounds_bcf": list(gas_bounds),
        "basin": basin,
        "window": window,
    }


# -----------------------------------------------------------------------------
# Core assembly: Black-oil residuals + Jacobian with EDFM and boundary/well ops
# -----------------------------------------------------------------------------

def assemble_jacobian_and_residuals_blackoil(
    state_vec: np.ndarray,
    grid: dict,
    rock: dict,
    pvt,
    relperm: dict,
    init: dict,
    schedule: dict,
    options: dict,
):
    """
    Assemble the global sparse Jacobian matrix A and residual vector R for the
    fully-implicit black-oil step. Includes:
      - Accumulation with φ(P) and Bw(P)
      - TPFA internal faces with gravity (+ optional capillary)
      - Fixed-P boundary faces
      - Simple “aquifer tank” coupling
      - BHP / TRUE-rate wells (TRUE-rate via extra unknown per well)
      - EDFM matrix↔fracture and optional fracture↔fracture transmissibilities

    Parameters
    ----------
    state_vec : np.ndarray
        Packed vector of unknowns: [3*N matrix vars] + [wells?] + [aquifers?] + [fractures?]
    grid : dict
        nx, ny, nz, dx, dy, dz
    rock : dict
        phi, kx_md, ky_md, (optional kz or kvkh), cr_1overpsi, p_ref_psi
    pvt : object
        Provides Bo(P), Bg(P), Rs(P), mu_o(P), mu_g(P), Bw(P), mu_w(P)
    relperm : dict
        Corey-like params: nw, no, krw_end, kro_end
    init, schedule, options : dict
        Step configuration, wells, aquifers, EDFM options, etc.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Global Jacobian
    R : np.ndarray
        Residual vector
    meta : dict
        Meta info about the assembly
    """
    # ---- toggles / params ----
    use_grav = bool(options.get("use_gravity", True))
    use_pc   = bool(options.get("use_pc", False))
    rho_o = float(options.get("rho_o_lbft3", DEFAULT_OPTIONS["rho_o_lbft3"]))
    rho_w = float(options.get("rho_w_lbft3", DEFAULT_OPTIONS["rho_w_lbft3"]))
    rho_g = float(options.get("rho_g_lbft3", DEFAULT_OPTIONS["rho_g_lbft3"]))

    # Unknown-map lookups
    well_map = options.get("well_unknown_map", {})         # {well_idx: col}
    aq_map   = options.get("aquifer_unknown_map", {})      # {aq_idx: col}
    aq_by_face = options.get("aquifer_face_map", {})       # face -> list of (aq_idx, pi_mult)
    fixed_by_face = options.get("fixed_p_face_map", {})    # face -> p_ext
    prev_aqP = options.get("prev_aquifer_pressures", {})   # {aq_idx: P_prev}

    dt_days = float(options.get("dt_days", 1.0))
    dt = max(1.0, dt_days * 86400.0)

    # ---- capillary Pcs (Corey-like) ----
    pcw_entry = float(options.get("pcw_entry_psi", 0.0))
    pcw_L     = float(options.get("pcw_lambda", 2.0))
    pcg_entry = float(options.get("pcg_entry_psi", 0.0))
    pcg_L     = float(options.get("pcg_lambda", 2.0))

    def Pcow(sw):
        if not use_pc:
            z = np.asarray(sw, float)
            return np.zeros_like(z), np.zeros_like(z)
        sw = np.clip(np.asarray(sw, float), 1e-9, 1-1e-9)
        pc = pcw_entry * (sw ** (-1.0/pcw_L) - 1.0)
        dpc_dsw = pcw_entry * (-(1.0/pcw_L)) * (sw ** (-1.0/pcw_L - 1.0))
        return pc, dpc_dsw

    def Pcg(sg):
        if not use_pc:
            z = np.asarray(sg, float)
            return np.zeros_like(z), np.zeros_like(z)
        sg = np.clip(np.asarray(sg, float), 1e-9, 1-1e-9)
        pc = pcg_entry * (sg ** (-1.0/pcg_L) - 1.0)
        dpc_dsg = pcg_entry * (-(1.0/pcg_L)) * (sg ** (-1.0/pcg_L - 1.0))
        return pc, dpc_dsg

    # ---- grid / indexing ----
    nx, ny, nz = [int(grid[k]) for k in ("nx", "ny", "nz")]
    dx, dy, dz = [float(grid[k]) for k in ("dx", "dy", "dz")]
    N = nx * ny * nz

    def lin(i, j, k): return (k * ny + j) * nx + i
    Vcell = dx * dy * dz

    # ---- fracture unknowns (EDFM) ----
    frac_cells = options.get("frac_cells", []) or []
    frac_map   = options.get("frac_unknown_map", {}) or {}
    Nf = len(frac_cells)

    # ---- unknowns in state_vec ----
    nW = len(well_map)
    nA = len(aq_map)
    expected_len = 3*N + nW + nA + Nf
    if state_vec.size != expected_len:
        raise ValueError(f"state_vec size {state_vec.size} != expected {expected_len}")

    # Matrix unknowns
    P  = state_vec[0:3*N:3].copy()
    Sw = state_vec[1:3*N:3].copy()
    Sg = state_vec[2:3*N:3].copy()
    So = 1.0 - Sw - Sg

    eps = 1e-9
    Sw[:] = np.clip(Sw, 0.0 + eps, 1.0 - eps)
    Sg[:] = np.clip(Sg, 0.0 + eps, 1.0 - eps)
    So[:] = np.clip(So,  0.0 + eps, 1.0 - eps)

    # Fracture pressures block (ordered by frac_map values)
    if Nf > 0:
        Pf = np.empty(Nf, float)
        frac_cols = [frac_map[i] for i in range(Nf)]
        base_off = min(frac_cols)
        Pf[:] = state_vec[base_off:base_off+Nf]
    else:
        Pf = np.empty(0, float)

    # ---- rock φ(P) ----
    phi0 = np.asarray(rock.get("phi", 0.08)).reshape(N)
    cr   = float(rock.get("cr_1overpsi", options.get("cr_1overpsi", 0.0)))
    p_ref = float(rock.get("p_ref_psi", options.get("p_ref_psi", init.get("p_init_psi", 5000.0))))
    if cr > 0.0:
        phi = phi0 * np.exp(cr * (P - p_ref))
        dphi_dP = cr * phi
    else:
        phi = phi0
        dphi_dP = np.zeros_like(P)

    # perms (geomech)
    kx  = np.asarray(rock.get("kx_md", 100.0)).reshape(N)
    ky  = np.asarray(rock.get("ky_md", 100.0)).reshape(N)
    kvkh = float(options.get("kvkh", 0.1))
    kz  = np.asarray(rock.get("kz_md", kvkh * 0.5 * (kx + ky))).reshape(N)

    p_init = float(init.get("p_init_psi", 5000.0))
    geo_alpha = float(options.get("geo_alpha", 0.0))
    kmult = k_multiplier_from_pressure(P, p_init, alpha=geo_alpha)
    kx *= kmult; ky *= kmult; kz *= kmult

    # ---- previous state for accumulation ----
    prev = options.get("prev", None)
    if prev is None:
        Pm, Swm, Sgm = P.copy(), Sw.copy(), Sg.copy()
    else:
        Pm = np.asarray(prev.get("P", P))
        Swm = np.asarray(prev.get("Sw", Sw))
        Sgm = np.asarray(prev.get("Sg", Sg))
    Som = 1.0 - Swm - Sgm

    # ---- PVT (+ water PVT) ----
    def numdiff_p(fn, p, eps=1e-3):
        return (np.asarray(fn(p + eps)) - np.asarray(fn(p - eps))) / (2.0 * eps)

    Bo = np.asarray(pvt.Bo(P));   dBo_dP = numdiff_p(pvt.Bo, P)
    Bg = np.asarray(pvt.Bg(P));   dBg_dP = numdiff_p(pvt.Bg, P)
    Rs = np.asarray(pvt.Rs(P));   dRs_dP = numdiff_p(pvt.Rs, P)
    mu_o = np.asarray(pvt.mu_o(P))
    mu_g = np.asarray(pvt.mu_g(P))

    # Water PVT optional
    Bw   = np.asarray(getattr(pvt, "Bw", lambda x: np.ones_like(x))(P))
    dBw_dP = numdiff_p(getattr(pvt, "Bw", lambda x: np.ones_like(x)), P)
    mu_w = np.asarray(getattr(pvt, "mu_w", lambda x: np.full_like(x, 0.5))(P))

    # ---- relperm & mobilities ----
    nw = float(relperm.get("nw", 2.0)); no = float(relperm.get("no", 2.0))
    krw_end = float(relperm.get("krw_end", 0.6)); kro_end = float(relperm.get("kro_end", 0.8))
    krw = relperm.get("krw_fn", lambda s: krw_end * (s ** nw))(Sw)
    kro = relperm.get("kro_fn", lambda s: kro_end * (s ** no))(So)
    krg = relperm.get("krg_fn", lambda s: s ** 2)(Sg)

    lam_w = krw / np.maximum(mu_w, 1e-12)
    lam_o = kro / np.maximum(mu_o, 1e-12)
    lam_g = krg / np.maximum(mu_g, 1e-12)

    # ---- accumulation (φ(P), Bw(P)) ----
    invBo = 1.0 / np.maximum(Bo, 1e-12)
    invBg = 1.0 / np.maximum(Bg, 1e-12)
    invBw = 1.0 / np.maximum(Bw, 1e-12)

    acc_o = phi * So * invBo
    acc_g = phi * (Sg * invBg + So * Rs * invBo)
    acc_w = phi * Sw * invBw

    Bom = np.asarray(pvt.Bo(Pm)); Bgm = np.asarray(pvt.Bg(Pm)); Rsm = np.asarray(pvt.Rs(Pm))
    Bwm = np.asarray(getattr(pvt, "Bw", lambda x: np.ones_like(x))(Pm))
    phim = phi0 * np.exp(cr*(Pm-p_ref)) if cr > 0.0 else phi0

    accm_o = phim * Som / np.maximum(Bom, 1e-12)
    accm_g = phim * (Sgm / np.maximum(Bgm, 1e-12) + Som * Rsm / np.maximum(Bom, 1e-12))
    accm_w = phim * Swm / np.maximum(Bwm, 1e-12)

    # ---- allocate A,R (note: include frac unknowns) ----
    nunk = 3*N + nW + nA + Nf
    A = lil_matrix((nunk, nunk), dtype=float)
    R = np.zeros(nunk, dtype=float)

    # Accumulation residuals for matrix cells
    scale = (dx * dy * dz) / dt
    R[0:3*N:3] += scale * (acc_o - accm_o)
    R[1:3*N:3] += scale * (acc_w - accm_w)
    R[2:3*N:3] += scale * (acc_g - accm_g)

    dinvBo_dP = -dBo_dP * (invBo ** 2)
    dinvBg_dP = -dBg_dP * (invBg ** 2)
    dinvBw_dP = -dBw_dP * (invBw ** 2)

    dacc_o_dP  = dphi_dP * So * invBo + phi * So * dinvBo_dP
    dacc_o_dSw = -phi * invBo
    dacc_o_dSg = -phi * invBo

    dacc_w_dP  = dphi_dP * Sw * invBw + phi * Sw * dinvBw_dP
    dacc_w_dSw =  phi * invBw

    dacc_g_dP  = dphi_dP * (Sg * invBg + So * Rs * invBo) \
               + phi * (Sg * dinvBg_dP + So * (dRs_dP * invBo + Rs * dinvBo_dP))
    dacc_g_dSw = -phi * (Rs * invBo)
    dacc_g_dSg =  phi * invBg - phi * (Rs * invBo)

    for c in range(N):
        A[_cell_idx(c),  _cell_idx(c)]      += scale * dacc_o_dP[c]
        A[_cell_idx(c),  _cell_idx_sw(c)]   += scale * dacc_o_dSw[c]
        A[_cell_idx(c),  _cell_idx_sg(c)]   += scale * dacc_o_dSg[c]

        A[_cell_idx_sw(c), _cell_idx(c)]    += scale * dacc_w_dP[c]
        A[_cell_idx_sw(c), _cell_idx_sw(c)] += scale * dacc_w_dSw[c]

        A[_cell_idx_sg(c), _cell_idx(c)]    += scale * dacc_g_dP[c]
        A[_cell_idx_sg(c), _cell_idx_sw(c)] += scale * dacc_g_dSw[c]
        A[_cell_idx_sg(c), _cell_idx_sg(c)] += scale * dacc_g_dSg[c]

    # ------------------ internal faces (gravity + capillary) ------------------

    def harm(a, b): return 2.0 * a * b / (a + b + 1e-30)

    area_x = dy * dz
    area_y = dx * dz
    area_z = dx * dy

    Pcw, dPcw_dSw = Pcow(Sw)
    Pcg, dPcg_dSg = Pcg(Sg)

    rho_o_grad = rho_o * PSI_PER_FT
    rho_w_grad = rho_w * PSI_PER_FT
    rho_g_grad = rho_g * PSI_PER_FT

    def add_face(c, n, T_face, dz_ft):
        """
        Assemble fluxes for an internal face between matrix cells c and n.
        Includes gravity head and optional capillary pressure corrections.
        Upstreaming is based on phase-potential differences.
        """
        dP = P[c] - P[n]
        up = c if dP >= 0.0 else n

        Bo_up, Bg_up, Bw_up = Bo[up], Bg[up], Bw[up]
        Rs_up = Rs[up]
        lam_o_up, lam_w_up, lam_g_up = lam_o[up], lam_w[up], lam_g[up]

        if use_grav and abs(dz_ft) > 0.0:
            dPhi_o = dP - rho_o_grad * dz_ft
            dPhi_w = (P[c]-Pcw[c]) - (P[n]-Pcw[n]) - rho_w_grad * dz_ft
            dPhi_g = (P[c]+Pcg[c]) - (P[n]+Pcg[n]) - rho_g_grad * dz_ft
        else:
            dPhi_o = dP
            dPhi_w = (P[c]-Pcw[c]) - (P[n]-Pcw[n])
            dPhi_g = (P[c]+Pcg[c]) - (P[n]+Pcg[n])

        invBo_up = 1.0 / max(Bo_up, 1e-12)
        invBg_up = 1.0 / max(Bg_up, 1e-12)
        invBw_up = 1.0 / max(Bw_up, 1e-12)

        Fo = T_face * lam_o_up * invBo_up * dPhi_o
        Fw = T_face * lam_w_up * invBw_up * dPhi_w
        Fg = T_face * (lam_g_up * invBg_up * dPhi_g + (Rs_up * lam_o_up * invBo_up) * dPhi_o)

        R[_cell_idx(c)]     += Fo; R[_cell_idx(n)]     -= Fo
        R[_cell_idx_sw(c)]  += Fw; R[_cell_idx_sw(n)]  -= Fw
        R[_cell_idx_sg(c)]  += Fg; R[_cell_idx_sg(n)]  -= Fg

        coef_o = T_face * lam_o_up * invBo_up
        coef_w = T_face * lam_w_up * invBw_up
        coef_g1 = T_face * lam_g_up * invBg_up
        coef_g2 = T_face * (Rs_up * lam_o_up * invBo_up)

        for row_c, row_n, coef in (
            (_cell_idx(c),    _cell_idx(n),    coef_o),
            (_cell_idx_sw(c), _cell_idx_sw(n), coef_w),
            (_cell_idx_sg(c), _cell_idx_sg(n), (coef_g1 + coef_g2)),
        ):
            A[row_c, _cell_idx(c)] += coef;  A[row_c, _cell_idx(n)] -= coef
            A[row_n, _cell_idx(c)] -= coef;  A[row_n, _cell_idx(n)] += coef

        # saturation derivatives (upstream only)
        So_up, Sw_up, Sg_up = So[up], Sw[up], Sg[up]
        muo_up, muw_up, mug_up = mu_o[up], mu_w[up], mu_g[up]
        dlamo_dSo = kro_end*no*max(So_up,1e-12)**(no-1.0)/max(muo_up,1e-12)
        dlamw_dSw = krw_end*nw*max(Sw_up,1e-12)**(nw-1.0)/max(muw_up,1e-12)
        dlamo_dSw = -dlamo_dSo; dlamo_dSg = -dlamo_dSo
        dlamg_dSg = 2.0 * max(Sg_up,0.0)/max(mug_up,1e-12)

        if up == c:
            A[_cell_idx(c),    _cell_idx_sw(c)] += T_face * dlamo_dSw * invBo_up * dPhi_o
            A[_cell_idx(n),    _cell_idx_sw(c)] -= T_face * dlamo_dSw * invBo_up * dPhi_o
            A[_cell_idx(c),    _cell_idx_sg(c)] += T_face * dlamo_dSg * invBo_up * dPhi_o
            A[_cell_idx(n),    _cell_idx_sg(c)] -= T_face * dlamo_dSg * invBo_up * dPhi_o

            A[_cell_idx_sw(c), _cell_idx_sw(c)] += T_face * dlamw_dSw * invBw_up * dPhi_w
            A[_cell_idx_sw(n), _cell_idx_sw(c)] -= T_face * dlamw_dSw * invBw_up * dPhi_w

            A[_cell_idx_sg(c), _cell_idx_sg(c)] += T_face * dlamg_dSg * invBg_up * dPhi_g
            A[_cell_idx_sg(n), _cell_idx_sg(c)] -= T_face * dlamg_dSg * invBg_up * dPhi_g

            A[_cell_idx_sg(c), _cell_idx_sw(c)] += T_face * (dlamo_dSw * Rs_up) * invBo_up * dPhi_o
            A[_cell_idx_sg(n), _cell_idx_sw(c)] -= T_face * (dlamo_dSw * Rs_up) * invBo_up * dPhi_o
            A[_cell_idx_sg(c), _cell_idx_sg(c)] += T_face * (dlamo_dSg * Rs_up) * invBo_up * dPhi_o
            A[_cell_idx_sg(n), _cell_idx_sg(c)] -= T_face * (dlamo_dSg * Rs_up) * invBo_up * dPhi_o
        else:
            A[_cell_idx(c),    _cell_idx_sw(n)] += T_face * dlamo_dSw * invBo_up * dPhi_o
            A[_cell_idx(n),    _cell_idx_sw(n)] -= T_face * dlamo_dSw * invBo_up * dPhi_o
            A[_cell_idx(c),    _cell_idx_sg(n)] += T_face * dlamo_dSg * invBo_up * dPhi_o
            A[_cell_idx(n),    _cell_idx_sg(n)] -= T_face * dlamo_dSg * invBo_up * dPhi_o

            A[_cell_idx_sw(c), _cell_idx_sw(n)] += T_face * dlamw_dSw * invBw_up * dPhi_w
            A[_cell_idx_sw(n), _cell_idx_sw(n)] -= T_face * dlamw_dSw * invBw_up * dPhi_w

            A[_cell_idx_sg(c), _cell_idx_sg(n)] += T_face * dlamg_dSg * invBg_up * dPhi_g
            A[_cell_idx_sg(n), _cell_idx_sg(n)] -= T_face * dlamg_dSg * invBg_up * dPhi_g

            A[_cell_idx_sg(c), _cell_idx_sw(n)] += T_face * (dlamo_dSw * Rs_up) * invBo_up * dPhi_o
            A[_cell_idx_sg(n), _cell_idx_sw(n)] -= T_face * (dlamo_dSw * Rs_up) * invBo_up * dPhi_o
            A[_cell_idx_sg(c), _cell_idx_sg(n)] += T_face * (dlamo_dSg * Rs_up) * invBo_up * dPhi_o
            A[_cell_idx_sg(n), _cell_idx_sg(n)] -= T_face * (dlamo_dSg * Rs_up) * invBo_up * dPhi_o

        # Extra Jacobian terms from P-dependence in PVT
        extra_o = T_face * lam_o_up * dPhi_o * (-dBo_dP[up] / (max(Bo_up,1e-12)**2))
        extra_w = T_face * lam_w_up * dPhi_w * (-dBw_dP[up] / (max(Bw_up,1e-12)**2))
        extra_g = T_face * (lam_g_up * dPhi_g * (-dBg_dP[up] / (max(Bg_up,1e-12)**2))
                 + lam_o_up * dPhi_o * ((dRs_dP[up]*Bo_up - Rs_up*dBo_dP[up]) / (max(Bo_up,1e-12)**2)))
        for row in (_cell_idx(c), _cell_idx(n)):
            A[row, _cell_idx(up)] += extra_o
        for row in (_cell_idx_sw(c), _cell_idx_sw(n)):
            A[row, _cell_idx(up)] += extra_w
        for row in (_cell_idx_sg(c), _cell_idx_sg(n)):
            A[row, _cell_idx(up)] += extra_g

        if use_pc:
            cwcoef = T_face * lam_w_up * (1.0 / max(Bw_up, 1e-12))
            cgcoef = T_face * lam_g_up * (1.0 / max(Bg_up, 1e-12))

            A[_cell_idx_sw(c), _cell_idx_sw(c)] += cwcoef * (-dPcw_dSw[c])
            A[_cell_idx_sw(n), _cell_idx_sw(c)] -= cwcoef * (-dPcw_dSw[c])
            A[_cell_idx_sw(c), _cell_idx_sw(n)] += cwcoef * (+dPcw_dSw[n])
            A[_cell_idx_sw(n), _cell_idx_sw(n)] -= cwcoef * (+dPcw_dSw[n])

            A[_cell_idx_sg(c), _cell_idx_sg(c)] += cgcoef * (+dPcg_dSg[c])
            A[_cell_idx_sg(n), _cell_idx_sg(c)] -= cgcoef * (+dPcg_dSg[c])
            A[_cell_idx_sg(c), _cell_idx_sg(n)] += cgcoef * (-dPcg_dSg[n])
            A[_cell_idx_sg(n), _cell_idx_sg(n)] -= cgcoef * (-dPcg_dSg[n])

    # Sweep internal faces (i,j,k adjacency)
    for kk in range(nz):
        for jj in range(ny):
            for ii in range(nx):
                c = lin(ii, jj, kk)
                if ii + 1 < nx:
                    n = lin(ii + 1, jj, kk)
                    T = (harm(kx[c], kx[n]) * (dy * dz)) / max(dx, 1e-30)
                    add_face(c, n, T, dz_ft=0.0)
                if jj + 1 < ny:
                    n = lin(ii, jj + 1, kk)
                    T = (harm(ky[c], ky[n]) * (dx * dz)) / max(dy, 1e-30)
                    add_face(c, n, T, dz_ft=0.0)
                if kk + 1 < nz:
                    n = lin(ii, jj, kk + 1)
                    T = (harm(kz[c], kz[n]) * (dx * dy)) / max(dz, 1e-30)
                    # upward neighbor is deeper by dz (positive dz_ft means “down”)
                    add_face(c, n, T, dz_ft=-dz)

    # -------------------------------------------------------------------------
    # Phase-2 EDFM: add a diagonal “leak” for matrix↔fracture coupling.
    # Safe no-op if options["edfm_connectivity"] is absent/empty.
    # -------------------------------------------------------------------------
    A = apply_edfm_leak(A, lam_o, lam_w, lam_g, Bo, Bw, Bg, options)

    # ---------------------- EDFM: matrix ↔ fracture ---------------------------

    if Nf > 0:
        frac_cols = [frac_map[i] for i in range(Nf)]
        fbase = min(frac_cols)

        for fi, f in enumerate(frac_cells):
            c = int(f["cell"])
            nx_u, ny_u, nz_u = _safe_unit(f.get("normal", (1.0, 0.0, 0.0)))
            area_ft2 = float(f["area_ft2"])
            aperture_ft = float(f.get("aperture_ft", 0.0))

            k_n_md = _proj_diag_perm_md(nx_u, ny_u, nz_u, kx[c], ky[c], kz[c])
            d_eq_ft = _half_cell_distance_ft(dx, dy, dz, (nx_u, ny_u, nz_u)) + 0.5*aperture_ft

            Tmf = (max(k_n_md, 1e-30) * area_ft2) / max(d_eq_ft, 1e-30)

            col_f = fbase + fi
            dP = P[c] - Pf[fi]
            up = c  # For matrix-fracture flow, take properties from matrix.

            invBo_up = 1.0 / max(Bo[up], 1e-12)
            invBg_up = 1.0 / max(Bg[up], 1e-12)
            invBw_up = 1.0 / max(Bw[up], 1e-12)
            lam_o_up, lam_w_up, lam_g_up = lam_o[up], lam_w[up], lam_g[up]
            Rs_up = Rs[up]

            Fo = Tmf * lam_o_up * invBo_up * dP
            Fw = Tmf * lam_w_up * invBw_up * dP
            Fg = Tmf * (lam_g_up * invBg_up * dP + (Rs_up * lam_o_up * invBo_up) * dP)

            # Fracture equation: conserve total (surface) volume flowing into fracture
            R[_cell_idx(c)]    += Fo
            R[_cell_idx_sw(c)] += Fw
            R[_cell_idx_sg(c)] += Fg
            R[col_f]           -= (Fo + Fw + Fg)

            coef_o = Tmf * lam_o_up * invBo_up
            coef_w = Tmf * lam_w_up * invBw_up
            coef_g_total = Tmf * (lam_g_up * invBg_up + Rs_up * lam_o_up * invBo_up)

            # Derivatives for matrix equations
            A[_cell_idx(c),    _cell_idx(c)] += coef_o
            A[_cell_idx(c),    col_f]        -= coef_o

            A[_cell_idx_sw(c), _cell_idx(c)] += coef_w
            A[_cell_idx_sw(c), col_f]        -= coef_w

            A[_cell_idx_sg(c), _cell_idx(c)] += coef_g_total
            A[_cell_idx_sg(c), col_f]        -= coef_g_total

            # Derivatives for fracture equation (total volume conservation)
            A[col_f, _cell_idx(c)] -= (coef_o + coef_w + coef_g_total)
            A[col_f, col_f]       += (coef_o + coef_w + coef_g_total)

    # -------------------- NEW: EDFM fracture ↔ fracture connections ------------------

    frac_links = options.get("frac_links", []) or []
    if Nf > 0 and len(frac_links) > 0:
        # Use water-like single-phase mobility as a stable, pragmatic proxy
        mu_w_avg = float(np.mean(mu_w)) if mu_w.size else 0.5
        Bw_avg = float(np.mean(Bw)) if Bw.size else 1.0
        invBw_mu_w_avg = 1.0 / (max(Bw_avg, 1e-12) * max(mu_w_avg, 1e-12))

        for link in frac_links:
            i = int(link["i"])
            j = int(link["j"])
            if not (0 <= i < Nf and 0 <= j < Nf):
                continue

            col_i = frac_map[i]
            col_j = frac_map[j]

            k_md = float(link.get("k_md", 50000.0))             # Perm of intersection
            L_ft = float(link.get("L_ft", 100.0))               # Trace length
            aperture_ft = float(link.get("aperture_ft", 0.001)) # Effective aperture
            d_ft = float(link.get("d_ft", 10.0))                # Center distance

            area_ft2 = L_ft * aperture_ft
            Tff_base = (max(k_md, 1e-30) * area_ft2) / max(d_ft, 1e-30)
            Tff = Tff_base * invBw_mu_w_avg

            dP_ff = Pf[i] - Pf[j]
            Fw_ff = Tff * dP_ff  # single-phase proxy flux

            # Residuals (flux leaves i, enters j)
            R[col_i] -= Fw_ff
            R[col_j] += Fw_ff

            # Jacobian
            A[col_i, col_i] += Tff
            A[col_i, col_j] -= Tff
            A[col_j, col_i] -= Tff
            A[col_j, col_j] += Tff

    # -------------------- boundary faces: fixed-P & aquifers ------------------

    kx_mat = np.asarray(rock.get("kx_md", 100.0)).reshape(nz, ny, nx)
    ky_mat = np.asarray(rock.get("ky_md", 100.0)).reshape(nz, ny, nx)

    def add_dirichlet_face(c, face, area, half_d, p_ext):
        """
        Weakly impose fixed boundary pressure by treating it as a half-cell
        with transmissibility T = k_dir * A / d_half and a ghost pressure p_ext.
        """
        if face in ("xmin","xmax"): kdir = kx[c]
        elif face in ("ymin","ymax"): kdir = ky[c]
        else: kdir = kz[c]

        T = (max(kdir, 1e-30) * area) / max(half_d, 1e-30)

        invBo_c = 1.0 / max(Bo[c], 1e-12)
        invBg_c = 1.0 / max(Bg[c], 1e-12)
        invBw_c = 1.0 / max(Bw[c], 1e-12)

        lam_o_c, lam_w_c, lam_g_c = lam_o[c], lam_w[c], lam_g[c]
        Rs_c = Rs[c]

        dP = P[c] - p_ext

        Fo = T * lam_o_c * invBo_c * dP
        Fw = T * lam_w_c * invBw_c * dP
        Fg = T * (lam_g_c * invBg_c * dP + (Rs_c * lam_o_c * invBo_c) * dP)

        R[_cell_idx(c)]     += Fo
        R[_cell_idx_sw(c)]  += Fw
        R[_cell_idx_sg(c)]  += Fg

        A[_cell_idx(c),    _cell_idx(c)]    += T * lam_o_c * invBo_c
        A[_cell_idx_sw(c), _cell_idx(c)]    += T * lam_w_c * invBw_c
        A[_cell_idx_sg(c), _cell_idx(c)]    += T * (lam_g_c * invBg_c + Rs_c * lam_o_c * invBo_c)

        # P-derivatives through PVT
        A[_cell_idx(c), _cell_idx(c)]     += T * lam_o_c * dP * (-dBo_dP[c] / (max(Bo[c],1e-12)**2))
        A[_cell_idx_sw(c), _cell_idx(c)]  += T * lam_w_c * dP * (-dBw_dP[c] / (max(Bw[c],1e-12)**2))
        A[_cell_idx_sg(c), _cell_idx(c)]  += T * (
            lam_g_c * dP * (-dBg_dP[c] / (max(Bg[c],1e-12)**2))
            + lam_o_c * dP * ((dRs_dP[c]*Bo[c] - Rs_c*dBo_dP[c]) / (max(Bo[c],1e-12)**2))
        )

    def add_aquifer_face(c, face, area, half_d, aq_idx, pi_mult, col_paq):
        """
        Simple “aquifer tank” coupling at a boundary face.
        """
        if face in ("xmin","xmax"): kdir = kx[c]
        elif face in ("ymin","ymax"): kdir = ky[c]
        else: kdir = kz[c]

        T0 = (max(kdir, 1e-30) * area) / max(half_d, 1e-30)
        T = pi_mult * T0

        invBw_c = 1.0 / max(Bw[c], 1e-12)
        lam_w_c = lam_w[c]

        # Water flux between cell c and aquifer unknown Paq
        A[_cell_idx_sw(c), _cell_idx(c)]   += T * lam_w_c * invBw_c
        A[_cell_idx_sw(c), col_paq]        += -T * lam_w_c * invBw_c
        R[_cell_idx_sw(c)]                 += T * lam_w_c * invBw_c * P[c]

        # d(Bw)/dP effect
        A[_cell_idx_sw(c), _cell_idx(c)]  += T * lam_w_c * (P[c]) * (-dBw_dP[c] / (max(Bw[c],1e-12)**2))

        row_aq = col_paq
        A[row_aq, _cell_idx(c)] += -T * lam_w_c * invBw_c
        A[row_aq, col_paq]      += +T * lam_w_c * invBw_c
        R[row_aq]               += 0.0  # storage term added below

    def _iter_boundary_cells(grid):
        """
        Generator over all boundary faces, yielding:
        (cell_index, face_name, face_area, half_distance)
        """
        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        dx, dy, dz = float(grid["dx"]), float(grid["dy"]), float(grid["dz"])

        def lin(i,j,k): return (k*ny + j)*nx + i

        for k in range(nz):
            for j in range(ny):
                yield lin(0, j, k), "xmin", (dy*dz), 0.5*dx
                yield lin(nx-1, j, k), "xmax", (dy*dz), 0.5*dx

        for k in range(nz):
            for i in range(nx):
                yield lin(i, 0, k), "ymin", (dx*dz), 0.5*dy
                yield lin(i, ny-1, k), "ymax", (dx*dz), 0.5*dy

        for j in range(ny):
            for i in range(nx):
                yield lin(i, j, 0), "zmin", (dx*dy), 0.5*dz
                yield lin(i, j, nz-1), "zmax", (dx*dy), 0.5*dz

    # Apply fixed pressure faces
    for c, face, area, half_d in _iter_boundary_cells(grid):
        if face in options.get("fixed_p_face_map", {}):
            add_dirichlet_face(c, face, area, half_d, options["fixed_p_face_map"][face])

    # Apply aquifer faces
    for c, face, area, half_d in _iter_boundary_cells(grid):
        if face in options.get("aquifer_face_map", {}):
            for aq_idx, pi_mult in options.get("aquifer_face_map", {})[face]:
                col_paq = aq_map[aq_idx]
                add_aquifer_face(c, face, area, half_d, aq_idx, pi_mult, col_paq)

    # Aquifer storage (tank compressibility) terms
    for aq_idx, col in aq_map.items():
        C = float(options["aquifers"][aq_idx].get("C_bbl_per_psi", 0.0))
        Paq_prev = float(options["prev_aquifer_pressures"].get(
            aq_idx, options["aquifers"][aq_idx].get("p_init_psi", 3000.0)
        ))
        A[col, col] += C / dt
        R[col]      += -C * Paq_prev / dt

    # Wells
    wells = options.get("prepared_wells", []) or []

    def wi_at(i, j, k, rw, skin):
        return peaceman_wi_cartesian(kx_mat[k, j, i], ky_mat[k, j, i], dz, dx, dy,
                                     rw_ft=rw, skin=skin)

    for w_idx, w in enumerate(wells):
        ctrl = str(w.get("control", "BHP")).upper()
        perfs = w["perfs"]
        q_tgt = float(w.get("rate_mscfd", w.get("rate_stbd", 0.0)))
        pw_col = options.get("well_unknown_map", {}).get(w_idx, None)
        limits = w.get("limits", {"pw": float(w.get("bhp_psi", 2500.0))})

        total_coef = 0.0
        if pw_col is not None:
            row_w = pw_col
        else:
            row_w = None

        for p in perfs:
            i, j, k = int(p["i"]), int(p["j"]), int(p["k"])
            c = (k*ny + j)*nx + i
            wi = wi_at(i, j, k, float(p.get("rw_ft", 0.35)), float(p.get("skin", 0.0)))

            muo = max(mu_o[c], 1e-12)
            mug = max(mu_g[c], 1e-12)
            muw = max(mu_w[c], 1e-12)

            lam_o_c = max(0.0, kro_end * (So[c] ** no) / muo)
            lam_w_c = max(0.0, krw_end * (Sw[c] ** nw) / muw)
            lam_g_c = max(0.0, (Sg[c] ** 2) / mug)

            Bo_c = max(Bo[c], 1e-12)
            Bw_c = max(Bw[c], 1e-12)
            Bg_c = max(Bg[c], 1e-12)
            Rs_c = Rs[c]

            co = wi * lam_o_c / Bo_c
            cw = wi * lam_w_c / Bw_c
            cg = wi * (lam_g_c / Bg_c + (Rs_c * lam_o_c) / Bo_c)

            if pw_col is None:
                # BHP controlled — no extra unknown, inject residual directly
                dP = P[c] - limits["pw"]
                R[_cell_idx(c)]     += co * dP
                R[_cell_idx_sw(c)]  += cw * dP
                R[_cell_idx_sg(c)]  += cg * dP

                A[_cell_idx(c),    _cell_idx(c)] += co
                A[_cell_idx_sw(c), _cell_idx(c)] += cw
                A[_cell_idx_sg(c), _cell_idx(c)] += cg

                # saturation sensitivities
                dlamo_dSo = kro_end*no*max(So[c],1e-12)**(no-1.0) / muo
                dlamw_dSw = krw_end*nw*max(Sw[c],1e-12)**(nw-1.0) / muw
                dlamo_dSw = -dlamo_dSo
                dlamo_dSg = -dlamo_dSo
                dlamg_dSg = 2.0 * max(Sg[c], 0.0) / mug

                A[_cell_idx(c),    _cell_idx_sw(c)] += wi * (dlamo_dSw / Bo_c) * dP
                A[_cell_idx(c),    _cell_idx_sg(c)] += wi * (dlamo_dSg / Bo_c) * dP

                A[_cell_idx_sw(c), _cell_idx_sw(c)] += wi * (dlamw_dSw / Bw_c) * dP

                A[_cell_idx_sg(c), _cell_idx_sg(c)] += (wi*(dlamg_dSg/Bg_c) + wi*((Rs_c*dlamo_dSg)/Bo_c)) * dP
                A[_cell_idx_sg(c), _cell_idx_sw(c)] += wi * ((Rs_c*dlamo_dSw)/Bo_c) * dP

            else:
                # TRUE-rate control — add equivalent BHP unknown
                A[_cell_idx(c),    _cell_idx(c)] += co
                A[_cell_idx(c),    pw_col]       += -co
                R[_cell_idx(c)]                  += co * P[c]

                A[_cell_idx_sw(c), _cell_idx(c)] += cw
                A[_cell_idx_sw(c), pw_col]       += -cw
                R[_cell_idx_sw(c)]               += cw * P[c]

                A[_cell_idx_sg(c), _cell_idx(c)] += cg
                A[_cell_idx_sg(c), pw_col]       += -cg
                R[_cell_idx_sg(c)]               += cg * P[c]

                # saturation sensitivities
                dlamo_dSo = kro_end*no*max(So[c],1e-12)**(no-1.0) / muo
                dlamw_dSw = krw_end*nw*max(Sw[c],1e-12)**(nw-1.0) / muw
                dlamo_dSw = -dlamo_dSo
                dlamo_dSg = -dlamo_dSo
                dlamg_dSg = 2.0 * max(Sg[c], 0.0) / mug

                A[_cell_idx(c),    _cell_idx_sw(c)] += wi * (dlamo_dSw/Bo_c) * P[c]
                A[_cell_idx(c),    _cell_idx_sg(c)] += wi * (dlamo_dSg/Bo_c) * P[c]

                A[_cell_idx_sw(c), _cell_idx_sw(c)] += wi * (dlamw_dSw/Bw_c) * P[c]

                A[_cell_idx_sg(c), _cell_idx_sg(c)] += (wi*(dlamg_dSg/Bg_c) + wi*((Rs_c*dlamo_dSg)/Bo_c)) * P[c]
                A[_cell_idx_sg(c), _cell_idx_sw(c)] += wi * ((Rs_c*dlamo_dSw)/Bo_c) * P[c]

            if ctrl == "RATE_GAS_MSCFD":
                total_coef += cg
            elif ctrl == "RATE_OIL_STBD":
                total_coef += co
            elif ctrl == "RATE_LIQ_STBD":
                total_coef += (co + cw)

        if pw_col is not None:
            # Rate constraint equation (sum of coef * (Pcell - pw) = q_target)
            p0 = perfs[0]
            c0 = (int(p0["k"])*ny + int(p0["j"])) * nx + int(p0["i"])
            A[pw_col, _cell_idx(c0)] += total_coef
            A[pw_col, pw_col]        += -total_coef
            R[pw_col]                += -q_tgt

    meta = {"note": "TPFA black-oil with φ(P), Bw(P), Pc (opt), gravity, fixed-P BCs, aquifer tanks, BHP/TRUE-rate wells, and EDFM (matrix–fracture, optional fracture–fracture) + Phase-2 diagonal leak."}
    return A.tocsr(), R, meta


# =============================================================================
# End Part 1/4
# =============================================================================
# core/full3d.py — UPDATED (Part 2/4)
# =============================================================================
# PVT model (replacement), scheduling helpers, fracture/well/aquifer prep
# =============================================================================


# --------------------------- REPLACEMENT: _SimplePVT --------------------------
class _SimplePVT:
    """Corrected PVT class with self argument."""

    def __init__(self, pb_psi=3000.0, Bo_pb=1.2, Rs_pb=600.0,
                 mu_o_cp=1.2, mu_g_cp=0.02, Bw_ref=1.0, cw_1overpsi=2.5e-6, mu_w_cp=0.5):
        self.pb = float(pb_psi)
        self.Bo_pb = float(Bo_pb)
        self.Rs_pb = float(Rs_pb)
        self.mu_o_cp = float(mu_o_cp)
        self.mu_g_cp = float(mu_g_cp)
        self.Bw_ref = float(Bw_ref)
        self.cw = float(cw_1overpsi)
        self.mu_w_cp = float(mu_w_cp)

    def Bo(self, P):
        P = np.asarray(P, float)
        return self.Bo_pb * np.exp(1e-5 * (self.pb - P))

    def Bg(self, P):
        P = np.asarray(P, float)
        return 0.005 * np.exp(-3e-4 * (self.pb - P))

    def Rs(self, P):
        P = np.asarray(P, float)
        # Use max(1.0, self.pb) in denominator to avoid blowups at pb=0
        return np.minimum(self.Rs_pb, self.Rs_pb * np.clip(P / max(1.0, self.pb), 0.0, None))

    def mu_o(self, P):
        return np.full_like(np.asarray(P, float), self.mu_o_cp)

    def mu_g(self, P):
        return np.full_like(np.asarray(P, float), self.mu_g_cp)

    def Bw(self, P):
        return self.Bw_ref * np.exp(-self.cw * (np.asarray(P, float) - self.pb))

    def mu_w(self, P):
        # NOTE: Removed stray trailing period that would cause a syntax error.
        return np.full_like(np.asarray(P, float), self.mu_w_cp)


# ------------------------------ Scheduling helpers ---------------------------

def _active_wells_at_time(t_days: float, schedule: dict):
    """
    Resolve “active wells” at the current time t_days using either:
      - an explicit event list, or
      - a static wells list on the schedule.
    Fallback: a single BHP-controlled well using schedule defaults.
    """
    if isinstance(schedule.get("well_events"), list) and len(schedule["well_events"]) > 0:
        active = []
        for ev in schedule["well_events"]:
            t0 = float(ev.get("start_day", -np.inf))
            t1 = float(ev.get("end_day", np.inf))
            if t0 <= t_days < t1:
                ws = ev.get("wells", [])
                active.extend(ws)
        if len(active) > 0:
            return active

    if isinstance(schedule.get("wells"), list) and len(schedule["wells"]) > 0:
        return schedule["wells"]

    return [{
        "control": schedule.get("control", "BHP"),
        "bhp_psi": float(schedule.get("bhp_psi", 2500.0)),
    }]


def _well_perf_list(w, nx, ny, nz):
    """
    Normalize a well’s perforation list into a list of dicts with fields:
    (i, j, k, rw_ft, skin). If missing, default to a single perf at grid center.
    """
    base = {"rw_ft": w.get("rw_ft", 0.35), "skin": w.get("skin", 0.0)}
    perfs = []

    if isinstance(w.get("perfs"), (list, tuple)) and len(w["perfs"]) > 0:
        for p in w["perfs"]:
            perfs.append({
                "i": int(p.get("i", nx // 2)),
                "j": int(p.get("j", ny // 2)),
                "k": int(p.get("k", nz // 2)),
                # BUGFIX from original: removed extra closing parenthesis
                "rw_ft": float(p.get("rw_ft", base["rw_ft"])),
                "skin": float(p.get("skin",  base["skin"])),
            })
    else:
        perfs.append({
            "i": int(w.get("i", nx // 2)),
            "j": int(w.get("j", ny // 2)),
            "k": int(w.get("k", nz // 2)),
            "rw_ft": float(base["rw_ft"]),
            "skin": float(base["skin"]),
        })
    return perfs


def _estimate_pw_for_rate(ctrl, Pcell, co, cw, cg, q_target):
    """
    Back-out an equivalent BHP if we are asked to enforce a TRUE rate.
    """
    if ctrl == "RATE_GAS_MSCFD":
        coef = max(cg, 1e-30)
    elif ctrl == "RATE_OIL_STBD":
        coef = max(co, 1e-30)
    elif ctrl == "RATE_LIQ_STBD":
        coef = max(co + cw, 1e-30)
    else:
        return Pcell
    return Pcell - q_target / coef


def _prepare_wells_for_step(P, grid, rock, relperm, wells, options, pvt):
    """
    Prepare well control for this time step. When in “true_rate” mode, we add a
    BHP unknown to the system for each rate-controlled well (one per well).
    """
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    dx, dy, dz = float(grid["dx"]), float(grid["dy"]), float(grid["dz"])

    kx_mat = np.asarray(rock.get("kx_md", 100.0)).reshape(nz, ny, nx)
    ky_mat = np.asarray(rock.get("ky_md", 100.0)).reshape(nz, ny, nx)

    nw = float(relperm.get("nw", 2.0))
    no = float(relperm.get("no", 2.0))
    krw_end = float(relperm.get("krw_end", 0.6))
    kro_end = float(relperm.get("kro_end", 0.8))

    prepared = []
    want_true = options.get("well_mode", "equivalent_bhp").lower() == "true_rate"

    for w in wells:
        ctrl = str(w.get("control", "BHP")).upper()
        perfs = _well_perf_list(w, nx, ny, nz)

        p0 = perfs[0]
        i, j, k = int(p0["i"]), int(p0["j"]), int(p0["k"])
        c = (k*ny + j)*nx + i

        wi = peaceman_wi_cartesian(kx_mat[k, j, i], ky_mat[k, j, i], dz, dx, dy,
                                   rw_ft=float(p0.get("rw_ft", 0.35)),
                                   skin=float(p0.get("skin", 0.0)))

        # Rough in-cell estimates for initial mobility
        Sw_est, Sg_est = 0.2, 0.05
        So_est = 1.0 - Sw_est - Sg_est
        mu_o = pvt.mu_o(P[c]); mu_g = pvt.mu_g(P[c]); mu_w = pvt.mu_w(P[c])
        Bo_c = pvt.Bo(P[c]);   Bg_c = pvt.Bg(P[c]);   Bw_c = pvt.Bw(P[c]); Rs_c = pvt.Rs(P[c])

        lam_o = max(0.0, kro_end * (So_est**no) / max(mu_o, 1e-12))
        lam_w = max(0.0, krw_end * (Sw_est**nw) / max(mu_w, 1e-12))
        lam_g = max(0.0, (Sg_est**2) / max(mu_g, 1e-12))

        co = wi * lam_o / max(Bo_c, 1e-12)
        cw = wi * lam_w / max(Bw_c, 1e-12)
        cg = wi * (lam_g / max(Bg_c, 1e-12) + (Rs_c * lam_o) / max(Bo_c, 1e-12))

        pw_min = w.get("pw_min_psi", None)
        pw_max = w.get("pw_max_psi", None)

        if want_true and ctrl.startswith("RATE"):
            q_tgt = float(w.get("rate_mscfd", w.get("rate_stbd", 0.0)))
            pw_est = _estimate_pw_for_rate(ctrl, P[c], co, cw, cg, q_tgt)

            violate_min = (pw_min is not None and pw_est < float(pw_min))
            violate_max = (pw_max is not None and pw_est > float(pw_max))

            if violate_min or violate_max:
                # Fallback to BHP if rate would violate the bounds
                eff_bhp = float(pw_min) if violate_min else float(pw_max)
                prepared.append({
                    "control": "BHP",
                    "perfs": perfs,
                    "bhp_psi": eff_bhp,
                    "limits": {"pw": eff_bhp},
                    "rate_mscfd": float(w.get("rate_mscfd", 0.0)),
                    "rate_stbd":  float(w.get("rate_stbd", 0.0)),
                })
            else:
                prepared.append({
                    "control": ctrl,
                    "perfs": perfs,
                    "rate_mscfd": float(w.get("rate_mscfd", 0.0)),
                    "rate_stbd":  float(w.get("rate_stbd", 0.0)),
                })
        else:
            # Always BHP-controlled in equivalent_bhp mode.
            eff_bhp = float(w.get("bhp_psi", options.get("bhp_psi", 2500.0)))
            prepared.append({
                "control": "BHP",
                "perfs": perfs,
                "bhp_psi": eff_bhp,
                "limits": {"pw": eff_bhp},
                "rate_mscfd": float(w.get("rate_mscfd", 0.0)),
                "rate_stbd":  float(w.get("rate_stbd", 0.0)),
            })

    # Build a compact unknown map for the “equivalent BHP” variables
    well_unknown_map = {}
    col = 3 * nx * ny * nz
    for w_idx, w in enumerate(prepared):
        if w["control"].startswith("RATE"):
            well_unknown_map[w_idx] = col
            col += 1

    return prepared, well_unknown_map


def _prepare_aquifers_for_step(grid, options, prev_aquifer_state):
    """
    Setup aquifer unknowns and face coupling for this step. Returns:
      aquifers, aquifer_unknown_map, face_map, prev_press
    """
    aquifers = options.get("aquifers", []) or []
    if len(aquifers) == 0:
        return [], {}, {}, {}

    aquifer_unknown_map = {}
    face_map = {}
    prev_press = {}

    for idx, aq in enumerate(aquifers):
        face = aq.get("face", "zmin")
        pi_mult = float(aq.get("pi_multiplier", 1.0))
        face_map.setdefault(face, []).append((idx, pi_mult))
        prev_press[idx] = float(prev_aquifer_state.get(idx, aq.get("p_init_psi", 3000.0)))

    return aquifers, aquifer_unknown_map, face_map, prev_press


def _prepare_fractures_for_step(P_matrix, grid, options, well_map, aquifer_map):
    """
    Returns (frac_cells, frac_unknown_map, Pf0_vec)

    • frac_cells: list of dicts, each with keys:
        cell, area_ft2, normal, aperture_ft, k_md (optional)
    • frac_unknown_map: {local_index -> global_col}
    • Pf0_vec: initial fracture pressures seeded from host matrix cell

    NOTE: We append fracture unknowns AFTER wells and aquifers.
    """
    frac_cells = options.get("frac_cells", []) or []
    Nf = len(frac_cells)
    if Nf == 0:
        return [], {}, np.empty(0, float)

    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    base = 3*nx*ny*nz + len(well_map) + len(aquifer_map)

    frac_unknown_map = {i: base + i for i in range(Nf)}
    Pf0 = _frac_initial_pressures(frac_cells, P_matrix)
    return frac_cells, frac_unknown_map, Pf0


def _compute_well_rates(P, Sw, Sg, So, Bo, Bg, Bw, Rs, mu_o, mu_g, mu_w,
                        grid, rock, relperm, prepared_wells, well_solution_pw=None):
    """
    Compute total well rates (surface units) by looping perfs and applying
    Peaceman WI + mobilities with the solved BHPs.
    """
    nx, ny, nz = [int(grid[k]) for k in ("nx", "ny", "nz")]
    dx, dy, dz = [float(grid[k]) for k in ("dx", "dy", "dz")]

    def lin(i, j, k): return (k*ny + j)*nx + i

    nw = float(relperm.get("nw", 2.0))
    no = float(relperm.get("no", 2.0))
    krw_end = float(relperm.get("krw_end", 0.6))
    kro_end = float(relperm.get("kro_end", 0.8))

    kx = np.asarray(rock.get("kx_md")).reshape(nz, ny, nx)
    ky = np.asarray(rock.get("ky_md")).reshape(nz, ny, nx)

    qo = qg = qw = 0.0

    for w_idx, w in enumerate(prepared_wells):
        ctrl = w["control"]
        perfs = w["perfs"]

        # Determine well pressure to use for this well
        if ctrl == "BHP":
            pw_val = float(w.get("bhp_psi", w.get("limits", {}).get("pw", 2500.0)))
        elif well_solution_pw and w_idx in well_solution_pw:
            pw_val = well_solution_pw[w_idx]
        else:
            # For rate-controlled well without a solved BHP, skip contribution
            # (should not happen post-solve).
            continue

        for p in perfs:
            i = int(p["i"]); j = int(p["j"]); k = int(p["k"])
            c = lin(i, j, k)

            wi = peaceman_wi_cartesian(kx[k, j, i], ky[k, j, i], dz, dx, dy,
                                       rw_ft=float(p.get("rw_ft", 0.35)),
                                       skin=float(p.get("skin", 0.0)))

            muo = max(mu_o[c], 1e-12)
            mug = max(mu_g[c], 1e-12)
            muw = max(mu_w[c], 1e-12)

            lam_o = max(0.0, kro_end*(So[c]**no) / muo)
            lam_w = max(0.0, krw_end*(Sw[c]**nw) / muw)
            lam_g = max(0.0, (Sg[c]**2) / mug)

            Bo_c = max(Bo[c], 1e-12)
            Bg_c = max(Bg[c], 1e-12)
            Bw_c = max(Bw[c], 1e-12)
            Rs_c = Rs[c]

            co = wi * lam_o / Bo_c
            cw = wi * lam_w / Bw_c
            cg = wi * (lam_g / Bg_c + (Rs_c * lam_o) / Bo_c)

            dP = P[c] - pw_val

            qo += co * dP
            qg += cg * dP
            qw += cw * dP

    return qo, qg, qw


# =============================================================================
# End Part 2/4
# =============================================================================
# core/full3d.py — UPDATED (Part 3/4)
# =============================================================================
# Inputs builder (accepts frac_cells/links), Newton driver (replacement), proxy
# =============================================================================


def _simulate_analytical_proxy(inputs: dict):
    """
    Lightweight built-in proxy used only if you explicitly select engine='analytical'
    and do not have engines.fast available via simulate()’s fallback.
    This is NOT intended to be a rigorous type-curve or ARPS model—just a
    stable, deterministic curve for UX checks and smoke tests.
    """
    t = np.linspace(1.0, 3650.0, 240)
    qi_g, di_g = 8000.0, 0.80
    qi_o, di_o = 1000.0, 0.70
    years = t/365.25
    qg = qi_g*np.exp(-di_g*years)
    qo = qi_o*np.exp(-di_o*years)
    return {"t": t, "qg": qg, "qo": qo, "press_matrix": None, "pm_mid_psi": None,
            "p_init_3d": None, "ooip_3d": None}


def _build_inputs_for_blackoil(inputs):
    """
    Construct typed inputs for the implicit engine from a flat dict.

    Returns
    -------
    state0, grid, rock, relperm, init, schedule, options, pvt
    """
    nx = int(inputs.get("nx", 10))
    ny = int(inputs.get("ny", 10))
    nz = int(inputs.get("nz", 5))

    dx = float(inputs.get("dx", 100.0))
    dy = float(inputs.get("dy", 100.0))
    dz = float(inputs.get("dz", 50.0))

    grid = {"nx": nx, "ny": ny, "nz": nz, "dx": dx, "dy": dy, "dz": dz}
    N = nx*ny*nz

    phi_val = float(inputs.get("phi", 0.08))
    kx_val  = float(inputs.get("kx_md", 100.0))
    ky_val  = float(inputs.get("ky_md", 100.0))

    rock = {
        "phi":   np.full(N, phi_val, float),
        "kx_md": np.full(N, kx_val,  float),
        "ky_md": np.full(N, ky_val,  float),
        "cr_1overpsi": float(inputs.get("cr_1overpsi", 0.0)),
        "p_ref_psi":   float(inputs.get("p_ref_psi", inputs.get("p_init_psi", 5000.0))),
    }

    p_init = float(inputs.get("p_init_psi", 5000.0))
    Sw0 = float(inputs.get("Sw0", 0.20))
    Sg0 = float(inputs.get("Sg0", 0.05))

    P0  = np.full(N, p_init, float)
    Sw  = np.full(N, Sw0,   float)
    Sg  = np.full(N, Sg0,   float)

    state0 = np.empty(3*N, float)
    state0[0::3] = P0
    state0[1::3] = Sw
    state0[2::3] = Sg

    init = {"p_init_psi": p_init}

    relperm = {
        "nw": float(inputs.get("nw", 2.0)),
        "no": float(inputs.get("no", 2.0)),
        "krw_end": float(inputs.get("krw_end", 0.6)),
        "kro_end": float(inputs.get("kro_end", 0.8)),
    }

    schedule = {
        "control":        str(inputs.get("control", "BHP")),
        "bhp_psi":        float(inputs.get("bhp_psi", 2500.0)),
        "pad_bhp_psi":    float(inputs.get("pad_bhp_psi", inputs.get("bhp_psi", 2500.0))),
        "rate_mscfd":     float(inputs.get("rate_mscfd", 0.0)),
        "pad_rate_mscfd": float(inputs.get("pad_rate_mscfd", inputs.get("rate_mscfd", 0.0))),
        "rate_stbd":      float(inputs.get("rate_stbd", 0.0)),
        "pad_rate_stbd":  float(inputs.get("pad_rate_stbd", inputs.get("rate_stbd", 0.0))),
        "wells": inputs.get("wells", None),
        "well_events": inputs.get("well_events", None),
    }

    options = {
        "dt_days": float(inputs.get("dt_days", 30.0)),
        "t_end_days": float(inputs.get("t_end_days", 3650.0)),

        "use_gravity": bool(inputs.get("use_gravity", True)),
        "rho_o_lbft3": float(inputs.get("rho_o_lbft3", DEFAULT_OPTIONS["rho_o_lbft3"])),
        "rho_w_lbft3": float(inputs.get("rho_w_lbft3", DEFAULT_OPTIONS["rho_w_lbft3"])),
        "rho_g_lbft3": float(inputs.get("rho_g_lbft3", DEFAULT_OPTIONS["rho_g_lbft3"])),

        "kvkh": float(inputs.get("kvkh", DEFAULT_OPTIONS["kvkh"])),
        "geo_alpha": float(inputs.get("geo_alpha", DEFAULT_OPTIONS["geo_alpha"])),
        "use_pc": bool(inputs.get("use_pc", DEFAULT_OPTIONS["use_pc"])),

        "well_mode": str(inputs.get("well_mode", DEFAULT_OPTIONS["well_mode"])),

        "fixed_p_bounds": inputs.get("fixed_p_bounds", []),
        "aquifers": inputs.get("aquifers", []),

        # ---- EDFM passthrough ----
        "frac_cells": inputs.get("frac_cells", []),
        "frac_links": inputs.get("frac_links", []),

        # ---- EUR options / guardrails ----
        "eur_oil_cutoff_stbd": float(inputs.get("eur_oil_cutoff_stbd", 30.0)),
        "eur_gas_cutoff_mscfd": float(inputs.get("eur_gas_cutoff_mscfd", 100.0)),
        "eur_cap_days": float(inputs.get("eur_cap_days", 10950.0)),  # ~30 years
        "eur_basin": str(inputs.get("eur_basin", "Midland")),
        "eur_window": str(inputs.get("eur_window", "oil")),
        "eur_oil_bounds_mmbo": tuple(inputs.get("eur_oil_bounds_mmbo", (0.3, 1.5))),
        "eur_gas_bounds_bcf": tuple(inputs.get("eur_gas_bounds_bcf", (0.3, 3.0))),
    }

    pvt = _SimplePVT(
        pb_psi=float(inputs.get("pb_psi", 3000.0)),
        Bo_pb=float(inputs.get("Bo_pb_rb_stb", 1.2)),
        Rs_pb=float(inputs.get("Rs_pb_scf_stb", 600.0)),
        mu_o_cp=float(inputs.get("mu_o_cp", 1.2)),
        mu_g_cp=float(inputs.get("mu_g_cp", 0.02)),
        Bw_ref=float(inputs.get("Bw_ref", 1.0)),
        cw_1overpsi=float(inputs.get("cw_1overpsi", 2.5e-6)),
        mu_w_cp=float(inputs.get("mu_w_cp", 0.5)),
    )
    return state0, grid, rock, relperm, init, schedule, options, pvt


# ------------------------- REPLACEMENT: newton_solve_blackoil ----------------

def newton_solve_blackoil(state0, grid, rock, relperm, init, schedule, options, pvt):
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    N = nx * ny * nz
    assert state0.size == 3 * N

    dx, dy, dz = float(grid["dx"]), float(grid["dy"]), float(grid["dz"])
    Vcell_ft3 = dx * dy * dz

    P0_vec  = state0[0::3].copy()
    Sw0_vec = state0[1::3].copy()
    Sg0_vec = state0[2::3].copy()
    So0_vec = 1.0 - Sw0_vec - Sg0_vec

    phi_vec = np.asarray(rock.get("phi", 0.08)).reshape(N)
    Bo0_vec = np.asarray(pvt.Bo(P0_vec))
    p_avg_hist: list[float] = []

    dt_days = float(options.get("dt_days", 30.0))
    t_end   = float(options.get("t_end_days", 3650.0))
    max_newton = int(options.get("max_newton", 10))
    tol = float(options.get("newton_tol", 1e-6))

    t_hist, qo_hist, qg_hist, qw_hist = [], [], [], []

    Pm = state0[0::3].copy()
    Swm = state0[1::3].copy()
    Sgm = state0[2::3].copy()

    t = 0.0
    state = state0.copy()

    aquifer_state_prev = {}

    while t < t_end - 1e-9:
        wells_for_step = _active_wells_at_time(t, schedule)

        prepared_wells, well_unknown_map = _prepare_wells_for_step(
            state[0::3], grid, rock, relperm, wells_for_step, options, pvt
        )

        aquifers, aquifer_unknown_map, aq_face_map, prev_aqP = _prepare_aquifers_for_step(
            grid, {**options, "aquifers": options.get("aquifers", [])}, aquifer_state_prev
        )

        frac_cells, frac_unknown_map, Pf0 = _prepare_fractures_for_step(
            state[0::3], grid, options, well_unknown_map, aquifer_unknown_map
        )

        nW = len(well_unknown_map)
        nA = len(aquifers)
        Nf = len(frac_cells)

        state_k = state.copy()
        next_col = 3*N

        if nW > 0:
            pw0 = []
            for w_idx in sorted(well_unknown_map.keys(), key=lambda x: well_unknown_map[x]):
                well_unknown_map[w_idx] = next_col
                next_col += 1
            for w_idx in sorted(well_unknown_map.keys(), key=lambda x: well_unknown_map[x]):
                w = prepared_wells[w_idx]
                p0 = w["perfs"][0]
                c0 = (int(p0["k"])*ny + int(p0["j"])) * nx + int(p0["i"])
                pw0.append(float(w.get("bhp_psi", state[0::3][c0])))
            state_k = np.concatenate([state_k, np.asarray(pw0, float)], axis=0)

        if nA > 0:
            for aq_idx in range(nA):
                aquifer_unknown_map[aq_idx] = next_col
                next_col += 1
            Paq0 = [
                float(prev_aqP.get(aq_idx, aquifers[aq_idx].get("p_init_psi", 3000.0)))
                for aq_idx in range(nA)
            ]
            state_k = np.concatenate([state_k, np.asarray(Paq0, float)], axis=0)

        if Nf > 0:
            for fi in range(Nf):
                frac_unknown_map[fi] = next_col
                next_col += 1
            state_k = np.concatenate([state_k, Pf0.astype(float)], axis=0)

        converged = False
        for _it in range(max_newton):
            step_opts = dict(options)
            step_opts["prev"] = {"P": Pm, "Sw": Swm, "Sg": Sgm}
            step_opts["prepared_wells"] = prepared_wells
            step_opts["well_unknown_map"] = well_unknown_map
            step_opts["aquifers"] = aquifers
            step_opts["aquifer_unknown_map"] = aquifer_unknown_map
            step_opts["aquifer_face_map"] = aq_face_map
            step_opts["prev_aquifer_pressures"] = prev_aqP
            step_opts["frac_cells"] = frac_cells
            step_opts["frac_unknown_map"] = frac_unknown_map

            fixed_bounds = options.get("fixed_p_bounds", []) or []
            step_opts["fixed_p_face_map"] = {
                fb["face"]: float(fb["p_psi"] if "p_psi" in fb else fb["p"])
                for fb in fixed_bounds
                if fb.get("face") in ("xmin","xmax","ymin","ymax","zmin","zmax")
                and ("p_psi" in fb or "p" in fb)
            }

            A, R, _ = assemble_jacobian_and_residuals_blackoil(
                state_k, grid, rock, pvt, relperm, init, schedule, step_opts
            )

            normR = float(np.linalg.norm(R, ord=np.inf))
            if normR < tol:
                converged = True
                break

            try:
                dx = spsolve(A, -R)
            except Exception:
                A = A.tolil()
                idx = np.arange(A.shape[0])
                A[idx, idx] = A[idx, idx] + 1e-12
                A = A.tocsr()
                dx = spsolve(A, -R)

            new_state = state_k + dx

            # Project saturations and renormalize So (matrix cells only)
            eps = 1e-9
            Pn  = new_state[0:3*N:3]
            Swn = np.clip(new_state[1:3*N:3], eps, 1.0 - eps)
            Sgn = np.clip(new_state[2:3*N:3], eps, 1.0 - eps)
            Son = 1.0 - Swn - Sgn
            bad = Son < eps
            if np.any(bad):
                total = np.maximum(Swn[bad] + Sgn[bad], 1e-12)
                scale = (1.0 - eps) / total
                Swn[bad] *= scale
                Sgn[bad] *= scale

            new_state[0:3*N:3] = Pn
            new_state[1:3*N:3] = Swn
            new_state[2:3*N:3] = Sgn
            state_k = new_state

        # Accept step (strip extra unknowns)
        state = state_k[:3*N]
        P = state[0::3]; Sw = state[1::3]; Sg = state[2::3]; So = 1.0 - Sw - Sg

        # Save aquifer pressures for next step
        if len(aquifer_unknown_map) > 0:
            for aq_idx, col in aquifer_unknown_map.items():
                aquifer_state_prev[aq_idx] = float(state_k[col])

        # Post-step well rates
        Bo = np.asarray(pvt.Bo(P)); Bg = np.asarray(pvt.Bg(P)); Rs = np.asarray(pvt.Rs(P))
        mu_o = np.asarray(pvt.mu_o(P)); mu_g = np.asarray(pvt.mu_g(P))
        mu_w = np.asarray(pvt.mu_w(P)); Bw = np.asarray(pvt.Bw(P))

        solved_pw = {w_idx: float(state_k[col]) for w_idx, col in well_unknown_map.items()}
        qo, qg, qw = _compute_well_rates(
            P, Sw, Sg, So, Bo, Bg, Bw, Rs, mu_o, mu_g, mu_w,
            grid, rock, relperm, prepared_wells, solved_pw
        )

        t += dt_days
        t_hist.append(t); qo_hist.append(qo); qg_hist.append(qg); qw_hist.append(qw)
        p_avg_hist.append(float(np.mean(P)))

        Pm, Swm, Sgm = P.copy(), Sw.copy(), Sg.copy()

    # HARMONIZE OUTPUT KEYS to match app expectations
    return {
        "t": np.asarray(t_hist, float),
        "qo": np.asarray(qo_hist, float),
        "qg": np.asarray(qg_hist, float),
        "qw": np.asarray(qw_hist, float),
        "p_initial": P0_vec.reshape(nz, ny, nx),
        "p_final": state[0::3].reshape(nz, ny, nx),
        "ooip_3d": ((Vcell_ft3 * phi_vec * So0_vec) / (Bo0_vec * 5.614583)).reshape(nz, ny, nx),
        "p_avg_psi": np.asarray(p_avg_hist, float),
    }


# =============================================================================
# End Part 3/4
# =============================================================================
# core/full3d.py — UPDATED (Part 4/4)
# =============================================================================
# Top-level simulate() API (replacement) + compute_eur_from_series + docs
# =============================================================================

# Optional fast engine import; safe fallback below in simulate()
try:
    from engines.fast import fallback_fast_solver  # type: ignore
except Exception:
    def fallback_fast_solver(inputs, rng):
        """
        Safe local stub if engines.fast.fallback_fast_solver is not available.
        It defers to _simulate_analytical_proxy() so the simulator can still run.
        """
        _ = rng  # unused in this stub
        return _simulate_analytical_proxy(inputs)


# ------------------------------ REPLACEMENT: simulate ------------------------

def simulate(inputs: dict):
    """Main simulation entry point."""
    engine = inputs.get("engine", "analytical").lower().strip()

    if engine == "analytical":
        # Call the robust analytical solver from engines/fast.py (or stub)
        rng = np.random.default_rng(inputs.get("rng_seed", 1234))
        return fallback_fast_solver(inputs, rng)

    elif engine == "implicit":
        state0, grid, rock, relperm, init, schedule, options, pvt = _build_inputs_for_blackoil(inputs)
        out = newton_solve_blackoil(state0, grid, rock, relperm, init, schedule, options, pvt)

        # Post-process for EUR and other derived quantities
        (t_cap, cum_o_stb, cum_g_mscf, eur_o_mmbo, eur_g_bcf) = _compute_eur_and_cum(
            out.get("t"), out.get("qo"), out.get("qg"),
            oil_cutoff_stbd=options.get("eur_oil_cutoff_stbd", 30.0),
            gas_cutoff_mscfd=options.get("eur_gas_cutoff_mscfd", 100.0),
            cap_days=options.get("eur_cap_days", 10950.0)
        )

        eur_validation = _validate_eur(
            eur_o_mmbo, eur_g_bcf,
            basin=options.get("eur_basin", "Midland"),
            window=options.get("eur_window", "oil"),
            oil_bounds=options.get("eur_oil_bounds_mmbo", (0.3, 1.5)),
            gas_bounds=options.get("eur_gas_bounds_bcf", (0.3, 3.0))
        )

        out.update({
            "t_cap": t_cap,
            "cum_o_stb": cum_o_stb,
            "cum_g_mscf": cum_g_mscf,
            "eur_o_mmbo": eur_o_mmbo,
            "eur_g_bcf": eur_g_bcf,
            "eur_validation": eur_validation,
        })
        return out

    else:
        raise ValueError(f"Unknown engine '{engine}'")


# ------------------------------ Public EUR helper ----------------------------

def compute_eur_from_series(t_days, qo_stbd, qg_mscfd,
                            oil_cutoff_stbd=30.0, gas_cutoff_mscfd=100.0,
                            cap_days=10950.0,
                            basin="Midland", window="oil",
                            oil_bounds_mmbo=(0.3, 1.5), gas_bounds_bcf=(0.3, 3.0)):
    """
    Convenience wrapper to compute EURs from arbitrary time/rate series,
    with the same defaults/guardrails as the implicit pipeline.
    """
    t_cap, cum_o_stb, cum_g_mscf, eur_o_mmbo, eur_g_bcf = _compute_eur_and_cum(
        t_days, qo_stbd, qg_mscfd,
        oil_cutoff_stbd=oil_cutoff_stbd,
        gas_cutoff_mscfd=gas_cutoff_mscfd,
        cap_days=cap_days,
    )
    eur_validation = _validate_eur(
        eur_o_mmbo, eur_g_bcf,
        basin=basin, window=window,
        oil_bounds=oil_bounds_mmbo, gas_bounds=gas_bounds_bcf
    )
    return {
        "t_cap": t_cap,
        "cum_o_stb": cum_o_stb,
        "cum_g_mscf": cum_g_mscf,
        "eur_o_mmbo": eur_o_mmbo,
        "eur_g_bcf": eur_g_bcf,
        "eur_validation": eur_validation,
        "cutoffs": {
            "oil_stbd": oil_cutoff_stbd,
            "gas_mscfd": gas_cutoff_mscfd,
            "cap_days": cap_days
        }
    }


# =============================================================================
# Developer Notes / Inline Docs (long-form, intentionally verbose)
# =============================================================================
#
# 1) Units and conventions
#    ----------------------
#    Pressure: psi
#    Length:   ft
#    Volume:   ft^3 (bulk), STB (surface oil), Mscf (surface gas)
#    Rates:    STB/day, Mscf/day (water/oil/gas respectively at surface)
#
#    FVF definitions:
#      Bo = reservoir bbl / STB
#      Bw = reservoir bbl / STB (water)
#      Bg = reservoir bbl / Mscf
#
#    Hydrostatic head:
#      Δp ≈ ρ(lb/ft^3) * (1/144) psi/ft * Δz_ft
#      where 1 psi = 144 lb/ft^2 and g/gc ≈ 1 in oilfield units.
#
# 2) Stability and convergence tips
#    -------------------------------
#    • If you encounter non-convergence, consider:
#        - Reducing dt_days
#        - Enabling a line search (not included here by default; easy to add)
#        - Loosening/tightening tolerances depending on stiffness
#        - Checking relperm endpoints (kro_end, krw_end)
#    • For EDFM, ensure fracture areas/normals/apertures are consistent with the
#      grid scale; extreme apertures or negative distances will destabilize the
#      linear system.
#
# 3) Extending PVT
#    --------------
#    The _SimplePVT class here is deliberately simple and deterministic:
#      Bo, Bg, Bw are smooth exponentials; Rs capped at bubble point ratio.
#    For more realistic behavior, plug in tabular PVT (pressure-dependent) or
#    EOS-based correlations. Maintain the same API (Bo, Bg, Rs, mu_o, mu_g, Bw, mu_w).
#
# 4) Wells and controls
#    -------------------
#    The “true_rate” mode introduces one extra unknown per rate-controlled well:
#    an equivalent BHP. This keeps the nonlinearity manageable and the Jacobian
#    structure simple. Rate targets are enforced via a single well equation:
#      sum_j coef_j * (P_cell_j - pw) = q_target
#
# 5) Aquifer tanks
#    --------------
#    The aquifer model here is a single-pressure “tank” per boundary face group,
#    with a simple storage term (C/dt) and transmissibility to the face cells
#    scaled by an optional productivity multiplier. It’s intentionally crude but
#    often sufficient to emulate edge-water drive.
#
# 6) EDFM
#    -----
#    Matrix↔fracture flux uses projected normal k and a half-cell distance plus
#    half-aperture. Fracture↔fracture links model exchange along an intersection
#    trace using a water-like single-phase mobility for stability.
#
# 7) Guardrails
#    ----------
#    The EUR validator ensures output stays within plausible windows per basin
#    and fluid window. Calling code can decide how to present or override these
#    messages to the end user.
#
# 8) Performance
#    -----------
#    The Jacobian is assembled in lil_matrix then converted to CSR for solving.
#    For large grids or dense fracture networks, consider:
#      - Switching to specialized sparse linear solvers (AMG, etc.)
#      - Domain decomposition or tiling
#      - Vectorizing internal-face loops further
#
# 9) Testing
#    -------
#    Minimal smoke tests:
#      >>> out = simulate({"engine":"analytical"})
#      >>> out = simulate({"engine":"implicit", "nx":4, "ny":4, "nz":2})
#      >>> assert "t" in out and "qo" in out and "qg" in out
#
# 10) API stability
#     --------------
#     The public entrypoints are:
#       • simulate(inputs: dict) -> dict
#       • compute_eur_from_series(...)
#     _build_inputs_for_blackoil() is considered internal and subject to change.
#
# =============================================================================
# End Part 4/4
# =============================================================================
