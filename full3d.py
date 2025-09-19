# core/full3d.py
# -----------------------------------------------------------------------------
# Minimal working proxy simulate() so the app runs end-to-end,
# plus Phase 1 scaffolds:
#   1.1 Peaceman WI
#   1.2 Geomech k(p) multiplier
#   1.3 EDFM connectivity scaffold
#   1.4 Black-oil residual/Jacobian skeleton (3× unknowns: P, Sw, Sg)
# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
# ---------------- Gravity & Well helpers ----------------
DEFAULT_OPTIONS = {
    "use_gravity": True,
    "rho_o_lbft3": 53.0,
    "rho_w_lbft3": 62.4,
    "rho_g_lbft3": 0.06,
    "kvkh": 0.10,       # vertical anisotropy (kv/kh) default
    "geo_alpha": 0.0,   # simple geomech k-multiplier (0 = off)
}

def _z_centers(nz, dz):
    # cell-center depths (ft). k=0 is shallowest if you use top-down indexing.
    return (np.arange(nz, dtype=float) + 0.5) * float(dz)

def _peaceman_WI(kx_md, ky_md, dx, dy, dz, rw_ft=0.328, skin=0.0):
    """
    Peaceman Well Index (field units).
    kx/ky in mD, dx/dy/dz in ft, rw in ft.
    Returns WI consistent with transmissibility: includes 0.001127 factor.
    """
    kx = float(max(kx_md, 1.0e-8))
    ky = float(max(ky_md, 1.0e-8))
    dx = float(dx); dy = float(dy); dz = float(dz)
    k_eff = np.sqrt(kx * ky)          # mD
    # anisotropic equivalent radius
    re = 0.28 * np.sqrt((dx**2) * np.sqrt(ky/kx) + (dy**2) * np.sqrt(kx/ky))
    denom = np.log(max(re / max(rw_ft, 1.0e-6), 1.0)) + skin
    denom = max(denom, 1.0e-8)
    WI = 0.001127 * 2.0 * np.pi * k_eff * dz / denom   # (md*ft)→ rb/day/psi with 0.001127
    return WI

def _build_single_well(grid, rock, schedule):
    """
    Builds a single well centered in the model (safe default) using Peaceman WI.
    Uses schedule['control'] in {'BHP','RATE'} and bhp_psi/rate_mscfd if present.
    """
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    dx, dy, dz = float(grid["dx"]), float(grid["dy"]), float(grid["dz"])
    i = nx // 2
    j = ny // 2
    k = nz // 2

    kx = float(rock.get("kx_md")[k, j, i]) if rock.get("kx_md") is not None else 0.05
    ky = float(rock.get("ky_md")[k, j, i]) if rock.get("ky_md") is not None else 0.05
    WI = _peaceman_WI(kx, ky, dx, dy, dz)

    ctrl = (schedule.get("control") or schedule.get("pad_ctrl") or "BHP").upper()
    bhp = float(schedule.get("bhp_psi") or schedule.get("pad_bhp_psi") or 3000.0)
    rate_mscfd = float(schedule.get("rate_mscfd") or schedule.get("pad_rate_mscfd") or 0.0)

    return {
        "indices": [(k, j, i)],
        "WI": [WI],
        "control": ctrl,
        "bhp_psi": bhp,
        "rate_mscfd": rate_mscfd,
    }

from scipy.sparse import lil_matrix
from scipy.sparse import lil_matrix

G_FT_S2 = 32.174  # gravity in ft/s^2

def k_multiplier_from_pressure(P, p_init, alpha=0.0):
    """Geomech-like multiplier exp(alpha * (p_init - P)), alpha>=0 shrinks k as pressure drops."""
    if alpha == 0.0:
        return np.ones_like(P, dtype=float)
    dP = np.maximum(0.0, float(p_init) - np.asarray(P, float))
    return np.exp(alpha * dP)


# -----------------------------------------------------------------------------
# 1.1 Peaceman Well Index (anisotropic)
# -----------------------------------------------------------------------------
def peaceman_wi(
    dx: float,
    dy: float,
    dz: float,
    kx_md: float,
    ky_md: float,
    mu_cp: float,
    rw_ft: float = 0.328,
    skin: float = 0.0,
) -> float:
    """
    Peaceman well index for an anisotropic grid (vertical well proxy; extend as needed).

    Parameters
    ----------
    dx, dy, dz : float
        Cell dimensions (ft)
    kx_md, ky_md : float
        Cell permeability (mD) in x and y directions
    mu_cp : float
        Phase viscosity (cP)
    rw_ft : float
        Wellbore radius (ft)
    skin : float
        Dimensionless skin

    Returns
    -------
    wi : float
        Well index (unit-consistent with your flow eqns).
    """
    # Effective drainage radius for anisotropic case (Peaceman)
    re = 0.28 * np.sqrt(np.sqrt(kx_md / ky_md) * dx * dx + np.sqrt(ky_md / kx_md) * dy * dy)
    # Here we assume unit-consistent coefficients elsewhere (trans forms, flow eqns, etc.)
    # If you use field units, carry conversion in your transmissibility/flow assembly.
    wi = 2.0 * np.pi * np.sqrt(kx_md * ky_md) * dz / (mu_cp * (np.log(re / rw_ft) + skin + 1e-12))
    return float(wi)


# -----------------------------------------------------------------------------
# 1.2 Simple Geomechanics: permeability multiplier k(p)
# -----------------------------------------------------------------------------
def k_multiplier_from_pressure(
    p_cell: np.ndarray,
    p_init: float,
    alpha: float = 0.0,
    min_mult: float = 0.2,
    max_mult: float = 1.0,
) -> np.ndarray:
    """
    Exponential compaction: k_eff = k0 * exp(-alpha * (p_init - p))
    Clipped to [min_mult, max_mult].

    Parameters
    ----------
    p_cell : np.ndarray
        Current cell pressure (psi)
    p_init : float
        Initial pressure (psi)
    alpha : float
        Compaction sensitivity (1/psi). 0 disables compaction.
    min_mult, max_mult : float
        Clamps for stability.

    Returns
    -------
    mult : np.ndarray
        Multiplier to apply to permeability arrays.
    """
    dp = (p_init - np.asarray(p_cell))
    mult = np.exp(-alpha * dp)
    return np.clip(mult, min_mult, max_mult)


# -----------------------------------------------------------------------------
# 1.3 EDFM connectivity scaffold (placeholder)
# -----------------------------------------------------------------------------
def build_edfm_connectivity(grid: dict, dfn_segments: np.ndarray | None):
    """
    Placeholder: compute EDFM matrix–fracture and fracture–fracture transmissibilities.

    Parameters
    ----------
    grid : dict  with keys {"nx","ny","nz","dx","dy","dz"}
    dfn_segments : np.ndarray [Nseg,6] of (x0,y0,z0,x1,y1,z1) in ft, or None

    Returns
    -------
    dict with:
      'mf_T': list of (cell_index, frac_index, T_mf)
      'ff_T': list of (frac_i, frac_j, T_ff)
      'frac_cells': optional data structure for fracture unknowns (None here)
    """
    if dfn_segments is None or len(dfn_segments) == 0:
        return {"mf_T": [], "ff_T": [], "frac_cells": None}
    # TODO: implement fracture-cell intersections and shape factors
    return {"mf_T": [], "ff_T": [], "frac_cells": None}

def peaceman_wi_cartesian(kx, ky, dz, dx, dy, rw_ft=0.35, skin=0.0):
    """
    Peaceman well index for a vertical well completed in one cell (i,j,k).
    Anisotropic form (Peaceman, 1978). Returns WI in md*ft (transmissibility units before viscosity/FVF).
    """
    # effective drainage radius (anisotropic)
    beta = np.sqrt(ky / np.maximum(kx, 1e-30))
    re = 0.28 * np.sqrt((dx**2 * beta + dy**2 / beta) / (beta + 1.0 / beta))
    kh = np.sqrt(kx * ky) * dz
    wi = 2.0 * np.pi * kh / (np.log(re / max(rw_ft, 1e-6)) + max(skin, 0.0))
    return wi


def apply_wells_blackoil(
    A, R,
    grid, rock,
    P, Sw, Sg, So,
    Bo, Bg, Bw, Rs,
    mu_o, mu_g, mu_w,
    relperm,  # for derivatives
    schedule, options,
):
    """
    Adds BHP-controlled well source terms (Peaceman WI).
    Component residuals:
      oil:  + q_o_comp =  WI * lam_o/Bo * (P - pw)
      wat:  + q_w_comp =  WI * lam_w/Bw * (P - pw)
      gas:  + q_g_comp =  WI * [lam_g/Bg*(P - pw) + Rs*lam_o/Bo*(P - pw)]
    where lam = kr/mu on the completed cell.
    """
    nx, ny, nz = [int(grid[k]) for k in ("nx", "ny", "nz")]
    dx, dy, dz = [float(grid[k]) for k in ("dx", "dy", "dz")]
    N = nx * ny * nz

    def lin(i, j, k):
        return (k * ny + j) * nx + i

    # kr endpoints & Corey exponents for derivatives
    nw = float(relperm.get("nw", 2.0))
    no = float(relperm.get("no", 2.0))
    krw_end = float(relperm.get("krw_end", 0.6))
    kro_end = float(relperm.get("kro_end", 0.8))

    def dkro_dSo(so):  # kro_end * no * so^(no-1)
        return kro_end * no * np.maximum(so, 1e-12) ** (no - 1.0)

    def dkrw_dSw(sw):  # krw_end * nw * sw^(nw-1)
        return krw_end * nw * np.maximum(sw, 1e-12) ** (nw - 1.0)

    # wells spec
    wells = schedule.get("wells")
    if not wells:
        # Fallback: single vertical producer at grid center using schedule["bhp_psi"] if present
        pw = float(schedule.get("bhp_psi", 2500.0))
        wells = [{
            "i": nx // 2, "j": ny // 2, "k": nz // 2,
            "bhp_psi": pw, "rw_ft": 0.35, "skin": 0.0,
        }]

    # rock k’s (we only need the completed cell values)
    kx = np.asarray(rock.get("kx_md", 100.0)).reshape(nz, ny, nx)
    ky = np.asarray(rock.get("ky_md", 100.0)).reshape(nz, ny, nx)

    for w in wells:
        i = int(w.get("i", nx // 2))
        j = int(w.get("j", ny // 2))
        k = int(w.get("k", nz // 2))
        c = lin(i, j, k)
        pw = float(w.get("bhp_psi", 2500.0))
        rw = float(w.get("rw_ft", 0.35))
        skin = float(w.get("skin", 0.0))

        wi = peaceman_wi_cartesian(kx[k, j, i], ky[k, j, i], dz, dx, dy, rw_ft=rw, skin=skin)

        # local props (completed cell)
        muo = max(mu_o[c], 1e-12); mug = max(mu_g[c], 1e-12); muw = max(mu_w[c], 1e-12)
        lam_o = np.maximum(0.0, (kro_end * (So[c] ** no)) / muo)
        lam_w = np.maximum(0.0, (krw_end * (Sw[c] ** nw)) / muw)
        lam_g = np.maximum(0.0, (Sg[c] ** 2) / mug)  # default gas kr ~ s^2

        # component coefficients
        co = wi * lam_o / max(Bo[c], 1e-12)
        cw = wi * lam_w / max(Bw[c], 1e-12)
        cg = wi * (lam_g / max(Bg[c], 1e-12) + (Rs[c] * lam_o) / max(Bo[c], 1e-12))

        dP = P[c] - pw

        # residual (outflow from cell is positive)
        R[3 * c + 0] += co * dP  # oil comp row iP(c)
        R[3 * c + 1] += cw * dP  # water comp row iSw(c)
        R[3 * c + 2] += cg * dP  # gas comp row iSg(c)

        # Jacobian wrt P(c)
        A[3 * c + 0, 3 * c + 0] += co
        A[3 * c + 1, 3 * c + 0] += cw
        A[3 * c + 2, 3 * c + 0] += cg

        # Saturation derivatives via lam’s
        dlamo_dSo = dkro_dSo(So[c]) / muo
        dlamo_dSw = -dlamo_dSo
        dlamo_dSg = -dlamo_dSo
        dlamw_dSw = dkrw_dSw(Sw[c]) / muw
        dlamg_dSg = 2.0 * np.maximum(Sg[c], 0.0) / mug

        dco_dSw = wi * (dlamo_dSw / max(Bo[c], 1e-12))
        dco_dSg = wi * (dlamo_dSg / max(Bo[c], 1e-12))
        dcw_dSw = wi * (dlamw_dSw / max(Bw[c], 1e-12))
        dcg_dSg = wi * (dlamg_dSg / max(Bg[c], 1e-12))
        dcg_dSw = wi * ((Rs[c] * dlamo_dSw) / max(Bo[c], 1e-12))
        dcg_dSg2 = wi * ((Rs[c] * dlamo_dSg) / max(Bo[c], 1e-12))

        # apply on residual rows (multiply by dP)
        A[3 * c + 0, 3 * c + 1] += dco_dSw * dP   # oil row vs Sw
        A[3 * c + 0, 3 * c + 2] += dco_dSg * dP   # oil row vs Sg
        A[3 * c + 1, 3 * c + 1] += dcw_dSw * dP   # water row vs Sw

        A[3 * c + 2, 3 * c + 2] += (dcg_dSg + dcg_dSg2) * dP  # gas row vs Sg
        A[3 * c + 2, 3 * c + 1] += dcg_dSw * dP               # gas row vs Sw




# -----------------------------------------------------------------------------
# 1.4 Black-oil residual/Jacobian skeleton (fully implicit)
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
    Black-oil TPFA with:
      • Accumulation (implicit) + derivatives wrt P/Sw/Sg
      • Flux upwinding
      • Gravity term in vertical faces: ΔΦ = ΔP - ρ_up g Δz (phase-wise)
      • Peaceman WI wells (BHP-controlled) added as source terms
    """
    # ---- helpers ----
    def iP(i):  return 3 * i
    def iSw(i): return 3 * i + 1
    def iSg(i): return 3 * i + 2

    def harm(a, b):
        return 2.0 * a * b / (a + b + 1e-30)

    def numdiff_p(fn, p, eps=1e-3):
        return (np.asarray(fn(p + eps)) - np.asarray(fn(p - eps))) / (2.0 * eps)

    nw = float(relperm.get("nw", 2.0))
    no = float(relperm.get("no", 2.0))
    krw_end = float(relperm.get("krw_end", 0.6))
    kro_end = float(relperm.get("kro_end", 0.8))

    def dkrw_dSw(sw):  return krw_end * nw * np.maximum(sw, 1e-12) ** (nw - 1.0)
    def dkro_dSo(so):  return kro_end * no * np.maximum(so, 1e-12) ** (no - 1.0)
    def dkrg_dSg(sg):  return 2.0 * np.maximum(sg, 0.0)

    # ---- grid & indexing ----
    nx, ny, nz = [int(grid[k]) for k in ("nx", "ny", "nz")]
    dx, dy, dz = [float(grid[k]) for k in ("dx", "dy", "dz")]
    N = nx * ny * nz

    def lin(i, j, k): return (k * ny + j) * nx + i

    Vcell = dx * dy * dz

    # cell centers (only z used for gravity)
    Zc = np.empty(N, dtype=float)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = lin(i, j, k)
                Zc[c] = (k + 0.5) * dz  # ft

    # ---- unknowns ----
    P  = state_vec[0::3].copy()
    Sw = state_vec[1::3].copy()
    Sg = state_vec[2::3].copy()
    So = 1.0 - Sw - Sg
    eps = 1e-9
    Sw[:] = np.clip(Sw, 0.0 + eps, 1.0 - eps)
    Sg[:] = np.clip(Sg, 0.0 + eps, 1.0 - eps)
    So[:] = np.clip(So,  0.0 + eps, 1.0 - eps)

    # ---- rock ----
    phi = np.asarray(rock.get("phi", 0.08)).reshape(N)
    kx  = np.asarray(rock.get("kx_md", 100.0)).reshape(N)
    ky  = np.asarray(rock.get("ky_md", 100.0)).reshape(N)
    # vertical perm: use provided or a fraction of horizontal (kv/kh ~ 0.1 default)
    kvkh = float(options.get("kvkh", 0.1))
    kz  = np.asarray(rock.get("kz_md", kvkh * 0.5 * (kx + ky))).reshape(N)

    # geomech multiplier
    p_init = float(init.get("p_init_psi", 5000.0))
    geo_alpha = float(options.get("geo_alpha", 0.0))
    kmult = k_multiplier_from_pressure(P, p_init, alpha=geo_alpha)
    kx *= kmult; ky *= kmult; kz *= kmult

    # ---- time discretization ----
    dt_days = float(options.get("dt_days", 1.0))
    dt = max(1.0, dt_days * 86400.0)

    prev = options.get("prev", None)
    if prev is None:
        Pm, Swm, Sgm = P.copy(), Sw.copy(), Sg.copy()
    else:
        Pm  = np.asarray(prev.get("P",  P))
        Swm = np.asarray(prev.get("Sw", Sw))
        Sgm = np.asarray(prev.get("Sg", Sg))
    Som = 1.0 - Swm - Sgm

    # ---- PVT ----
    Bo = np.asarray(pvt.Bo(P));   dBo_dP = numdiff_p(pvt.Bo, P)
    Bg = np.asarray(pvt.Bg(P));   dBg_dP = numdiff_p(pvt.Bg, P)
    Rs = np.asarray(pvt.Rs(P));   dRs_dP = numdiff_p(pvt.Rs, P)
    mu_o = np.asarray(pvt.mu_o(P))
    mu_g = np.asarray(pvt.mu_g(P))
    mu_w = np.full_like(P, 0.5)  # placeholder
    Bw   = np.ones_like(P)
    dBw_dP = np.zeros_like(P)

    # ---- relperm + phase mobilities ----
    krw = relperm.get("krw_fn", lambda s: krw_end * (s ** nw))(Sw)
    kro = relperm.get("kro_fn", lambda s: kro_end * (s ** no))(So)
    krg = relperm.get("krg_fn", lambda s: s ** 2)(Sg)

    lam_w = krw / np.maximum(mu_w, 1e-12)
    lam_o = kro / np.maximum(mu_o, 1e-12)
    lam_g = krg / np.maximum(mu_g, 1e-12)

    # ---- accumulations ----
    acc_o = phi * So / np.maximum(Bo, 1e-12)
    acc_g = phi * (Sg / np.maximum(Bg, 1e-12) + So * Rs / np.maximum(Bo, 1e-12))
    acc_w = phi * Sw / np.maximum(Bw, 1e-12)

    Bom = np.asarray(pvt.Bo(Pm))
    Bgm = np.asarray(pvt.Bg(Pm))
    Rsm = np.asarray(pvt.Rs(Pm))
    Bwm = np.ones_like(Pm)

    accm_o = phi * Som / np.maximum(Bom, 1e-12)
    accm_g = phi * (Sgm / np.maximum(Bgm, 1e-12) + Som * Rsm / np.maximum(Bom, 1e-12))
    accm_w = phi * Swm / np.maximum(Bwm, 1e-12)

    # ---- allocate system ----
    nunk = 3 * N
    A = lil_matrix((nunk, nunk), dtype=float)
    R = np.zeros(nunk, dtype=float)

    # accumulation residual & Jacobian (diagonals)
    scale = Vcell / dt
    R[0::3] += scale * (acc_o - accm_o)
    R[1::3] += scale * (acc_w - accm_w)
    R[2::3] += scale * (acc_g - accm_g)

    dacc_o_dP  = phi * So * (-dBo_dP / (np.maximum(Bo, 1e-12) ** 2))
    dacc_o_dSw = -phi / np.maximum(Bo, 1e-12)
    dacc_o_dSg = -phi / np.maximum(Bo, 1e-12)

    dacc_w_dP  = phi * Sw * (-dBw_dP / (np.maximum(Bw, 1e-12) ** 2))
    dacc_w_dSw = phi / np.maximum(Bw, 1e-12)

    term1 = Sg * (-dBg_dP / (np.maximum(Bg, 1e-12) ** 2))
    term2 = So * (dRs_dP / np.maximum(Bo, 1e-12) - Rs * dBo_dP / (np.maximum(Bo, 1e-12) ** 2))
    dacc_g_dP  = phi * (term1 + term2)
    dacc_g_dSw = -phi * (Rs / np.maximum(Bo, 1e-12))
    dacc_g_dSg =  phi / np.maximum(Bg, 1e-12) - phi * (Rs / np.maximum(Bo, 1e-12))

    for c in range(N):
        A[iP(c),  iP(c)]  += scale * dacc_o_dP[c]
        A[iP(c),  iSw(c)] += scale * dacc_o_dSw[c]
        A[iP(c),  iSg(c)] += scale * dacc_o_dSg[c]

        A[iSw(c), iP(c)]  += scale * dacc_w_dP[c]
        A[iSw(c), iSw(c)] += scale * dacc_w_dSw[c]

        A[iSg(c), iP(c)]  += scale * dacc_g_dP[c]
        A[iSg(c), iSw(c)] += scale * dacc_g_dSw[c]
        A[iSg(c), iSg(c)] += scale * dacc_g_dSg[c]

    # ---- fluxes with gravity (vertical faces only) ----
    use_grav = bool(options.get("use_gravity", True))
    rho_o = float(options.get("rho_o_lbft3", 53.0))
    rho_w = float(options.get("rho_w_lbft3", 62.4))
    rho_g = float(options.get("rho_g_lbft3", 0.06))

    area_x = dy * dz
    area_y = dx * dz
    area_z = dx * dy

    def add_face(c, n, T_face, dz_ft):
        dP = P[c] - P[n]
        up = c if dP >= 0.0 else n

        Bo_up = Bo[up]; Bg_up = Bg[up]; Bw_up = Bw[up]
        Rs_up = Rs[up]
        lam_o_up = lam_o[up]; lam_w_up = lam_w[up]; lam_g_up = lam_g[up]

        # potential drops
        if use_grav and abs(dz_ft) > 0.0:
            dPhi_o = dP - rho_o * G_FT_S2 * dz_ft
            dPhi_w = dP - rho_w * G_FT_S2 * dz_ft
            dPhi_g = dP - rho_g * G_FT_S2 * dz_ft
        else:
            dPhi_o = dPhi_w = dPhi_g = dP

        # component fluxes (c → n is positive)
        Fo = T_face * lam_o_up / max(Bo_up, 1e-12) * dPhi_o
        Fw = T_face * lam_w_up / max(Bw_up, 1e-12) * dPhi_w
        Fg = T_face * (lam_g_up / max(Bg_up, 1e-12) * dPhi_g + (Rs_up * lam_o_up) / max(Bo_up, 1e-12) * dPhi_o)

        # residual
        R[iP(c)]  += Fo; R[iP(n)]  -= Fo
        R[iSw(c)] += Fw; R[iSw(n)] -= Fw
        R[iSg(c)] += Fg; R[iSg(n)] -= Fg

        # pressure coupling Jacobian (treat upwind props frozen)
        coef_o = T_face * lam_o_up / max(Bo_up, 1e-12)
        coef_w = T_face * lam_w_up / max(Bw_up, 1e-12)
        coef_g1 = T_face * lam_g_up / max(Bg_up, 1e-12)
        coef_g2 = T_face * (Rs_up * lam_o_up) / max(Bo_up, 1e-12)

        # rows at c
        A[iP(c),  iP(c)]  += coef_o
        A[iP(c),  iP(n)]  -= coef_o
        A[iSw(c), iP(c)]  += coef_w
        A[iSw(c), iP(n)]  -= coef_w
        A[iSg(c), iP(c)]  += (coef_g1 + coef_g2)
        A[iSg(c), iP(n)]  -= (coef_g1 + coef_g2)

        # rows at n (symmetric)
        A[iP(n),  iP(c)]  -= coef_o
        A[iP(n),  iP(n)]  += coef_o
        A[iSw(n), iP(c)]  -= coef_w
        A[iSw(n), iP(n)]  += coef_w
        A[iSg(n), iP(c)]  -= (coef_g1 + coef_g2)
        A[iSg(n), iP(n)]  += (coef_g1 + coef_g2)

        # saturation derivatives (upstream cell only)
        if up == c:
            So_up = So[c]; Sw_up = Sw[c]; Sg_up = Sg[c]
            muo_up = mu_o[c]; muw_up = mu_w[c]; mug_up = mu_g[c]

            dlamo_dSo = dkro_dSo(So_up) / max(muo_up, 1e-12)
            dlamo_dSw = -dlamo_dSo
            dlamo_dSg = -dlamo_dSo

            dlamw_dSw = dkrw_dSw(Sw_up) / max(muw_up, 1e-12)
            dlamg_dSg = dkrg_dSg(Sg_up) / max(mug_up, 1e-12)

            A[iP(c),  iSw(c)] += T_face * dlamo_dSw / max(Bo_up, 1e-12) * dPhi_o
            A[iP(n),  iSw(c)] -= T_face * dlamo_dSw / max(Bo_up, 1e-12) * dPhi_o
            A[iP(c),  iSg(c)] += T_face * dlamo_dSg / max(Bo_up, 1e-12) * dPhi_o
            A[iP(n),  iSg(c)] -= T_face * dlamo_dSg / max(Bo_up, 1e-12) * dPhi_o

            A[iSw(c), iSw(c)] += T_face * dlamw_dSw / max(Bw_up, 1e-12) * dPhi_w
            A[iSw(n), iSw(c)] -= T_face * dlamw_dSw / max(Bw_up, 1e-12) * dPhi_w

            # gas: free + dissolved-in-oil
            A[iSg(c), iSg(c)] += T_face * dlamg_dSg / max(Bg_up, 1e-12) * dPhi_g
            A[iSg(n), iSg(c)] -= T_face * dlamg_dSg / max(Bg_up, 1e-12) * dPhi_g
            A[iSg(c), iSw(c)] += T_face * (dlamo_dSw * Rs_up) / max(Bo_up, 1e-12) * dPhi_o
            A[iSg(n), iSw(c)] -= T_face * (dlamo_dSw * Rs_up) / max(Bo_up, 1e-12) * dPhi_o
            A[iSg(c), iSg(c)] += T_face * (dlamo_dSg * Rs_up) / max(Bo_up, 1e-12) * dPhi_o
            A[iSg(n), iSg(c)] -= T_face * (dlamo_dSg * Rs_up) / max(Bo_up, 1e-12) * dPhi_o

        elif up == n:
            So_up = So[n]; Sw_up = Sw[n]; Sg_up = Sg[n]
            muo_up = mu_o[n]; muw_up = mu_w[n]; mug_up = mu_g[n]

            dlamo_dSo = dkro_dSo(So_up) / max(muo_up, 1e-12)
            dlamo_dSw = -dlamo_dSo
            dlamo_dSg = -dlamo_dSo
            dlamw_dSw = dkrw_dSw(Sw_up) / max(muw_up, 1e-12)
            dlamg_dSg = dkrg_dSg(Sg_up) / max(mug_up, 1e-12)

            A[iP(c),  iSw(n)] += T_face * dlamo_dSw / max(Bo_up, 1e-12) * dPhi_o
            A[iP(n),  iSw(n)] -= T_face * dlamo_dSw / max(Bo_up, 1e-12) * dPhi_o
            A[iP(c),  iSg(n)] += T_face * dlamo_dSg / max(Bo_up, 1e-12) * dPhi_o
            A[iP(n),  iSg(n)] -= T_face * dlamo_dSg / max(Bo_up, 1e-12) * dPhi_o

            A[iSw(c), iSw(n)] += T_face * dlamw_dSw / max(Bw_up, 1e-12) * dPhi_w
            A[iSw(n), iSw(n)] -= T_face * dlamw_dSw / max(Bw_up, 1e-12) * dPhi_w

            A[iSg(c), iSg(n)] += T_face * dlamg_dSg / max(Bg_up, 1e-12) * dPhi_g
            A[iSg(n), iSg(n)] -= T_face * dlamg_dSg / max(Bg_up, 1e-12) * dPhi_g
            A[iSg(c), iSw(n)] += T_face * (dlamo_dSw * Rs_up) / max(Bo_up, 1e-12) * dPhi_o
            A[iSg(n), iSw(n)] -= T_face * (dlamo_dSw * Rs_up) / max(Bo_up, 1e-12) * dPhi_o
            A[iSg(c), iSg(n)] += T_face * (dlamo_dSg * Rs_up) / max(Bo_up, 1e-12) * dPhi_o
            A[iSg(n), iSg(n)] -= T_face * (dlamo_dSg * Rs_up) / max(Bo_up, 1e-12) * dPhi_o

    # sweep faces (i+1, j+1, k+1)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = lin(i, j, k)

                # i-face
                if i + 1 < nx:
                    n = lin(i + 1, j, k)
                    T = (harm(kx[c], kx[n]) * area_x) / max(dx, 1e-30)
                    add_face(c, n, T, dz_ft=0.0)

                # j-face
                if j + 1 < ny:
                    n = lin(i, j + 1, k)
                    T = (harm(ky[c], ky[n]) * area_y) / max(dy, 1e-30)
                    add_face(c, n, T, dz_ft=0.0)

                # k-face (vertical, gravity)
                if k + 1 < nz:
                    n = lin(i, j, k + 1)
                    T = (harm(kz[c], kz[n]) * area_z) / max(dz, 1e-30)
                    add_face(c, n, T, dz_ft=(Zc[c] - Zc[n]))

    # ---- wells (BHP-controlled, Peaceman WI) ----
    apply_wells_blackoil(
        A, R, grid, rock,
        P, Sw, Sg, So,
        Bo, Bg, Bw, Rs,
        mu_o, mu_g, mu_w,
        relperm, schedule, options
    )

    meta = {"note": "TPFA black-oil with gravity (vertical faces) + BHP Peaceman wells."}
    return A.tocsr(), R, meta
