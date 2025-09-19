# core/full3d.py
# -----------------------------------------------------------------------------
# Minimal working proxy simulate() so the app runs end-to-end,
# plus Phase 1 scaffolds:
#   • Gravity + Peaceman WI helpers
#   • Simple geomech k(p) multiplier
#   • EDFM connectivity placeholder
#   • Black-oil residual/Jacobian skeleton (P, Sw, Sg)
#   • Implicit solver skeleton + analytical proxy/dispatch
# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from scipy.sparse import lil_matrix

# ---------------------------- constants/options ------------------------------
G_FT_S2 = 32.174  # ft/s^2 (used for gravity head)

DEFAULT_OPTIONS = {
    "use_gravity": True,
    "rho_o_lbft3": 53.0,
    "rho_w_lbft3": 62.4,
    "rho_g_lbft3": 0.06,
    "kvkh": 0.10,       # vertical anisotropy (kv/kh)
    "geo_alpha": 0.0,   # simple geomech k-multiplier (0 = off)
}

# ----------------------------- small helpers --------------------------------
def _z_centers(nz, dz):
    """Cell-center depths (ft). k=0 is shallowest if you use top-down indexing."""
    return (np.arange(nz, dtype=float) + 0.5) * float(dz)

def peaceman_wi_cartesian(kx, ky, dz, dx, dy, rw_ft=0.35, skin=0.0) -> float:
    """
    Peaceman well index for a vertical well completed in one cell (i,j,k).
    Anisotropic form (Peaceman, 1978). Returns WI in md*ft (before viscosity/FVF).
    """
    beta = np.sqrt(ky / np.maximum(kx, 1e-30))
    re = 0.28 * np.sqrt((dx**2 * beta + dy**2 / beta) / (beta + 1.0 / beta))
    kh = np.sqrt(kx * ky) * dz
    wi = 2.0 * np.pi * kh / (np.log(re / max(rw_ft, 1e-6)) + max(skin, 0.0))
    return float(wi)

# ------------------------ simple geomech: k(p) multiplier --------------------
def k_multiplier_from_pressure(
    p_cell: np.ndarray,
    p_init: float,
    alpha: float = 0.0,
    min_mult: float = 0.2,
    max_mult: float = 1.0,
) -> np.ndarray:
    """
    Exponential compaction: k_eff = k0 * exp(-alpha * (p_init - p)), clipped.
    alpha=0 disables compaction.
    """
    dp = (p_init - np.asarray(p_cell))
    mult = np.exp(-alpha * dp)
    return np.clip(mult, min_mult, max_mult)

# ----------------------------- EDFM placeholder ------------------------------
def build_edfm_connectivity(grid: dict, dfn_segments: np.ndarray | None):
    """
    Placeholder: compute EDFM matrix–fracture and fracture–fracture
    transmissibilities. Returns empty connectivity for now.
    """
    if dfn_segments is None or len(dfn_segments) == 0:
        return {"mf_T": [], "ff_T": [], "frac_cells": None}
    return {"mf_T": [], "ff_T": [], "frac_cells": None}

# ---------------------- BHP-controlled Peaceman wells ------------------------
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
    lam = kr/mu evaluated in the completed cell (upstream/frozen for Jacobian).
    """
    nx, ny, nz = [int(grid[k]) for k in ("nx", "ny", "nz")]
    dx, dy, dz = [float(grid[k]) for k in ("dx", "dy", "dz")]

    def lin(i, j, k):
        return (k * ny + j) * nx + i

    # relperm endpoints & Corey exponents for derivatives
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
        pw = float(schedule.get("bhp_psi", 2500.0))
        wells = [{
            "i": nx // 2, "j": ny // 2, "k": nz // 2,
            "bhp_psi": pw, "rw_ft": 0.35, "skin": 0.0,
        }]

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
        lam_o = np.maximum(0.0, (kro_end * ((So[c]) ** no)) / muo)
        lam_w = np.maximum(0.0, (krw_end * ((Sw[c]) ** nw)) / muw)
        lam_g = np.maximum(0.0, (Sg[c] ** 2) / mug)  # default gas kr ~ s^2

        co = wi * lam_o / max(Bo[c], 1e-12)
        cw = wi * lam_w / max(Bw[c], 1e-12)
        cg = wi * (lam_g / max(Bg[c], 1e-12) + (Rs[c] * lam_o) / max(Bo[c], 1e-12))

        dP = P[c] - pw

        # residual (outflow from cell is positive)
        R[3 * c + 0] += co * dP  # oil comp
        R[3 * c + 1] += cw * dP  # water comp
        R[3 * c + 2] += cg * dP  # gas comp

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
        A[3 * c + 0, 3 * c + 1] += dco_dSw * dP
        A[3 * c + 0, 3 * c + 2] += dco_dSg * dP
        A[3 * c + 1, 3 * c + 1] += dcw_dSw * dP
        A[3 * c + 2, 3 * c + 2] += (dcg_dSg + dcg_dSg2) * dP
        A[3 * c + 2, 3 * c + 1] += dcg_dSw * dP

# ------------------- Black-oil residual/Jacobian (TPFA) ----------------------
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
      • Gravity term on vertical faces: ΔΦ = ΔP − ρ_up * g * Δz (phase-wise)
      • Peaceman WI wells (BHP-controlled)
    """
    # ---- index helpers ----
    def iP(i):  return 3 * i
    def iSw(i): return 3 * i + 1
    def iSg(i): return 3 * i + 2

    def harm(a, b): return 2.0 * a * b / (a + b + 1e-30)

    def numdiff_p(fn, p, eps=1e-3):
        return (np.asarray(fn(p + eps)) - np.asarray(fn(p - eps))) / (2.0 * eps)

    # relperm derivatives
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

    # cell centers z (for gravity)
    Zc = np.empty(N, dtype=float)
    for kk in range(nz):
        for jj in range(ny):
            for ii in range(nx):
                c = lin(ii, jj, kk)
                Zc[c] = (kk + 0.5) * dz

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

    # ---- accumulation terms ----
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

    nunk = 3 * N
    A = lil_matrix((nunk, nunk), dtype=float)
    R = np.zeros(nunk, dtype=float)

    scale = (dx * dy * dz) / dt
    R[0::3] += scale * (acc_o - accm_o)
    R[1::3] += scale * (acc_w - accm_w)
    R[2::3] += scale * (acc_g - accm_g)

    dacc_o_dP  = phi * So * (-dBo_dP / (np.maximum(Bo, 1e-12) ** 2))
    dacc_o_dSw = -phi / np.maximum(Bo, 1e-12)
    dacc_o_dSg = -phi / np.maximum(Bo, 1e-12)

    dacc_w_dP  = phi * Sw * (-dBw_dP / (np.maximum(Bw, 1e-12) ** 2))
    dacc_w_dSw =  phi / np.maximum(Bw, 1e-12)

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

        # phase potentials
        if use_grav and abs(dz_ft) > 0.0:
            dPhi_o = dP - rho_o * G_FT_S2 * dz_ft
            dPhi_w = dP - rho_w * G_FT_S2 * dz_ft
            dPhi_g = dP - rho_g * G_FT_S2 * dz_ft
        else:
            dPhi_o = dPhi_w = dPhi_g = dP

        # component fluxes (c → n positive)
        Fo = T_face * lam_o_up / max(Bo_up, 1e-12) * dPhi_o
        Fw = T_face * lam_w_up / max(Bw_up, 1e-12) * dPhi_w
        Fg = T_face * (lam_g_up / max(Bg_up, 1e-12) * dPhi_g
                       + (Rs_up * lam_o_up) / max(Bo_up, 1e-12) * dPhi_o)

        # residual
        R[iP(c)]  += Fo; R[iP(n)]  -= Fo
        R[iSw(c)] += Fw; R[iSw(n)] -= Fw
        R[iSg(c)] += Fg; R[iSg(n)] -= Fg

        # pressure coupling Jacobian (upwind props frozen)
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

        # saturation derivatives (only upstream cell contributes)
        if up == c:
            So_up = So[c]; Sw_up = Sw[c]; Sg_up = Sg[c]
            muo_up = mu_o[c]; muw_up = mu_w[c]; mug_up = mu_g[c]

            dlamo_dSo = (kro_end * no * np.maximum(So_up, 1e-12) ** (no - 1.0)) / max(muo_up, 1e-12)
            dlamo_dSw = -dlamo_dSo
            dlamo_dSg = -dlamo_dSo
            dlamw_dSw = (krw_end * nw * np.maximum(Sw_up, 1e-12) ** (nw - 1.0)) / max(muw_up, 1e-12)
            dlamg_dSg = (2.0 * np.maximum(Sg_up, 0.0)) / max(mug_up, 1e-12)

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

            dlamo_dSo = (kro_end * no * np.maximum(So_up, 1e-12) ** (no - 1.0)) / max(muo_up, 1e-12)
            dlamo_dSw = -dlamo_dSo
            dlamo_dSg = -dlamo_dSo
            dlamw_dSw = (krw_end * nw * np.maximum(Sw_up, 1e-12) ** (nw - 1.0)) / max(muw_up, 1e-12)
            dlamg_dSg = (2.0 * np.maximum(Sg_up, 0.0)) / max(mug_up, 1e-12)

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

# ===== BEGIN 3-PHASE IMPLICIT SKELETON =====
# (kept minimal; raises NotImplementedError so the app still runs via proxy)

def assemble_jacobian_and_residuals(state, grid, rock, fluid, schedule, options):
    """
    Build residuals (Rw, Ro, Rg) and the 3x3 Jacobian blocks for (p, Sw, Sg).
    TODO (Phase 1):
      - Loop faces (x, y, z), compute TPFA transmissibilities per face
      - Upwind phase fluxes
      - On z-faces, add hydrostatic term
      - Accumulate residuals and fill Jacobian sub-blocks
      - Add well contributions
    """
    raise NotImplementedError("3-phase assembler pending")

def newton_solve(state0, grid, rock, fluid, schedule, options):
    """
    Time-march with Newton iterations using assemble_jacobian_and_residuals.
    TODO (Phase 1): implement driver, timestep control, convergence, outputs.
    """
    raise NotImplementedError("Implicit Newton driver pending")

def _simulate_analytical_proxy(inputs: dict):
    """Fast decline-proxy so the UI works while implicit solver is built."""
    t = np.linspace(1.0, 3650.0, 240)  # days (~10 years)
    qi_g, di_g = 8000.0, 0.80   # Mcf/d, 1/yr
    qi_o, di_o = 1000.0, 0.70   # stb/d, 1/yr
    years = t / 365.25
    qg = qi_g * np.exp(-di_g * years)
    qo = qi_o * np.exp(-di_o * years)
    return {
        "t": t,
        "qg": qg,
        "qo": qo,
        "press_matrix": None,
        "pm_mid_psi": None,
        "p_init_3d": None,
        "ooip_3d": None,
    }

def simulate(inputs: dict):
    """
    Dispatch: analytical proxy OR implicit engine.
    Set inputs['engine'] to 'analytical' or 'implicit'.
    """
    engine = inputs.get("engine", "analytical").lower()
    if engine == "analytical":
        return _simulate_analytical_proxy(inputs)
    elif engine == "implicit":
        # TODO: build grid/rock/fluid/state/schedule/options and call newton_solve(...)
        raise NotImplementedError("Implicit engine wiring pending")
    else:
        raise ValueError(f"Unknown engine '{engine}'")
# ===== END 3-PHASE IMPLICIT SKELETON =====
