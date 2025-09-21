# core/full3d.py
# -----------------------------------------------------------------------------
# Full 3D Unconventional / Black-Oil Reservoir Core
# Minimal-but-robust implicit engine + analytical proxy dispatch.
#
# What you get:
#   • Gravity + Peaceman WI wells
#   • Simple geomech k(p) multiplier (k = k0 * exp(-alpha*(p_init - p)))
#   • EDFM placeholder for future DFN
#   • Black-oil assembler (P, Sw, Sg) with:
#       - Accumulation + Jacobian (incl. φ(P) and Bw(P))
#       - TPFA face fluxes with upwinding
#       - Gravity on vertical faces
#       - Optional capillary pressure Pcow(Sw), Pcg(Sg) (off by default)
#       - Pressure-derivative terms in flux (1/Bo, 1/Bg, 1/Bw, Rs/Bo)
#   • Wells: BHP or RATE controls (gas/oil/liquid) via equivalent-BHP mapping
#   • Implicit Newton with backtracking and adaptive dt
#   • Analytical proxy fallback (fast preview)
# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# ---------------------------- constants/options ------------------------------
G_FT_S2 = 32.174  # ft/s^2 (gravity head)

DEFAULT_OPTIONS = {
    "use_gravity": True,
    "rho_o_lbft3": 53.0,
    "rho_w_lbft3": 62.4,
    "rho_g_lbft3": 0.06,
    "kvkh": 0.10,        # vertical anisotropy (kv/kh)
    "geo_alpha": 0.0,    # simple geomech k-multiplier (0 = off)
    "use_pc": False,     # capillary pressure off by default
    # rock compressibility can be supplied either in rock["cr_1overpsi"] or here:
    "cr_1overpsi": 0.0,
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
    Exponential compaction: k_eff = k0 * exp(-alpha * (p_init - p)), then clipped.
    alpha=0 disables compaction.
    """
    dp = (p_init - np.asarray(p_cell))
    mult = np.exp(-alpha * dp)
    return np.clip(mult, min_mult, max_mult)

# ----------------------------- EDFM placeholder ------------------------------
def build_edfm_connectivity(grid: dict, dfn_segments: np.ndarray | None):
    """
    Placeholder for EDFM matrix–fracture and fracture–fracture transmissibilities.
    Returns empty connectivity for now.
    """
    if dfn_segments is None or len(dfn_segments) == 0:
        return {"mf_T": [], "ff_T": [], "frac_cells": None}
    return {"mf_T": [], "ff_T": [], "frac_cells": None}

# ---------- helper: pick equivalent pw for BHP or RATE-controlled wells ------
def _effective_pw_for_control(ctrl, Pcell, co, cw, cg, w, schedule):
    """
    Map a RATE control to an equivalent BHP 'pw' (so well behaves like a BHP well
    that exactly hits the requested rate using current upwind properties).
    Controls:
      - BHP                  : use bhp_psi
      - RATE_GAS_MSCFD      : target total gas component rate (free + Rs*oil)
      - RATE_OIL_STBD       : target oil component rate
      - RATE_LIQ_STBD       : target liquid (oil + water) component rate
    """
    ctrl = (ctrl or "").upper()

    def _bhp_default():
        return float(
            w.get(
                "bhp_psi",
                schedule.get("bhp_psi", schedule.get("pad_bhp_psi", 2500.0)),
            )
        )

    if ctrl == "BHP":
        return _bhp_default()

    if ctrl == "RATE_GAS_MSCFD":
        q_tgt = float(
            w.get(
                "rate_mscfd",
                schedule.get("rate_mscfd", schedule.get("pad_rate_mscfd", 0.0)),
            )
        )
        dP_req = q_tgt / max(cg, 1e-30)
        return Pcell - dP_req

    if ctrl == "RATE_OIL_STBD":
        q_tgt = float(w.get("rate_stbd", schedule.get("rate_stbd", 0.0)))
        dP_req = q_tgt / max(co, 1e-30)
        return Pcell - dP_req

    if ctrl == "RATE_LIQ_STBD":
        q_tgt = float(w.get("rate_stbd", schedule.get("rate_stbd", 0.0)))
        cl = co + cw
        dP_req = q_tgt / max(cl, 1e-30)
        return Pcell - dP_req

    # unknown → fallback to BHP
    return _bhp_default()

# ---------------------- BHP/RATE-controlled Peaceman wells -------------------
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
    Add well source terms into (A, R) for BHP or RATE controls using Peaceman WI.
    RATE modes are mapped to an equivalent BHP inside the same completed cell.
    Jacobian treatment uses "frozen upwind properties", which is a robust first
    step before a full well-equation approach (with extra unknown per well).
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

    # well list (fallback single producer at center)
    wells = schedule.get("wells")
    if not wells:
        ctrl = str(schedule.get("control", schedule.get("pad_ctrl", "BHP"))).upper()
        wells = [{
            "i": nx // 2, "j": ny // 2, "k": nz // 2,
            "control": ctrl,
            "bhp_psi": float(schedule.get("bhp_psi", schedule.get("pad_bhp_psi", 2500.0))),
            "rate_mscfd": float(schedule.get("rate_mscfd", schedule.get("pad_rate_mscfd", 0.0))),
            "rate_stbd":  float(schedule.get("rate_stbd",  schedule.get("pad_rate_stbd",  0.0))),
            "rw_ft": 0.35, "skin": 0.0,
        }]

    kx = np.asarray(rock.get("kx_md", 100.0)).reshape(nz, ny, nx)
    ky = np.asarray(rock.get("ky_md", 100.0)).reshape(nz, ny, nx)

    for w in wells:
        i = int(w.get("i", nx // 2))
        j = int(w.get("j", ny // 2))
        k = int(w.get("k", nz // 2))
        c = lin(i, j, k)

        ctrl = str(w.get("control", schedule.get("control", schedule.get("pad_ctrl", "BHP")))).upper()
        rw = float(w.get("rw_ft", 0.35))
        skin = float(w.get("skin", 0.0))

        wi = peaceman_wi_cartesian(kx[k, j, i], ky[k, j, i], dz, dx, dy, rw_ft=rw, skin=skin)

        # local props (completed cell)
        muo = max(mu_o[c], 1e-12); mug = max(mu_g[c], 1e-12); muw = max(mu_w[c], 1e-12)
        lam_o = np.maximum(0.0, (kro_end * (So[c] ** no)) / muo)
        lam_w = np.maximum(0.0, (krw_end * (Sw[c] ** nw)) / muw)
        lam_g = np.maximum(0.0, (Sg[c] ** 2) / mug)

        Bo_c = max(Bo[c], 1e-12)
        Bw_c = max(Bw[c], 1e-12)
        Bg_c = max(Bg[c], 1e-12)
        Rs_c = Rs[c]

        # ΔP → component-rate coefficients
        co = wi * lam_o / Bo_c
        cw = wi * lam_w / Bw_c
        cg = wi * (lam_g / Bg_c + (Rs_c * lam_o) / Bo_c)

        # pick equivalent pw for selected control
        pw = _effective_pw_for_control(ctrl, P[c], co, cw, cg, w, schedule)
        dP = P[c] - pw  # positive → production

        # residual
        R[3 * c + 0] += co * dP  # oil component
        R[3 * c + 1] += cw * dP  # water component
        R[3 * c + 2] += cg * dP  # gas component

        # Jacobian wrt local pressure (frozen upwind props)
        A[3 * c + 0, 3 * c + 0] += co
        A[3 * c + 1, 3 * c + 0] += cw
        A[3 * c + 2, 3 * c + 0] += cg

        # Saturation sensitivities via kr (frozen at cell)
        dlamo_dSo = dkro_dSo(So[c]) / muo
        dlamo_dSw = -dlamo_dSo
        dlamo_dSg = -dlamo_dSo
        dlamw_dSw = dkrw_dSw(Sw[c]) / muw
        dlamg_dSg = 2.0 * np.maximum(Sg[c], 0.0) / mug

        dco_dSw = wi * (dlamo_dSw / Bo_c)
        dco_dSg = wi * (dlamo_dSg / Bo_c)
        dcw_dSw = wi * (dlamw_dSw / Bw_c)
        dcg_dSg = wi * (dlamg_dSg / Bg_c)
        dcg_dSw = wi * ((Rs_c * dlamo_dSw) / Bo_c)
        dcg_dSg2 = wi * ((Rs_c * dlamo_dSg) / Bo_c)

        # apply saturation parts (multiply by dP)
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
    Build sparse Jacobian (A) and residual (R) for black-oil (P, Sw, Sg).
    Includes: φ(P), Bw(P), gravity, optional Pc, pressure-derivative terms,
    and BHP/RATE wells (applied after flux assembly).
    """
    # ---- toggles / params ----
    use_grav = bool(options.get("use_gravity", True))
    use_pc   = bool(options.get("use_pc", False))
    rho_o = float(options.get("rho_o_lbft3", 53.0))
    rho_w = float(options.get("rho_w_lbft3", 62.4))
    rho_g = float(options.get("rho_g_lbft3", 0.06))

    # Simple Corey-style capillary (off by default)
    pcw_entry = float(options.get("pcw_entry_psi", 0.0))
    pcw_L     = float(options.get("pcw_lambda", 2.0))
    pcg_entry = float(options.get("pcg_entry_psi", 0.0))
    pcg_L     = float(options.get("pcg_lambda", 2.0))

    def Pcow(sw):
        if not use_pc: return np.zeros_like(sw), np.zeros_like(sw)
        sw = np.clip(np.asarray(sw, float), 1e-9, 1.0 - 1e-9)
        pc = pcw_entry * (sw ** (-1.0 / pcw_L) - 1.0)
        dpc_dsw = pcw_entry * (-(1.0 / pcw_L)) * (sw ** (-1.0 / pcw_L - 1.0))
        return pc, dpc_dsw

    def Pcg(sg):
        if not use_pc: return np.zeros_like(sg), np.zeros_like(sg)
        sg = np.clip(np.asarray(sg, float), 1e-9, 1.0 - 1e-9)
        pc = pcg_entry * (sg ** (-1.0 / pcg_L) - 1.0)
        dpc_dsg = pcg_entry * (-(1.0 / pcg_L)) * (sg ** (-1.0 / pcg_L - 1.0))
        return pc, dpc_dsg

    # ---- helpers ----
    def iP(i):  return 3 * i
    def iSw(i): return 3 * i + 1
    def iSg(i): return 3 * i + 2
    def harm(a, b): return 2.0 * a * b / (a + b + 1e-30)

    def numdiff_p(fn, p, eps=1e-3):
        return (np.asarray(fn(p + eps)) - np.asarray(fn(p - eps))) / (2.0 * eps)

    # relperm params & derivs
    nw = float(relperm.get("nw", 2.0))
    no = float(relperm.get("no", 2.0))
    krw_end = float(relperm.get("krw_end", 0.6))
    kro_end = float(relperm.get("kro_end", 0.8))

    # ---- grid & indexing ----
    nx, ny, nz = [int(grid[k]) for k in ("nx", "ny", "nz")]
    dx, dy, dz = [float(grid[k]) for k in ("dx", "dy", "dz")]
    N = nx * ny * nz
    def lin(i, j, k): return (k * ny + j) * nx + i

    Vcell = dx * dy * dz

    # z centers (for gravity)
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

    # ---- rock φ(P) ----
    phi0 = np.asarray(rock.get("phi", 0.08)).reshape(N)
    cr   = float(rock.get("cr_1overpsi", options.get("cr_1overpsi", 0.0)))
    p_ref = float(rock.get("p_ref_psi", init.get("p_init_psi", 5000.0)))
    if cr > 0.0:
        phi = phi0 * np.exp(cr * (P - p_ref))
        dphi_dP = cr * phi
    else:
        phi = phi0
        dphi_dP = np.zeros_like(P)

    # perms (apply geomech k(p) multiplier)
    kx  = np.asarray(rock.get("kx_md", 100.0)).reshape(N)
    ky  = np.asarray(rock.get("ky_md", 100.0)).reshape(N)
    kvkh = float(options.get("kvkh", 0.1))
    kz  = np.asarray(rock.get("kz_md", kvkh * 0.5 * (kx + ky))).reshape(N)

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

    # ---- PVT (with water) ----
    Bo = np.asarray(pvt.Bo(P));   dBo_dP = numdiff_p(pvt.Bo, P)
    Bg = np.asarray(pvt.Bg(P));   dBg_dP = numdiff_p(pvt.Bg, P)
    Rs = np.asarray(pvt.Rs(P));   dRs_dP = numdiff_p(pvt.Rs, P)

    mu_o = np.asarray(pvt.mu_o(P))
    mu_g = np.asarray(pvt.mu_g(P))
    mu_w = np.asarray(pvt.mu_w(P))

    Bw   = np.asarray(pvt.Bw(P)); dBw_dP = numdiff_p(pvt.Bw, P)

    # ---- relperm + mobilities ----
    krw = relperm.get("krw_fn", lambda s: krw_end * (s ** nw))(Sw)
    kro = relperm.get("kro_fn", lambda s: kro_end * (s ** no))(So)
    krg = relperm.get("krg_fn", lambda s: s ** 2)(Sg)

    lam_w = krw / np.maximum(mu_w, 1e-12)
    lam_o = kro / np.maximum(mu_o, 1e-12)
    lam_g = krg / np.maximum(mu_g, 1e-12)

    # ---- accumulations (with φ(P) and Bw(P)) ----
    invBo = 1.0 / np.maximum(Bo, 1e-12)
    invBg = 1.0 / np.maximum(Bg, 1e-12)
    invBw = 1.0 / np.maximum(Bw, 1e-12)

    acc_o = phi * So * invBo
    acc_g = phi * (Sg * invBg + So * Rs * invBo)
    acc_w = phi * Sw * invBw

    Bom = np.asarray(pvt.Bo(Pm))
    Bgm = np.asarray(pvt.Bg(Pm))
    Rsm = np.asarray(pvt.Rs(Pm))
    Bwm = np.asarray(pvt.Bw(Pm))
    phi_prev = phi0 * (np.exp(cr * (Pm - p_ref)) if cr > 0.0 else 1.0)

    accm_o = phi_prev * Som / np.maximum(Bom, 1e-12)
    accm_g = phi_prev * (Sgm / np.maximum(Bgm, 1e-12) + Som * Rsm / np.maximum(Bom, 1e-12))
    accm_w = phi_prev * Swm / np.maximum(Bwm, 1e-12)

    # system
    nunk = 3 * N
    A = lil_matrix((nunk, nunk), dtype=float)
    R = np.zeros(nunk, dtype=float)

    scale = Vcell / dt
    R[0::3] += scale * (acc_o - accm_o)
    R[1::3] += scale * (acc_w - accm_w)
    R[2::3] += scale * (acc_g - accm_g)

    # accumulation Jacobian (includes dφ/dP and dBw/dP)
    dinvBo_dP = -dBo_dP * (invBo ** 2)
    dinvBg_dP = -dBg_dP * (invBg ** 2)
    dinvBw_dP = -dBw_dP * (invBw ** 2)

    dacc_o_dP  = dphi_dP * So * invBo + phi * So * dinvBo_dP
    dacc_o_dSw = -phi * invBo
    dacc_o_dSg = -phi * invBo

    dacc_w_dP  = dphi_dP * Sw * invBw + phi * Sw * dinvBw_dP
    dacc_w_dSw =  phi * invBw

    dacc_g_dP  = dphi_dP * (Sg * invBg + So * Rs * invBo) + phi * (Sg * dinvBg_dP + So * (dRs_dP * invBo + Rs * dinvBo_dP))
    dacc_g_dSw = -phi * (Rs * invBo)
    dacc_g_dSg =  phi * invBg - phi * (Rs * invBo)

    for c in range(N):
        A[iP(c),  iP(c)]  += scale * dacc_o_dP[c]
        A[iP(c),  iSw(c)] += scale * dacc_o_dSw[c]
        A[iP(c),  iSg(c)] += scale * dacc_o_dSg[c]

        A[iSw(c), iP(c)]  += scale * dacc_w_dP[c]
        A[iSw(c), iSw(c)] += scale * dacc_w_dSw[c]

        A[iSg(c), iP(c)]  += scale * dacc_g_dP[c]
        A[iSg(c), iSw(c)] += scale * dacc_g_dSw[c]
        A[iSg(c), iSg(c)] += scale * dacc_g_dSg[c]

    # ---- faces (TPFA) with gravity; optional Pc in potentials ----
    def add_face(c, n, T_face, dz_ft):
        dP = P[c] - P[n]
        up = c if dP >= 0.0 else n

        Bo_up = Bo[up]; Bg_up = Bg[up]; Bw_up = Bw[up]
        Rs_up = Rs[up]
        lam_o_up = lam_o[up]; lam_w_up = lam_w[up]; lam_g_up = lam_g[up]

        # capillary pressure at c and n
        Pcw_c, dPcw_dSw_c = Pcow(Sw[c])
        Pcw_n, dPcw_dSw_n = Pcow(Sw[n])
        Pcg_c, dPcg_dSg_c = Pcg(Sg[c])
        Pcg_n, dPcg_dSg_n = Pcg(Sg[n])

        # phase potentials
        if use_grav and abs(dz_ft) > 0.0:
            dPhi_o = dP - rho_o * G_FT_S2 * dz_ft
            dPhi_w = (P[c] - Pcw_c) - (P[n] - Pcw_n) - rho_w * G_FT_S2 * dz_ft
            dPhi_g = (P[c] + Pcg_c) - (P[n] + Pcg_n) - rho_g * G_FT_S2 * dz_ft
        else:
            dPhi_o = dP
            dPhi_w = (P[c] - Pcw_c) - (P[n] - Pcw_n)
            dPhi_g = (P[c] + Pcg_c) - (P[n] + Pcg_n)

        invBo_up = 1.0 / max(Bo_up, 1e-12)
        invBg_up = 1.0 / max(Bg_up, 1e-12)
        invBw_up = 1.0 / max(Bw_up, 1e-12)

        # component fluxes (c → n positive)
        Fo = T_face * lam_o_up * invBo_up * dPhi_o
        Fw = T_face * lam_w_up * invBw_up * dPhi_w
        Fg = T_face * (lam_g_up * invBg_up * dPhi_g + (Rs_up * lam_o_up * invBo_up) * dPhi_o)

        # residual
        Arow_o_c = iP(c); Arow_o_n = iP(n)
        Arow_w_c = iSw(c); Arow_w_n = iSw(n)
        Arow_g_c = iSg(c); Arow_g_n = iSg(n)

        R[Arow_o_c] += Fo; R[Arow_o_n] -= Fo
        R[Arow_w_c] += Fw; R[Arow_w_n] -= Fw
        R[Arow_g_c] += Fg; R[Arow_g_n] -= Fg

        # pressure coupling Jacobian (frozen upwind props)
        coef_o = T_face * lam_o_up * invBo_up
        coef_w = T_face * lam_w_up * invBw_up
        coef_g1 = T_face * lam_g_up * invBg_up
        coef_g2 = T_face * (Rs_up * lam_o_up * invBo_up)

        # rows at c
        A[Arow_o_c, iP(c)] += coef_o
        A[Arow_o_c, iP(n)] -= coef_o
        A[Arow_w_c, iP(c)] += coef_w
        A[Arow_w_c, iP(n)] -= coef_w
        A[Arow_g_c, iP(c)] += (coef_g1 + coef_g2)
        A[Arow_g_c, iP(n)] -= (coef_g1 + coef_g2)

        # rows at n (symmetric)
        A[Arow_o_n, iP(c)] -= coef_o
        A[Arow_o_n, iP(n)] += coef_o
        A[Arow_w_n, iP(c)] -= coef_w
        A[Arow_w_n, iP(n)] += coef_w
        A[Arow_g_n, iP(c)] -= (coef_g1 + coef_g2)
        A[Arow_g_n, iP(n)] += (coef_g1 + coef_g2)

        # NEW: pressure derivatives from denominators and Rs/Bo (on upwind pressure)
        dinvBo_up = -dBo_dP[up] / (max(Bo_up, 1e-12) ** 2)
        dinvBg_up = -dBg_dP[up] / (max(Bg_up, 1e-12) ** 2)
        dinvBw_up = -dBw_dP[up] / (max(Bw_up, 1e-12) ** 2)
        dRs_over_Bo = (dRs_dP[up] * Bo_up - Rs_up * dBo_dP[up]) / (max(Bo_up, 1e-12) ** 2)

        extra_o = T_face * lam_o_up * dPhi_o * dinvBo_up
        extra_w = T_face * lam_w_up * dPhi_w * dinvBw_up
        extra_g = T_face * (lam_g_up * dPhi_g * dinvBg_up + lam_o_up * dPhi_o * dRs_over_Bo)

        A[Arow_o_c, iP(up)] += extra_o
        A[Arow_o_n, iP(up)] -= extra_o
        A[Arow_w_c, iP(up)] += extra_w
        A[Arow_w_n, iP(up)] -= extra_w
        A[Arow_g_c, iP(up)] += extra_g
        A[Arow_g_n, iP(up)] -= extra_g

        # saturation derivatives (only upstream cell contributes)
        if up == c:
            So_up = So[c]; Sw_up = Sw[c]; Sg_up = Sg[c]
            muo_up = mu_o[c]; muw_up = mu_w[c]; mug_up = mu_g[c]

            dlamo_dSo = (kro_end * no * np.maximum(So_up, 1e-12) ** (no - 1.0)) / max(muo_up, 1e-12)
            dlamo_dSw = -dlamo_dSo
            dlamo_dSg = -dlamo_dSo
            dlamw_dSw = (krw_end * nw * np.maximum(Sw_up, 1e-12) ** (nw - 1.0)) / max(muw_up, 1e-12)
            dlamg_dSg = (2.0 * np.maximum(Sg_up, 0.0)) / max(mug_up, 1e-12)

            A[Arow_o_c, iSw(c)] += T_face * dlamo_dSw * invBo_up * dPhi_o
            A[Arow_o_n, iSw(c)] -= T_face * dlamo_dSw * invBo_up * dPhi_o
            A[Arow_o_c, iSg(c)] += T_face * dlamo_dSg * invBo_up * dPhi_o
            A[Arow_o_n, iSg(c)] -= T_face * dlamo_dSg * invBo_up * dPhi_o

            A[Arow_w_c, iSw(c)] += T_face * dlamw_dSw * invBw_up * dPhi_w
            A[Arow_w_n, iSw(c)] -= T_face * dlamw_dSw * invBw_up * dPhi_w

            # gas: free + dissolved-in-oil
            A[Arow_g_c, iSg(c)] += T_face * dlamg_dSg * invBg_up * dPhi_g
            A[Arow_g_n, iSg(c)] -= T_face * dlamg_dSg * invBg_up * dPhi_g
            A[Arow_g_c, iSw(c)] += T_face * (dlamo_dSw * Rs_up) * invBo_up * dPhi_o
            A[Arow_g_n, iSw(c)] -= T_face * (dlamo_dSw * Rs_up) * invBo_up * dPhi_o
            A[Arow_g_c, iSg(c)] += T_face * (dlamo_dSg * Rs_up) * invBo_up * dPhi_o
            A[Arow_g_n, iSg(c)] -= T_face * (dlamo_dSg * Rs_up) * invBo_up * dPhi_o

        else:  # up == n
            So_up = So[n]; Sw_up = Sw[n]; Sg_up = Sg[n]
            muo_up = mu_o[n]; muw_up = mu_w[n]; mug_up = mu_g[n]

            dlamo_dSo = (kro_end * no * np.maximum(So_up, 1e-12) ** (no - 1.0)) / max(muo_up, 1e-12)
            dlamo_dSw = -dlamo_dSo
            dlamo_dSg = -dlamo_dSo
            dlamw_dSw = (krw_end * nw * np.maximum(Sw_up, 1e-12) ** (nw - 1.0)) / max(muw_up, 1e-12)
            dlamg_dSg = (2.0 * np.maximum(Sg_up, 0.0)) / max(mug_up, 1e-12)

            A[Arow_o_c, iSw(n)] += T_face * dlamo_dSw * invBo_up * dPhi_o
            A[Arow_o_n, iSw(n)] -= T_face * dlamo_dSw * invBo_up * dPhi_o
            A[Arow_o_c, iSg(n)] += T_face * dlamo_dSg * invBo_up * dPhi_o
            A[Arow_o_n, iSg(n)] -= T_face * dlamo_dSg * invBo_up * dPhi_o

            A[Arow_w_c, iSw(n)] += T_face * dlamw_dSw * invBw_up * dPhi_w
            A[Arow_w_n, iSw(n)] -= T_face * dlamw_dSw * invBw_up * dPhi_w

            A[Arow_g_c, iSg(n)] += T_face * dlamg_dSg * invBg_up * dPhi_g
            A[Arow_g_n, iSg(n)] -= T_face * dlamg_dSg * invBg_up * dPhi_g
            A[Arow_g_c, iSw(n)] += T_face * (dlamo_dSw * Rs_up) * invBo_up * dPhi_o
            A[Arow_g_n, iSw(n)] -= T_face * (dlamo_dSw * Rs_up) * invBo_up * dPhi_o
            A[Arow_g_c, iSg(n)] += T_face * (dlamo_dSg * Rs_up) * invBo_up * dPhi_o
            A[Arow_g_n, iSg(n)] -= T_face * (dlamo_dSg * Rs_up) * invBo_up * dPhi_o

        # Capillary pressure derivatives (independent of upwind selection)
        if use_pc:
            cwcoef = T_face * lam_w_up * invBw_up
            # ∂dPhi_w/∂Sw(c) = -dPcw/dSw(c)
            A[Arow_w_c, iSw(c)] += cwcoef * (-dPcw_dSw_c)
            A[Arow_w_n, iSw(c)] -= cwcoef * (-dPcw_dSw_c)
            # ∂dPhi_w/∂Sw(n) = +dPcw/dSw(n)
            A[Arow_w_c, iSw(n)] += cwcoef * (+dPcw_dSw_n)
            A[Arow_w_n, iSw(n)] -= cwcoef * (+dPcw_dSw_n)

            cgcoef = T_face * lam_g_up * invBg_up
            # ∂dPhi_g/∂Sg(c) = +dPcg/dSg(c)
            A[Arow_g_c, iSg(c)] += cgcoef * (+dPcg_dSg_c)
            A[Arow_g_n, iSg(c)] -= cgcoef * (+dPcg_dSg_c)
            # ∂dPhi_g/∂Sg(n) = −dPcg/dSg(n)
            A[Arow_g_c, iSg(n)] += cgcoef * (-dPcg_dSg_n)
            A[Arow_g_n, iSg(n)] -= cgcoef * (-dPcg_dSg_n)

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

    # ---- wells (BHP/RATE via equivalent BHP) ----
    apply_wells_blackoil(
        A, R, grid, rock,
        P, Sw, Sg, So,
        Bo, Bg, Bw, Rs,
        mu_o, mu_g, mu_w,
        relperm, schedule, options
    )

    meta = {"note": "TPFA black-oil with φ(P), Bw(P), optional Pc, gravity, and BHP/RATE wells."}
    return A.tocsr(), R, meta

# ===== SKELETON stubs kept (not used by driver below) =====
def assemble_jacobian_and_residuals(state, grid, rock, fluid, schedule, options):
    raise NotImplementedError("3-phase assembler pending")

def newton_solve(state0, grid, rock, fluid, schedule, options):
    raise NotImplementedError("Implicit Newton driver pending")

# ===== Simple PVT model (dev/testing) WITH water PVT =========================
class _SimplePVT:
    """
    Lightweight PVT used for development/testing:
      - Bo(P), Bg(P) mild exponentials
      - Rs(P) capped linear to pb
      - Bw(P) exponential with small cw
      - μo, μg, μw constants (derivatives = 0)
    """
    def __init__(
        self,
        pb_psi=3000.0,
        Bo_pb=1.2, Rs_pb=600.0,
        mu_o_cp=1.2, mu_g_cp=0.02,
        Bw_ref=1.00, cw_1overpsi=2.5e-6,
        mu_w_cp=0.5
    ):
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
        c_o = 1.0e-5
        return self.Bo_pb * np.exp(c_o * (self.pb - P))

    def Bg(self, P):
        P = np.asarray(P, float)
        c_g = 3.0e-4
        return 0.005 * np.exp(-c_g * (self.pb - P))

    def Rs(self, P):
        P = np.asarray(P, float)
        return np.minimum(self.Rs_pb, self.Rs_pb * np.clip(P / self.pb, 0.0, None))

    def mu_o(self, P):
        P = np.asarray(P, float)
        return np.full_like(P, self.mu_o_cp)

    def mu_g(self, P):
        P = np.asarray(P, float)
        return np.full_like(P, self.mu_g_cp)

    def Bw(self, P):
        P = np.asarray(P, float)
        # Bw = Bw_ref * exp(-cw*(P - Pref)); choose Pref=self.pb
        return self.Bw_ref * np.exp(-self.cw * (P - self.pb))

    def mu_w(self, P):
        P = np.asarray(P, float)
        return np.full_like(P, self.mu_w_cp)

# ===== Build inputs + initial state ==========================================
def _build_inputs_for_blackoil(inputs):
    """
    Create grid/rock/relperm/init/schedule/options/pvt and initial state from 'inputs'.
    All arrays are shaped so assembler's reshape(N) lines work.
    """
    # --- grid ---
    nx = int(inputs.get("nx", 10))
    ny = int(inputs.get("ny", 10))
    nz = int(inputs.get("nz", 5))
    dx = float(inputs.get("dx", 100.0))
    dy = float(inputs.get("dy", 100.0))
    dz = float(inputs.get("dz", 50.0))
    grid = {"nx": nx, "ny": ny, "nz": nz, "dx": dx, "dy": dy, "dz": dz}
    N = nx * ny * nz

    # --- rock ---
    phi_val = float(inputs.get("phi", 0.08))
    kx_val = float(inputs.get("kx_md", 100.0))
    ky_val = float(inputs.get("ky_md", 100.0))
    rock = {
        "phi":   np.full(N, phi_val, dtype=float),
        "kx_md": np.full(N, kx_val, dtype=float),
        "ky_md": np.full(N, ky_val, dtype=float),
        # Optional rock compressibility parameters:
        # "cr_1overpsi": inputs.get("cr_1overpsi", 0.0),
        # "p_ref_psi": inputs.get("p_ref_psi", inputs.get("p_init_psi", 5000.0)),
    }

    # --- initial conditions ---
    p_init = float(inputs.get("p_init_psi", 5000.0))
    Sw0 = float(inputs.get("Sw0", 0.20))
    Sg0 = float(inputs.get("Sg0", 0.05))
    P0  = np.full(N, p_init, dtype=float)
    Sw  = np.full(N, Sw0, dtype=float)
    Sg  = np.full(N, Sg0, dtype=float)
    state0 = np.empty(3 * N, dtype=float)
    state0[0::3] = P0
    state0[1::3] = Sw
    state0[2::3] = Sg
    init = {"p_init_psi": p_init}

    # --- relperm params ---
    relperm = {
        "nw":       float(inputs.get("nw", 2.0)),
        "no":       float(inputs.get("no", 2.0)),
        "krw_end":  float(inputs.get("krw_end", 0.6)),
        "kro_end":  float(inputs.get("kro_end", 0.8)),
    }

    # --- schedule (BHP or RATE controlled) ---
    schedule = {
        "control":        str(inputs.get("control", "BHP")),
        "bhp_psi":        float(inputs.get("bhp_psi", 2500.0)),
        "pad_bhp_psi":    float(inputs.get("pad_bhp_psi", inputs.get("bhp_psi", 2500.0))),
        # Optional RATE targets:
        "rate_mscfd":     float(inputs.get("rate_mscfd", 0.0)),  # for RATE_GAS_MSCFD
        "pad_rate_mscfd": float(inputs.get("pad_rate_mscfd", inputs.get("rate_mscfd", 0.0))),
        "rate_stbd":      float(inputs.get("rate_stbd", 0.0)),   # for RATE_OIL_STBD / RATE_LIQ_STBD
        "pad_rate_stbd":  float(inputs.get("pad_rate_stbd", inputs.get("rate_stbd", 0.0))),
        # "wells": [...]  # optional explicit completions
    }

    # --- options / controls ---
    options = {
        "dt_days":     float(inputs.get("dt_days", 30.0)),   # 1-month step
        "t_end_days":  float(inputs.get("t_end_days", 3650.0)),
        "use_gravity": bool(inputs.get("use_gravity", True)),
        "use_pc":      bool(inputs.get("use_pc", DEFAULT_OPTIONS["use_pc"])),
        "rho_o_lbft3": float(inputs.get("rho_o_lbft3", DEFAULT_OPTIONS["rho_o_lbft3"])),
        "rho_w_lbft3": float(inputs.get("rho_w_lbft3", DEFAULT_OPTIONS["rho_w_lbft3"])),
        "rho_g_lbft3": float(inputs.get("rho_g_lbft3", DEFAULT_OPTIONS["rho_g_lbft3"])),
        "kvkh":        float(inputs.get("kvkh", DEFAULT_OPTIONS["kvkh"])),
        "geo_alpha":   float(inputs.get("geo_alpha", DEFAULT_OPTIONS["geo_alpha"])),
        # "cr_1overpsi": float(inputs.get("cr_1overpsi", DEFAULT_OPTIONS["cr_1overpsi"])),
    }

    # --- simple PVT ---
    pvt = _SimplePVT(
        pb_psi=float(inputs.get("pb_psi", 3000.0)),
        Bo_pb=float(inputs.get("Bo_pb_rb_stb", 1.2)),
        Rs_pb=float(inputs.get("Rs_pb_scf_stb", 600.0)),
        mu_o_cp=float(inputs.get("mu_o_cp", 1.2)),
        mu_g_cp=float(inputs.get("mu_g_cp", 0.02)),
        Bw_ref=float(inputs.get("Bw_ref", 1.00)),
        cw_1overpsi=float(inputs.get("cw_1overpsi", 2.5e-6)),
        mu_w_cp=float(inputs.get("mu_w_cp", 0.5)),
    )

    return state0, grid, rock, relperm, init, schedule, options, pvt

# ===== Well rate recomputation (for reporting) ===============================
def _compute_bhp_well_q(P, Sw, Sg, So, Bo, Bg, Bw, Rs, mu_o, mu_g, mu_w,
                        grid, rock, relperm, schedule):
    """
    Recompute well component rates (oil/gas/water). Positive = production.
    Mirrors the same Peaceman WI and control logic as apply_wells_blackoil().
    """
    nx, ny, nz = [int(grid[k]) for k in ("nx", "ny", "nz")]
    dx, dy, dz = [float(grid[k]) for k in ("dx", "dy", "dz")]
    def lin(i, j, k): return (k * ny + j) * nx + i

    nw = float(relperm.get("nw", 2.0))
    no = float(relperm.get("no", 2.0))
    krw_end = float(relperm.get("krw_end", 0.6))
    kro_end = float(relperm.get("kro_end", 0.8))

    kx = np.asarray(rock.get("kx_md")).reshape(nz, ny, nx)
    ky = np.asarray(rock.get("ky_md")).reshape(nz, ny, nx)

    wells = schedule.get("wells")
    if not wells:
        ctrl = str(schedule.get("control", schedule.get("pad_ctrl", "BHP"))).upper()
        wells = [{
            "i": nx // 2, "j": ny // 2, "k": nz // 2,
            "control": ctrl,
            "bhp_psi": float(schedule.get("bhp_psi", schedule.get("pad_bhp_psi", 2500.0))),
            "rate_mscfd": float(schedule.get("rate_mscfd", schedule.get("pad_rate_mscfd", 0.0))),
            "rate_stbd":  float(schedule.get("rate_stbd",  schedule.get("pad_rate_stbd",  0.0))),
            "rw_ft": 0.35, "skin": 0.0,
        }]

    q_o, q_g, q_w = 0.0, 0.0, 0.0
    for w in wells:
        i = int(w.get("i", nx // 2)); j = int(w.get("j", ny // 2)); k = int(w.get("k", nz // 2))
        c = lin(i, j, k)

        ctrl = str(w.get("control", schedule.get("control", schedule.get("pad_ctrl", "BHP")))).upper()
        rw = float(w.get("rw_ft", 0.35))
        skin = float(w.get("skin", 0.0))

        wi = peaceman_wi_cartesian(kx[k, j, i], ky[k, j, i], dz, dx, dy, rw_ft=rw, skin=skin)

        muo = max(mu_o[c], 1e-12); mug = max(mu_g[c], 1e-12); muw = max(mu_w[c], 1e-12)
        lam_o = max(0.0, kro_end * (So[c] ** no) / muo)
        lam_w = max(0.0, krw_end * (Sw[c] ** nw) / muw)
        lam_g = max(0.0, (Sg[c] ** 2) / mug)

        Bo_c = max(Bo[c], 1e-12)
        Bw_c = max(Bw[c], 1e-12)
        Bg_c = max(Bg[c], 1e-12)
        Rs_c = Rs[c]

        co = wi * lam_o / Bo_c
        cw = wi * lam_w / Bw_c
        cg = wi * (lam_g / Bg_c + (Rs_c * lam_o) / Bo_c)

        pw = _effective_pw_for_control(ctrl, P[c], co, cw, cg, w, schedule)
        dP = P[c] - pw

        q_o += co * dP
        q_g += cg * dP
        q_w += cw * dP

    return q_o, q_g, q_w

# ===== Newton solver (time marching) =========================================
def newton_solve_blackoil(state0, grid, rock, relperm, init, schedule, options, pvt):
    """
    Implicit time-marching Newton solver with:
      • Backtracking line search (stability)
      • Adaptive timestep control (grow/shrink)
    Returns basic production time series (qo, qg, qw).
    """
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    N = nx * ny * nz
    assert state0.size == 3 * N

    # controls
    dt_days = float(options.get("dt_days", 30.0))
    t_end   = float(options.get("t_end_days", 3650.0))
    max_newton = int(options.get("max_newton", 12))
    tol = float(options.get("newton_tol", 1e-6))
    dt_min_days = float(options.get("dt_min_days", 0.5))
    dt_max_days = float(options.get("dt_max_days", 120.0))
    grow = float(options.get("dt_grow", 1.5))
    shrink = float(options.get("dt_shrink", 0.5))

    # outputs
    t_hist, qo_hist, qg_hist, qw_hist = [], [], [], []

    # previous state for accumulation
    Pm = state0[0::3].copy(); Swm = state0[1::3].copy(); Sgm = state0[2::3].copy()
    t = 0.0
    state = state0.copy()

    while t < t_end - 1e-12:
        step_ok = False
        last_iters = 0

        # Try up to 10 retries per step with smaller dt if needed
        for _attempt in range(10):
            step_opts = dict(options)
            step_opts["dt_days"] = dt_days
            step_opts["prev"] = {"P": Pm, "Sw": Swm, "Sg": Sgm}

            state_k = state.copy()
            converged = False
            for it in range(max_newton):
                A, R, _ = assemble_jacobian_and_residuals_blackoil(
                    state_k, grid, rock, pvt, relperm, init, schedule, step_opts
                )
                normR = float(np.linalg.norm(R, ord=np.inf))
                if normR < tol:
                    converged = True
                    last_iters = it
                    break

                try:
                    dx = spsolve(A, -R)
                except Exception:
                    A = A.tolil()
                    idx = np.arange(A.shape[0])
                    A[idx, idx] = A[idx, idx] + 1e-12
                    A = A.tocsr()
                    dx = spsolve(A, -R)

                # Backtracking line search to ensure residual reduction
                alpha = 1.0
                for _ls in range(8):
                    trial = state_k + alpha * dx
                    _, R_try, _ = assemble_jacobian_and_residuals_blackoil(
                        trial, grid, rock, pvt, relperm, init, schedule, step_opts
                    )
                    if np.linalg.norm(R_try, ord=np.inf) <= 0.9 * normR:
                        state_k = trial
                        break
                    alpha *= 0.5
                else:
                    state_k = trial  # take the last (damped) trial

            if converged:
                step_ok = True
                state = state_k
                break
            else:
                # shrink dt and retry
                dt_days = max(dt_min_days, dt_days * shrink)

        if not step_ok:
            # give up further marching; return what we have
            break

        # update time & outputs
        P = state[0::3]; Sw = state[1::3]; Sg = state[2::3]; So = 1.0 - Sw - Sg
        Bo = np.asarray(pvt.Bo(P)); Bg = np.asarray(pvt.Bg(P)); Bw = np.asarray(pvt.Bw(P)); Rs = np.asarray(pvt.Rs(P))
        mu_o = np.asarray(pvt.mu_o(P)); mu_g = np.asarray(pvt.mu_g(P)); mu_w = np.asarray(pvt.mu_w(P))

        qo, qg, qw = _compute_bhp_well_q(
            P, Sw, Sg, So, Bo, Bg, Bw, Rs, mu_o, mu_g, mu_w, grid, rock, relperm, schedule
        )

        t += dt_days
        t_hist.append(t); qo_hist.append(qo); qg_hist.append(qg); qw_hist.append(qw)

        # roll prev
        Pm, Swm, Sgm = P.copy(), Sw.copy(), Sg.copy()

        # adapt dt for next step
        if last_iters <= 3:
            dt_days = min(dt_max_days, dt_days * grow)
        elif last_iters >= max_newton - 1:
            dt_days = max(dt_min_days, dt_days * shrink)

    return {
        "t": np.asarray(t_hist, float),
        "qo": np.asarray(qo_hist, float),   # STB/d (component oil)
        "qg": np.asarray(qg_hist, float),   # Mscf/d (component gas)
        "qw": np.asarray(qw_hist, float),   # STB/d (component water)
        "press_matrix": None,
        "pm_mid_psi": None,
        "p_init_3d": init.get("p_init_psi", None),
        "ooip_3d": None,
    }

# --- analytical proxy (kept so the app can still run fast) -------------------
def _simulate_analytical_proxy(inputs: dict):
    """Simple exponential decline proxy so the UI can run quickly for previews."""
    t = np.linspace(1.0, 3650.0, 240)  # days (~10 years)
    qi_g, di_g = 8000.0, 0.80   # Mcf/d, 1/yr
    qi_o, di_o = 1000.0, 0.70   # stb/d, 1/yr
    years = t / 365.25
    qg = qi_g * np.exp(-di_g * years)
    qo = qi_o * np.exp(-di_o * years)
    qw = np.zeros_like(qo)
    return {
        "t": t,
        "qg": qg,
        "qo": qo,
        "qw": qw,
        "press_matrix": None,
        "pm_mid_psi": None,
        "p_init_3d": None,
        "ooip_3d": None,
    }

# --- simulate() dispatch: analytical vs implicit ------------------------------
def simulate(inputs: dict):
    """
    Dispatch between the analytical proxy and the implicit engine.
    Set inputs['engine'] to 'analytical' or 'implicit'.
    """
    engine = inputs.get("engine", "analytical").lower().strip()
    if engine == "analytical":
        return _simulate_analytical_proxy(inputs)
    elif engine == "implicit":
        state0, grid, rock, relperm, init, schedule, options, pvt = _build_inputs_for_blackoil(inputs)
        return newton_solve_blackoil(state0, grid, rock, relperm, init, schedule, options, pvt)
    else:
        raise ValueError(f"Unknown engine '{engine}'")
