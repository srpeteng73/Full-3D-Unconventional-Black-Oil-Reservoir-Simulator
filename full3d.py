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
from scipy.sparse import lil_matrix


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
    Fully implicit black-oil skeleton with TPFA fluxes and a usable sparse Jacobian.
    Unknowns per cell: [P, Sw, Sg]  (So = 1 - Sw - Sg)

    Equations mapped per cell:
      • Row iP(c):   OIL component balance      (phi*So/Bo,  +  oil-phase flux / Bo)
      • Row iSw(c):  WATER component balance    (phi*Sw/Bw,  +  water-phase flux / Bw)
      • Row iSg(c):  GAS component balance      (phi*(Sg/Bg + So*Rs/Bo),  + gas-phase flux/Bg + Rs*oil-flux/Bo)

    Notes
    -----
    • Time discretization: backward Euler. If previous state is not provided, the transient term is zero.
      Pass previous state via options["prev"] = {"P":..., "Sw":..., "Sg":...}.
    • Gravity is omitted here for clarity (easy to add later).
    • Flux upwinding: phase mobility and Rs/Bo/Bg on the *upstream* cell of the face.
    • Jacobian: includes accumulation derivatives and pressure neighbor coupling; saturation
      derivatives include the effect of relperm on the *upstream* cell’s mobility.
    """

    # ---------- helpers ----------
    def iP(i):  return 3 * i
    def iSw(i): return 3 * i + 1
    def iSg(i): return 3 * i + 2

    def harm(a, b):
        return 2.0 * a * b / (a + b + 1e-30)

    # numerical derivative for PVT vs pressure (central diff)
    def numdiff_p(fn, p, eps=1e-3):
        return (np.asarray(fn(p + eps)) - np.asarray(fn(p - eps))) / (2.0 * eps)

    # relperm shapes & derivatives (Corey-like defaults)
    nw = float(relperm.get("nw", 2.0))
    no = float(relperm.get("no", 2.0))
    krw_end = float(relperm.get("krw_end", 0.6))
    kro_end = float(relperm.get("kro_end", 0.8))
    # gas kr default ~ s^2; derivative 2s
    def dkrw_dSw(sw):  # d(krw)/dSw
        return krw_end * nw * np.maximum(sw, 1e-12) ** (nw - 1.0)
    def dkro_dSo(so):  # d(kro)/dSo
        return kro_end * no * np.maximum(so, 1e-12) ** (no - 1.0)
    def dkrg_dSg(sg):  # d(krg)/dSg, with default s^2
        return 2.0 * np.maximum(sg, 0.0)

    # ---------- unpack/grid ----------
    nx, ny, nz = [int(grid[k]) for k in ("nx", "ny", "nz")]
    dx, dy, dz = [float(grid[k]) for k in ("dx", "dy", "dz")]
    N = nx * ny * nz
    nunk = 3 * N

    def idx(k, j, i):
        return (k * ny + j) * nx + i

    Vcell = dx * dy * dz

    # ---------- unknowns ----------
    P  = state_vec[0::3].copy()
    Sw = state_vec[1::3].copy()
    Sg = state_vec[2::3].copy()
    So = 1.0 - Sw - Sg

    # bounding
    eps = 1e-9
    Sw[:] = np.clip(Sw, 0.0 + eps, 1.0 - eps)
    Sg[:] = np.clip(Sg, 0.0 + eps, 1.0 - eps)
    So[:] = np.clip(So,  0.0 + eps, 1.0 - eps)

    # ---------- rock ----------
    phi = np.asarray(rock.get("phi", 0.08)).reshape(N)
    kx  = np.asarray(rock.get("kx_md", 100.0)).reshape(N)
    ky  = np.asarray(rock.get("ky_md", 100.0)).reshape(N)

    # optional geomech multiplier
    p_init = float(init.get("p_init_psi", 5000.0))
    geo_alpha = float(options.get("geo_alpha", 0.0))
    kmult = k_multiplier_from_pressure(P, p_init, alpha=geo_alpha)
    kx *= kmult
    ky *= kmult

    # ---------- time stepping ----------
    dt_days = float(options.get("dt_days", 1.0))
    dt = dt_days * 86400.0

    prev = options.get("prev", None)
    if prev is None:
        Pm, Swm, Sgm = P.copy(), Sw.copy(), Sg.copy()
    else:
        Pm  = np.asarray(prev.get("P",  P))
        Swm = np.asarray(prev.get("Sw", Sw))
        Sgm = np.asarray(prev.get("Sg", Sg))
    Som = 1.0 - Swm - Sgm

    # ---------- PVT ----------
    Bo = np.asarray(pvt.Bo(P))
    Bg = np.asarray(pvt.Bg(P))
    Rs = np.asarray(pvt.Rs(P))
    mu_o = np.asarray(pvt.mu_o(P))
    mu_g = np.asarray(pvt.mu_g(P))
    mu_w = np.full_like(P, 0.5)  # placeholder
    Bw   = np.ones_like(P)       # placeholder

    dBo_dP = numdiff_p(pvt.Bo, P)
    dBg_dP = numdiff_p(pvt.Bg, P)
    dRs_dP = numdiff_p(pvt.Rs, P)
    # water FVF derivative ~ 0 in placeholder
    dBw_dP = np.zeros_like(P)

    # ---------- relperm & mobilities ----------
    krw = relperm.get("krw_fn", lambda s: krw_end * (s ** nw))(Sw)
    kro = relperm.get("kro_fn", lambda s: kro_end * (s ** no))(So)
    krg = relperm.get("krg_fn", lambda s: s ** 2)(Sg)

    lam_w = krw / np.maximum(mu_w, 1e-12)
    lam_o = kro / np.maximum(mu_o, 1e-12)
    lam_g = krg / np.maximum(mu_g, 1e-12)

    # ---------- accumulations (new and old) ----------
    # oil comp:   phi * So / Bo
    # gas comp:   phi * (Sg / Bg + So * Rs / Bo)
    # water comp: phi * Sw / Bw
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

    # ---------- allocate system ----------
    A = lil_matrix((nunk, nunk), dtype=float)
    R = np.zeros(nunk, dtype=float)

    # add accumulation residuals & their Jacobian (local diagonals)
    # R += V/dt * (acc_new - acc_old)
    scale = Vcell / max(dt, 1e-30)
    R[0::3] += scale * (acc_o - accm_o)       # oil comp → row iP
    R[1::3] += scale * (acc_w - accm_w)       # water comp → row iSw
    R[2::3] += scale * (acc_g - accm_g)       # gas comp → row iSg

    # accumulation derivatives wrt NEW state
    # d(acc_o)/dP = phi * So * d(1/Bo)/dP = phi * So * ( - dBo/dP / Bo^2 )
    dacc_o_dP  = phi * So * (-dBo_dP / (np.maximum(Bo, 1e-12) ** 2))
    # d(acc_o)/dSw = phi * dSo/dSw / Bo = -phi / Bo ; d(acc_o)/dSg = -phi / Bo
    dacc_o_dSw = -phi / np.maximum(Bo, 1e-12)
    dacc_o_dSg = -phi / np.maximum(Bo, 1e-12)

    # water
    dacc_w_dP  = phi * Sw * (-dBw_dP / (np.maximum(Bw, 1e-12) ** 2))  # ~ 0 with Bw=1
    dacc_w_dSw = phi / np.maximum(Bw, 1e-12)

    # gas
    # ∂/∂P[ phi*(Sg/Bg + So*Rs/Bo) ] =
    #   phi*( Sg * ( -dBg/dP / Bg^2 ) + So*( dRs/dP / Bo - Rs * dBo/dP / Bo^2 ) )
    term1 = Sg * (-dBg_dP / (np.maximum(Bg, 1e-12) ** 2))
    term2 = So * (dRs_dP / np.maximum(Bo, 1e-12) - Rs * dBo_dP / (np.maximum(Bo, 1e-12) ** 2))
    dacc_g_dP  = phi * (term1 + term2)
    dacc_g_dSw = -phi * (Rs / np.maximum(Bo, 1e-12))   # from So = 1-Sw-Sg
    dacc_g_dSg =  phi / np.maximum(Bg, 1e-12) - phi * (Rs / np.maximum(Bo, 1e-12))

    # put accumulation Jacobian on diagonals
    for c in range(N):
        A[iP(c),  iP(c)]  += scale * dacc_o_dP[c]
        A[iP(c),  iSw(c)] += scale * dacc_o_dSw[c]
        A[iP(c),  iSg(c)] += scale * dacc_o_dSg[c]

        A[iSw(c), iP(c)]  += scale * dacc_w_dP[c]
        A[iSw(c), iSw(c)] += scale * dacc_w_dSw[c]

        A[iSg(c), iP(c)]  += scale * dacc_g_dP[c]
        A[iSg(c), iSw(c)] += scale * dacc_g_dSw[c]
        A[iSg(c), iSg(c)] += scale * dacc_g_dSg[c]

    # ---------- face fluxes (TPFA, upwind) ----------
    # F_o^comp =  T * lam_o_up * (Pc - Pn) / Bo_up
    # F_w^comp =  T * lam_w_up * (Pc - Pn) / Bw_up
    # F_g^comp =  T * [lam_g_up * (Pc - Pn) / Bg_up  +  Rs_up * lam_o_up * (Pc - Pn) / Bo_up]
    #
    # For residuals we add +F at cell c (outflow positive) and -F at cell n.
    #
    area_x = dy * dz
    area_y = dx * dz
    # (you can add k_z and z-direction similarly if/when you include vertical flow)

    def add_face(c, n, T_face):
        # pressure drop & upstream cell
        dP = P[c] - P[n]
        up = c if dP >= 0.0 else n

        # upstream phase mobilities
        lam_o_up = lam_o[up]
        lam_w_up = lam_w[up]
        lam_g_up = lam_g[up]

        # upstream PVT factors
        Bo_up = Bo[up]
        Bg_up = Bg[up]
        Bw_up = Bw[up]
        Rs_up = Rs[up]

        # component fluxes (c -> n, positive when leaving c)
        base = T_face * dP
        Fo = base * lam_o_up / np.maximum(Bo_up, 1e-12)
        Fw = base * lam_w_up / np.maximum(Bw_up, 1e-12)
        Fg = base * (lam_g_up / np.maximum(Bg_up, 1e-12) + (Rs_up * lam_o_up) / np.maximum(Bo_up, 1e-12))

        # residual add (c gets +, n gets -)
        R[iP(c)]  += Fo
        R[iP(n)]  -= Fo
        R[iSw(c)] += Fw
        R[iSw(n)] -= Fw
        R[iSg(c)] += Fg
        R[iSg(n)] -= Fg

        # Jacobian: pressure coupling (hold upwind props constant)
        coef_o = T_face * (lam_o_up / np.maximum(Bo_up, 1e-12))
        coef_w = T_face * (lam_w_up / np.maximum(Bw_up, 1e-12))
        coef_g = T_face * (lam_g_up / np.maximum(Bg_up, 1e-12) + (Rs_up * lam_o_up) / np.maximum(Bo_up, 1e-12))

        # rows for c
        A[iP(c),  iP(c)]  += coef_o
        A[iP(c),  iP(n)]  -= coef_o
        A[iSw(c), iP(c)]  += coef_w
        A[iSw(c), iP(n)]  -= coef_w
        A[iSg(c), iP(c)]  += coef_g
        A[iSg(c), iP(n)]  -= coef_g

        # rows for n (negative of above)
        A[iP(n),  iP(c)]  -= coef_o
        A[iP(n),  iP(n)]  += coef_o
        A[iSw(n), iP(c)]  -= coef_w
        A[iSw(n), iP(n)]  += coef_w
        A[iSg(n), iP(c)]  -= coef_g
        A[iSg(n), iP(n)]  += coef_g

        # Saturation dependence via relperm on the *upstream* cell
        # dFo/dSw_up etc.  (∂lam/∂sat * dP / Bo_up)
        if up == c:
            So_up = So[c]; Sw_up = Sw[c]; Sg_up = Sg[c]
            muo_up = mu_o[c]; muw_up = mu_w[c]; mug_up = mu_g[c]

            dlamo_dSo = dkro_dSo(So_up) / max(muo_up, 1e-12)
            dlamo_dSw = -dlamo_dSo    # since So = 1 - Sw - Sg
            dlamo_dSg = -dlamo_dSo

            dlamw_dSw = dkrw_dSw(Sw_up) / max(muw_up, 1e-12)
            dlamg_dSg = dkrg_dSg(Sg_up) / max(mug_up, 1e-12)

            dFo_dSw_up = T_face * dP * dlamo_dSw / max(Bo_up, 1e-12)
            dFo_dSg_up = T_face * dP * dlamo_dSg / max(Bo_up, 1e-12)

            dFw_dSw_up = T_face * dP * dlamw_dSw / max(Bw_up, 1e-12)

            dFg_dSg_up = T_face * dP * dlamg_dSg / max(Bg_up, 1e-12)
            dFg_dSw_up = T_face * dP * (dlamo_dSw * Rs_up) / max(Bo_up, 1e-12)
            dFg_dSg_o  = T_face * dP * (dlamo_dSg * Rs_up) / max(Bo_up, 1e-12)

            # apply to rows (c gets +, n gets -), columns are upstream cell's sats
            A[iP(c),  iSw(c)] += dFo_dSw_up
            A[iP(n),  iSw(c)] -= dFo_dSw_up
            A[iP(c),  iSg(c)] += dFo_dSg_up
            A[iP(n),  iSg(c)] -= dFo_dSg_up

            A[iSw(c), iSw(c)] += dFw_dSw_up
            A[iSw(n), iSw(c)] -= dFw_dSw_up

            A[iSg(c), iSg(c)] += dFg_dSg_up
            A[iSg(n), iSg(c)] -= dFg_dSg_up
            A[iSg(c), iSw(c)] += dFg_dSw_up
            A[iSg(n), iSw(c)] -= dFg_dSw_up
            A[iSg(c), iSg(c)] += dFg_dSg_o  # additional oil-carried gas term via kro
            A[iSg(n), iSg(c)] -= dFg_dSg_o

        elif up == n:
            So_up = So[n]; Sw_up = Sw[n]; Sg_up = Sg[n]
            muo_up = mu_o[n]; muw_up = mu_w[n]; mug_up = mu_g[n]

            dlamo_dSo = dkro_dSo(So_up) / max(muo_up, 1e-12)
            dlamo_dSw = -dlamo_dSo
            dlamo_dSg = -dlamo_dSo

            dlamw_dSw = dkrw_dSw(Sw_up) / max(muw_up, 1e-12)
            dlamg_dSg = dkrg_dSg(Sg_up) / max(mug_up, 1e-12)

            dFo_dSw_up = T_face * dP * dlamo_dSw / max(Bo_up, 1e-12)
            dFo_dSg_up = T_face * dP * dlamo_dSg / max(Bo_up, 1e-12)

            dFw_dSw_up = T_face * dP * dlamw_dSw / max(Bw_up, 1e-12)

            dFg_dSg_up = T_face * dP * dlamg_dSg / max(Bg_up, 1e-12)
            dFg_dSw_up = T_face * dP * (dlamo_dSw * Rs_up) / max(Bo_up, 1e-12)
            dFg_dSg_o  = T_face * dP * (dlamo_dSg * Rs_up) / max(Bo_up, 1e-12)

            # apply to rows, columns are neighbor's sats
            A[iP(c),  iSw(n)] += dFo_dSw_up
            A[iP(n),  iSw(n)] -= dFo_dSw_up
            A[iP(c),  iSg(n)] += dFo_dSg_up
            A[iP(n),  iSg(n)] -= dFo_dSg_up

            A[iSw(c), iSw(n)] += dFw_dSw_up
            A[iSw(n), iSw(n)] -= dFw_dSw_up

            A[iSg(c), iSg(n)] += dFg_dSg_up
            A[iSg(n), iSg(n)] -= dFg_dSg_up
            A[iSg(c), iSw(n)] += dFg_dSw_up
            A[iSg(n), iSw(n)] -= dFg_dSw_up
            A[iSg(c), iSg(n)] += dFg_dSg_o
            A[iSg(n), iSg(n)] -= dFg_dSg_o

    # sweep faces once (i+1, j+1). Add k-direction later when you add kz.
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = idx(k, j, i)

                # i-face (to i+1)
                if i + 1 < nx:
                    n = idx(k, j, i + 1)
                    k_face = harm(kx[c], kx[n])
                    T = (k_face * area_x) / max(dx, 1e-30)
                    add_face(c, n, T)

                # j-face (to j+1)
                if j + 1 < ny:
                    n = idx(k, j + 1, i)
                    k_face = harm(ky[c], ky[n])
                    T = (k_face * area_y) / max(dy, 1e-30)
                    add_face(c, n, T)

    meta = {
        "note": "TPFA black-oil with accumulation+pressure coupling; "
                "includes upwind relperm saturation derivatives on fluxes. "
                "Gravity & wells can be added next."
    }
    return A.tocsr(), R, meta
