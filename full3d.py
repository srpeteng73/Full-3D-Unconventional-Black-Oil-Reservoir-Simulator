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
    Skeleton for a fully-implicit black-oil 3-phase assembly.
    Unknowns per cell: [P, Sw, Sg]  (So = 1 - Sw - Sg)

    This function compiles as-is and provides clear TODOs for:
      • Component mass balances (oil/gas/water)
      • Fluxes with upwinding
      • Full 3×3 block derivatives dR/d[P,Sw,Sg] for each connection
      • Wells (BHP or RATE) using Peaceman WI

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Jacobian (sparse)
    R : np.ndarray
        Residual vector
    meta : dict
        Debug metadata
    """
    nx, ny, nz = [int(grid[k]) for k in ("nx", "ny", "nz")]
    N = nx * ny * nz
    nunk = 3 * N

    def iP(i):  return 3 * i
    def iSw(i): return 3 * i + 1
    def iSg(i): return 3 * i + 2

    # Unpack state vector
    P = state_vec[0::3].copy()
    Sw = state_vec[1::3].copy()
    Sg = state_vec[2::3].copy()
    So = 1.0 - Sw - Sg

    eps = 1e-9
    Sw = np.clip(Sw, 0.0 + eps, 1.0 - eps)
    Sg = np.clip(Sg, 0.0 + eps, 1.0 - eps)
    So = np.clip(So,  0.0 + eps, 1.0 - eps)

    # Rock props (flatten to 1D)
    phi = np.asarray(rock.get("phi", 0.08)).reshape(N)               # porosity
    kx  = np.asarray(rock.get("kx_md", 100.0)).reshape(N)            # mD
    ky  = np.asarray(rock.get("ky_md", 100.0)).reshape(N)            # mD
    # TODO: kz if needed

    # PVT callables (must be provided by the app layer)
    Bo = np.asarray(pvt.Bo(P))                # rb/STB
    Bg = np.asarray(pvt.Bg(P))                # rb/scf (or consistent units)
    Rs = np.asarray(pvt.Rs(P))                # scf/STB
    mu_o = np.asarray(pvt.mu_o(P))            # cP
    mu_g = np.asarray(pvt.mu_g(P))            # cP
    # simple placeholder for water viscosity & FVF
    mu_w = np.full_like(P, 0.5)               # cP
    Bw   = np.ones_like(P)                    # rb/STB

    # RelPerm functions (provide in relperm dict or default Corey-like)
    krw_fn = relperm.get("krw_fn", lambda s: (s ** relperm.get("nw", 2.0)) * relperm.get("krw_end", 0.6))
    kro_fn = relperm.get("kro_fn", lambda s: (s ** relperm.get("no", 2.0)) * relperm.get("kro_end", 0.8))
    krg_fn = relperm.get("krg_fn", lambda s: s ** 2)

    kro = np.asarray(kro_fn(So))
    krw = np.asarray(krw_fn(Sw))
    krg = np.asarray(krg_fn(Sg))

    # Mobilities
    lam_o = kro / np.maximum(mu_o, 1e-12)
    lam_w = krw / np.maximum(mu_w, 1e-12)
    lam_g = krg / np.maximum(mu_g, 1e-12)
    lam_t = lam_o + lam_w + lam_g  # (for total-velocity upwinding if used)

    # Geomechanics multiplier (optional)
    p_init = float(init.get("p_init_psi", 5000.0))
    geo_alpha = float(options.get("geo_alpha", 0.0))
    kmult = k_multiplier_from_pressure(P, p_init, alpha=geo_alpha)
    kx *= kmult
    ky *= kmult

    # Time step and compressibility (skeleton)
    dt_days = float(options.get("dt_days", 1.0))
    dt = dt_days * 86400.0
    ct = float(options.get("ct_1_over_psi", 1e-5))  # rock+fluid compressibility placeholder

    # Accumulations (black-oil components; previous state needed in full implicit)
    # oil:   phi * So / Bo
    # gas:   phi * (Sg / Bg + So * Rs / Bo)
    # water: phi * Sw / Bw
    acc_o = phi * So / np.maximum(Bo, 1e-12)
    acc_g = phi * (Sg / np.maximum(Bg, 1e-12) + So * Rs / np.maximum(Bo, 1e-12))
    acc_w = phi * Sw / np.maximum(Bw, 1e-12)
    # TODO: subtract previous accumulations and scale by 1/dt for transient terms

    # Grid helpers
    dx, dy, dz = grid["dx"], grid["dy"], grid["dz"]
    def cell_index(k, j, i):
        return (k * ny + j) * nx + i

    # Allocate sparse Jacobian and residual
    A = lil_matrix((nunk, nunk), dtype=float)
    R = np.zeros(nunk, dtype=float)

    # -------------------------------------------------------------------------
    # Neighbor loops (skeleton). Fill TPFA fluxes and full derivatives here.
    # -------------------------------------------------------------------------
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = cell_index(k, j, i)

                # Diagonal stabilization to keep matrix nonsingular in skeleton
                A[iP(c),  iP(c)]  += 1.0 + acc_o[c] + acc_g[c] + acc_w[c]
                A[iSw(c), iSw(c)] += 1.0
                A[iSg(c), iSg(c)] += 1.0
                # R[...] stays zero for now; replace with actual mass balances:
                #   R_oil(c)   = accum_oil + sum(flux_oil) + well_src_oil - target
                #   R_gas(c)   = accum_gas + sum(flux_gas) + well_src_gas - target
                #   R_water(c) = accum_wtr + sum(flux_wtr) + well_src_wtr - target
                # and fill corresponding A[...] entries for dR/dP, dR/dSw, dR/dSg

                # Example neighbor (i+1) pattern to expand when implementing fluxes:
                # if i + 1 < nx:
                #     n = cell_index(k, j, i + 1)
                #     # harmonic perms, face area / distance, upwinding based on pressure + gravity, etc.
                #     # add flux contributions to R and derivatives to A

    meta = {"note": "black-oil skeleton; fill full 3×3 residuals and flux derivatives"}
    return A.tocsr(), R, meta


# -----------------------------------------------------------------------------
# Minimal working proxy so Streamlit app runs end-to-end today
# -----------------------------------------------------------------------------
def simulate(inputs: dict) -> dict:
    """
    Minimal proxy so the app runs on Streamlit/Colab while you complete
    the implicit engine. Replace with calls to the full 3D solver when ready.
    """
    # time in days
    t = np.linspace(1.0, 3650.0, 240)  # ~10 years
    # crude declines (proxy)
    qi_g, di_g = 8000.0, 0.80   # Mcf/d, 1/yr
    qi_o, di_o = 1000.0, 0.70   # stb/d, 1/yr
    years = t / 365.25
    qg = qi_g * np.exp(-di_g * years)
    qo = qi_o * np.exp(-di_o * years)

    return {
        "t": t,
        "qg": qg,
        "qo": qo,
        # Optional fields used by downstream tabs:
        "press_matrix": None,
        "pm_mid_psi": None,
        "p_init_3d": None,
        "ooip_3d": None,
    }


