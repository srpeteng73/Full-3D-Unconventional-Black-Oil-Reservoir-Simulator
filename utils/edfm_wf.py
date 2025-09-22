# utils/edfm_wf.py
from typing import Dict, List

def build_edfm_line_along_x(grid: dict, j_mid: int, k_mid: int,
                            i0: int, i1: int, T_each: float) -> Dict[str, List[dict]]:
    """
    Return an edfm_connectivity dict with links on cells (i0..i1) in a fixed j_mid,k_mid row,
    each with the same transmissibility `T_each` (already in your TPFA units).
    """
    nx = int(grid["nx"]); ny = int(grid.get("ny", 1)); nz = int(grid.get("nz", 1))
    def lin(ii, jj, kk):  # 0-based -> linear index, ii fastest
        return (kk*ny + jj)*nx + ii

    mf_T = []
    for ii in range(i0, i1 + 1):
        mf_T.append({"cell": int(lin(ii, j_mid, k_mid)), "T": float(T_each)})
    return {"mf_T": mf_T}

def apply_edfm_leak(A, lam_o, lam_w, lam_g, Bo, Bw, Bg, options=None):
    """
    Phase #2 diagonal EDFM: add a per-cell diagonal 'leak' using fracture/matrix
    connectivity (embedded EDFM) without adding new unknowns.
    Works for dense NumPy arrays or scipy.sparse CSR.
    """
    edfm = (options or {}).get("edfm_connectivity")
    if not edfm or (options or {}).get("edfm_mode", "embedded") != "embedded":
        return A

    # Convert CSR->LIL for efficient in-place diagonal updates
    need_back_to_csr = False
    try:
        import scipy.sparse as sp
        if sp.isspmatrix_csr(A):
            A = A.tolil(copy=False)
            need_back_to_csr = True
    except Exception:
        pass  # A may be a dense ndarray

    for link in edfm.get("mf_T", []):
        c = int(link["cell"])        # 0-based cell index
        Tf = float(link["T"])        # fracture "conductance"
        # add diagonal with per-phase mobilities and FVFs
        A[c, c] += Tf * (
            lam_o[c] / max(Bo[c], 1e-12) +
            lam_w[c] / max(Bw[c], 1e-12) +
            lam_g[c] / max(Bg[c], 1e-12)
        )

    if need_back_to_csr:
        A = A.tocsr(copy=False)
    return A
