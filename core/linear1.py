# core/linear1.py
from __future__ import annotations
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, spilu, gmres, LinearOperator

def solve_linear(J, b, tol=1e-8, max_iter=2000):
    """
    Solve J x = b with ILU-preconditioned GMRES; fall back to SuperLU if needed.
    J: scipy.sparse matrix (csr/csc)
    """
    Jcsc = J if isinstance(J, csc_matrix) else J.tocsc()
    try:
        # ILU(0) with modest fill. Tune drop_tol/fill_factor if needed.
        ilu = spilu(Jcsc, drop_tol=1e-4, fill_factor=10)
        M = LinearOperator(Jcsc.shape, matvec=lambda x: ilu.solve(x))
        x, info = gmres(Jcsc, b, M=M, atol=0.0, tol=tol, restart=50, maxiter=max_iter)
        if info == 0:
            return x
        # If GMRES didn’t converge, use direct
        return spsolve(Jcsc, b)
    except Exception:
        # SuperLU direct solve (robust fallback)
        return spsolve(Jcsc, b)

