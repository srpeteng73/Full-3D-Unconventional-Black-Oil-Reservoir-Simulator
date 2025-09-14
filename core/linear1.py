# core/linear1.py
from __future__ import annotations
import numpy as np
from typing import Optional
from scipy.sparse.linalg import spsolve, cg
from scipy.sparse import isspmatrix

def solve_linear(A, b, max_iter: int = 200):
    """Small, robust linear solve. Uses direct spsolve if sparse; falls back to CG."""
    if isspmatrix(A):
        try:
            return spsolve(A, b)
        except Exception:
            x, info = cg(A, b, maxiter=max_iter)
            if info != 0:
                raise RuntimeError(f"CG failed (info={info})")
            return x
    else:
        # dense fallback
        return np.linalg.solve(A, b)
