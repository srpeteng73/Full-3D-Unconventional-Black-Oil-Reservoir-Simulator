# core/linear.py
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

def solve_linear(J, rhs, max_iter=200):
    """Small wrapper so we can swap solvers later (PyAMG, MKL, cuSPARSE)."""
    if not hasattr(J, "tocsc"):
        J = csc_matrix(J)
    return spsolve(J, rhs)
