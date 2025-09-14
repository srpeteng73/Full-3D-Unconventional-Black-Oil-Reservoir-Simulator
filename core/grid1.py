# core/grid1.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

_EPS = 1e-12

@dataclass
class Grid:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    num_cells: int
    # Rock properties (flattened length = num_cells)
    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray
    phi: np.ndarray

    @staticmethod
    def _flatten_or_fill(arr, shape, fill):
        if arr is None:
            return np.full(np.prod(shape), fill, dtype=float)
        a = np.asarray(arr, dtype=float)
        if a.ndim == 3 and a.shape == shape:
            return a.reshape(-1)
        if a.ndim == 1 and a.size == np.prod(shape):
            return a.copy()
        # Try broadcast to shape then flatten
        return np.full(np.prod(shape), fill, dtype=float)

    @staticmethod
    def from_inputs(g: dict, rock: dict | None = None) -> "Grid":
        nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
        dx, dy, dz = float(g["dx"]), float(g["dy"]), float(g["dz"])
        shape = (nz, ny, nx)
        num = nx * ny * nz

        rock = rock or {}
        kx = Grid._flatten_or_fill(rock.get("kx_md"), shape, 0.1)
        ky = Grid._flatten_or_fill(rock.get("ky_md"), shape, 0.1)
        # If kz not provided, use a vertical multiplier (0.1 × kx)
        kz_raw = rock.get("kz_md")
        kz = Grid._flatten_or_fill(kz_raw, shape, 0.1) if kz_raw is not None else 0.1 * kx
        phi = Grid._flatten_or_fill(rock.get("phi"),   shape, 0.12)

        return Grid(
            nx=nx, ny=ny, nz=nz,
            dx=dx, dy=dy, dz=dz,
            num_cells=num,
            kx=np.maximum(kx, _EPS),
            ky=np.maximum(ky, _EPS),
            kz=np.maximum(kz, _EPS),
            phi=np.clip(phi, 1e-3, 0.35),
        )

    def get_idx(self, i: int, j: int, k: int) -> int:
        if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
            return k * (self.nx * self.ny) + j * self.nx + i
        return -1
