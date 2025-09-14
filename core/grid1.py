# core/grid.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Grid:
    nx: int; ny: int; nz: int
    dx: float; dy: float; dz: float

    @property
    def num_cells(self): return self.nx * self.ny * self.nz

    def get_idx(self, i, j, k):
        if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
            return k * (self.nx * self.ny) + j * self.nx + i
        return -1

    @staticmethod
    def from_inputs(d: dict) -> "Grid":
        return Grid(int(d["nx"]), int(d["ny"]), int(d["nz"]),
                    float(d["dx"]), float(d["dy"]), float(d["dz"]))
