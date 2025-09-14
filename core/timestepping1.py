# core/timestepping.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TimeStepper:
    total_days: float
    nsteps: int

    def __iter__(self):
        dt = self.total_days / self.nsteps
        t = 0.0
        for _ in range(self.nsteps):
            t += dt
            yield dt, t
