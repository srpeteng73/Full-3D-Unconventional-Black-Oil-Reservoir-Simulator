# core/timestepping1.py
from __future__ import annotations

class TimeStepping:
    def __init__(self, total_days: float, nsteps: int):
        self.total_days = float(total_days)
        self.nsteps = int(nsteps)

    def __iter__(self):
        dt = self.total_days / float(self.nsteps)
        t = 0.0
        for _ in range(self.nsteps):
            t += dt
            yield dt, t
