"""Impulsive maneuvers: Hohmann transfer between coplanar circular orbits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import MU_EARTH
from .dynamics import ForceModel
from .propagate import propagate


@dataclass
class HohmannPlan:
    r1: float
    r2: float
    dv1: float      # km/s, burn at periapsis of the transfer ellipse
    dv2: float      # km/s, circularisation burn at apoapsis
    tof: float      # s, time of flight (half the transfer-ellipse period)

    @property
    def dv_total(self):
        return abs(self.dv1) + abs(self.dv2)


def hohmann(r1, r2, mu=MU_EARTH) -> HohmannPlan:
    a_t = 0.5 * (r1 + r2)
    dv1 = np.sqrt(mu / r1) * (np.sqrt(2 * r2 / (r1 + r2)) - 1)
    dv2 = np.sqrt(mu / r2) * (1 - np.sqrt(2 * r1 / (r1 + r2)))
    tof = np.pi * np.sqrt(a_t**3 / mu)
    return HohmannPlan(r1, r2, dv1, dv2, tof)


def apply_impulse(y, dv):
    """Add a delta-v [km/s] along the current velocity direction (prograde burn)."""
    v = y[3:]
    out = np.array(y, float)
    out[3:] = v + dv * v / np.linalg.norm(v)
    return out


def simulate_hohmann(r1, r2, model: ForceModel = ForceModel()):
    """Numerically fly the transfer: circular orbit -> burn 1 -> coast -> burn 2 -> one final orbit."""
    plan = hohmann(r1, r2)
    y0 = np.array([r1, 0, 0, 0, np.sqrt(MU_EARTH / r1), 0.0])
    t_park = 2 * np.pi * np.sqrt(r1**3 / MU_EARTH)
    t_a, y_a = propagate(y0, t_park, model, n_out=400)
    y_b0 = apply_impulse(y_a[-1], plan.dv1)
    t_b, y_b = propagate(y_b0, plan.tof, model, n_out=800)
    y_c0 = apply_impulse(y_b[-1], plan.dv2)
    t_final = 2 * np.pi * np.sqrt(r2**3 / MU_EARTH)
    t_c, y_c = propagate(y_c0, t_final, model, n_out=800)
    legs = {"parking": (t_a, y_a), "transfer": (t_b + t_a[-1], y_b),
            "final": (t_c + t_a[-1] + t_b[-1], y_c)}
    return plan, legs
