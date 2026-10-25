"""Equations of motion: two-body gravity, J2 oblateness and atmospheric drag.

State vector y = [x, y, z, vx, vy, vz] in the Earth-centred inertial (ECI) frame, km and km/s.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import J2, MU_EARTH, OMEGA_EARTH, R_EARTH

# Exponential atmosphere, Vallado Table 8-4: base altitude [km], base density [kg/m^3], scale height [km]
_ATMOS = np.array([
    [0, 1.225, 7.249], [25, 3.899e-2, 6.349], [30, 1.774e-2, 6.682], [40, 3.972e-3, 7.554],
    [50, 1.057e-3, 8.382], [60, 3.206e-4, 7.714], [70, 8.770e-5, 6.549], [80, 1.905e-5, 5.799],
    [90, 3.396e-6, 5.382], [100, 5.297e-7, 5.877], [110, 9.661e-8, 7.263], [120, 2.438e-8, 9.473],
    [130, 8.484e-9, 12.636], [140, 3.845e-9, 16.149], [150, 2.070e-9, 22.523], [180, 5.464e-10, 29.740],
    [200, 2.789e-10, 37.105], [250, 7.248e-11, 45.546], [300, 2.418e-11, 53.628], [350, 9.518e-12, 53.298],
    [400, 3.725e-12, 58.515], [450, 1.585e-12, 60.828], [500, 6.967e-13, 63.822], [600, 1.454e-13, 71.835],
    [700, 3.614e-14, 88.667], [800, 1.170e-14, 124.64], [900, 5.245e-15, 181.05], [1000, 3.019e-15, 268.00],
])


def atmospheric_density(alt_km: float) -> float:
    """Density [kg/m^3] from the piecewise exponential model rho = rho0 * exp(-(h - h0)/H)."""
    if alt_km < 0:
        alt_km = 0.0
    idx = np.searchsorted(_ATMOS[:, 0], alt_km, side="right") - 1
    h0, rho0, H = _ATMOS[max(idx, 0)]
    return rho0 * np.exp(-(alt_km - h0) / H)


@dataclass
class ForceModel:
    """Which perturbations to include and the spacecraft's drag properties."""
    j2: bool = False
    drag: bool = False
    cd: float = 2.2          # drag coefficient
    area_m2: float = 1.0     # cross-sectional area [m^2]
    mass_kg: float = 100.0   # mass [kg]

    @property
    def ballistic(self) -> float:
        """Cd*A/m [m^2/kg]."""
        return self.cd * self.area_m2 / self.mass_kg


def accel_two_body(r):
    rn = np.linalg.norm(r)
    return -MU_EARTH * r / rn**3


def accel_j2(r):
    x, y, z = r
    rn = np.linalg.norm(r)
    k = -1.5 * J2 * MU_EARTH * R_EARTH**2 / rn**5
    zr2 = 5 * z**2 / rn**2
    return k * np.array([x * (1 - zr2), y * (1 - zr2), z * (3 - zr2)])


def accel_drag(r, v, ballistic):
    """Drag acceleration [km/s^2]. Air co-rotates with Earth, so use velocity relative to the atmosphere."""
    alt = np.linalg.norm(r) - R_EARTH
    rho = atmospheric_density(alt)
    v_rel = v - np.cross([0.0, 0.0, OMEGA_EARTH], r)
    # 0.5*rho*B*|v|v with v in m/s gives m/s^2; converting km/s in and km/s^2 out gives a factor of 1e3.
    return -0.5 * rho * ballistic * np.linalg.norm(v_rel) * v_rel * 1e3


def derivatives(t, y, model: ForceModel):
    """dy/dt for the chosen force model."""
    r, v = y[:3], y[3:]
    a = accel_two_body(r)
    if model.j2:
        a = a + accel_j2(r)
    if model.drag:
        a = a + accel_drag(r, v, model.ballistic)
    return np.concatenate([v, a])


def specific_energy(y):
    r, v = y[:3], y[3:]
    return 0.5 * np.dot(v, v) - MU_EARTH / np.linalg.norm(r)
