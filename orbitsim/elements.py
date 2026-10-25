"""Conversions between Cartesian state vectors and classical orbital elements.

Classical orbital elements (COEs):
    a     semi-major axis [km]
    e     eccentricity [-]
    i     inclination [rad]
    raan  right ascension of the ascending node, Omega [rad]
    argp  argument of periapsis, omega [rad]
    nu    true anomaly [rad]

References: Curtis, "Orbital Mechanics for Engineering Students", Ch. 4;
Vallado, "Fundamentals of Astrodynamics and Applications", Algs. 9-10.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import MU_EARTH


@dataclass
class OrbitalElements:
    a: float
    e: float
    i: float
    raan: float
    argp: float
    nu: float

    @property
    def period(self) -> float:
        """Orbital period [s] from Kepler's third law (elliptical orbits)."""
        return 2.0 * np.pi * np.sqrt(self.a**3 / MU_EARTH)


def _rot1(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def _rot3(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def coe_to_rv(el: OrbitalElements, mu: float = MU_EARTH):
    """Classical elements -> position r [km] and velocity v [km/s] in the inertial (ECI) frame."""
    p = el.a * (1.0 - el.e**2)                      # semi-latus rectum
    r_pf = p / (1 + el.e * np.cos(el.nu)) * np.array([np.cos(el.nu), np.sin(el.nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(el.nu), el.e + np.cos(el.nu), 0.0])
    # Perifocal -> ECI: R = R3(-raan) R1(-i) R3(-argp)
    q = _rot3(-el.raan) @ _rot1(-el.i) @ _rot3(-el.argp)
    return q @ r_pf, q @ v_pf


def rv_to_coe(r, v, mu: float = MU_EARTH) -> OrbitalElements:
    """Position/velocity in ECI -> classical orbital elements."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    n = np.cross([0.0, 0.0, 1.0], h)                 # node vector
    nn = np.linalg.norm(n)
    e_vec = ((vn**2 - mu / rn) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)
    energy = vn**2 / 2 - mu / rn
    a = -mu / (2 * energy)
    i = np.arccos(np.clip(h[2] / hn, -1, 1))

    eps = 1e-11
    raan = np.arccos(np.clip(n[0] / nn, -1, 1)) if nn > eps else 0.0
    if nn > eps and n[1] < 0:
        raan = 2 * np.pi - raan

    if e > eps and nn > eps:
        argp = np.arccos(np.clip(np.dot(n, e_vec) / (nn * e), -1, 1))
        if e_vec[2] < 0:
            argp = 2 * np.pi - argp
    else:
        argp = 0.0

    if e > eps:
        nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * rn), -1, 1))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu
    else:  # circular: use argument of latitude measured from the node
        ref = n / nn if nn > eps else np.array([1.0, 0.0, 0.0])
        nu = np.arccos(np.clip(np.dot(ref, r) / rn, -1, 1))
        if r[2] < 0 or (nn <= eps and r[1] < 0):
            nu = 2 * np.pi - nu
    return OrbitalElements(a, e, i, raan, argp, nu)


def kepler_propagate(el: OrbitalElements, dt: float, mu: float = MU_EARTH) -> OrbitalElements:
    """Analytic two-body propagation by solving Kepler's equation M = E - e sin E (elliptical)."""
    e = el.e
    n = np.sqrt(mu / el.a**3)
    E0 = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(el.nu / 2), np.sqrt(1 + e) * np.cos(el.nu / 2))
    M = E0 - e * np.sin(E0) + n * dt
    E = M if e < 0.8 else np.pi
    for _ in range(50):                               # Newton-Raphson
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if abs(dE) < 1e-14:
            break
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return OrbitalElements(el.a, e, el.i, el.raan, el.argp, nu % (2 * np.pi))
