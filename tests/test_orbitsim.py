"""Verification tests: every model is checked against an analytic or textbook result."""
from datetime import datetime, timezone

import numpy as np
import pytest

from orbitsim import (HUNTSVILLE, MU_EARTH, R_EARTH, J2, ForceModel, OrbitalElements, coe_to_rv,
                      eci_to_ecef, find_passes, hohmann, kepler_propagate, look_angles, propagate,
                      rk4, rv_to_coe, simulate_hohmann, specific_energy)
from orbitsim.groundtrack import gmst

DEG = np.pi / 180


def test_element_round_trip():
    el = OrbitalElements(a=8000, e=0.15, i=40 * DEG, raan=75 * DEG, argp=30 * DEG, nu=110 * DEG)
    out = rv_to_coe(*coe_to_rv(el))
    for name in ("a", "e", "i", "raan", "argp", "nu"):
        assert getattr(out, name) == pytest.approx(getattr(el, name), rel=1e-10, abs=1e-10)


def test_textbook_state_vector():
    # Curtis Example 4.3: r = [-6045, -3490, 2500] km, v = [-3.457, 6.618, 2.533] km/s
    el = rv_to_coe([-6045, -3490, 2500], [-3.457, 6.618, 2.533])
    assert el.e == pytest.approx(0.1712, abs=1e-4)
    assert np.degrees(el.i) == pytest.approx(153.2, abs=0.1)
    assert np.degrees(el.raan) == pytest.approx(255.3, abs=0.1)
    assert np.degrees(el.argp) == pytest.approx(20.07, abs=0.05)
    assert np.degrees(el.nu) == pytest.approx(28.45, abs=0.05)


def test_energy_and_momentum_conserved_two_body():
    el = OrbitalElements(7000, 0.05, 30 * DEG, 0, 0, 0)
    y0 = np.concatenate(coe_to_rv(el))
    t, ys = propagate(y0, 10 * el.period)
    E = np.array([specific_energy(y) for y in ys])
    h = np.linalg.norm(np.cross(ys[:, :3], ys[:, 3:]), axis=1)
    assert np.max(np.abs(E / E[0] - 1)) < 1e-9
    assert np.max(np.abs(h / h[0] - 1)) < 1e-9


def test_numerical_matches_kepler_solution():
    el = OrbitalElements(12000, 0.3, 63.4 * DEG, 40 * DEG, 270 * DEG, 0)
    y0 = np.concatenate(coe_to_rv(el))
    t, ys = propagate(y0, 3 * el.period, n_out=50)
    for tk, yk in zip(t, ys):
        r_ana, _ = coe_to_rv(kepler_propagate(el, tk))
        assert np.linalg.norm(yk[:3] - r_ana) < 1e-3   # better than 1 metre


def test_rk4_is_fourth_order():
    el = OrbitalElements(7000, 0.01, 0, 0, 0, 0)
    y0 = np.concatenate(coe_to_rv(el))
    r_true, _ = coe_to_rv(kepler_propagate(el, el.period))
    errs = []
    for dt in (60.0, 30.0):
        steps = round(el.period / dt)
        _, ys = rk4(y0, el.period, el.period / steps)
        errs.append(np.linalg.norm(ys[-1, :3] - r_true))
    assert 12 < errs[0] / errs[1] < 20               # halving dt cuts error ~16x


def test_j2_nodal_regression_matches_theory():
    el = OrbitalElements(R_EARTH + 800, 0.001, 98.6 * DEG, 0, 0, 0)
    y0 = np.concatenate(coe_to_rv(el))
    days = 5
    t, ys = propagate(y0, days * 86400, ForceModel(j2=True), n_out=3000, rtol=1e-10, atol=1e-10)
    raan = np.unwrap([rv_to_coe(y[:3], y[3:]).raan for y in ys])
    sim_rate = np.polyfit(t, raan, 1)[0] * 86400 / DEG       # deg/day
    n = np.sqrt(MU_EARTH / el.a**3)
    p = el.a * (1 - el.e**2)
    theory = -1.5 * n * J2 * (R_EARTH / p) ** 2 * np.cos(el.i) * 86400 / DEG
    assert sim_rate == pytest.approx(theory, rel=0.02)
    assert theory == pytest.approx(0.9856, abs=0.03)         # sun-synchronous


def test_hohmann_leo_to_geo():
    plan = hohmann(R_EARTH + 300, 42164.0)
    assert plan.dv_total == pytest.approx(3.893, abs=0.005)   # Curtis Example 6.1 class result
    assert plan.tof / 3600 == pytest.approx(5.275, abs=0.005)
    _, legs = simulate_hohmann(R_EARTH + 300, 42164.0)
    r_final = np.linalg.norm(legs["final"][1][:, :3], axis=1)
    assert np.max(np.abs(r_final - 42164.0)) < 0.5            # circular within 0.5 km


def test_drag_decay_matches_analytic_rate():
    model = ForceModel(drag=True, cd=2.2, area_m2=1.0, mass_kg=100.0)
    r0 = R_EARTH + 300
    y0 = np.array([r0, 0, 0, 0, np.sqrt(MU_EARTH / r0), 0.0])
    days = 2
    t, ys = propagate(y0, days * 86400, model, n_out=4000, rtol=1e-10, atol=1e-10)
    a = np.array([rv_to_coe(y[:3], y[3:]).a for y in ys])
    sim_rate = np.polyfit(t, a, 1)[0]                          # km/s
    from orbitsim import atmospheric_density
    rho = atmospheric_density(np.mean(a) - R_EARTH)
    # Circular-orbit decay: da/dt = -rho * B * sqrt(mu * a)  (units handled with the 1e3 factor)
    # The equatorial atmosphere co-rotates with Earth, so the relative wind is v - omega*r:
    # drag scales with (1 - omega*r/v)^2 (about 0.88 here).
    from orbitsim import OMEGA_EARTH
    am = np.mean(a)
    corot = (1 - OMEGA_EARTH * am / np.sqrt(MU_EARTH / am)) ** 2
    theory = -rho * model.ballistic * np.sqrt(MU_EARTH * am) * 1e3 * corot
    assert sim_rate < 0
    assert sim_rate == pytest.approx(theory, rel=0.03)


def test_gmst_reference_value():
    # Vallado Example 3-5: 1992-08-20 12:14 UT1 -> GMST = 152.578788 deg
    g = np.degrees(gmst(datetime(1992, 8, 20, 12, 14, tzinfo=timezone.utc)))
    assert g == pytest.approx(152.578788, abs=1e-3)


def test_station_sees_overhead_satellite():
    st = HUNTSVILLE
    r = st.ecef() * (1 + 400 / R_EARTH)
    elev, _, rng = look_angles(r[None, :], st)
    assert elev[0] == pytest.approx(90, abs=0.3)
    assert rng[0] == pytest.approx(400, abs=25)


def test_passes_found_for_iss_like_orbit():
    epoch = datetime(2026, 10, 23, tzinfo=timezone.utc)
    el = OrbitalElements(R_EARTH + 420, 0.0005, 51.6 * DEG, 0, 0, 0)
    y0 = np.concatenate(coe_to_rv(el))
    t, ys = propagate(y0, 86400, ForceModel(j2=True), n_out=8640)
    elev, _, _ = look_angles(eci_to_ecef(ys[:, :3], t, epoch), HUNTSVILLE)
    passes = find_passes(t, elev, HUNTSVILLE, epoch)
    assert len(passes) >= 2
    assert all(0 < p["duration_min"] < 12 for p in passes)
