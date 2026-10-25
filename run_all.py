"""Run every scenario, produce the plots in results/ and a verification & validation summary.

Usage:  python run_all.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orbitsim import (HUNTSVILLE, J2, MU_EARTH, OMEGA_EARTH, R_EARTH, ForceModel, OrbitalElements,
                      atmospheric_density, coe_to_rv, eci_to_ecef, ecef_to_latlon, find_passes, hohmann,
                      kepler_propagate, look_angles, propagate, rk4, rv_to_coe, simulate_hohmann,
                      specific_energy)

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
DEG = np.pi / 180
EPOCH = datetime(2026, 10, 23, 0, 0, tzinfo=timezone.utc)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
summary = {}


def earth_sphere(ax):
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(R_EARTH * np.cos(u) * np.sin(v), R_EARTH * np.sin(u) * np.sin(v),
                    R_EARTH * np.cos(v), color="#9db4c0", alpha=0.5, linewidth=0)


# 1. Three representative orbits in 3D ------------------------------------------------------------
orbits = {
    "LEO (ISS-like, 420 km, 51.6 deg)": OrbitalElements(R_EARTH + 420, 0.0005, 51.6 * DEG, 0, 0, 0),
    "Molniya (e = 0.74, 63.4 deg)": OrbitalElements(26600, 0.74, 63.4 * DEG, 40 * DEG, 270 * DEG, 0),
    "GEO (35,786 km)": OrbitalElements(42164, 0.0, 0.0, 0, 0, 0),
}
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection="3d")
earth_sphere(ax)
for name, el in orbits.items():
    _, ys = propagate(np.concatenate(coe_to_rv(el)), el.period, n_out=600)
    ax.plot(*ys[:, :3].T, label=name, lw=1.2)
lim = 45000
ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim), xlabel="x [km]", ylabel="y [km]",
       zlabel="z [km]", title="Representative Earth orbits (ECI frame)")
ax.set_box_aspect((1, 1, 1))
ax.legend(loc="upper left", fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "01_orbits_3d.png")
plt.close(fig)

# 2. Energy conservation: hand-written RK4 vs adaptive DOP853 -------------------------------------
el = OrbitalElements(7000, 0.05, 30 * DEG, 0, 0, 0)
y0 = np.concatenate(coe_to_rv(el))
t_rk, y_rk = rk4(y0, 10 * el.period, 30.0)
t_dp, y_dp = propagate(y0, 10 * el.period)
E0 = specific_energy(y0)
err_rk = np.abs([specific_energy(y) / E0 - 1 for y in y_rk])
err_dp = np.abs([specific_energy(y) / E0 - 1 for y in y_dp])
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.semilogy(t_rk / el.period, err_rk + 1e-17, label="RK4, fixed 30 s step")
ax.semilogy(t_dp / el.period, err_dp + 1e-17, label="DOP853, adaptive (rtol 1e-11)")
ax.set(xlabel="Orbits", ylabel="|E/E0 - 1|", title="Specific orbital energy conservation (two-body)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "02_energy_conservation.png")
plt.close(fig)
summary["energy_rel_error_rk4_30s"] = float(err_rk.max())
summary["energy_rel_error_dop853"] = float(err_dp.max())

# 3. RK4 convergence order ----------------------------------------------------------------------
el_c = OrbitalElements(7000, 0.01, 0, 0, 0, 0)
y0c = np.concatenate(coe_to_rv(el_c))
r_true, _ = coe_to_rv(kepler_propagate(el_c, el_c.period))
dts, errs = [], []
for steps in (40, 80, 160, 320, 640):
    dt = el_c.period / steps
    _, ys = rk4(y0c, el_c.period, dt)
    dts.append(dt)
    errs.append(np.linalg.norm(ys[-1, :3] - r_true))
order = np.polyfit(np.log(dts), np.log(errs), 1)[0]
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.loglog(dts, errs, "o-", label=f"RK4 position error after 1 orbit (slope {order:.2f})")
ax.loglog(dts, errs[0] * (np.array(dts) / dts[0]) ** 4, "--", label="Ideal 4th order")
ax.set(xlabel="Step size dt [s]", ylabel="Error vs Kepler solution [km]", title="RK4 convergence study")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "03_rk4_convergence.png")
plt.close(fig)
summary["rk4_observed_order"] = round(float(order), 3)

# 4. J2 nodal regression -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.6))
j2_rows = []
for name, el in {"Sun-synchronous, 800 km, i = 98.6 deg": OrbitalElements(R_EARTH + 800, 0.001, 98.6 * DEG, 0, 0, 0),
                 "ISS-like, 420 km, i = 51.6 deg": OrbitalElements(R_EARTH + 420, 0.0005, 51.6 * DEG, 0, 0, 0)}.items():
    days = 5
    t, ys = propagate(np.concatenate(coe_to_rv(el)), days * 86400, ForceModel(j2=True), n_out=3000,
                      rtol=1e-10, atol=1e-10)
    raan = np.degrees(np.unwrap([rv_to_coe(y[:3], y[3:]).raan for y in ys]))
    n = np.sqrt(MU_EARTH / el.a**3)
    p = el.a * (1 - el.e**2)
    theory = np.degrees(-1.5 * n * J2 * (R_EARTH / p) ** 2 * np.cos(el.i)) * 86400
    sim = np.polyfit(t / 86400, raan, 1)[0]
    j2_rows.append({"orbit": name, "theory_deg_per_day": round(theory, 4), "simulated_deg_per_day": round(sim, 4),
                    "percent_difference": round(100 * (sim - theory) / theory, 3)})
    ax.plot(t / 86400, raan - raan[0], label=f"{name} (sim)")
    ax.plot(t / 86400, theory * t / 86400, "k--", lw=0.8)
ax.set(xlabel="Days", ylabel="Change in RAAN [deg]", title="J2 nodal regression: simulation vs theory (dashed)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "04_j2_nodal_regression.png")
plt.close(fig)
summary["j2_nodal_regression"] = j2_rows

# 5. Hohmann transfer LEO -> GEO -----------------------------------------------------------------
r1, r2 = R_EARTH + 300, 42164.0
plan, legs = simulate_hohmann(r1, r2)
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.add_patch(plt.Circle((0, 0), R_EARTH, color="#9db4c0"))
for (name, (t, ys)), style in zip(legs.items(), ["-", "--", "-"]):
    ax.plot(ys[:, 0], ys[:, 1], style, label=name.capitalize())
ax.plot(*legs["transfer"][1][0, :2], "r^", label=f"Burn 1: {plan.dv1:.3f} km/s")
ax.plot(*legs["transfer"][1][-1, :2], "rv", label=f"Burn 2: {plan.dv2:.3f} km/s")
ax.set(aspect="equal", xlabel="x [km]", ylabel="y [km]",
       title=f"Hohmann transfer 300 km LEO -> GEO\nTotal dv = {plan.dv_total:.3f} km/s, time of flight = {plan.tof/3600:.2f} h")
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "05_hohmann_transfer.png")
plt.close(fig)
r_final = np.linalg.norm(legs["final"][1][:, :3], axis=1)
summary["hohmann"] = {"r1_km": r1, "r2_km": r2, "dv1_km_s": round(plan.dv1, 4), "dv2_km_s": round(plan.dv2, 4),
                      "dv_total_km_s": round(plan.dv_total, 4), "tof_hours": round(plan.tof / 3600, 3),
                      "final_orbit_radius_error_km": round(float(np.max(np.abs(r_final - r2))), 4)}

# 6. Atmospheric drag: orbit decay to re-entry ----------------------------------------------------
model = ForceModel(drag=True, j2=True, cd=2.2, area_m2=0.01, mass_kg=1.33)   # 1U CubeSat (10 cm x 10 cm, 1.33 kg)
r0 = R_EARTH + 350
el_d = OrbitalElements(r0, 0.0005, 51.6 * DEG, 0, 0, 0)
t, ys = propagate(np.concatenate(coe_to_rv(el_d)), 400 * 86400, model, n_out=20000, rtol=1e-9, atol=1e-9,
                  stop_alt_km=120)
alt = np.linalg.norm(ys[:, :3], axis=1) - R_EARTH
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(t / 86400, alt, lw=0.6)
ax.set(xlabel="Days after deployment", ylabel="Altitude [km]",
       title=f"1U CubeSat orbit decay from 350 km (Cd 2.2, A/m = 0.0075 m^2/kg)\nRe-entry (120 km) after {t[-1]/86400:.0f} days")
fig.tight_layout()
fig.savefig(OUT / "06_drag_decay_cubesat.png")
plt.close(fig)
summary["cubesat_decay_days_350km"] = round(float(t[-1] / 86400), 1)

# 7. Ground track and Huntsville passes ----------------------------------------------------------
el_g = OrbitalElements(R_EARTH + 420, 0.0005, 51.6 * DEG, 0, 0, 0)
t, ys = propagate(np.concatenate(coe_to_rv(el_g)), 86400, ForceModel(j2=True), n_out=17280)
r_ecef = eci_to_ecef(ys[:, :3], t, EPOCH)
lat, lon = ecef_to_latlon(r_ecef)
elev, az, rng = look_angles(r_ecef, HUNTSVILLE)
passes = find_passes(t, elev, HUNTSVILLE, EPOCH)
fig, ax = plt.subplots(figsize=(8, 4))
first = t <= 3 * el_g.period
lon_plot = np.where(np.abs(np.diff(lon, prepend=lon[0])) > 180, np.nan, lon)
ax.plot(lon_plot[first], lat[first], lw=0.9, label="Ground track, first 3 orbits")
vis = elev >= HUNTSVILLE.min_elev_deg
ax.plot(lon[vis], lat[vis], ".", ms=2, color="red", label="Visible from Huntsville (> 10 deg elev), 24 h")
ax.plot(HUNTSVILLE.lon_deg, HUNTSVILLE.lat_deg, "k*", ms=10, label="Huntsville, AL")
ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude [deg]", ylabel="Latitude [deg]",
       title="ISS-like orbit ground track and Huntsville ground-station visibility (epoch 2026-10-23 00:00 UTC)")
ax.set_xticks(range(-180, 181, 30))
ax.set_yticks(range(-90, 91, 30))
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "07_ground_track.png")
plt.close(fig)
summary["huntsville_passes_24h"] = passes

(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
