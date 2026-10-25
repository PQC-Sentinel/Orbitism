# Orbitsim: Earth-Orbit Flight Dynamics Simulator

Author: Jason Jenkins (built with AI assistance)

Orbitsim numerically propagates satellite orbits around Earth under two-body gravity, the J2 oblateness perturbation and atmospheric drag. It also plans and flies Hohmann transfers, computes ground tracks, and predicts passes over a ground station in Huntsville, Alabama. Every model is verified against an analytic or textbook result.

An interactive 3D version of the simulator runs in the browser: see `web/orbitsim.html`.

## Quick start

```bash
pip install numpy scipy matplotlib pytest
python -m pytest -q          # 11 verification tests
python run_all.py            # runs every scenario, writes plots + summary.json to results/
```

## What it does

| Module | Capability |
|---|---|
| `orbitsim/elements.py` | State vector to/from classical orbital elements; Kepler's-equation analytic propagation |
| `orbitsim/dynamics.py` | Two-body, J2 and drag accelerations; piecewise exponential atmosphere (Vallado Table 8-4) |
| `orbitsim/propagate.py` | Hand-written fixed-step RK4 and adaptive Dormand-Prince 8(5,3) (SciPy) with a re-entry stop event |
| `orbitsim/maneuvers.py` | Hohmann transfer planning (delta-v, time of flight) and a numerically flown transfer with impulsive burns |
| `orbitsim/groundtrack.py` | GMST Earth rotation, ECI to ECEF, ground track, WGS-84 ground station, elevation/azimuth/range, pass finder |

## Results (from `python run_all.py`)

| Scenario | Result |
|---|---|
| Hohmann 300 km LEO to GEO | dv1 = 2.4257 km/s, dv2 = 1.4668 km/s, total 3.8926 km/s, time of flight 5.275 h |
| 1U CubeSat (1.33 kg, 0.01 m^2, Cd 2.2) from 350 km | Re-entry (120 km) after about 66 days, average solar activity |
| ISS-like orbit, 24 h from 2026-10-23 00:00 UTC | 4 Huntsville passes above 10 deg; best is 6.6 min at 54 deg max elevation |
| Sun-synchronous 800 km, i = 98.6 deg | RAAN drifts +0.99 deg/day, matching the Sun's apparent motion |

## Verification and validation

| Check | Reference | Simulated | Pass criterion |
|---|---|---|---|
| Energy conservation, 10 orbits (DOP853) | 0 | 3.3e-11 relative | < 1e-9 |
| Angular momentum conservation | 0 | < 1e-9 relative | < 1e-9 |
| Numerical vs Kepler's equation, 3 orbits, e = 0.3 | 0 km | < 0.001 km | < 1 m |
| RK4 convergence order | 4 | 4.21 | ratio 12-20 when dt halves |
| J2 nodal regression, SSO 800 km | 0.9853 deg/day | 0.9895 deg/day | within 2% |
| J2 nodal regression, 420 km, 51.6 deg | -4.951 deg/day | -4.972 deg/day | within 2% |
| Hohmann 300 km to GEO total dv | 3.893 km/s (textbook) | 3.8926 km/s | within 0.005 km/s |
| Hohmann final orbit circularity | r = 42,164 km | error < 0.0001 km | < 0.5 km |
| Drag decay rate, 300 km | analytic da/dt with co-rotating air | within 3% | within 3% |
| GMST | 152.578788 deg (Vallado Ex. 3-5) | 152.5788 deg | within 0.001 deg |
| State vector to elements | Curtis Ex. 4.3 | matches | within 0.1 deg |
| Overhead satellite look angle | 90 deg elevation | 90 deg | within 0.3 deg |

The browser simulator (JavaScript RK4) independently predicts the same four Huntsville pass times as the Python DOP853 reference, to the minute.

## Assumptions and limitations

- Gravity: point mass plus J2 only; no higher harmonics, Sun/Moon third-body gravity or solar radiation pressure.
- Atmosphere: static exponential model for average solar activity; real decay is faster near solar maximum.
- Burns are impulsive (instantaneous delta-v).
- Ground-track latitude is geocentric; the ground station uses the WGS-84 ellipsoid.
- The ISS-like orbit uses representative elements, not a live ISS ephemeris (TLE).

## Possible extensions

SGP4/TLE input for real satellites, Sun/Moon perturbations, a Jacchia or NRLMSISE-00 atmosphere, finite-burn maneuvers, Lambert's problem for rendezvous, and Monte Carlo dispersion of drag parameters.

## References

- H. Curtis, *Orbital Mechanics for Engineering Students*, 4th ed.
- D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.
