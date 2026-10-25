"""Earth rotation, ground tracks and ground-station visibility (passes)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from .constants import FLATTENING, OMEGA_EARTH, R_EARTH


def julian_date(dt: datetime) -> float:
    dt = dt.astimezone(timezone.utc)
    y, m = dt.year, dt.month
    d = dt.day + (dt.hour + (dt.minute + (dt.second + dt.microsecond / 1e6) / 60) / 60) / 24
    if m <= 2:
        y, m = y - 1, m + 12
    A = y // 100
    B = 2 - A + A // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5


def gmst(dt: datetime) -> float:
    """Greenwich Mean Sidereal Time [rad] (IAU-82 expression, Vallado Eq. 3-47)."""
    T = (julian_date(dt) - 2451545.0) / 36525.0
    sec = (67310.54841 + (876600 * 3600 + 8640184.812866) * T + 0.093104 * T**2 - 6.2e-6 * T**3)
    return np.deg2rad((sec % 86400) / 240.0) % (2 * np.pi)


def eci_to_ecef(r_eci, t, epoch: datetime):
    """Rotate inertial positions (N,3) at times t [s] after epoch into the Earth-fixed frame."""
    theta = gmst(epoch) + OMEGA_EARTH * np.asarray(t)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = r_eci[:, 0], r_eci[:, 1], r_eci[:, 2]
    return np.column_stack([c * x + s * y, -s * x + c * y, z])


def ecef_to_latlon(r_ecef):
    """Geocentric latitude and longitude [deg] (adequate for plotting a ground track)."""
    x, y, z = r_ecef[:, 0], r_ecef[:, 1], r_ecef[:, 2]
    lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


@dataclass
class GroundStation:
    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0
    min_elev_deg: float = 10.0

    def ecef(self):
        """Station position on the WGS-84 ellipsoid [km]."""
        lat, lon = np.radians(self.lat_deg), np.radians(self.lon_deg)
        e2 = FLATTENING * (2 - FLATTENING)
        N = R_EARTH / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        return np.array([(N + self.alt_km) * np.cos(lat) * np.cos(lon),
                         (N + self.alt_km) * np.cos(lat) * np.sin(lon),
                         (N * (1 - e2) + self.alt_km) * np.sin(lat)])


HUNTSVILLE = GroundStation("Huntsville, AL", 34.7304, -86.5861, 0.19)


def look_angles(r_ecef, station: GroundStation):
    """Elevation, azimuth [deg] and range [km] from the station to each satellite position (ENU frame)."""
    lat, lon = np.radians(station.lat_deg), np.radians(station.lon_deg)
    rho = r_ecef - station.ecef()
    east = np.array([-np.sin(lon), np.cos(lon), 0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    e, n, u = rho @ east, rho @ north, rho @ up
    rng = np.linalg.norm(rho, axis=1)
    elev = np.degrees(np.arcsin(u / rng))
    az = np.degrees(np.arctan2(e, n)) % 360
    return elev, az, rng


def find_passes(t, elev, station: GroundStation, epoch: datetime):
    """List passes above the station's minimum elevation: rise, set, peak elevation and duration."""
    above = elev >= station.min_elev_deg
    passes = []
    k = 0
    while k < len(t):
        if above[k]:
            start = k
            while k < len(t) and above[k]:
                k += 1
            end = k - 1

            def cross(i0, i1):  # linear interpolation of the threshold crossing time
                if i0 < 0 or i1 >= len(t):
                    return t[max(min(i0, len(t) - 1), 0)]
                f = (station.min_elev_deg - elev[i0]) / (elev[i1] - elev[i0])
                return t[i0] + f * (t[i1] - t[i0])

            t_rise, t_set = cross(start - 1, start), cross(end, end + 1)
            peak = start + int(np.argmax(elev[start:end + 1]))
            passes.append({
                "rise_utc": (epoch + timedelta(seconds=float(t_rise))).strftime("%Y-%m-%d %H:%M:%S"),
                "set_utc": (epoch + timedelta(seconds=float(t_set))).strftime("%Y-%m-%d %H:%M:%S"),
                "duration_min": round((t_set - t_rise) / 60, 2),
                "max_elev_deg": round(float(elev[peak]), 1),
            })
        k += 1
    return passes
