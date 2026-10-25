"""Numerical propagators: a hand-written fixed-step RK4 and SciPy's adaptive DOP853."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .constants import R_EARTH
from .dynamics import ForceModel, derivatives


def rk4(y0, t_end, dt, model: ForceModel = ForceModel()):
    """Classic 4th-order Runge-Kutta with a fixed step. Returns times (N,) and states (N, 6)."""
    n = int(np.ceil(t_end / dt))
    t = np.linspace(0.0, n * dt, n + 1)
    ys = np.empty((n + 1, 6))
    ys[0] = y0
    y = np.array(y0, float)
    for k in range(n):
        tk = t[k]
        k1 = derivatives(tk, y, model)
        k2 = derivatives(tk + dt / 2, y + dt / 2 * k1, model)
        k3 = derivatives(tk + dt / 2, y + dt / 2 * k2, model)
        k4 = derivatives(tk + dt, y + dt * k3, model)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys[k + 1] = y
    return t, ys


def propagate(y0, t_end, model: ForceModel = ForceModel(), n_out=2000, rtol=1e-11, atol=1e-12,
              stop_alt_km=None):
    """High-accuracy adaptive propagation (Dormand-Prince 8(5,3)). Optionally stop at re-entry altitude."""
    events = None
    if stop_alt_km is not None:
        def reentry(t, y, *_):
            return np.linalg.norm(y[:3]) - (R_EARTH + stop_alt_km)
        reentry.terminal = True
        reentry.direction = -1
        events = reentry
    t_eval = np.linspace(0, t_end, n_out)
    sol = solve_ivp(derivatives, (0, t_end), y0, method="DOP853", t_eval=t_eval,
                    rtol=rtol, atol=atol, args=(model,), events=events)
    return sol.t, sol.y.T
