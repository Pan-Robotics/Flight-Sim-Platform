"""
X4 Quadcopter — vehicle dynamics for sim_platform.

Implements the VehicleDynamics interface:
  .state_dim, .control_dim, .state_names, .control_names
  .initial_state() -> np.ndarray (17,)
  .get_position(X) -> np.ndarray (3,)   NED position (north, east, down)
  .apply_constraints(X) -> np.ndarray   (normalises quat at X[6:10] + ground clamp)
  .terminal_condition(t, X) -> str|None ('crash' / 'departure')
  .derivatives(t, X, U) -> np.ndarray
  .describe() -> dict

State vector (17):
  [0]  x      NED North [m]
  [1]  xdot   NED North vel [m/s]
  [2]  y      NED East  [m]
  [3]  ydot   NED East  vel [m/s]
  [4]  z      NED Down  [m]  (z < 0 = above datum)
  [5]  zdot   NED Down  vel [m/s]
  [6]  qw     quaternion scalar (body→NED)
  [7]  qx
  [8]  qy
  [9]  qz
  [10] p      roll  rate [rad/s]
  [11] q_ang  pitch rate [rad/s]
  [12] r      yaw   rate [rad/s]
  [13] w1     motor 1 speed [rad/s]
  [14] w2     motor 2 speed [rad/s]
  [15] w3     motor 3 speed [rad/s]
  [16] w4     motor 4 speed [rad/s]

Control (4):  motor PWM commands 0–800 per motor
"""
import numpy as np


# --- Nominal vehicle parameters (from CAD / MATLAB control design) ---
# Per-instance values may be overridden via X4Dynamics(params={...}) for
# Monte Carlo dispersion; these module-level nominals feed the controller's
# equilibrium so the *controller* always uses the design-point plant.
NOMINAL_PARAMS = {
    'M':        0.857945,
    'g':        9.81,
    'L':        0.16319,
    'Ixx':      1.061e5 / (1000 * 10000),
    'Iyy':      1.061e5 / (1000 * 10000),
    'Izz':      2.011e5 / (1000 * 10000),
    'Ktau':     7.708e-10 * 2,
    'Kthrust':  1.812e-7,
    'Kthrust2': 0.0007326,
    'Mtau':     1.0 / 44.22,
    'Ku':       515.5,
    'Dxx':      0.01212,
    'Dyy':      0.01212,
    'Dzz':      0.0648,
}

_M        = NOMINAL_PARAMS['M']
_g        = NOMINAL_PARAMS['g']
_Kthrust  = NOMINAL_PARAMS['Kthrust']
_Kthrust2 = NOMINAL_PARAMS['Kthrust2']
_Mtau     = NOMINAL_PARAMS['Mtau']
_Ku       = NOMINAL_PARAMS['Ku']

# Nominal hover equilibrium motor speed and command (design point)
W_e   = ((-4*_Kthrust2) + np.sqrt((4*_Kthrust2)**2 + 4*4*_Kthrust*_M*_g)) / (2*4*_Kthrust)
U_e   = W_e / (_Ku * _Mtau)
U_e_eq = U_e   # alias used by candidates/x4_lqg.py


def _quat_deriv(qw, qx, qy, qz, p, q_ang, r):
    return 0.5 * np.array([
        -qx*p  - qy*q_ang - qz*r,
         qw*p  + qy*r     - qz*q_ang,
         qw*q_ang - qx*r  + qz*p,
         qw*r  + qx*q_ang - qy*p,
    ])


class X4Dynamics:
    state_dim   = 17
    control_dim = 4
    state_names = [
        'x', 'xdot', 'y', 'ydot', 'z', 'zdot',
        'qw', 'qx', 'qy', 'qz',
        'p', 'q_ang', 'r',
        'w1', 'w2', 'w3', 'w4',
    ]
    control_names = ['m1', 'm2', 'm3', 'm4']

    def __init__(self, params=None):
        """params: optional dict overriding NOMINAL_PARAMS entries
        (e.g. {'M': 0.94} for Monte Carlo dispersion)."""
        self.params = {**NOMINAL_PARAMS, **(params or {})}
        p = self.params
        # This instance's true hover equilibrium (differs from the nominal
        # W_e when the plant is perturbed — the controller keeps nominal)
        self.W_e = ((-4*p['Kthrust2'])
                    + np.sqrt((4*p['Kthrust2'])**2
                              + 4*4*p['Kthrust']*p['M']*p['g'])) \
                   / (2*4*p['Kthrust'])
        self._wind_ned = np.zeros(3)

    def set_wind_ned(self, w):
        """Steady/gust wind, NED [m/s]. Drag acts on air-relative velocity."""
        self._wind_ned = np.asarray(w, dtype=float)

    def initial_state(self, x0=0.0, y0=0.0, z0=0.0):
        X = np.zeros(17)
        X[0], X[2], X[4] = x0, y0, z0
        X[6]    = 1.0        # identity quaternion (level, north-heading)
        X[13:17] = self.W_e  # motors at this plant's hover equilibrium
        return X

    def get_position(self, X):
        """NED position (north, east, down) [m]. X4 interleaves pos/vel."""
        return np.array([X[0], X[2], X[4]])

    def terminal_condition(self, t, X):
        """'crash' / 'departure' / None. Only armed once the vehicle has flown."""
        alt = -X[4]
        if alt > 1.0:
            self._was_airborne = True
        if not getattr(self, '_was_airborne', False):
            return None
        # Crash: back at ground level with a hard sink rate (zdot NED-down +)
        if alt <= 0.05 and X[5] > 1.5:
            return 'crash'
        # Departure: tilted past 90 deg from vertical (cos(tilt) = 1-2(qx²+qy²))
        if 1.0 - 2.0 * (X[7]**2 + X[8]**2) < 0.0:
            return 'departure'
        return None

    def apply_constraints(self, X):
        """Quaternion normalisation + ground clamp (z_NED >= 0 = on/below ground)."""
        Xn = X.copy()
        n = np.linalg.norm(Xn[6:10])
        if n > 1e-10:
            Xn[6:10] /= n
        if Xn[4] > 0.0:
            Xn[4] = 0.0
            if Xn[5] > 0.0:
                Xn[5] = 0.0
        return Xn

    def derivatives(self, t, X, U):
        pr = self.params
        M, g, L          = pr['M'], pr['g'], pr['L']
        Ixx, Iyy, Izz    = pr['Ixx'], pr['Iyy'], pr['Izz']
        Ktau, Kth, Kth2  = pr['Ktau'], pr['Kthrust'], pr['Kthrust2']
        Mtau, Ku         = pr['Mtau'], pr['Ku']
        Dxx, Dyy, Dzz    = pr['Dxx'], pr['Dyy'], pr['Dzz']

        x, xd, y, yd, z, zd, qw, qx, qy, qz, p, q_ang, r, w1, w2, w3, w4 = X
        U = np.asarray(U).flatten()
        w = np.array([w1, w2, w3, w4])
        F  = Kth * w**2 + Kth2 * w
        Fn = F.sum()
        Tn = Ktau * (w1**2 - w2**2 - w3**2 + w4**2)

        # Drag acts on air-relative velocity (wind toggle-able; zero when off)
        if self._wind_ned.any():
            wxa = xd - self._wind_ned[0]
            wya = yd - self._wind_ned[1]
            wza = zd - self._wind_ned[2]
        else:
            wxa, wya, wza = xd, yd, zd

        dX = np.zeros(17)
        dX[0] = xd
        dX[2] = yd
        dX[4] = zd
        dX[6:10] = _quat_deriv(qw, qx, qy, qz, p, q_ang, r)

        dX[1] = -(Fn/M) * 2*(qx*qz + qw*qy) - Dxx/M * wxa
        dX[3] = -(Fn/M) * 2*(qy*qz - qw*qx)          - Dyy/M * wya
        dX[5] = g - (Fn/M) * (1 - 2*(qx**2 + qy**2)) - Dzz/M * wza

        dX[10] = (L/Ixx) * (F[0]+F[1]-F[2]-F[3]) - (Izz-Iyy)/Ixx * r * q_ang
        dX[11] = (L/Iyy) * (F[0]-F[1]+F[2]-F[3]) - (Izz-Ixx)/Iyy * p * r
        dX[12] = Tn/Izz                            - (Iyy-Ixx)/Izz * p * q_ang

        dX[13:17] = -(1.0/Mtau) * w + Ku * U
        return dX

    def describe(self):
        pr = self.params
        return {
            'model':      'X4 quadcopter (quaternion, NED z-down)',
            'mass_kg':    pr['M'],
            'arm_m':      pr['L'],
            'Ixx_kgm2':  pr['Ixx'],
            'Izz_kgm2':  pr['Izz'],
            'W_e_rad_s':  self.W_e,
            'U_e_cmd':    U_e,
        }
