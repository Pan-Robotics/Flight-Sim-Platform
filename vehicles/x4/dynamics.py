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


# --- Vehicle parameters (from CAD / MATLAB control design) ---
_M       = 0.857945
_g       = 9.81
_L       = 0.16319
_Ixx     = 1.061e5 / (1000 * 10000)
_Iyy     = 1.061e5 / (1000 * 10000)
_Izz     = 2.011e5 / (1000 * 10000)
_Ktau    = 7.708e-10 * 2
_Kthrust  = 1.812e-7
_Kthrust2 = 0.0007326
_Mtau    = 1.0 / 44.22
_Ku      = 515.5
_Dxx     = 0.01212
_Dyy     = 0.01212
_Dzz     = 0.0648

# Hover equilibrium motor speed and command
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

    def initial_state(self, x0=0.0, y0=0.0, z0=0.0):
        X = np.zeros(17)
        X[0], X[2], X[4] = x0, y0, z0
        X[6]    = 1.0     # identity quaternion (level, north-heading)
        X[13:17] = W_e    # motors at hover equilibrium
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
        x, xd, y, yd, z, zd, qw, qx, qy, qz, p, q_ang, r, w1, w2, w3, w4 = X
        U = np.asarray(U).flatten()
        w = np.array([w1, w2, w3, w4])
        F  = _Kthrust * w**2 + _Kthrust2 * w
        Fn = F.sum()
        Tn = _Ktau * (w1**2 - w2**2 - w3**2 + w4**2)

        dX = np.zeros(17)
        dX[0] = xd
        dX[2] = yd
        dX[4] = zd
        dX[6:10] = _quat_deriv(qw, qx, qy, qz, p, q_ang, r)

        dX[1] = -(Fn/_M) * 2*(qx*qz + qw*qy) - _Dxx/_M * xd
        dX[3] = -(Fn/_M) * 2*(qy*qz - qw*qx)          - _Dyy/_M * yd
        dX[5] = _g - (Fn/_M) * (1 - 2*(qx**2 + qy**2)) - _Dzz/_M * zd

        dX[10] = (_L/_Ixx) * (F[0]+F[1]-F[2]-F[3]) - (_Izz-_Iyy)/_Ixx * r * q_ang
        dX[11] = (_L/_Iyy) * (F[0]-F[1]+F[2]-F[3]) - (_Izz-_Ixx)/_Iyy * p * r
        dX[12] = Tn/_Izz                             - (_Iyy-_Ixx)/_Izz * p * q_ang

        dX[13:17] = -(1.0/_Mtau) * w + _Ku * U
        return dX

    def describe(self):
        return {
            'model':      'X4 quadcopter (quaternion, NED z-down)',
            'mass_kg':    _M,
            'arm_m':      _L,
            'Ixx_kgm2':  _Ixx,
            'Izz_kgm2':  _Izz,
            'W_e_rad_s':  W_e,
            'U_e_cmd':    U_e,
        }
