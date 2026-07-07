"""
Spearhead VTOL — vehicle dynamics.

Implements the VehicleDynamics interface expected by sim.runner.SimRunner:
  .state_dim, .control_dim, .state_names, .control_names
  .initial_state() -> np.ndarray
  .get_position(X) -> np.ndarray(3)   NED position (north, east, down)
  .apply_constraints(X) -> np.ndarray (quat normalise at X[9:13] + ground clamp)
  .envelope_violations(X) -> list[str] (aero DB validity: |alpha|,|beta| <= 30 deg)
  .terminal_condition(t, X) -> str|None ('crash' / 'departure')
  .derivatives(t, X, U) -> np.ndarray
  .describe() -> dict
"""
import os
import numpy as np

from sim.quaternion import (
    normalize_quaternion, quat_to_euler, quat_kinematics,
    rotate_inertial_to_body, rotate_body_to_inertial,
)


# ---------------------------------------------------------------------------
# ADB loading helpers
# ---------------------------------------------------------------------------

def _load_adb(filename):
    raw = np.loadtxt(filename)
    rows, cols = raw.shape
    if cols != 6 or rows % 9 != 0:
        raise ValueError(f'{filename}: expected multiple-of-9 rows, 6 cols.')
    nBeta = rows // 9
    adb   = raw.reshape((nBeta, 9, 6))
    beta_breaks = np.array([
        -180, -175, -170, -165, -160, -150, -140, -130, -120, -110, -100,
         -90,  -80,  -70,  -60,  -50,  -40,  -30,  -25,  -21,  -18,  -15,
         -12,  -10,   -8,   -6,   -4,   -2,    0,    2,    4,    6,    8,
          10,   12,   15,   18,   21,   25,   30,   40,   50,   60,   70,
          80,   90,  100,  110,  120,  130,  140,  150,  160,  170,  175,  180,
    ], dtype=float)
    if len(beta_breaks) != nBeta:
        raise ValueError(
            f'Beta breakpoint count ({len(beta_breaks)}) != ADB groups ({nBeta}).')
    return adb, beta_breaks


def _load_poly(filename):
    arr = np.loadtxt(filename)
    if arr.shape != (9, 6):
        raise ValueError(f'{filename}: expected 9×6.')
    return arr


def _aero_coeffs(alpha_deg, beta_deg, adb, beta_breaks):
    alpha_deg = np.clip(alpha_deg, -30.0, 30.0)
    beta_deg  = np.clip(beta_deg,  -30.0, 30.0)
    alpha_rad = np.radians(alpha_deg)
    apoly     = np.array([alpha_rad**p for p in range(9)])
    # Linearly interpolate the aero table across the two bracketing beta
    # breakpoints (was nearest-neighbour, which stepped at each breakpoint).
    j  = int(np.clip(np.searchsorted(beta_breaks, beta_deg),
                     1, len(beta_breaks) - 1))
    b0, b1 = beta_breaks[j - 1], beta_breaks[j]
    w  = 0.0 if b1 == b0 else (beta_deg - b0) / (b1 - b0)
    c0 = apoly @ adb[j - 1, :, :]
    c1 = apoly @ adb[j, :, :]
    return (1.0 - w) * c0 + w * c1


def _surface_coeffs(angle_deg, poly_9x6):
    apoly = np.array([angle_deg**p for p in range(9)])
    return apoly @ poly_9x6


# ---------------------------------------------------------------------------
# SpearheadDynamics
# ---------------------------------------------------------------------------

class SpearheadDynamics:
    """
    21-state quaternion dynamics for the Spearhead VTOL.

    State vector X (21):
      [0:3]  body velocities  (u, v, w)  m/s
      [3:6]  body rates       (p, q, r)  rad/s
      [6:9]  NED position     (x, y, z)  m
      [9:13] quaternion       (q0,q1,q2,q3)
      [13:18] rotor speeds    (w1..w5)   rad/s
      [18:21] servo angles    (dl,dr,drd) deg  (deg cmd, unity actuator DC gain)

    Control vector U (8):
      [0:4] vertical motor commands  (m1..m4)  0–1000
      [4]   pusher motor command     (m5)       0–1000
      [5:8] surface commands         (le,re,rud) deg
    """

    state_dim   = 21
    control_dim = 8

    state_names = [
        'u', 'v', 'w', 'p', 'q', 'r',
        'x', 'y', 'z',
        'q0', 'q1', 'q2', 'q3',
        'w1', 'w2', 'w3', 'w4', 'w5',
        'dl', 'dr', 'drd',
    ]
    control_names = ['m1', 'm2', 'm3', 'm4', 'm5', 'servo_le', 'servo_re', 'servo_rud']

    # Vehicle constants (used by both dynamics and controller)
    params = {
        'M':    20.0,
        'g':    9.81,
        'rho':  1.225,
        'Ixx':  8.734,
        'Iyy':  5.592,
        'Izz':  13.623,
        'L1':   1.65,    # lateral moment arm [m]
        'L2':   1.425,   # longitudinal moment arm [m]
        'C':    0.32,    # wing chord [m]
        'S':    2.24,    # wing reference area [m²]
        'K':    [1.123e-6, 2.25e-6, 7.708e-7, 18.708e-7],
        'Ku':   [44.2205 * 8.18, 44.2205 * 7.02, 20.0],
        'Mtau': [1/44.22, 1/44.22, 1/20],
        'W_HOVER': 808.0,   # hover equilibrium cmd per vertical motor
        'CFZ0': 0.04888,    # wing lift coeff at alpha=beta=0
        'CFX0': 0.00374,    # wing forward force coeff at alpha=beta=0
        'CMY0': 0.03716,    # pitching moment coeff at alpha=beta=0
    }

    def __init__(self, data_dir=None):
        if data_dir is None:
            # Default: ADB files live alongside this file in vehicles/spearhead/
            data_dir = os.path.dirname(os.path.abspath(__file__))
        self._adb, self._beta_breaks = _load_adb(
            os.path.join(data_dir, 'adb_w_hat.txt'))
        self._le     = _load_poly(os.path.join(data_dir, 'le_w_hat.txt'))
        self._re     = _load_poly(os.path.join(data_dir, 're_w_hat.txt'))
        self._rudder = _load_poly(os.path.join(data_dir, 'rudder_w_hat.txt'))

    def initial_state(self):
        X = np.zeros(self.state_dim)
        X[9] = 1.0   # quaternion identity (level, zero heading)
        return X

    def get_position(self, X):
        """NED position (north, east, down) [m]."""
        return np.asarray(X[6:9], dtype=float)

    def apply_constraints(self, X):
        """Quaternion normalisation + ground clamp (z_NED >= 0 = on/below ground)."""
        Xn = X.copy()
        Xn[9:13] = normalize_quaternion(Xn[9:13])
        if Xn[8] > 0.0:          # NED down-position at/below ground datum
            Xn[8] = 0.0
            if Xn[2] > 0.0:      # body-frame w (downward component)
                Xn[2] = 0.0
        return Xn

    # Aero DB is a 9th-order alpha polynomial fitted over ±30 deg (and clipped
    # there in _aero_coeffs) — anything integrated past that edge is fiction.
    ENV_ALPHA_MAX_DEG = 30.0
    ENV_BETA_MAX_DEG  = 30.0
    ENV_MIN_SPEED     = 3.0    # m/s — below this, aero forces are negligible
                               # and alpha/beta are numerically meaningless

    def envelope_violations(self, X):
        """Non-empty list of reasons when outside the validated aero envelope."""
        u, v, w = X[0:3]
        V = np.sqrt(u*u + v*v + w*w)
        if V < self.ENV_MIN_SPEED:
            return []
        alpha = np.degrees(np.arctan2(w, u))
        beta  = np.degrees(np.arcsin(np.clip(v / V, -1.0, 1.0)))
        out = []
        if abs(alpha) > self.ENV_ALPHA_MAX_DEG:
            out.append(f'alpha = {alpha:+.1f} deg outside +/-'
                       f'{self.ENV_ALPHA_MAX_DEG:.0f} deg (V = {V:.1f} m/s)')
        if abs(beta) > self.ENV_BETA_MAX_DEG:
            out.append(f'beta  = {beta:+.1f} deg outside +/-'
                       f'{self.ENV_BETA_MAX_DEG:.0f} deg (V = {V:.1f} m/s)')
        return out

    def terminal_condition(self, t, X):
        """'crash' / 'departure' / None. Only armed once the vehicle has flown."""
        alt = -X[8]
        if alt > 1.0:
            self._was_airborne = True
        if not getattr(self, '_was_airborne', False):
            return None
        quat = normalize_quaternion(X[9:13])
        # Crash: ground contact with a hard sink rate (inertial vz NED-down +)
        if alt <= 0.05:
            vz = rotate_body_to_inertial(X[0:3], quat)[2]
            if vz > 2.0:
                return 'crash'
        # Departure: rolled or pitched past 85 deg while airborne
        phi, theta, _ = quat_to_euler(quat)
        if abs(phi) > np.radians(85.0) or abs(theta) > np.radians(85.0):
            return 'departure'
        return None

    def describe(self):
        p = self.params
        return {
            'mass (kg)':         p['M'],
            'Ixx Iyy Izz (kg·m²)': f"{p['Ixx']}  {p['Iyy']}  {p['Izz']}",
            'L1 L2 (m)':         f"{p['L1']}  {p['L2']}",
            'S C (m² m)':        f"{p['S']}  {p['C']}",
            'K (thrust coeffs)': p['K'],
            'Ku (cmd→speed)':    p['Ku'],
            'W_HOVER (cmd)':     p['W_HOVER'],
        }

    def derivatives(self, t, X, U):
        p = self.params
        M, g, rho  = p['M'], p['g'], p['rho']
        Ixx, Iyy, Izz = p['Ixx'], p['Iyy'], p['Izz']
        L1, L2, C, S  = p['L1'], p['L2'], p['C'], p['S']
        K, Ku, Mtau   = p['K'], p['Ku'], p['Mtau']

        u, v, w          = X[0:3]
        p_r, q_om, r_r   = X[3:6]
        q0, q1, q2, q3   = X[9:13]
        w1, w2, w3, w4, w5 = X[13:18]
        dl, dr, drd      = X[18:21]
        quat = np.array([q0, q1, q2, q3])

        # Motor forces / torques
        F1 = K[0] * w1**2;  F2 = K[0] * w2**2
        F3 = K[0] * w3**2;  F4 = K[0] * w4**2
        F5 = K[1] * w5**2
        Tau1 =  K[2] * w1**2;  Tau2 = -K[2] * w2**2
        Tau3 = -K[2] * w3**2;  Tau4 =  K[2] * w4**2
        Tau5 =  K[3] * w5**2
        Fn   = F1 + F2 + F3 + F4
        Taun = Tau1 + Tau2 + Tau3 + Tau4

        # Aerodynamics
        V = np.sqrt(u*u + v*v + w*w)
        alpha_deg = np.degrees(np.arctan2(w, u)) if V > 1e-6 else 0.0
        beta_deg  = np.degrees(np.arcsin(v / V)) if V > 1e-6 else 0.0

        adb_C = _aero_coeffs(alpha_deg, beta_deg, self._adb, self._beta_breaks)
        le_C  = _surface_coeffs(dl,  self._le)
        re_C  = _surface_coeffs(dr,  self._re)
        rd_C  = _surface_coeffs(drd, self._rudder)

        CFX = adb_C[0] + le_C[0] + re_C[0] + rd_C[0]
        CFY = adb_C[1] + le_C[1] + re_C[1] + rd_C[1]
        CFZ = adb_C[2] + le_C[2] + re_C[2] + rd_C[2]
        CMX = adb_C[3] + le_C[3] + re_C[3] + rd_C[3]
        CMY = adb_C[4] + le_C[4] + re_C[4] + rd_C[4]
        CMZ = adb_C[5] + le_C[5] + re_C[5] + rd_C[5]

        qS  = 0.5 * rho * V**2
        FAx = qS * S * CFX;  FAy = qS * S * CFY;  FAz = qS * S * CFZ
        LA  = qS * S * C * CMX
        MA  = qS * S * C * CMY
        NA  = qS * S * C * CMZ

        # Thrust forces (pusher = +x body, vertical rotors = -z body)
        FTx, FTy, FTz = F5, 0.0, -Fn

        # Gravity in body frame
        FGx, FGy, FGz = rotate_inertial_to_body(np.array([0.0, 0.0, M * g]), quat)

        # Structural moments
        LT = L1 * ((F1 + F3) - (F2 + F4)) + Tau5
        MT = L2 * ((F1 + F2) - (F3 + F4))
        NT = Taun

        # Translational acceleration (body frame)
        dudt = (FGx + FAx + FTx) / M - q_om * w  + r_r * v
        dvdt = (FGy + FAy + FTy) / M - r_r  * u  + p_r * w
        dwdt = (FGz + FAz + FTz) / M - p_r  * v  + q_om * u

        # Rotational acceleration
        dpdt = (LA + LT - q_om * r_r  * (Izz - Iyy)) / Ixx
        dqdt = (MA + MT - p_r  * r_r  * (Ixx - Izz)) / Iyy
        drdt = (NA + NT - p_r  * q_om * (Iyy - Ixx)) / Izz

        # Position kinematics (inertial frame)
        dxdt, dydt, dzdt = rotate_body_to_inertial(np.array([u, v, w]), quat)

        # Quaternion kinematics
        qdot = quat_kinematics(quat, p_r, q_om, r_r)

        # Actuator dynamics (first-order lag: dw/dt = -(1/tau)*w + Ku*U)
        # Mtau stores time constants [tau_vert, tau_pusher, tau_servo];
        # 1/Mtau[i] gives the bandwidth (44.22, 44.22, 20 rad/s).
        dw1dt  = -(1.0/Mtau[0]) * w1  + Ku[0] * U[0]
        dw2dt  = -(1.0/Mtau[0]) * w2  + Ku[0] * U[1]
        dw3dt  = -(1.0/Mtau[0]) * w3  + Ku[0] * U[2]
        dw4dt  = -(1.0/Mtau[0]) * w4  + Ku[0] * U[3]
        dw5dt  = -(1.0/Mtau[1]) * w5  + Ku[1] * U[4]
        dl_dt  = -(1.0/Mtau[2]) * dl  + Ku[2] * U[5]
        dr_dt  = -(1.0/Mtau[2]) * dr  + Ku[2] * U[6]
        drd_dt = -(1.0/Mtau[2]) * drd + Ku[2] * U[7]

        dX = np.zeros(self.state_dim)
        dX[0:6]  = [dudt, dvdt, dwdt, dpdt, dqdt, drdt]
        dX[6:9]  = [dxdt, dydt, dzdt]
        dX[9:13] = qdot
        dX[13:18] = [dw1dt, dw2dt, dw3dt, dw4dt, dw5dt]
        dX[18:21] = [dl_dt, dr_dt, drd_dt]
        return dX
