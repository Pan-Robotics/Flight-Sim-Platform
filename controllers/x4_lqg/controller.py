"""
X4 LQR+I (full-state feedback) controller for sim_platform.

Implements the Controller interface:
  .step(t, X) -> (U: np.ndarray(4), info: dict)
  .reset()
  .describe() -> dict

The controller runs at T_ctrl Hz (default 100 Hz). Between control updates
the runner drives the dynamics at the integration step; the controller
holds its last command and re-computes only every int(T_ctrl/dt) calls.

Reference management
--------------------
set_waypoints(list_of_ref4) — load a list of [x,y,z_NED,psi] waypoints.
Waypoints are tracked in sequence; each one is rate-limited with
generate_ref_step so large position steps never cause integrator wind-up.
"""
import numpy as np

from sim.quaternion import quat_to_euler
# Hover-equilibrium rotor speed — imported from the vehicle so the controller's
# equilibrium can never drift out of sync with the plant it was designed for.
from vehicles.x4.dynamics import W_e


def _generate_ref_step(current, target, max_rate):
    delta = np.asarray(target) - np.asarray(current)
    return current + np.clip(delta, -max_rate, max_rate)


class X4LQGController:
    """
    Parameters
    ----------
    Adt, Bdt, Cdt  : discrete-time plant matrices
    Kdt            : LQR state-feedback gain (4×16)
    Kidt           : integral gain            (4×4)
    Ldt            : Kalman gain (unused in full_state mode)
    U_e            : equilibrium commands     (4,)
    ref            : initial reference [x, y, z_NED, psi]
    T_ctrl         : controller sample period [s]  (default 0.01 → 100 Hz)
    max_ref_rate   : max reference change per ctrl step [m or rad]
    waypoints      : optional list of reference arrays (4,) — visited in order
    wp_tol         : distance tolerance to switch to next waypoint [m]
    """

    def __init__(self, Adt, Bdt, Cdt, Kdt, Kidt, Ldt, U_e,
                 ref=None, T_ctrl=0.01, max_ref_rate=0.05,
                 waypoints=None, wp_tol=0.15):
        self.Adt  = Adt
        self.Bdt  = Bdt
        self.Cdt  = Cdt
        self.Kdt  = Kdt
        self.Kidt = Kidt
        self.Ldt  = Ldt
        self.U_e  = np.asarray(U_e).reshape(-1)
        self.T_ctrl      = T_ctrl
        self.max_ref_rate = max_ref_rate
        self.wp_tol      = wp_tol

        n = Adt.shape[0]
        m = Cdt.shape[0]
        self.Xest = np.zeros(n)
        self.Xe   = np.zeros(m)

        self.ref = (np.zeros(4) if ref is None
                    else np.asarray(ref, dtype=float))
        self._active_ref = self.ref.copy()   # rate-limited reference

        self.waypoints  = list(waypoints) if waypoints is not None else []
        self._wp_idx    = 0

        self._last_U    = self.U_e.copy()
        self._t_last    = None

        self.X_eq = np.array([0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0,
                               W_e, W_e, W_e, W_e])

    # ------------------------------------------------------------------
    def set_ref(self, ref):
        self.ref = np.asarray(ref, dtype=float)

    def set_waypoints(self, waypoints):
        self.waypoints = list(waypoints)
        self._wp_idx   = 0
        if waypoints:
            self.ref = np.asarray(waypoints[0], dtype=float)

    def reset(self):
        self.Xest[:] = 0.0
        self.Xe[:]   = 0.0
        self._last_U = self.U_e.copy()
        self._t_last = None
        self._active_ref = self.ref.copy()
        self._wp_idx = 0

    # ------------------------------------------------------------------
    def _to_lqg_state(self, X_full):
        """17-state quaternion plant → 16-state LQG deviation vector."""
        x, xd, y, yd, z, zd, qw, qx, qy, qz, p, q_ang, r, w1, w2, w3, w4 = X_full
        phi, theta, psi = quat_to_euler((qw, qx, qy, qz))
        X_lqg = np.array([x, xd, y, yd, z, zd,
                           phi, p, theta, q_ang, psi, r,
                           w1, w2, w3, w4])
        return X_lqg - self.X_eq

    # ------------------------------------------------------------------
    def step(self, t, X_full):
        """
        Called every integration step (dt).  Controller recomputes only
        at integer multiples of T_ctrl; otherwise returns the held command.
        """
        # Determine if this is a controller tick
        if self._t_last is None:
            do_ctrl = True
        else:
            elapsed = t - self._t_last
            do_ctrl = elapsed >= self.T_ctrl - 1e-9

        if not do_ctrl:
            phase = self._current_phase(X_full)
            return self._last_U.copy(), self._make_info(X_full, phase)

        self._t_last = t

        # Waypoint sequencing — advance based on actual vehicle position
        if self.waypoints and self._wp_idx < len(self.waypoints):
            tgt = np.asarray(self.waypoints[self._wp_idx], dtype=float)
            veh_pos = np.array([X_full[0], X_full[2], X_full[4]])  # NED
            pos_err = np.linalg.norm(veh_pos - tgt[:3])
            if pos_err < self.wp_tol:
                self._wp_idx += 1
                if self._wp_idx < len(self.waypoints):
                    self.ref = np.asarray(
                        self.waypoints[self._wp_idx], dtype=float)
            else:
                self.ref = tgt

        # Rate-limit the reference
        self._active_ref = _generate_ref_step(
            self._active_ref, self.ref, self.max_ref_rate)

        # LQR + integral control (full-state feedback, no Kalman).
        X_dev   = self._to_lqg_state(X_full)
        Xe_next = self.Xe + (self._active_ref - self.Cdt @ X_dev)
        dU      = -(self.Kdt @ X_dev) - (self.Kidt @ Xe_next)
        U_raw   = self.U_e + dU
        U_abs   = np.clip(U_raw, 0.0, 800.0)

        # Anti-windup (conditional integration): commit the integral step only
        # when the raw command is unsaturated; otherwise hold Xe so it cannot
        # wind up against the actuator clip. In unsaturated flight this is
        # identical to a plain accumulator.
        if np.array_equal(U_raw, U_abs):
            self.Xe = Xe_next

        self._last_U = U_abs
        phase = self._current_phase(X_full)
        return U_abs, self._make_info(X_full, phase)

    # ------------------------------------------------------------------
    def _current_phase(self, X_full):
        alt_agl = -X_full[4]   # NED: z_NED = -alt_AGL
        if alt_agl < 0.05:
            return 'ground'
        ref_z = self._active_ref[2]
        if abs(alt_agl - (-ref_z)) > 0.5:
            return 'climbing'
        return 'hover'

    def _make_info(self, X_full, phase):
        phi, theta, psi = quat_to_euler(X_full[6:10])
        return {
            'phase':     phase,
            'alt_m':     -X_full[4],
            'x_m':       X_full[0],
            'y_m':       X_full[2],
            'roll_deg':  np.degrees(phi),
            'pitch_deg': np.degrees(theta),
            'yaw_deg':   np.degrees(psi),
            'wp_idx':    self._wp_idx,
        }

    def describe(self):
        return {
            'type':         'LQR+I (full-state feedback)',
            'T_ctrl_s':     self.T_ctrl,
            'max_ref_rate': self.max_ref_rate,
            'waypoints':    len(self.waypoints),
            'wp_tol_m':     self.wp_tol,
        }
