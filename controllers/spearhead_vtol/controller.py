"""
Spearhead VTOL nested-PID controller.

Implements the Controller interface expected by sim.runner.SimRunner:
  .step(t, X) -> (U: np.ndarray, info: dict)
  .reset()
  .describe() -> dict
"""
import numpy as np

from sim.pid import PID
from sim.quaternion import (
    normalize_quaternion, quat_to_euler, rotate_body_to_inertial,
)


# ---------------------------------------------------------------------------
# PID sub-controllers (forward-flight and hover loops)
# ---------------------------------------------------------------------------

class _AltCtrl:
    """Outer loop: altitude error (m NED) → desired vz (m/s NED)."""
    def __init__(self, dt):
        self.pid = PID(Kp=0.15, Ki=0.005, Kd=0.08, dt=dt, integral_limit=50)

    def update(self, ref_alt, z):
        return self.pid.update(ref_alt - z)


class _VelCtrl:
    """Middle loop: velocity errors → thrust cmd + pusher cmd."""
    def __init__(self, dt):
        self.vert_pid      = PID(Kp=35,   Ki=6,     Kd=8,    dt=dt, integral_limit=15)
        self.fwd_pid       = PID(Kp=20,   Ki=0.5,   Kd=0,    dt=dt, integral_limit=50)
        self.fwd_pitch_pid = PID(Kp=0.03, Ki=0.005, Kd=0.01, dt=dt, integral_limit=0.5)

    def update_vertical(self, desired_vz, current_vz):
        return self.vert_pid.update(current_vz - desired_vz)

    def update_forward(self, desired_vx, current_vx):
        return self.fwd_pid.update(desired_vx - current_vx)


class _HoverAttCtrl:
    """Hover middle loop: angle error → desired body rate."""
    def __init__(self, dt):
        self.roll_pid  = PID(Kp=1.8, Ki=0.02, Kd=0.05, dt=dt, integral_limit=30)
        self.pitch_pid = PID(Kp=1.8, Ki=0.02, Kd=0.05, dt=dt, integral_limit=30)
        self.yaw_pid   = PID(Kp=1.2, Ki=0.02, Kd=0.04, dt=dt, integral_limit=25)

    def update(self, ref_roll, ref_pitch, ref_yaw, phi, theta, psi):
        return (self.roll_pid.update(ref_roll   - phi),
                self.pitch_pid.update(ref_pitch - theta),
                self.yaw_pid.update(ref_yaw     - psi))


class _FwdAttCtrl:
    """Forward-flight middle loop: angle error → desired body rate."""
    def __init__(self, dt):
        self.roll_pid  = PID(Kp=1.5, Ki=0.02, Kd=0.05, dt=dt, integral_limit=30)
        self.pitch_pid = PID(Kp=2.0, Ki=0.04, Kd=0.08, dt=dt, integral_limit=30)
        self.yaw_pid   = PID(Kp=1.5, Ki=0.02, Kd=0.04, dt=dt, integral_limit=25)

    def update(self, ref_roll, ref_pitch, ref_yaw, phi, theta, psi):
        return (self.roll_pid.update(ref_roll   - phi),
                self.pitch_pid.update(ref_pitch - theta),
                self.yaw_pid.update(ref_yaw     - psi))


class _HoverRateCtrl:
    """Hover inner loop: body-rate error → motor differential command."""
    def __init__(self, dt):
        self.roll_rate_pid  = PID(Kp=22, Ki=4, Kd=0.3, dt=dt, integral_limit=25)
        self.pitch_rate_pid = PID(Kp=18, Ki=3, Kd=0.3, dt=dt, integral_limit=15)
        self.yaw_rate_pid   = PID(Kp=40, Ki=3, Kd=0.8, dt=dt, integral_limit=14)

    def update(self, des_p, des_q, des_r, p, q, r):
        return (self.roll_rate_pid.update(des_p  - p),
                self.pitch_rate_pid.update(des_q - q),
                self.yaw_rate_pid.update(des_r   - r))


class _FwdRateCtrl:
    """
    Forward-flight inner loop: body-rate error → surface deflection (deg).
    Gains designed at V_design = 15 m/s; gain-scheduled by (15/V)² at runtime.
    """
    def __init__(self, dt):
        self.roll_rate_pid  = PID(Kp=2.2, Ki=0.4, Kd=0.04, dt=dt, integral_limit=37)
        self.pitch_rate_pid = PID(Kp=2.0, Ki=0.8, Kd=0.04, dt=dt, integral_limit=37)
        self.yaw_rate_pid   = PID(Kp=3.0, Ki=0.3, Kd=0.06, dt=dt, integral_limit=50)

    def update(self, des_p, des_q, des_r, p, q, r):
        return (self.roll_rate_pid.update(des_p  - p),
                self.pitch_rate_pid.update(des_q - q),
                self.yaw_rate_pid.update(des_r   - r))


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class SpearheadVTOLController:
    """
    Full nested-PID controller for the Spearhead VTOL.

    vehicle_params: dict from SpearheadDynamics.params
    config:         SimConfig instance
    """

    def __init__(self, vehicle_params, config):
        self.vp  = vehicle_params
        self.cfg = config
        dt = config.dt

        # Phase time boundaries
        phases = config.phases
        self._T_SPINUP = phases.get('spinup',     5.0)
        self._T_HOVER  = phases.get('hover',      30.0)
        self._T_TRANS  = phases.get('transition', 90.0)

        # Reference signals
        refs = config.references
        self._ref_alt     = refs.get('alt',     -10.0)
        self._ref_fwd_vel = refs.get('fwd_vel',  54.0)
        self._ref_roll    = refs.get('roll',      0.0)
        self._ref_yaw     = refs.get('yaw',       0.0)

        # PID controllers
        self._alt_ctrl      = _AltCtrl(dt)
        self._vel_ctrl      = _VelCtrl(dt)
        self._att_ctrl      = _HoverAttCtrl(dt)
        self._att_ctrl_fwd  = _FwdAttCtrl(dt)
        self._rate_ctrl     = _HoverRateCtrl(dt)
        self._rate_ctrl_fwd = _FwdRateCtrl(dt)

        # Phase-transition state
        self._trans_entered    = False
        self._fwd_entered      = False
        self._fwd_stack_active = False
        self._fwd_vx_entry     = 0.0
        self._fwd_t_entry      = 0.0

    # ------------------------------------------------------------------
    def reset(self):
        for attr in ('_alt_ctrl', '_vel_ctrl', '_att_ctrl', '_att_ctrl_fwd',
                     '_rate_ctrl', '_rate_ctrl_fwd'):
            obj = getattr(self, attr)
            for sub in vars(obj).values():
                if isinstance(sub, PID):
                    sub.reset()
        self._trans_entered    = False
        self._fwd_entered      = False
        self._fwd_stack_active = False
        self._fwd_vx_entry     = 0.0
        self._fwd_t_entry      = 0.0

    # ------------------------------------------------------------------
    def describe(self):
        def _fmt(pid):
            return f'Kp={pid.Kp}  Ki={pid.Ki}  Kd={pid.Kd}  lim=±{pid.integral_limit}'
        ac = self._alt_ctrl
        vc = self._vel_ctrl
        ah = self._att_ctrl
        af = self._att_ctrl_fwd
        rh = self._rate_ctrl
        rf = self._rate_ctrl_fwd
        return {
            'AltCtrl':             _fmt(ac.pid),
            'VelCtrl (vert)':      _fmt(vc.vert_pid),
            'VelCtrl (fwd)':       _fmt(vc.fwd_pid),
            'AttCtrl roll (hover)':    _fmt(ah.roll_pid),
            'AttCtrl pitch (hover)':   _fmt(ah.pitch_pid),
            'AttCtrl yaw  (hover)':    _fmt(ah.yaw_pid),
            'AttCtrl roll (fwd)':      _fmt(af.roll_pid),
            'AttCtrl pitch (fwd)':     _fmt(af.pitch_pid),
            'AttCtrl yaw  (fwd)':      _fmt(af.yaw_pid),
            'RateCtrl roll  (hover)':  _fmt(rh.roll_rate_pid),
            'RateCtrl pitch (hover)':  _fmt(rh.pitch_rate_pid),
            'RateCtrl yaw   (hover)':  _fmt(rh.yaw_rate_pid),
            'RateCtrl roll  (fwd)':    _fmt(rf.roll_rate_pid),
            'RateCtrl pitch (fwd)':    _fmt(rf.pitch_rate_pid),
            'RateCtrl yaw   (fwd)':    _fmt(rf.yaw_rate_pid),
        }

    # ------------------------------------------------------------------
    def step(self, t, X):
        """
        Compute control inputs for the current timestep.
        Returns (U, info) where info is a dict of logged fields.
        """
        vp  = self.vp
        M, g   = vp['M'],  vp['g']
        K, Ku  = vp['K'],  vp['Ku']
        L1, L2 = vp['L1'], vp['L2']
        S      = vp['S']
        CFZ0   = vp['CFZ0']
        CFX0   = vp['CFX0']
        CMY0   = vp['CMY0']
        W_HOVER = vp['W_HOVER']
        rho    = vp['rho']

        T_SPINUP = self._T_SPINUP
        T_HOVER  = self._T_HOVER
        T_TRANS  = self._T_TRANS
        ref_fwd_vel = self._ref_fwd_vel

        # -- Extract state --
        u_b, v_b, w_b = X[0:3]
        p,   q,   r   = X[3:6]
        z             = X[8]
        quat = normalize_quaternion(X[9:13])
        phi, theta, psi = quat_to_euler(quat)

        # Inertial velocities
        vel_i  = rotate_body_to_inertial(np.array([u_b, v_b, w_b]), quat)
        vz     = vel_i[2]                           # NED vertical (+ = down)
        vx     = np.sqrt(vel_i[0]**2 + vel_i[1]**2)  # horizontal speed

        # Aerodynamic angles (for logging)
        V_log     = np.sqrt(u_b**2 + v_b**2 + w_b**2)
        alpha_log = np.degrees(np.arctan2(w_b, u_b)) if V_log > 1e-6 else 0.0
        beta_log  = np.degrees(np.arcsin(
            np.clip(v_b / V_log, -1, 1))) if V_log > 1e-6 else 0.0

        # ----------------------------------------------------------------
        # Phase 1 — SPINUP: ramp vertical rotors to W_HOVER
        # ----------------------------------------------------------------
        if t <= T_SPINUP:
            base = (t / T_SPINUP) * W_HOVER
            m1 = m2 = m3 = m4 = base
            m5 = 0.0
            servo_le = servo_re = servo_rud = 0.0
            phase = 'spinup'
            surf_blend = 0.0

        # ----------------------------------------------------------------
        # Phases 2–4 — UNIFIED altitude / attitude / surfaces
        # ----------------------------------------------------------------
        else:
            # -- One-shot FWD entry reset --
            if t > T_TRANS and not self._fwd_entered:
                self._fwd_entered   = True
                self._fwd_vx_entry  = vx
                self._fwd_t_entry   = t
                self._alt_ctrl.pid.integral      = 0.0
                self._vel_ctrl.vert_pid.integral = 0.0
                self._vel_ctrl.fwd_pid.reset()
                for _ctl in (self._att_ctrl_fwd.roll_pid,
                             self._att_ctrl_fwd.pitch_pid,
                             self._att_ctrl_fwd.yaw_pid,
                             self._rate_ctrl_fwd.roll_rate_pid,
                             self._rate_ctrl_fwd.pitch_rate_pid,
                             self._rate_ctrl_fwd.yaw_rate_pid):
                    _ctl.reset()
                self._fwd_stack_active = False

            # Altitude loop
            _vz_lim    = 0.6 if self._fwd_entered else 1.5
            desired_vz = np.clip(self._alt_ctrl.update(self._ref_alt, z),
                                 -_vz_lim, _vz_lim)
            thrust_cmd = self._vel_ctrl.update_vertical(desired_vz, vz)

            # Wing lift fraction (1.0 at vx = full-lift speed ~54 m/s)
            wing_lift_frac = np.clip(
                CFZ0 * 0.5 * rho * vx**2 * S / (M * g), 0.0, 1.0)
            hover_cmd_frac = np.sqrt(max(0.0, 1.0 - wing_lift_frac))

            # Pitch altitude reference (only in forward flight)
            if self._fwd_entered:
                ref_pitch_alt = -desired_vz * wing_lift_frac / max(vx, 15.0)
            else:
                ref_pitch_alt = 0.0

            # Speed-based pitch (trim at current speed + overspeed brake)
            _pitch_trim_vx   = (CFX0 * 0.5 * rho * vx**2 * S) / (M * g)
            _pitch_spd_brake = 0.020 * max(0.0, vx - ref_fwd_vel)
            ref_pitch_acc    = np.clip(_pitch_trim_vx + _pitch_spd_brake, 0.0, 0.30)

            # Pitch rate damper (phugoid suppression)
            _pitch_rate_damp = -1.0 * q
            ref_pitch = np.clip(
                ref_pitch_alt + ref_pitch_acc + _pitch_rate_damp, -0.25, 0.30)

            # -- Hover attitude + rate loop (m1-m4) --
            des_p_h, des_q_h, des_r_h = self._att_ctrl.update(
                self._ref_roll, ref_pitch, self._ref_yaw, phi, theta, psi)
            roll_cmd, pitch_cmd, yaw_cmd = self._rate_ctrl.update(
                des_p_h, des_q_h, des_r_h, p, q, r)

            # -- Direct force allocation: front/rear split to cancel aero CMY --
            _K0   = K[0]
            _Ku0v = Ku[0] / 44.2205   # = 8.18
            _base_raw = hover_cmd_frac * (W_HOVER + thrust_cmd)
            if t > T_HOVER:
                _base_raw = max(_base_raw, 100.0)
            _T_total_N = 4.0 * _K0 * (_Ku0v * _base_raw) ** 2

            _MA_aero   = 0.5 * rho * vx**2 * S * vp['C'] * CMY0
            _F_budget  = (1.0 - wing_lift_frac) * M * g
            _MA_cancel = min(_MA_aero, _F_budget * L2)
            _delta_N   = _MA_cancel / L2

            _F_front = max(0.0, (_T_total_N - _delta_N) / 2.0)
            _F_rear  = (_T_total_N + _delta_N) / 2.0
            _cmd_front = (np.sqrt(_F_front / 2.0 / _K0) / _Ku0v
                          if _F_front > 0 else 0.0)
            _cmd_rear  = np.sqrt(_F_rear  / 2.0 / _K0) / _Ku0v
            base_throttle = (_cmd_front + _cmd_rear) / 2.0

            # -- Pusher reaction-torque roll feedforward --
            _w5     = X[17]
            _Tau5   = K[3] * _w5**2
            if base_throttle > 1.0:
                _roll_ff_raw = -_Tau5 / (L1 * _K0 * 8.0 * _Ku0v**2 * base_throttle)
                _roll_ff_lim = max(_cmd_front * 0.9, 1.0)
                _roll_ff = np.clip(_roll_ff_raw, -_roll_ff_lim, _roll_ff_lim)
            else:
                _roll_ff = 0.0

            # Blend out hover pitch/yaw authority as surfaces take over
            surf_blend = np.clip((vx - 5.0) / (ref_fwd_vel - 5.0), 0.0, 1.0)
            if self._fwd_entered:
                _hpy_blend = max(0.0, 1.0 - surf_blend)
                pitch_cmd *= _hpy_blend
                yaw_cmd   *= _hpy_blend

            # Motor mixing (m1-m4)
            m1 = np.clip(_cmd_front + ( roll_cmd + _roll_ff + pitch_cmd + yaw_cmd), 0, 1000)
            m2 = np.clip(_cmd_front + (-roll_cmd - _roll_ff + pitch_cmd - yaw_cmd), 0, 1000)
            m3 = np.clip(_cmd_rear  + ( roll_cmd + _roll_ff - pitch_cmd - yaw_cmd), 0, 1000)
            m4 = np.clip(_cmd_rear  + (-roll_cmd - _roll_ff - pitch_cmd + yaw_cmd), 0, 1000)

            # -- Pusher: phase-dependent --
            if t <= T_HOVER:
                m5    = 0.0
                phase = 'hover'

            elif t <= T_TRANS:
                phase = 'transition'
                if not self._trans_entered:
                    self._trans_entered = True
                    self._vel_ctrl.fwd_pid.reset()
                throttle_trans = self._vel_ctrl.update_forward(ref_fwd_vel, vx)
                m5 = np.clip(throttle_trans, 0, 330)

            else:
                phase = 'fwd_flight'
                throttle = self._vel_ctrl.update_forward(ref_fwd_vel, vx)
                m5 = np.clip(throttle, 0, 150)

            # -- Forward surfaces --
            if surf_blend > 0.10:
                if not self._fwd_stack_active:
                    self._fwd_stack_active = True
                    for _ctl in (self._att_ctrl_fwd.roll_pid,
                                 self._att_ctrl_fwd.pitch_pid,
                                 self._att_ctrl_fwd.yaw_pid,
                                 self._rate_ctrl_fwd.roll_rate_pid,
                                 self._rate_ctrl_fwd.pitch_rate_pid,
                                 self._rate_ctrl_fwd.yaw_rate_pid):
                        _ctl.reset()

                roll_r_f, pitch_r_f, yaw_r_f = self._att_ctrl_fwd.update(
                    self._ref_roll, ref_pitch, self._ref_yaw, phi, theta, psi)
                roll_f, pitch_f, yaw_f = self._rate_ctrl_fwd.update(
                    roll_r_f, pitch_r_f, yaw_r_f, p, q, r)

                # Gain scheduling: aero effectiveness ∝ V²; designed at V_d = 15 m/s
                _V_design = 15.0
                _gs = max(0.25, (vx / _V_design) ** 2)
                roll_f  /= _gs
                pitch_f /= _gs
                yaw_f   /= _gs
            else:
                roll_f = pitch_f = yaw_f = 0.0

            # Surface feedforwards
            # (a) pitch: cancel residual aero CMY not handled by rotor thrust diff
            _CMY_residual = max(0.0, _MA_aero - _MA_cancel)
            _vx_pitch     = max(vx, 15.0)
            _CMY_eff_v15  = 7.77   # N·m/deg at V=15 (Iyy · α_pitch_aero)
            _ff_pitch = np.clip(
                -_CMY_residual / (_CMY_eff_v15 * (_vx_pitch / 15.0) ** 2),
                -5.0, 0.0)

            # (b) roll: cancel residual Tau5 not already handled by rotor roll_ff
            _Tau5_rotor_cancelled = (abs(_roll_ff)
                                     * (8.0 * L1 * _K0 * _Ku0v**2 * base_throttle))
            _Tau5_residual = max(0.0, _Tau5 - _Tau5_rotor_cancelled)
            _CMX_eff_v15   = 8.909  # N·m/deg at V=15 (Ixx · α_roll_aero)
            _vx_roll       = max(vx, 1.0)
            _ff_roll = np.clip(
                -_Tau5_residual / (_CMX_eff_v15 * (_vx_roll / 15.0) ** 2),
                -3.0, 3.0)

            servo_le  = np.clip(surf_blend * (pitch_f + _ff_pitch + roll_f) + _ff_roll, -20, 20)
            servo_re  = np.clip(surf_blend * (pitch_f + _ff_pitch - roll_f) - _ff_roll, -20, 20)
            servo_rud = np.clip(surf_blend * yaw_f, -20, 20)

        # ----------------------------------------------------------------
        # Build output
        # ----------------------------------------------------------------
        U = np.array([m1, m2, m3, m4, m5, servo_le, servo_re, servo_rud])

        info = {
            'phase':           phase,
            'alt_m':           -z,
            'vx_ms':           vx,
            'vz_ms':           vz,
            'roll_deg':        np.degrees(phi),
            'pitch_deg':       np.degrees(theta),
            'yaw_deg':         np.degrees(psi),
            'p_dps':           np.degrees(p),
            'q_dps':           np.degrees(q),
            'r_dps':           np.degrees(r),
            'alpha_deg':       alpha_log,
            'beta_deg':        beta_log,
            'V_ms':            V_log,
            'm1_cmd':          m1,
            'm2_cmd':          m2,
            'm3_cmd':          m3,
            'm4_cmd':          m4,
            'm5_cmd':          m5,
            'servo_le_deg':    servo_le,
            'servo_re_deg':    servo_re,
            'servo_rud_deg':   servo_rud,
            'surf_blend':      surf_blend,
            'alt_error_m':     self._ref_alt - z,
            'fwd_vel_error_ms': ref_fwd_vel - vx,
        }
        return U, info
