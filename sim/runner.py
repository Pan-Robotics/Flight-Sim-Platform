"""
Generic RK4 simulation runner.

Interfaces (duck-typed):

  VehicleDynamics:
    .state_dim      int
    .control_dim    int
    .state_names    list[str]    (len == state_dim)
    .control_names  list[str]    (len == control_dim)
    .initial_state() -> np.ndarray
    .derivatives(t, X, U) -> np.ndarray   (dXdt)
    .describe()      -> dict               (for log header)

  Controller:
    .step(t, X) -> (U: np.ndarray, info: dict)
        info MUST contain key 'phase': str
        info MAY contain any additional scalar fields for CSV logging
    .reset()
    .describe() -> dict   (for log header, e.g. PID gains)
"""
import csv
import datetime
import os

import numpy as np

from sim.quaternion import normalize_quaternion


class SimRunner:

    def __init__(self, dynamics, controller, config):
        self.dynamics   = dynamics
        self.controller = controller
        self.config     = config

    # ------------------------------------------------------------------
    def run(self):
        cfg = self.config
        dyn = self.dynamics
        ctl = self.controller

        dt        = cfg.dt
        tf        = cfg.tf
        time_arr  = np.arange(0.0, tf + dt, dt)
        N         = len(time_arr)
        log_every = max(1, int(round(1.0 / (cfg.log_hz * dt))))

        # Log paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir    = os.path.normpath(os.path.join(script_dir, '..', cfg.log_dir))
        os.makedirs(log_dir, exist_ok=True)
        ts         = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path   = os.path.join(log_dir, f'flight_{ts}.log')
        csv_path   = os.path.join(log_dir, f'flight_{ts}.csv')

        # State / control histories (for plotting)
        X = dyn.initial_state().copy()
        X_hist = np.zeros((N, dyn.state_dim))
        U_hist = np.zeros((N, dyn.control_dim))

        # Phase tracking + per-phase accumulators
        _prev_phase  = None
        _phase_stats = {}   # phase -> list[info_dict]

        # CSV: columns determined on first step
        _csv_cols   = None
        _extra_cols = []
        log_rows    = []

        # Write log header
        self._write_header(log_path, cfg, dyn, ctl)

        # ----------------------------------------------------------------
        # Main loop
        # ----------------------------------------------------------------
        for i, t in enumerate(time_arr):
            X_hist[i, :] = X

            # Generic Spearhead-style quaternion normalisation (indices 9:13).
            # Vehicles that supply apply_constraints() handle their own normalisation there.
            if not hasattr(dyn, 'apply_constraints') and dyn.state_dim >= 13:
                X[9:13] = normalize_quaternion(X[9:13])

            # Controller step
            U, info = ctl.step(t, X)
            U_hist[i, :] = U

            phase = info.get('phase', 'unknown')

            # Build CSV column list on first step
            if _csv_cols is None:
                _extra_cols = [k for k in info if k != 'phase']
                _csv_cols   = (['time_s', 'phase']
                               + dyn.state_names
                               + dyn.control_names
                               + _extra_cols)

            # Accumulate per-phase stats
            if phase not in _phase_stats:
                _phase_stats[phase] = []
            _phase_stats[phase].append(info)

            # Phase-change event -> log file
            if phase != _prev_phase:
                with open(log_path, 'a') as lf:
                    lf.write(f'  t={t:8.3f}s  ENTER {phase.upper():<16}')
                    for key in ('alt_m', 'vx_ms', 'roll_deg', 'pitch_deg'):
                        v = info.get(key)
                        if v is not None:
                            lf.write(f'  {key}={v:+.2f}')
                    lf.write('\n')
                _prev_phase = phase

            # Decimated CSV row
            if i % log_every == 0:
                row  = [f'{t:.3f}', phase]
                row += [f'{v:.5g}' for v in X]
                row += [f'{v:.5g}' for v in U]
                for k in _extra_cols:
                    v = info.get(k, '')
                    row.append(f'{v:.5g}' if isinstance(v, (int, float)) else str(v))
                log_rows.append(row)

            # RK4
            k1 = dyn.derivatives(t,            X,            U)
            k2 = dyn.derivatives(t + 0.5*dt,   X + 0.5*dt*k1, U)
            k3 = dyn.derivatives(t + 0.5*dt,   X + 0.5*dt*k2, U)
            k4 = dyn.derivatives(t + dt,        X + dt*k3,     U)
            X  = X + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            # Post-step: quaternion normalisation and ground clamp.
            # Vehicle may supply apply_constraints(X) for custom logic.
            if hasattr(dyn, 'apply_constraints'):
                X = dyn.apply_constraints(X)
            else:
                # Generic Spearhead-style ground clamp (z at index 8)
                if dyn.state_dim >= 9 and X[8] > 0.0:
                    X[8] = 0.0
                    if X[2] > 0.0:
                        X[2] = 0.0

        # ----------------------------------------------------------------
        # Flush CSV and write summary
        # ----------------------------------------------------------------
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(_csv_cols or [])
            writer.writerows(log_rows)

        self._write_summary(log_path, csv_path, X, _phase_stats, tf)

        print(f'[LOG] {log_path}')
        print(f'[CSV] {csv_path}')

        return X_hist, U_hist, log_path, csv_path

    # ------------------------------------------------------------------
    def _write_header(self, log_path, cfg, dyn, ctl):
        with open(log_path, 'w') as lf:
            lf.write('=' * 72 + '\n')
            lf.write(f'  UAV Simulation — {cfg.vehicle_name} / {cfg.controller_name}\n')
            lf.write(f'  Generated : {datetime.datetime.now().isoformat()}\n')
            lf.write('=' * 72 + '\n\n')

            lf.write('[SIMULATION PARAMETERS]\n')
            lf.write(f'  dt           = {cfg.dt} s\n')
            lf.write(f'  tf           = {cfg.tf} s\n')
            lf.write(f'  phases       = {cfg.phases}\n')
            lf.write(f'  references   = {cfg.references}\n')
            lf.write(f'  log_hz       = {cfg.log_hz} Hz\n\n')

            lf.write('[VEHICLE]\n')
            for k, v in dyn.describe().items():
                lf.write(f'  {k:<28} = {v}\n')
            lf.write('\n')

            lf.write('[CONTROLLER]\n')
            for k, v in ctl.describe().items():
                lf.write(f'  {k:<36} = {v}\n')
            lf.write('\n')

            lf.write('[PHASE EVENTS]\n')

    # ------------------------------------------------------------------
    def _write_summary(self, log_path, csv_path, X_final, phase_stats, tf):
        with open(log_path, 'a') as lf:
            lf.write('\n[PHASE PERFORMANCE SUMMARY]\n')
            for phase, rows in phase_stats.items():
                if not rows:
                    continue
                lf.write(f'\n  {phase.upper()}\n')
                # Summarise every numeric field that appears in info
                keys = [k for k in rows[0]
                        if k not in ('phase',) and isinstance(rows[0][k], (int, float))]
                for key in keys:
                    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
                    if vals:
                        lf.write(f'    {key:<28}: '
                                 f'mean={np.mean(vals):10.4g}  '
                                 f'min={np.min(vals):10.4g}  '
                                 f'max={np.max(vals):10.4g}\n')

            lf.write('\n[END STATE]\n')
            lf.write(f'  t         = {tf:.1f} s\n')
            if len(X_final) >= 9:
                lf.write(f'  pos (NED) = ({X_final[6]:.1f}, {X_final[7]:.1f}, {X_final[8]:.1f}) m\n')
                lf.write(f'  alt AGL   = {-X_final[8]:.2f} m\n')

            lf.write(f'\n[OUTPUT FILES]\n')
            lf.write(f'  Log : {log_path}\n')
            lf.write(f'  CSV : {csv_path}\n')
            lf.write('=' * 72 + '\n')
