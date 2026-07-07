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
    .get_position(X) -> np.ndarray(3)      NED (north, east, down) [m]
                                           (optional; lets the runner report
                                            position without assuming layout)
    .apply_constraints(X) -> np.ndarray    (optional; per-step constraints such
                                            as quaternion normalisation and
                                            ground clamp — the runner calls it
                                            after each integration step and
                                            assumes no state layout of its own)
    .envelope_violations(X) -> list[str]   (optional; non-empty list = state is
                                            outside the model's validated
                                            envelope, e.g. aero-table alpha/beta
                                            range. The runner logs the first
                                            exit and marks all subsequent CSV
                                            rows data_valid=0.)
    .terminal_condition(t, X) -> str|None  (optional; return 'crash' or
                                            'departure' to end the run early —
                                            gated by config.terminate_on)

  Controller:
    .step(t, X) -> (U: np.ndarray, info: dict)
        info MUST contain key 'phase': str
        info MAY contain any additional scalar fields for CSV logging
    .reset()
    .describe() -> dict   (for log header, e.g. PID gains)

run() returns a SimResult. It unpacks like the legacy 4-tuple
(X_hist, U_hist, log_path, csv_path) and additionally carries
.verdict / .reason / .metrics / .t_end / .json_path.
"""
import csv
import datetime
import json
import os
import platform
import subprocess
from dataclasses import dataclass, field, asdict

import numpy as np


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """Outcome of a simulation run.

    verdict is one of:
      PASS / FAIL   — ran to tf, judged against config.pass_criteria
      COMPLETE      — ran to tf, no pass_criteria configured
      CRASHED / DEPARTED — ended early by the vehicle's terminal_condition
      DIVERGED      — non-finite state (always terminates)
    """
    x_hist:    np.ndarray
    u_hist:    np.ndarray
    log_path:  str
    csv_path:  str
    verdict:   str = 'COMPLETE'
    reason:    str = ''
    t_end:     float = 0.0
    metrics:   dict = field(default_factory=dict)
    json_path: str = ''

    @property
    def passed(self):
        return self.verdict in ('PASS', 'COMPLETE')

    def __iter__(self):
        # Legacy unpacking: X_hist, U_hist, log_path, csv_path = runner.run()
        return iter((self.x_hist, self.u_hist, self.log_path, self.csv_path))


def _git_metadata():
    """Commit hash + dirty flag of the repo this file lives in (best effort)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        h = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'],
                           capture_output=True, text=True, timeout=3)
        if h.returncode != 0:
            return {'commit': 'unknown (not a git repo)', 'dirty': None}
        d = subprocess.run(['git', '-C', root, 'status', '--porcelain'],
                           capture_output=True, text=True, timeout=3)
        return {'commit': h.stdout.strip(),
                'dirty': bool(d.stdout.strip()) if d.returncode == 0 else None}
    except (OSError, subprocess.TimeoutExpired):
        return {'commit': 'unknown (git unavailable)', 'dirty': None}


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

        # Wind / disturbances (toggle-able; None = OFF, bit-identical path)
        wind = None
        if cfg.wind is not None:
            from sim.wind import WindModel
            if hasattr(dyn, 'set_wind_ned'):
                wind = WindModel(cfg.wind, dt)
            else:
                print(f'[WARN] config.wind set but {type(dyn).__name__} has no '
                      f'set_wind_ned() — wind ignored')

        # Envelope validity: latched False on first exit — everything after an
        # envelope exit is model fiction and stays flagged even on re-entry.
        _has_envelope   = hasattr(dyn, 'envelope_violations')
        _data_valid     = True
        _env_first_exit = None
        _env_valid_steps = 0

        # Early-termination outcome: (verdict, reason) or None
        terminated = None
        last_info  = {}
        t_end      = time_arr[-1]

        # Write log header
        self._write_header(log_path, cfg, dyn, ctl)

        # ----------------------------------------------------------------
        # Main loop
        # ----------------------------------------------------------------
        for i, t in enumerate(time_arr):
            X_hist[i, :] = X

            # Wind: one filter step per integration step, held constant across
            # the RK4 substeps (zero-order hold).
            if wind is not None:
                w_ned = wind.step()
                dyn.set_wind_ned(w_ned)

            # Controller step
            U, info = ctl.step(t, X)
            U_hist[i, :] = U
            last_info = info

            phase = info.get('phase', 'unknown')

            # Build CSV column list on first step
            if _csv_cols is None:
                _extra_cols = [k for k in info if k != 'phase']
                _csv_cols   = (['time_s', 'phase']
                               + dyn.state_names
                               + dyn.control_names
                               + _extra_cols
                               + (['wind_n', 'wind_e', 'wind_d']
                                  if wind is not None else [])
                               + ['data_valid'])

            # Envelope validity check (vehicle-defined)
            if _has_envelope and _data_valid:
                _viol = dyn.envelope_violations(X)
                if _viol:
                    _data_valid     = False
                    _env_first_exit = t
                    with open(log_path, 'a') as lf:
                        lf.write(f'  t={t:8.3f}s  ENVELOPE EXIT — data '
                                 f'invalid from here on\n')
                        for v in _viol:
                            lf.write(f'      {v}\n')
                    if cfg.terminate_on.get('envelope_exit', False):
                        terminated = ('DEPARTED',
                                      f'envelope exit at t={t:.3f}s: '
                                      + '; '.join(_viol))
            if _data_valid:
                _env_valid_steps += 1

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
                if wind is not None:
                    row += [f'{v:.5g}' for v in w_ned]
                row.append('1' if _data_valid else '0')
                log_rows.append(row)

            # Early termination decided on this step's checks
            if terminated is not None:
                t_end = t
                N = i + 1
                break

            # RK4
            k1 = dyn.derivatives(t,            X,            U)
            k2 = dyn.derivatives(t + 0.5*dt,   X + 0.5*dt*k1, U)
            k3 = dyn.derivatives(t + 0.5*dt,   X + 0.5*dt*k2, U)
            k4 = dyn.derivatives(t + dt,        X + dt*k3,     U)
            X  = X + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            # Post-step constraints (quaternion normalisation, ground clamp, …)
            # are the vehicle's responsibility. The runner makes no assumptions
            # about state layout and simply delegates when the hook is present.
            if hasattr(dyn, 'apply_constraints'):
                X = dyn.apply_constraints(X)

            # Divergence always terminates — NaN/Inf cannot be integrated.
            if not np.all(np.isfinite(X)):
                bad = [dyn.state_names[j] for j in
                       np.where(~np.isfinite(X))[0][:6]]
                terminated = ('DIVERGED',
                              f'non-finite state at t={t + dt:.3f}s '
                              f'({", ".join(bad)})')
                t_end = t + dt
                N = i + 1
                break

            # Vehicle-detected terminal condition (crash / departure)
            if hasattr(dyn, 'terminal_condition'):
                tc = dyn.terminal_condition(t + dt, X)
                if tc is not None and cfg.terminate_on.get(tc, True):
                    verdict_code = 'CRASHED' if tc == 'crash' else 'DEPARTED'
                    terminated = (verdict_code,
                                  f'{tc} detected at t={t + dt:.3f}s')
                    t_end = t + dt
                    N = i + 1
                    break

        # ----------------------------------------------------------------
        # Truncate histories if the run ended early
        # ----------------------------------------------------------------
        X_hist = X_hist[:N]
        U_hist = U_hist[:N]

        # ----------------------------------------------------------------
        # Verdict
        # ----------------------------------------------------------------
        # Final metrics: controller info + runner-computed position keys.
        final_metrics = {k: v for k, v in last_info.items()
                         if isinstance(v, (int, float, np.integer, np.floating))}
        if hasattr(dyn, 'get_position'):
            n_p, e_p, d_p = (float(v) for v in dyn.get_position(X))
            final_metrics.update(north_m=n_p, east_m=e_p, down_m=d_p,
                                 alt_agl_m=-d_p)

        criteria_results = []
        if terminated is not None:
            verdict, reason = terminated
        elif cfg.pass_criteria:
            fails = []
            for key, (lo, hi) in cfg.pass_criteria.items():
                val = final_metrics.get(key)
                ok  = val is not None and lo <= val <= hi
                criteria_results.append(
                    {'metric': key, 'lo': lo, 'hi': hi,
                     'value': None if val is None else float(val),
                     'pass': bool(ok)})
                if not ok:
                    fails.append(f'{key}={val if val is not None else "missing"}'
                                 f' not in [{lo}, {hi}]')
            verdict = 'PASS' if not fails else 'FAIL'
            reason  = 'all criteria met' if not fails else '; '.join(fails)
        else:
            verdict, reason = 'COMPLETE', 'reached tf (no pass_criteria set)'

        env_stats = None
        if _has_envelope:
            env_stats = {'first_exit_t': _env_first_exit,
                         'valid_fraction': round(_env_valid_steps / N, 4)}

        # ----------------------------------------------------------------
        # Flush CSV, write summary, write machine-readable JSON verdict
        # ----------------------------------------------------------------
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(_csv_cols or [])
            writer.writerows(log_rows)

        self._write_summary(log_path, csv_path, X, _phase_stats, t_end,
                            verdict, reason, criteria_results, env_stats)

        json_path = os.path.splitext(log_path)[0] + '.json'
        with open(json_path, 'w') as jf:
            json.dump({
                'schema':      'flight-sim-platform/run-summary/v1',
                'generated':   datetime.datetime.now().isoformat(),
                'vehicle':     cfg.vehicle_name,
                'controller':  cfg.controller_name,
                'git':         _git_metadata(),
                'python':      platform.python_version(),
                'numpy':       np.__version__,
                'config':      {k: v for k, v in asdict(cfg).items()},
                'verdict':     verdict,
                'reason':      reason,
                't_end':       float(t_end),
                'criteria':    criteria_results,
                'envelope':    env_stats,
                'metrics':     {k: float(v) for k, v in final_metrics.items()},
                'files':       {'log': log_path, 'csv': csv_path},
            }, jf, indent=2)

        print(f'[LOG] {log_path}')
        print(f'[CSV] {csv_path}')
        print(f'[VERDICT] {verdict} — {reason}')

        return SimResult(X_hist, U_hist, log_path, csv_path,
                         verdict=verdict, reason=reason, t_end=float(t_end),
                         metrics=final_metrics, json_path=json_path)

    # ------------------------------------------------------------------
    def _write_header(self, log_path, cfg, dyn, ctl):
        with open(log_path, 'w') as lf:
            lf.write('=' * 72 + '\n')
            lf.write(f'  UAV Simulation — {cfg.vehicle_name} / {cfg.controller_name}\n')
            lf.write(f'  Generated : {datetime.datetime.now().isoformat()}\n')
            lf.write('=' * 72 + '\n\n')

            git = _git_metadata()
            lf.write('[REPRODUCIBILITY]\n')
            lf.write(f'  git commit   = {git["commit"]}'
                     + ('  (DIRTY working tree)' if git['dirty'] else '') + '\n')
            lf.write(f'  python       = {platform.python_version()}\n')
            lf.write(f'  numpy        = {np.__version__}\n\n')

            lf.write('[SIMULATION PARAMETERS]\n')
            for k, v in asdict(cfg).items():
                lf.write(f'  {k:<14} = {v}\n')
            lf.write('\n')

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
    def _write_summary(self, log_path, csv_path, X_final, phase_stats, t_end,
                       verdict, reason, criteria_results, env_stats):
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
            lf.write(f'  t         = {t_end:.1f} s\n')
            # Ask the vehicle for its NED position — never assume state layout.
            if hasattr(self.dynamics, 'get_position'):
                n_pos, e_pos, d_pos = self.dynamics.get_position(X_final)
                lf.write(f'  pos (NED) = ({n_pos:.1f}, {e_pos:.1f}, {d_pos:.1f}) m\n')
                lf.write(f'  alt AGL   = {-d_pos:.2f} m\n')

            if env_stats is not None:
                lf.write('\n[ENVELOPE]\n')
                if env_stats['first_exit_t'] is None:
                    lf.write('  within validated envelope for entire run\n')
                else:
                    lf.write(f'  FIRST EXIT at t = {env_stats["first_exit_t"]:.3f} s '
                             f'— data after this point is outside the validated '
                             f'model envelope\n')
                lf.write(f'  valid fraction  = {env_stats["valid_fraction"]*100:.1f}%\n')

            lf.write('\n[VERDICT]\n')
            lf.write(f'  {verdict} — {reason}\n')
            for c in criteria_results:
                mark = 'PASS' if c['pass'] else 'FAIL'
                lf.write(f'    [{mark}] {c["metric"]:<16} = {c["value"]}'
                         f'   (required {c["lo"]} .. {c["hi"]})\n')

            lf.write(f'\n[OUTPUT FILES]\n')
            lf.write(f'  Log : {log_path}\n')
            lf.write(f'  CSV : {csv_path}\n')
            lf.write('=' * 72 + '\n')
