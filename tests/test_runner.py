"""Tests for SimRunner termination / verdict / envelope machinery (stub vehicle)."""
import csv
import json
import os
import tempfile
import unittest

import numpy as np

from sim.config import SimConfig
from sim.runner import SimRunner, SimResult


class _StubDynamics:
    """1-D vehicle: X = [z_NED, zdot]. Rises at 1 m/s (z becomes negative)."""
    state_dim   = 2
    control_dim = 1
    state_names   = ['z', 'zdot']
    control_names = ['u0']

    def __init__(self, diverge_after=None):
        self.diverge_after = diverge_after

    def initial_state(self):
        return np.array([0.0, -1.0])       # climbing at 1 m/s

    def derivatives(self, t, X, U):
        if self.diverge_after is not None and t > self.diverge_after:
            return np.array([np.inf, np.inf])
        return np.array([X[1], 0.0])

    def get_position(self, X):
        return np.array([0.0, 0.0, X[0]])

    def describe(self):
        return {'model': 'stub 1-D'}


class _CrashingDynamics(_StubDynamics):
    def __init__(self, crash_at):
        super().__init__()
        self.crash_at = crash_at

    def terminal_condition(self, t, X):
        return 'crash' if t >= self.crash_at else None


class _EnvelopeDynamics(_StubDynamics):
    """Violates its envelope once above 0.2 m altitude (z < -0.2)."""
    def envelope_violations(self, X):
        return ['above validated ceiling'] if X[0] < -0.2 else []


class _StubController:
    def step(self, t, X):
        return np.zeros(1), {'phase': 'up', 'alt_m': -X[0]}

    def reset(self):
        pass

    def describe(self):
        return {'type': 'stub'}


def _cfg(tmp, **kw):
    base = dict(dt=0.01, tf=1.0, phases={'up': 0.0}, references={},
                vehicle_name='stub', controller_name='stub',
                log_dir=tmp, log_hz=100.0)
    base.update(kw)
    return SimConfig(**base)


class TestRunnerMachinery(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_complete_without_criteria(self):
        r = SimRunner(_StubDynamics(), _StubController(), _cfg(self.tmp)).run()
        self.assertEqual(r.verdict, 'COMPLETE')
        self.assertTrue(r.passed)

    def test_pass_and_fail_criteria(self):
        # climbs to ~1 m by tf=1 s
        ok = _cfg(self.tmp, pass_criteria={'alt_m': (0.9, 1.1)})
        r = SimRunner(_StubDynamics(), _StubController(), ok).run()
        self.assertEqual(r.verdict, 'PASS')

        bad = _cfg(self.tmp, pass_criteria={'alt_m': (5.0, 6.0)})
        r = SimRunner(_StubDynamics(), _StubController(), bad).run()
        self.assertEqual(r.verdict, 'FAIL')
        self.assertFalse(r.passed)
        self.assertIn('alt_m', r.reason)

    def test_missing_metric_fails(self):
        cfg = _cfg(self.tmp, pass_criteria={'no_such_metric': (0, 1)})
        r = SimRunner(_StubDynamics(), _StubController(), cfg).run()
        self.assertEqual(r.verdict, 'FAIL')

    def test_crash_terminates_and_truncates(self):
        cfg = _cfg(self.tmp)
        r = SimRunner(_CrashingDynamics(0.5), _StubController(), cfg).run()
        self.assertEqual(r.verdict, 'CRASHED')
        self.assertLess(r.t_end, 0.6)
        self.assertLess(r.x_hist.shape[0], int(1.0 / 0.01))

    def test_terminate_on_gating(self):
        cfg = _cfg(self.tmp, terminate_on={'crash': False})
        r = SimRunner(_CrashingDynamics(0.5), _StubController(), cfg).run()
        self.assertEqual(r.verdict, 'COMPLETE')
        self.assertAlmostEqual(r.t_end, 1.0, places=6)

    def test_divergence_always_terminates(self):
        cfg = _cfg(self.tmp)
        r = SimRunner(_StubDynamics(diverge_after=0.3), _StubController(),
                      cfg).run()
        self.assertEqual(r.verdict, 'DIVERGED')
        self.assertLess(r.t_end, 0.5)

    def test_envelope_marks_data_invalid(self):
        cfg = _cfg(self.tmp)
        r = SimRunner(_EnvelopeDynamics(), _StubController(), cfg).run()
        self.assertEqual(r.verdict, 'COMPLETE')   # marking, not termination
        with open(r.csv_path) as f:
            rows = list(csv.DictReader(f))
        flags = [int(row['data_valid']) for row in rows]
        self.assertEqual(flags[0], 1)
        self.assertEqual(flags[-1], 0)            # invalid after the exit
        self.assertEqual(sorted(set(flags), reverse=True), [1, 0])

    def test_envelope_exit_can_terminate(self):
        cfg = _cfg(self.tmp, terminate_on={'envelope_exit': True})
        r = SimRunner(_EnvelopeDynamics(), _StubController(), cfg).run()
        self.assertEqual(r.verdict, 'DEPARTED')
        self.assertLess(r.t_end, 0.35)

    def test_json_sidecar(self):
        cfg = _cfg(self.tmp, pass_criteria={'alt_m': (0.9, 1.1)})
        r = SimRunner(_StubDynamics(), _StubController(), cfg).run()
        self.assertTrue(os.path.exists(r.json_path))
        with open(r.json_path) as f:
            doc = json.load(f)
        self.assertEqual(doc['verdict'], 'PASS')
        self.assertIn('commit', doc['git'])
        self.assertEqual(doc['config']['dt'], 0.01)
        self.assertEqual(doc['criteria'][0]['metric'], 'alt_m')

    def test_legacy_tuple_unpacking(self):
        r = SimRunner(_StubDynamics(), _StubController(), _cfg(self.tmp)).run()
        self.assertIsInstance(r, SimResult)
        X_hist, U_hist, log_path, csv_path = r
        self.assertEqual(X_hist.shape[1], 2)
        self.assertTrue(log_path.endswith('.log'))


if __name__ == '__main__':
    unittest.main()
