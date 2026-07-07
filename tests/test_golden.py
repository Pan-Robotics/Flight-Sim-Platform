"""Golden-run regression tests.

Short deterministic runs of both candidates, pinned to reference end states.
If a change to dynamics, controllers, or the runner shifts these numbers, the
test fails — qualitative behavior changes must be *seen*, not discovered later.

To intentionally re-pin after a deliberate behavior change:
  PYTHONPATH=. python3 tests/test_golden.py --repin
and commit the new numbers together with the change that caused them.

Full-length mission tests (slow: minutes) are gated behind RUN_SLOW=1.
"""
import importlib
import os
import sys
import tempfile
import unittest

import numpy as np

from sim.runner import SimRunner

# ---------------------------------------------------------------------------
# Pinned references (regenerate with --repin; commit alongside the change)
# ---------------------------------------------------------------------------
GOLDEN = {
    'candidates.x4_lqg': {
        'tf': 3.0,
        'final_pos': [2.920002906, 2.724067276, -1.498891949],
        'x_norm':    3999.327179,
        'verdict':   'COMPLETE',
    },
    'candidates.spearhead_vtol': {
        'tf': 8.0,
        'final_pos': [0.005297790066, 0.0001031872852, -2.708252135],
        'x_norm':    13229.44964,
        'verdict':   'COMPLETE',
    },
}

# Same-machine runs are bit-deterministic; tolerance only allows for
# cross-platform libm/BLAS variation.
RTOL, ATOL = 1e-6, 1e-8


def _run(mod_name, tf):
    mod = importlib.import_module(mod_name)
    dyn, ctl, cfg = mod.build()
    cfg.tf = tf
    cfg.log_dir = tempfile.mkdtemp(prefix='golden_')
    cfg.pass_criteria = {}          # goldens judge state, not mission criteria
    result = SimRunner(dyn, ctl, cfg).run()
    return dyn, result


class TestGoldenRuns(unittest.TestCase):

    def _check(self, mod_name):
        ref = GOLDEN[mod_name]
        dyn, r = _run(mod_name, ref['tf'])
        self.assertEqual(r.verdict, ref['verdict'])
        self.assertTrue(np.all(np.isfinite(r.x_hist)))
        pos = dyn.get_position(r.x_hist[-1])
        np.testing.assert_allclose(pos, ref['final_pos'], rtol=RTOL, atol=ATOL,
                                   err_msg=f'{mod_name}: end position drifted')
        np.testing.assert_allclose(np.linalg.norm(r.x_hist[-1]), ref['x_norm'],
                                   rtol=RTOL,
                                   err_msg=f'{mod_name}: end state-norm drifted')

    def test_x4_lqg_golden(self):
        self._check('candidates.x4_lqg')

    def test_spearhead_golden(self):
        self._check('candidates.spearhead_vtol')


@unittest.skipUnless(os.environ.get('RUN_SLOW') == '1',
                     'full-length missions; set RUN_SLOW=1 to run')
class TestFullMissions(unittest.TestCase):

    def test_x4_full_mission_passes(self):
        mod = importlib.import_module('candidates.x4_lqg')
        dyn, ctl, cfg = mod.build()
        cfg.log_dir = tempfile.mkdtemp(prefix='full_')
        r = SimRunner(dyn, ctl, cfg).run()
        self.assertEqual(r.verdict, 'PASS', r.reason)

    def test_spearhead_full_mission_runs(self):
        mod = importlib.import_module('candidates.spearhead_vtol')
        dyn, ctl, cfg = mod.build()
        cfg.log_dir = tempfile.mkdtemp(prefix='full_')
        r = SimRunner(dyn, ctl, cfg).run()
        # Known open issue: the Spearhead sinks to the ground in cruise, so we
        # assert integrity (finite, no divergence), not mission PASS.
        self.assertIn(r.verdict, ('PASS', 'FAIL', 'CRASHED'))
        self.assertTrue(np.all(np.isfinite(r.x_hist)))


def _repin():
    lines = []
    for mod_name, ref in GOLDEN.items():
        dyn, r = _run(mod_name, ref['tf'])
        pos = [float(f'{v:.10g}') for v in dyn.get_position(r.x_hist[-1])]
        xn  = float(f'{np.linalg.norm(r.x_hist[-1]):.10g}')
        lines.append(f"    '{mod_name}': {{\n"
                     f"        'tf': {ref['tf']},\n"
                     f"        'final_pos': {pos},\n"
                     f"        'x_norm':    {xn},\n"
                     f"        'verdict':   '{r.verdict}',\n    }},")
    print('GOLDEN = {')
    print('\n'.join(lines))
    print('}')


if __name__ == '__main__':
    if '--repin' in sys.argv:
        _repin()
    else:
        unittest.main()
