"""Tests for overrides, the sweep runner machinery, and trim/linearization."""
import importlib
import os
import tempfile
import unittest

import numpy as np

import run_sweep
from sim.analysis import trim, linearize, eig_report
from vehicles.x4.dynamics import X4Dynamics, W_e, U_e


class TestOverrides(unittest.TestCase):

    def test_x4_build_with_overrides(self):
        mod = importlib.import_module('candidates.x4_lqg')
        dyn, ctl, cfg = mod.build({'vehicle': {'M': 0.95},
                                   'config': {'tf': 12.0},
                                   'controller': {'wp_tol': 0.3}})
        self.assertEqual(dyn.params['M'], 0.95)
        self.assertEqual(cfg.tf, 12.0)
        self.assertEqual(ctl.wp_tol, 0.3)
        # perturbed plant hovers at a different W_e than nominal
        self.assertGreater(dyn.W_e, W_e)

    def test_spearhead_controller_stays_nominal(self):
        mod = importlib.import_module('candidates.spearhead_vtol')
        dyn, ctl, cfg = mod.build({'vehicle': {'M': 25.0}})
        self.assertEqual(dyn.params['M'], 25.0)     # plant perturbed
        self.assertEqual(ctl.vp['M'], 20.0)         # controller at design point

    def test_no_overrides_is_default(self):
        mod = importlib.import_module('candidates.x4_lqg')
        d1, _, c1 = mod.build()
        d2, _, c2 = mod.build(None)
        self.assertEqual(d1.params, d2.params)
        self.assertEqual(c1.tf, c2.tf)


class TestSweepMachinery(unittest.TestCase):

    def test_set_dotted(self):
        d = {}
        run_sweep.set_dotted(d, 'vehicle.M', 0.9)
        run_sweep.set_dotted(d, 'config.tf', 5.0)
        self.assertEqual(d, {'vehicle': {'M': 0.9}, 'config': {'tf': 5.0}})

    def test_grid_matrix(self):
        mod = importlib.import_module('candidates.x4_lqg')
        spec = {'candidate': 'candidates.x4_lqg',
                'grid': {'vehicle.M': [0.8, 0.9], 'config.tf': [1.0, 2.0, 3.0]}}
        runs = run_sweep.build_run_matrix(spec, mod)
        self.assertEqual(len(runs), 6)
        self.assertEqual(runs[0]['overrides']['vehicle']['M'], 0.8)

    def test_monte_carlo_matrix_seeded(self):
        mod = importlib.import_module('candidates.x4_lqg')
        spec = {'candidate': 'candidates.x4_lqg',
                'monte_carlo': {'n': 5, 'seed': 11, 'dispersions': {
                    'vehicle.M': {'type': 'normal_pct', 'sigma_pct': 5}}}}
        a = run_sweep.build_run_matrix(spec, mod)
        b = run_sweep.build_run_matrix(spec, mod)
        self.assertEqual(len(a), 5)
        va = [r['flat']['vehicle.M'] for r in a]
        vb = [r['flat']['vehicle.M'] for r in b]
        np.testing.assert_array_equal(va, vb)       # seeded -> reproducible
        self.assertGreater(np.std(va), 0.0)         # actually dispersed
        m0 = X4Dynamics().params['M']
        self.assertLess(abs(np.mean(va) - m0) / m0, 0.10)

    def test_run_one_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = run_sweep.run_one({
                'candidate': 'candidates.x4_lqg', 'run_id': 0,
                'overrides': {'config': {'tf': 1.0, 'pass_criteria': {}}},
                'flat': {'config.tf': 1.0}, 'run_dir': tmp})
        self.assertEqual(row['verdict'], 'COMPLETE')
        self.assertAlmostEqual(row['t_end'], 1.0, places=6)
        self.assertIn('alt_m', row)

    def test_run_one_survives_bad_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = run_sweep.run_one({
                'candidate': 'candidates.does_not_exist', 'run_id': 1,
                'overrides': {}, 'flat': {}, 'run_dir': tmp})
        self.assertEqual(row['verdict'], 'ERROR')


class TestAnalysis(unittest.TestCase):

    def test_x4_hover_trim_matches_analytic(self):
        dyn = X4Dynamics()
        res = trim(dyn, dyn.initial_state(), np.full(4, U_e),
                   free_states=['w1', 'w2', 'w3', 'w4'],
                   free_controls=['m1', 'm2', 'm3', 'm4'],
                   residual_states=['xdot', 'ydot', 'zdot', 'p', 'q_ang', 'r',
                                    'w1', 'w2', 'w3', 'w4'])
        self.assertTrue(res.converged)
        np.testing.assert_allclose(res.X[13:17], W_e, rtol=1e-8)
        np.testing.assert_allclose(res.U, U_e, rtol=1e-8)

    def test_x4_linearization_structure(self):
        dyn = X4Dynamics()
        X0 = dyn.initial_state()
        U0 = np.full(4, U_e)
        A, B, f0 = linearize(dyn, X0, U0)
        self.assertEqual(A.shape, (17, 17))
        self.assertEqual(B.shape, (17, 4))
        np.testing.assert_allclose(f0, 0.0, atol=1e-7)   # hover is equilibrium
        p = dyn.params
        # analytically known entries
        self.assertAlmostEqual(A[1, 1], -p['Dxx'] / p['M'], places=6)
        self.assertAlmostEqual(A[13, 13], -1.0 / p['Mtau'], places=4)
        self.assertAlmostEqual(B[13, 0], p['Ku'], places=4)

    def test_x4_hover_modes_all_stable_or_neutral(self):
        dyn = X4Dynamics()
        A, _, _ = linearize(dyn, dyn.initial_state(), np.full(4, U_e))
        modes = eig_report(A)
        self.assertEqual(sum(m['unstable'] for m in modes), 0)

    def test_trim_bounds_respected(self):
        dyn = X4Dynamics()
        res = trim(dyn, dyn.initial_state(), np.full(4, U_e),
                   free_states=['w1', 'w2', 'w3', 'w4'],
                   free_controls=['m1', 'm2', 'm3', 'm4'],
                   residual_states=['xdot', 'ydot', 'zdot', 'p', 'q_ang', 'r',
                                    'w1', 'w2', 'w3', 'w4'],
                   bounds={'m1': (0.0, 100.0)})       # too little authority
        self.assertLessEqual(res.U[0], 100.0 + 1e-9)
        self.assertFalse(res.converged)               # can't hover at 100 cmd

    def test_unknown_name_raises(self):
        dyn = X4Dynamics()
        with self.assertRaises(ValueError):
            trim(dyn, dyn.initial_state(), np.zeros(4),
                 free_states=['nope'], free_controls=[],
                 residual_states=['xdot'])


if __name__ == '__main__':
    unittest.main()
