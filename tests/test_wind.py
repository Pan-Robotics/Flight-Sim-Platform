"""Tests for the toggle-able wind/disturbance model."""
import unittest
import numpy as np

from sim.wind import WindModel
from vehicles.x4.dynamics import X4Dynamics
from vehicles.spearhead.dynamics import SpearheadDynamics


class TestWindModel(unittest.TestCase):

    def test_constant_only(self):
        w = WindModel({'constant_ned': [5.0, -2.0, 0.5]}, dt=0.001)
        for _ in range(3):
            np.testing.assert_allclose(w.step(), [5.0, -2.0, 0.5])

    def test_dryden_requires_seed(self):
        with self.assertRaises(ValueError):
            WindModel({'dryden': {'V': 20.0}}, dt=0.001)

    def test_dryden_seeded_reproducible(self):
        spec = {'dryden': {'V': 20.0, 'sigma': [1.0, 1.0, 0.5],
                           'L': [200.0, 200.0, 50.0]}, 'seed': 7}
        a = WindModel(spec, dt=0.001)
        b = WindModel(spec, dt=0.001)
        for _ in range(100):
            np.testing.assert_array_equal(a.step(), b.step())

    def test_dryden_seed_changes_history(self):
        base = {'dryden': {'V': 20.0}, 'seed': 1}
        other = {'dryden': {'V': 20.0}, 'seed': 2}
        a = WindModel(base, dt=0.001)
        b = WindModel(other, dt=0.001)
        seq_a = np.array([a.step().copy() for _ in range(50)])
        seq_b = np.array([b.step().copy() for _ in range(50)])
        self.assertFalse(np.allclose(seq_a, seq_b))

    def test_dryden_statistics(self):
        # Long-run std of the gust filter must approach configured sigma.
        # Tolerance is statistical: with phi ~ 0.999 the correlation time is
        # ~1e3 steps, so 2e5 steps is only a few hundred independent samples.
        spec = {'dryden': {'V': 20.0, 'sigma': [2.0, 2.0, 1.0],
                           'L': [200.0, 200.0, 50.0]}, 'seed': 3}
        w = WindModel(spec, dt=0.01)
        hist = np.array([w.step().copy() for _ in range(200000)])
        np.testing.assert_allclose(hist.std(axis=0), [2.0, 2.0, 1.0], rtol=0.10)

    def test_bad_specs_rejected(self):
        with self.assertRaises(ValueError):
            WindModel({'constant_ned': [1.0, 2.0]}, dt=0.001)
        with self.assertRaises(ValueError):
            WindModel({'dryden': {'V': -1.0}, 'seed': 1}, dt=0.001)
        with self.assertRaises(TypeError):
            WindModel('5 m/s north', dt=0.001)


class TestVehicleWindCoupling(unittest.TestCase):

    def test_x4_wind_off_is_bitexact(self):
        d = X4Dynamics()
        X = d.initial_state()
        U = np.full(4, 300.0)
        dx0 = d.derivatives(0.0, X, U)
        d.set_wind_ned([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(d.derivatives(0.0, X, U), dx0)

    def test_x4_drag_uses_air_relative_velocity(self):
        d = X4Dynamics()
        X = d.initial_state()
        U = np.full(4, 300.0)
        dx0 = d.derivatives(0.0, X, U)
        d.set_wind_ned([10.0, 0.0, 0.0])
        dx1 = d.derivatives(0.0, X, U)
        p = d.params
        self.assertAlmostEqual(dx1[1] - dx0[1], p['Dxx'] / p['M'] * 10.0,
                               places=10)

    def test_spearhead_headwind_changes_aero(self):
        s = SpearheadDynamics()
        X = s.initial_state()
        X[0] = 20.0                    # flying north at 20 m/s
        U = np.zeros(8)
        f0 = s.derivatives(0.0, X, U).copy()
        s.set_wind_ned([-10.0, 0.0, 0.0])   # 10 m/s headwind -> V_air = 30
        f1 = s.derivatives(0.0, X, U)
        self.assertFalse(np.allclose(f0[0:3], f1[0:3]))

    def test_spearhead_envelope_uses_air_relative_flow(self):
        s = SpearheadDynamics()
        X = s.initial_state()
        X[0] = 20.0                    # alpha = 0 over ground
        self.assertEqual(s.envelope_violations(X), [])
        s.set_wind_ned([0.0, 0.0, -15.0])   # strong updraft -> large alpha_air
        self.assertTrue(any('alpha' in v for v in s.envelope_violations(X)))


if __name__ == '__main__':
    unittest.main()
