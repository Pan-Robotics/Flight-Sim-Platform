"""Unit tests for the Spearhead aero table lookup."""
import unittest
import numpy as np

from vehicles.spearhead.dynamics import SpearheadDynamics, _aero_coeffs


class TestAeroLookup(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dyn = SpearheadDynamics()

    def _coeffs(self, alpha, beta):
        return _aero_coeffs(alpha, beta, self.dyn._adb, self.dyn._beta_breaks)

    def test_continuous_across_beta_breakpoints(self):
        # Regression for the nearest-neighbour step discontinuity.
        for bp in (-10.0, -4.0, 0.0, 4.0, 10.0):
            below = self._coeffs(5.0, bp - 1e-4)
            above = self._coeffs(5.0, bp + 1e-4)
            np.testing.assert_allclose(below, above, atol=1e-3,
                                       err_msg=f'discontinuity at beta={bp}')

    def test_exact_at_breakpoint_matches_table(self):
        # At an exact breakpoint the interpolation must return that table row.
        alpha = 3.0
        apoly = np.array([np.radians(alpha) ** p for p in range(9)])
        for bp_idx in (26, 28, 30):     # beta = -4, 0, +4
            expected = apoly @ self.dyn._adb[bp_idx, :, :]
            got = self._coeffs(alpha, float(self.dyn._beta_breaks[bp_idx]))
            np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_alpha_clipped_beyond_30(self):
        np.testing.assert_allclose(self._coeffs(45.0, 0.0),
                                   self._coeffs(30.0, 0.0), atol=1e-12)

    def test_beta_clipped_beyond_30(self):
        np.testing.assert_allclose(self._coeffs(5.0, 60.0),
                                   self._coeffs(5.0, 30.0), atol=1e-12)

    def test_envelope_flags_high_alpha(self):
        X = np.zeros(21)
        X[0], X[2] = 5.0, 8.0            # u=5, w=8 -> alpha = 58 deg, V ~ 9.4
        v = self.dyn.envelope_violations(X)
        self.assertTrue(any('alpha' in s for s in v))

    def test_envelope_silent_at_low_speed(self):
        X = np.zeros(21)
        X[0], X[2] = 0.5, 1.0            # extreme alpha but V ~ 1.1 m/s
        self.assertEqual(self.dyn.envelope_violations(X), [])

    def test_envelope_clean_in_cruise(self):
        X = np.zeros(21)
        X[0], X[2] = 50.0, 2.0           # alpha ~ 2.3 deg
        self.assertEqual(self.dyn.envelope_violations(X), [])


if __name__ == '__main__':
    unittest.main()
