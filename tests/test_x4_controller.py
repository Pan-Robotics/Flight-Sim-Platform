"""Unit tests for the X4 LQR+I controller (incl. anti-windup)."""
import unittest
import numpy as np

import candidates.x4_lqg as x4mod
from vehicles.x4.dynamics import X4Dynamics


def _build():
    dyn, ctl, cfg = x4mod.build()
    return dyn, ctl


class TestX4Controller(unittest.TestCase):

    def test_hover_equilibrium_near_zero_error_command(self):
        dyn, ctl = _build()
        X = dyn.initial_state()
        U, info = ctl.step(0.0, X)
        self.assertEqual(U.shape, (4,))
        self.assertTrue(np.all(np.isfinite(U)))
        self.assertIn('phase', info)

    def test_antiwindup_freezes_integrator_under_saturation(self):
        dyn, ctl = _build()
        # Huge position error -> commands guaranteed to hit the 0..800 clip
        X = dyn.initial_state()
        X[0] = 500.0                       # 500 m north of the reference
        ctl.step(0.0, X)
        Xe_after_first = ctl.Xe.copy()
        ctl.step(ctl.T_ctrl, X)            # second controller tick, same error
        U, _ = ctl.step(2 * ctl.T_ctrl, X)
        saturated = np.any((U <= 0.0) | (U >= 800.0))
        self.assertTrue(saturated, 'test premise: command must saturate')
        np.testing.assert_allclose(ctl.Xe, Xe_after_first,
                                   err_msg='integrator wound up while saturated')

    def test_integrator_active_when_unsaturated(self):
        dyn, ctl = _build()
        X = dyn.initial_state()
        # NOTE: the LQR gains are stiff — even ~1 m of position deviation
        # saturates the 0..800 clip. Use a small deviation to stay linear.
        X[4] = -0.05
        U1, _ = ctl.step(0.0, X)
        self.assertFalse(np.any((U1 <= 0.0) | (U1 >= 800.0)),
                         'test premise: command must stay unsaturated')
        Xe1 = ctl.Xe.copy()
        ctl.step(ctl.T_ctrl, X)
        self.assertFalse(np.allclose(ctl.Xe, Xe1),
                         'integrator should advance when unsaturated')

    def test_reset_clears_state(self):
        dyn, ctl = _build()
        X = dyn.initial_state()
        X[0] = 3.0
        ctl.step(0.0, X)
        ctl.reset()
        np.testing.assert_allclose(ctl.Xe, 0.0)
        self.assertIsNone(ctl._t_last)
        self.assertEqual(ctl._wp_idx, 0)


class TestX4Dynamics(unittest.TestCase):

    def test_get_position_layout(self):
        dyn = X4Dynamics()
        X = np.arange(17.0)
        np.testing.assert_allclose(dyn.get_position(X), [0.0, 2.0, 4.0])

    def test_constraints_normalize_and_clamp(self):
        dyn = X4Dynamics()
        X = np.zeros(17)
        X[6:10] = [2.0, 0, 0, 0]           # unnormalized quat
        X[4], X[5] = 1.0, 2.0              # below ground, sinking
        Xc = dyn.apply_constraints(X)
        self.assertAlmostEqual(np.linalg.norm(Xc[6:10]), 1.0)
        self.assertEqual(Xc[4], 0.0)
        self.assertEqual(Xc[5], 0.0)

    def test_terminal_condition_arms_only_after_flight(self):
        dyn = X4Dynamics()
        X = dyn.initial_state()
        X[5] = 5.0                         # fast sink but never flew
        self.assertIsNone(dyn.terminal_condition(0.0, X))
        X[4] = -2.0                        # now airborne
        dyn.terminal_condition(1.0, X)
        X[4], X[5] = 0.0, 5.0              # ground contact, hard sink
        self.assertEqual(dyn.terminal_condition(2.0, X), 'crash')

    def test_terminal_condition_departure_on_inversion(self):
        dyn = X4Dynamics()
        X = dyn.initial_state()
        X[4] = -2.0
        dyn.terminal_condition(1.0, X)     # arm
        X[6:10] = [0.0, 1.0, 0.0, 0.0]     # rolled 180 deg
        self.assertEqual(dyn.terminal_condition(2.0, X), 'departure')


if __name__ == '__main__':
    unittest.main()
