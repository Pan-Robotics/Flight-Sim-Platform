"""Unit tests for sim.pid."""
import unittest

from sim.pid import PID


class TestPID(unittest.TestCase):

    def test_proportional_only(self):
        pid = PID(Kp=2.0, Ki=0.0, Kd=0.0, dt=0.01)
        self.assertAlmostEqual(pid.update(3.0), 6.0)

    def test_integral_accumulates(self):
        pid = PID(Kp=0.0, Ki=1.0, Kd=0.0, dt=0.5)
        pid.update(1.0)
        out = pid.update(1.0)
        self.assertAlmostEqual(out, 1.0)     # integral = 2 steps * 0.5 s * 1.0

    def test_integral_clamps_at_limit(self):
        pid = PID(Kp=0.0, Ki=1.0, Kd=0.0, dt=1.0, integral_limit=2.0)
        for _ in range(10):
            out = pid.update(5.0)
        self.assertAlmostEqual(out, 2.0)     # Ki * clamped accumulator
        self.assertAlmostEqual(pid.integral, 2.0)

    def test_derivative(self):
        pid = PID(Kp=0.0, Ki=0.0, Kd=1.0, dt=0.1)
        pid.update(1.0)
        self.assertAlmostEqual(pid.update(2.0), 10.0)   # (2-1)/0.1

    def test_reset(self):
        pid = PID(Kp=1.0, Ki=1.0, Kd=1.0, dt=0.1)
        pid.update(4.0)
        pid.reset()
        self.assertEqual(pid.integral, 0.0)
        self.assertEqual(pid.prev_error, 0.0)


if __name__ == '__main__':
    unittest.main()
