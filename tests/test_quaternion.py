"""Unit tests for sim.quaternion."""
import unittest
import numpy as np

from sim.quaternion import (
    normalize_quaternion, quat_to_euler, quat_kinematics, quat_multiply,
    rotate_inertial_to_body, rotate_body_to_inertial,
)


class TestQuaternion(unittest.TestCase):

    def test_normalize_unit(self):
        q = normalize_quaternion(np.array([2.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(q, [1, 0, 0, 0])

    def test_normalize_degenerate_returns_identity(self):
        q = normalize_quaternion(np.zeros(4))
        np.testing.assert_allclose(q, [1, 0, 0, 0])

    def test_identity_euler(self):
        np.testing.assert_allclose(quat_to_euler(np.array([1, 0, 0, 0])),
                                   [0, 0, 0], atol=1e-12)

    def test_known_rotations(self):
        # 90 deg yaw: q = [cos45, 0, 0, sin45]
        c = np.cos(np.pi / 4)
        phi, theta, psi = quat_to_euler(np.array([c, 0, 0, c]))
        np.testing.assert_allclose([phi, theta, psi], [0, 0, np.pi / 2],
                                   atol=1e-12)
        # 90 deg roll
        phi, theta, psi = quat_to_euler(np.array([c, c, 0, 0]))
        np.testing.assert_allclose([phi, theta, psi], [np.pi / 2, 0, 0],
                                   atol=1e-12)

    def test_multiply_identity(self):
        q = normalize_quaternion(np.array([0.9, 0.1, -0.2, 0.3]))
        np.testing.assert_allclose(quat_multiply(np.array([1, 0, 0, 0]), q), q,
                                   atol=1e-12)

    def test_rotations_are_inverses(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            q = normalize_quaternion(rng.normal(size=4))
            v = rng.normal(size=3)
            v2 = rotate_body_to_inertial(rotate_inertial_to_body(v, q), q)
            np.testing.assert_allclose(v2, v, atol=1e-10)

    def test_rotation_preserves_norm(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            q = normalize_quaternion(rng.normal(size=4))
            v = rng.normal(size=3)
            self.assertAlmostEqual(np.linalg.norm(rotate_inertial_to_body(v, q)),
                                   np.linalg.norm(v), places=10)

    def test_kinematics_preserves_norm_to_first_order(self):
        # qdot must be orthogonal to q (norm-preserving flow)
        q = normalize_quaternion(np.array([0.8, 0.2, -0.3, 0.4]))
        qd = quat_kinematics(q, 0.3, -0.5, 0.7)
        self.assertAlmostEqual(float(np.dot(q, qd)), 0.0, places=12)

    def test_yaw_90_maps_north_to_east(self):
        # Guards the body->NED quaternion convention: the rotate_* helpers
        # must agree in direction with quat_kinematics and quat_to_euler.
        # (They were crossed before 2026-07 — see sim/quaternion.py header.)
        c = np.cos(np.pi / 4)
        q = np.array([c, 0, 0, c])          # body yawed +90 deg (nose East)
        # body x-axis expressed in inertial frame -> East
        v_i = rotate_body_to_inertial(np.array([1.0, 0, 0]), q)
        np.testing.assert_allclose(v_i, [0, 1, 0], atol=1e-12)
        # and North expressed in the body frame -> -y (left wing)
        v_b = rotate_inertial_to_body(np.array([1.0, 0, 0]), q)
        np.testing.assert_allclose(v_b, [0, -1, 0], atol=1e-12)

    def test_kinematics_euler_helpers_agree_in_direction(self):
        # Integrate a pure yaw rate and require euler extraction and the
        # rotation helper to report the SAME heading — the invariant that was
        # broken by the crossed helpers.
        q = np.array([1.0, 0, 0, 0])
        dt = 1e-4
        for _ in range(5000):                       # r = +1 rad/s for 0.5 s
            q = normalize_quaternion(q + dt * quat_kinematics(q, 0, 0, 1.0))
        psi_euler = quat_to_euler(q)[2]
        nose_i = rotate_body_to_inertial(np.array([1.0, 0, 0]), q)
        psi_helper = np.arctan2(nose_i[1], nose_i[0])
        self.assertAlmostEqual(psi_euler, psi_helper, places=6)
        self.assertGreater(psi_euler, 0.4)          # ~+0.5 rad, right-handed


if __name__ == '__main__':
    unittest.main()
