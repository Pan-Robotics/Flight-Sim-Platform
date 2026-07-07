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

    @unittest.expectedFailure
    def test_yaw_90_maps_north_to_east(self):
        # KNOWN ISSUE: quat_kinematics and quat_to_euler follow the standard
        # body->NED convention, but rotate_body_to_inertial /
        # rotate_inertial_to_body are inverted relative to it (they return the
        # conjugate rotation). The Spearhead model uses the swapped pair
        # self-consistently, so it flies, but its world-frame trajectory is
        # mirrored w.r.t. its reported attitude. X4 hand-codes its rotations
        # (standard) and is unaffected. Fixing this flips Spearhead
        # trajectories, so it must land as its own golden-gated change.
        c = np.cos(np.pi / 4)
        q = np.array([c, 0, 0, c])          # body yawed +90 deg
        # body x-axis expressed in inertial frame -> East
        v_i = rotate_body_to_inertial(np.array([1.0, 0, 0]), q)
        np.testing.assert_allclose(v_i, [0, 1, 0], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
