"""Quaternion utilities shared across vehicle models.

Convention: q = [q0, q1, q2, q3] (Hamilton, scalar-first) is the body->NED
attitude quaternion — v_NED = q (x) v_body (x) q*.  This matches
quat_kinematics (qdot = 0.5 q (x) [0, omega_body]) and quat_to_euler
(standard ZYX extraction), and the hand-coded rotation terms in the X4 model.

History: prior to 2026-07 the two rotate_* helpers implemented the opposite
(NED->body) convention while keeping these names, i.e. their sandwich
products were crossed relative to the kinematics.  The Spearhead model used
the crossed pair self-consistently, which made its world-frame response
correspond to the conjugate of its reported attitude.  Fixed by swapping the
implementations; regression-gated by tests/test_quaternion.py and the golden
runs in tests/test_golden.py.
"""
import numpy as np


def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_to_euler(q):
    """Returns (phi, theta, psi) in radians from unit quaternion [q0,q1,q2,q3]."""
    q0, q1, q2, q3 = q
    sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    phi = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (q0 * q2 - q3 * q1)
    theta = np.sign(sinp) * (np.pi / 2.0) if abs(sinp) >= 1.0 else np.arcsin(sinp)
    siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    psi = np.arctan2(siny_cosp, cosy_cosp)
    return phi, theta, psi


def quat_kinematics(q, p, q_omega, r):
    """Quaternion derivative from body rates p, q_omega, r."""
    Omega = np.array([
        [ 0.0,    -p,  -q_omega, -r   ],
        [ p,       0.0,  r,      -q_omega],
        [ q_omega, -r,   0.0,     p   ],
        [ r,       q_omega, -p,   0.0 ],
    ])
    return 0.5 * (Omega @ q)


def quat_multiply(qA, qB):
    w1, x1, y1, z1 = qA
    w2, x2, y2, z2 = qB
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def rotate_inertial_to_body(v, q):
    """v_body = q* (x) v_NED (x) q   (q is the body->NED attitude quaternion)."""
    wv = np.array([0.0, v[0], v[1], v[2]])
    qc = np.array([q[0], -q[1], -q[2], -q[3]])
    return quat_multiply(quat_multiply(qc, wv), q)[1:]


def rotate_body_to_inertial(v, q):
    """v_NED = q (x) v_body (x) q*   (q is the body->NED attitude quaternion)."""
    wv = np.array([0.0, v[0], v[1], v[2]])
    qc = np.array([q[0], -q[1], -q[2], -q[3]])
    return quat_multiply(quat_multiply(q, wv), qc)[1:]
