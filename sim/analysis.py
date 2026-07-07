"""Trim and linearization tools.

Works on any duck-typed VehicleDynamics (state_names / control_names /
derivatives). Typical flow:

    X0, U0, res = trim(dyn, X0_guess, U0_guess,
                       free_states=['w1','w2','w3','w4'],
                       free_controls=['m1','m2','m3','m4'],
                       residual_states=['xdot','ydot','zdot','p','q_ang','r',
                                        'w1','w2','w3','w4'])
    A, B, f0 = linearize(dyn, X0, U0)
    for m in eig_report(A):
        print(format_mode(m))

Candidates may expose trim_specs() -> {name: spec-dict} consumed by
analyze_candidate.py; spec keys: X0, U0, free_states, free_controls,
residual_states, and optional quat_states (unit-norm constraint appended
automatically when any of them is free).
"""
import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Trim
# ---------------------------------------------------------------------------

class TrimResult:
    def __init__(self, X, U, converged, resnorm, message):
        self.X, self.U = X, U
        self.converged, self.resnorm, self.message = converged, resnorm, message

    def __repr__(self):
        return (f'TrimResult(converged={self.converged}, '
                f'resnorm={self.resnorm:.3e})')


def _idx(names, wanted, kind):
    out = []
    for n in wanted:
        if n not in names:
            raise ValueError(f'unknown {kind} name {n!r} (have {names})')
        out.append(names.index(n))
    return out


def trim(dyn, X0, U0, free_states, free_controls, residual_states,
         quat_states=None, bounds=None, tol=1e-12):
    """Find (X*, U*) such that dX/dt is zero on the residual states.

    free_states / free_controls: names the solver may vary.
    residual_states: names whose derivative must be driven to zero.
    quat_states: names of the quaternion elements; if any is free, a
      ||q|| - 1 residual is appended so trim stays on the unit sphere.
    bounds: optional {name: (lo, hi)} for free states/controls — use to keep
      the solver physical (e.g. rotor speeds >= 0).
    """
    fx = _idx(dyn.state_names, free_states, 'state')
    fu = _idx(dyn.control_names, free_controls, 'control')
    ri = _idx(dyn.state_names, residual_states, 'state')

    qi = _idx(dyn.state_names, quat_states or [], 'state')
    quat_constrained = bool(set(qi) & set(fx))

    X0 = np.asarray(X0, dtype=float).copy()
    U0 = np.asarray(U0, dtype=float).copy()

    def unpack(z):
        X = X0.copy()
        U = U0.copy()
        X[fx] = z[:len(fx)]
        U[fu] = z[len(fx):]
        return X, U

    def residual(z):
        X, U = unpack(z)
        r = dyn.derivatives(0.0, X, U)[ri]
        if quat_constrained:
            r = np.append(r, np.linalg.norm(X[qi]) - 1.0)
        return r

    z_names = list(free_states) + list(free_controls)
    lo = np.full(len(z_names), -np.inf)
    hi = np.full(len(z_names), np.inf)
    for k, (l, h) in (bounds or {}).items():
        for j, n in enumerate(z_names):
            if n == k:
                lo[j], hi[j] = l, h

    z0 = np.clip(np.concatenate([X0[fx], U0[fu]]), lo, hi)
    sol = least_squares(residual, z0, method='trf', bounds=(lo, hi),
                        xtol=1e-14, ftol=1e-14, gtol=1e-14)
    Xs, Us = unpack(sol.x)
    resnorm = float(np.linalg.norm(residual(sol.x)))
    return TrimResult(Xs, Us, resnorm < np.sqrt(tol), resnorm, sol.message)


# ---------------------------------------------------------------------------
# Linearization
# ---------------------------------------------------------------------------

def linearize(dyn, X0, U0, eps=1e-6):
    """Central-difference Jacobians about (X0, U0): returns (A, B, f0).

    A = df/dX (n x n), B = df/dU (n x m). Step per channel is scaled by the
    operating-point magnitude so fast states (rotor speeds ~1e3) and slow
    states (quaternions ~1) are both differentiated sensibly.
    """
    X0 = np.asarray(X0, dtype=float)
    U0 = np.asarray(U0, dtype=float)
    n, m = len(X0), len(U0)
    f0 = np.asarray(dyn.derivatives(0.0, X0, U0), dtype=float)

    A = np.zeros((n, n))
    for j in range(n):
        h = eps * max(1.0, abs(X0[j]))
        Xp, Xm = X0.copy(), X0.copy()
        Xp[j] += h
        Xm[j] -= h
        A[:, j] = (dyn.derivatives(0.0, Xp, U0)
                   - dyn.derivatives(0.0, Xm, U0)) / (2 * h)

    B = np.zeros((n, m))
    for j in range(m):
        h = eps * max(1.0, abs(U0[j]))
        Up, Um = U0.copy(), U0.copy()
        Up[j] += h
        Um[j] -= h
        B[:, j] = (dyn.derivatives(0.0, X0, Up)
                   - dyn.derivatives(0.0, X0, Um)) / (2 * h)
    return A, B, f0


# ---------------------------------------------------------------------------
# Eigen reporting
# ---------------------------------------------------------------------------

def eig_report(A, tol_im=1e-9):
    """Eigenvalues of A as mode dicts, most unstable first.

    Complex pairs are reported once with wn [rad/s] and damping ratio zeta;
    real modes with their time constant (stable) or doubling time (unstable).
    """
    vals = np.linalg.eigvals(A)
    modes = []
    seen = set()
    for i, lam in enumerate(vals):
        if i in seen:
            continue
        re, im = float(lam.real), float(lam.imag)
        if abs(im) > tol_im:
            # find and consume the conjugate partner
            for j in range(i + 1, len(vals)):
                if j not in seen and abs(vals[j] - lam.conjugate()) < 1e-8:
                    seen.add(j)
                    break
            wn = float(abs(lam))
            modes.append({'re': re, 'im': abs(im), 'type': 'oscillatory',
                          'wn_rad_s': wn, 'zeta': -re / wn if wn > 0 else 0.0,
                          'unstable': re > tol_im})
        else:
            m = {'re': re, 'im': 0.0, 'type': 'real', 'unstable': re > tol_im}
            if re > tol_im:
                m['doubling_time_s'] = np.log(2) / re
            elif re < -tol_im:
                m['time_constant_s'] = -1.0 / re
            modes.append(m)
    modes.sort(key=lambda m: -m['re'])
    return modes


def format_mode(m):
    if m['type'] == 'oscillatory':
        s = (f"  {'UNSTABLE' if m['unstable'] else 'stable':<9} "
             f"osc   re={m['re']:+10.4f}  im=±{m['im']:8.4f}  "
             f"wn={m['wn_rad_s']:8.4f} rad/s  zeta={m['zeta']:+7.4f}")
    else:
        extra = (f"T2={m['doubling_time_s']:.3f}s" if m['unstable'] else
                 f"tau={m['time_constant_s']:.3f}s" if 'time_constant_s' in m
                 else 'neutral')
        s = (f"  {'UNSTABLE' if m['unstable'] else 'stable':<9} "
             f"real  re={m['re']:+10.4f}  {extra}")
    return s
