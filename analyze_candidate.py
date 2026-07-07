#!/usr/bin/env python3
"""
Trim + linearization analysis of a candidate.

Usage:
  python analyze_candidate.py [candidate_module] [--condition NAME] [--json OUT]

For each trim condition the candidate declares (via trim_specs(dynamics)):
  1. solve the trim (scipy least-squares on the specified residual states)
  2. linearize the plant about the trim point (central differences)
  3. report the eigenvalues — instabilities, damping, frequencies

This answers "is this flight condition an equilibrium, and is it stable?"
in milliseconds instead of watching a 180 s simulation diverge.

Exit code 0 if all requested trims converged, 2 otherwise.
"""
import argparse
import importlib
import json
import sys

import numpy as np

from sim.analysis import trim, linearize, eig_report, format_mode


def main():
    ap = argparse.ArgumentParser(description='trim + linearization analysis')
    ap.add_argument('candidate', nargs='?', default='candidates.spearhead_vtol')
    ap.add_argument('--condition', default=None,
                    help='trim condition name (default: all declared)')
    ap.add_argument('--json', default=None, help='write results to this file')
    args = ap.parse_args()

    mod = importlib.import_module(args.candidate)
    dyn, _, _ = mod.build()
    if not hasattr(mod, 'trim_specs'):
        print(f'{args.candidate} declares no trim_specs() — nothing to analyze')
        sys.exit(0)

    specs = mod.trim_specs(dyn)
    names = [args.condition] if args.condition else list(specs)
    all_ok = True
    doc = {}

    for name in names:
        spec = specs[name]
        print(f'\n=== {args.candidate} :: {name} ===')
        res = trim(dyn, spec['X0'], spec['U0'],
                   spec['free_states'], spec['free_controls'],
                   spec['residual_states'],
                   quat_states=spec.get('quat_states'),
                   bounds=spec.get('bounds'))
        status = 'converged' if res.converged else 'NOT CONVERGED'
        print(f'[TRIM] {status}  |residual| = {res.resnorm:.3e}')
        all_ok &= res.converged

        show = [(n, v) for n, v in zip(dyn.state_names, res.X)
                if abs(v) > 1e-6]
        print('  X*: ' + '  '.join(f'{n}={v:.4g}' for n, v in show))
        showu = [(n, v) for n, v in zip(dyn.control_names, res.U)
                 if abs(v) > 1e-6]
        print('  U*: ' + ('  '.join(f'{n}={v:.4g}' for n, v in showu) or '(all zero)'))

        A, B, f0 = linearize(dyn, res.X, res.U)
        modes = eig_report(A)
        n_unstable = sum(m['unstable'] for m in modes)
        print(f'[MODES] {len(modes)} modes, {n_unstable} unstable:')
        for m in modes:
            print(format_mode(m))

        doc[name] = {
            'converged': bool(res.converged),
            'resnorm': res.resnorm,
            'X_trim': dict(zip(dyn.state_names, map(float, res.X))),
            'U_trim': dict(zip(dyn.control_names, map(float, res.U))),
            'n_unstable': int(n_unstable),
            'modes': modes,
        }

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(doc, f, indent=2, default=float)
        print(f'\n[JSON] {args.json}')

    sys.exit(0 if all_ok else 2)


if __name__ == '__main__':
    main()
