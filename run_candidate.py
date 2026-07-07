#!/usr/bin/env python3
"""
Generic simulation entry point.

Usage:
  python run_candidate.py [candidate_module] [--show]

  candidate_module  dotted module path under candidates/
                    default: candidates.spearhead_vtol
  --show            open interactive plot windows (default: headless — figures
                    are saved as PNGs next to the run's log files)

Examples:
  python run_candidate.py
  python run_candidate.py candidates.x4_lqg
  python run_candidate.py candidates.spearhead_vtol --show

A candidate module must expose:
  build()  -> (dynamics, controller, config)
  plot(X_hist, U_hist, config, show=True) -> list[Figure]  [optional]

Exit codes: 0 = PASS/COMPLETE, 2 = FAIL, 3 = CRASHED/DEPARTED/DIVERGED.

Run from the repo root so that package imports resolve correctly.
"""
import argparse
import importlib
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('candidate', nargs='?', default='candidates.spearhead_vtol',
                    help='dotted candidate module (default: %(default)s)')
    ap.add_argument('--show', action='store_true',
                    help='open interactive plot windows instead of running headless')
    args = ap.parse_args()

    # Headless by default: force a non-interactive backend BEFORE the candidate
    # module imports matplotlib, so runs work without a display.
    if not args.show:
        os.environ.setdefault('MPLBACKEND', 'Agg')

    from sim.runner import SimRunner   # after backend env is settled

    candidate = importlib.import_module(args.candidate)

    dynamics, controller, config = candidate.build()
    result = SimRunner(dynamics, controller, config).run()

    if hasattr(candidate, 'plot'):
        figs = candidate.plot(result.x_hist, result.u_hist, config,
                              show=args.show) or []
        base = os.path.splitext(result.log_path)[0]
        for k, fig in enumerate(figs, start=1):
            png = f'{base}_fig{k}.png'
            fig.savefig(png, dpi=130, bbox_inches='tight')
            print(f'[FIG] {png}')

    sys.exit(0 if result.passed else
             2 if result.verdict == 'FAIL' else 3)


if __name__ == '__main__':
    main()
