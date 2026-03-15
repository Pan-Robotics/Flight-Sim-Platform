#!/usr/bin/env python3
"""
Generic simulation entry point.

Usage:
  python run_candidate.py [candidate_module]

  candidate_module  dotted module path under candidates/
                    default: candidates.spearhead_vtol

Examples:
  python run_candidate.py
  python run_candidate.py candidates.spearhead_vtol

A candidate module must expose:
  build()  -> (dynamics, controller, config)
  plot(X_hist, U_hist, config)  [optional]

Run from the python/ directory so that package imports resolve correctly.
"""
import sys
import importlib

from sim.runner import SimRunner


def main():
    module_name = sys.argv[1] if len(sys.argv) > 1 else 'candidates.spearhead_vtol'
    candidate   = importlib.import_module(module_name)

    dynamics, controller, config = candidate.build()
    runner = SimRunner(dynamics, controller, config)
    X_hist, U_hist, log_path, csv_path = runner.run()

    if hasattr(candidate, 'plot'):
        candidate.plot(X_hist, U_hist, config)


if __name__ == '__main__':
    main()
