#!/usr/bin/env python3
"""
Batch / sweep / Monte Carlo runner.

Usage:
  python run_sweep.py spec.yaml [-j N] [--dry-run]

Spec file (YAML or JSON):

  candidate: candidates.x4_lqg
  name: mass_wind_study            # optional — sweep directory name
  base:                            # overrides applied to every run (optional)
    config: {tf: 30.0}
  grid:                            # cartesian product over listed values
    vehicle.M: [0.80, 0.86, 0.94]
    config.wind: [null, {constant_ned: [5, 0, 0]}]
  monte_carlo:                     # n seeded draws per grid point (optional)
    n: 25
    seed: 1
    dispersions:
      vehicle.M:       {type: normal_pct,  sigma_pct: 5}
      vehicle.Kthrust: {type: uniform_pct, pct: 10}

Dotted keys address the candidate's build(overrides) dict: 'vehicle.M' ->
overrides['vehicle']['M'], 'config.tf' -> overrides['config']['tf'].
Dispersion types: normal_pct (sigma as % of nominal), uniform_pct (+/- % of
nominal), normal_abs {sigma}, uniform_abs {lo, hi}. Percent types require the
nominal to be resolvable from the vehicle's params.

Outputs one directory logs/sweeps/<name>_<timestamp>/ containing:
  run_0000/ ... per-run flight_*.{log,csv,json}
  summary.csv   one row per run: overrides, verdict, t_end, final metrics
  sweep.json    spec + verdict histogram + pass rate

Exit code 0 if every run PASS/COMPLETE, 2 otherwise.
"""
import argparse
import concurrent.futures as cf
import copy
import csv
import datetime
import importlib
import itertools
import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Spec handling
# ---------------------------------------------------------------------------

def load_spec(path):
    with open(path) as f:
        text = f.read()
    if path.endswith(('.yaml', '.yml')):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def set_dotted(d, dotted, value):
    """set_dotted({}, 'vehicle.M', 0.9) -> {'vehicle': {'M': 0.9}}"""
    keys = dotted.split('.')
    # config.wind is a single override value (possibly a dict), not a path
    # into the wind schema — treat everything after the first two levels as
    # nested dict path anyway; two levels cover all current uses.
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
    return d


def _nominal_for(candidate_mod, dotted):
    """Resolve a nominal value for percent-type dispersions."""
    parts = dotted.split('.')
    if parts[0] == 'vehicle':
        dyn, _, _ = candidate_mod.build()
        val = dyn.params
        for k in parts[1:]:
            val = val[k] if isinstance(val, dict) else val[int(k)]
        return float(val)
    raise ValueError(f'cannot resolve nominal for {dotted!r} — use an '
                     f'absolute dispersion type for non-vehicle keys')


def draw_dispersions(dispersions, nominals, rng):
    out = {}
    for key, spec in dispersions.items():
        typ = spec['type']
        if typ == 'normal_pct':
            out[key] = nominals[key] * (1 + rng.normal() * spec['sigma_pct'] / 100.0)
        elif typ == 'uniform_pct':
            out[key] = nominals[key] * (1 + rng.uniform(-1, 1) * spec['pct'] / 100.0)
        elif typ == 'normal_abs':
            out[key] = spec.get('mean', nominals.get(key, 0.0)) + rng.normal() * spec['sigma']
        elif typ == 'uniform_abs':
            out[key] = rng.uniform(spec['lo'], spec['hi'])
        else:
            raise ValueError(f'unknown dispersion type {typ!r}')
    return out


def build_run_matrix(spec, candidate_mod):
    """Expand grid x monte_carlo into a list of (run_id, overrides-dict)."""
    base = spec.get('base', {}) or {}

    grid = spec.get('grid', {}) or {}
    grid_keys = list(grid.keys())
    grid_points = (list(itertools.product(*(grid[k] for k in grid_keys)))
                   if grid_keys else [()])

    mc = spec.get('monte_carlo')
    draws_per_point = 1
    dispersions, nominals, rng = {}, {}, None
    if mc:
        draws_per_point = int(mc['n'])
        dispersions = mc.get('dispersions', {})
        rng = np.random.default_rng(mc.get('seed', 0))
        pct_types = ('normal_pct', 'uniform_pct')
        nominals = {k: _nominal_for(candidate_mod, k)
                    for k, s in dispersions.items() if s['type'] in pct_types}

    runs = []
    rid = 0
    for point in grid_points:
        for _ in range(draws_per_point):
            ov = copy.deepcopy(base)
            flat = {}
            for k, v in zip(grid_keys, point):
                set_dotted(ov, k, v)
                flat[k] = v
            if mc:
                for k, v in draw_dispersions(dispersions, nominals, rng).items():
                    set_dotted(ov, k, v)
                    flat[k] = v
            runs.append({'run_id': rid, 'overrides': ov, 'flat': flat})
            rid += 1
    return runs


# ---------------------------------------------------------------------------
# Worker (top-level: must be picklable)
# ---------------------------------------------------------------------------

def run_one(task):
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from sim.runner import SimRunner

    row = {'run_id': task['run_id']}
    for k, v in task['flat'].items():
        row[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    try:
        mod = importlib.import_module(task['candidate'])
        dynamics, controller, config = mod.build(task['overrides'])
        config.log_dir = task['run_dir']          # absolute path
        result = SimRunner(dynamics, controller, config).run()
        row.update(verdict=result.verdict, reason=result.reason,
                   t_end=result.t_end)
        row.update({k: v for k, v in result.metrics.items()})
    except Exception as e:                        # a failed run must not kill the sweep
        row.update(verdict='ERROR', reason=f'{type(e).__name__}: {e}',
                   t_end=float('nan'))
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='sweep / Monte Carlo batch runner')
    ap.add_argument('spec', help='YAML or JSON sweep spec')
    ap.add_argument('-j', '--jobs', type=int, default=max(1, os.cpu_count() - 1),
                    help='parallel workers (default: %(default)s)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the run matrix and exit')
    args = ap.parse_args()

    spec = load_spec(args.spec)
    candidate = spec['candidate']
    candidate_mod = importlib.import_module(candidate)
    runs = build_run_matrix(spec, candidate_mod)

    print(f'[SWEEP] {candidate}: {len(runs)} runs, {args.jobs} workers')
    if args.dry_run:
        for r in runs:
            print(f"  run_{r['run_id']:04d}  {r['flat'] or '(base)'}")
        return

    root = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    name = spec.get('name', 'sweep')
    sweep_dir = os.path.join(root, 'logs', 'sweeps', f'{name}_{ts}')
    os.makedirs(sweep_dir, exist_ok=True)

    tasks = []
    for r in runs:
        run_dir = os.path.join(sweep_dir, f"run_{r['run_id']:04d}")
        os.makedirs(run_dir, exist_ok=True)
        tasks.append({'candidate': candidate, 'run_id': r['run_id'],
                      'overrides': r['overrides'], 'flat': r['flat'],
                      'run_dir': run_dir})

    rows = []
    with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(run_one, t): t['run_id'] for t in tasks}
        for fut in cf.as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(f"  run_{row['run_id']:04d}  {row['verdict']:<9} "
                  f"t_end={row.get('t_end', float('nan')):7.2f}  "
                  f"{row.get('reason', '')[:60]}")
    rows.sort(key=lambda r: r['run_id'])

    # summary.csv — union of keys, stable order
    cols = ['run_id', 'verdict', 'reason', 't_end']
    for row in rows:
        cols += [k for k in row if k not in cols]
    csv_path = os.path.join(sweep_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    counts = {}
    for row in rows:
        counts[row['verdict']] = counts.get(row['verdict'], 0) + 1
    n_ok = counts.get('PASS', 0) + counts.get('COMPLETE', 0)
    with open(os.path.join(sweep_dir, 'sweep.json'), 'w') as f:
        json.dump({'spec': spec, 'n_runs': len(rows), 'verdicts': counts,
                   'pass_rate': n_ok / len(rows) if rows else None,
                   'summary_csv': csv_path}, f, indent=2)

    print(f'\n[SWEEP SUMMARY] {counts}  pass rate = {n_ok}/{len(rows)}')
    print(f'[CSV] {csv_path}')
    sys.exit(0 if n_ok == len(rows) else 2)


if __name__ == '__main__':
    main()
