# UAV Simulation Platform

A modular, candidate-based simulation framework for validating vehicle dynamics and control systems together, before hardware. Each *candidate* pairs a vehicle dynamics model with a controller and a set of simulation parameters. The infrastructure — integrator, logger, verdict system, disturbances, sweep/Monte-Carlo runner, trim/linearization tools — is shared and vehicle-agnostic.

Every run is judged (PASS/FAIL/CRASHED/DEPARTED/DIVERGED), envelope-checked, reproducible (git commit + full config + seeds recorded), and machine-readable (JSON summary + exit codes). A golden-run regression suite pins simulation behavior; disturbances and parameter dispersion are strictly opt-in.

Two candidates are included: the Spearhead VTOL with a nested-PID controller, and the X4 quadcopter with an LQR+I full-state controller.

---

## Directory Structure

```
.
├── run_candidate.py              Run one candidate (headless by default, --show, --set overrides)
├── run_sweep.py                  Batch runner: parameter grid x Monte Carlo -> summary.csv
├── analyze_candidate.py          Trim + linearization + eigenvalue report per flight condition
│
├── sim/                          Simulation infrastructure (vehicle-agnostic)
│   ├── config.py                 SimConfig dataclass (incl. terminate_on, pass_criteria, wind)
│   ├── runner.py                 SimRunner: RK4 loop, envelope/termination, verdicts, logging
│   ├── wind.py                   Toggle-able disturbances: constant wind + seeded Dryden gusts
│   ├── analysis.py               trim(), linearize(), eig_report()
│   ├── pid.py                    Generic PID with integral anti-windup
│   └── quaternion.py             Quaternion utilities (body->NED convention; see header)
│
├── vehicles/                     One sub-package per vehicle
│   ├── spearhead/
│   │   ├── dynamics.py           SpearheadDynamics — 21-state quaternion 6-DOF VTOL
│   │   └── *.txt                 Aerodynamic database tables
│   └── x4/
│       ├── dynamics.py           X4Dynamics — 17-state quaternion quadcopter
│       └── data/                 X4 plant matrices + tuned LQR/LQG gains (*.txt)
│
├── controllers/                  One sub-package per controller type
│   ├── spearhead_vtol/
│   │   └── controller.py         SpearheadVTOLController — 4-loop nested PID
│   └── x4_lqg/
│       └── controller.py         X4LQGController — discrete-time LQR+I (full-state)
│
├── candidates/                   Wiring: vehicle + controller + SimConfig
│   ├── spearhead_vtol.py         build() + plot() + trim_specs() for Spearhead VTOL
│   └── x4_lqg.py                 build() + plot() + trim_specs() for X4 quadcopter
│
├── sweeps/                       Example sweep/Monte-Carlo specs (YAML)
└── tests/                        Unit + machinery + golden-run regression tests
```

Logs are written to `logs/` at the repo root (created automatically, gitignored).

---

## Quick Start

Install dependencies, then run from the repo root:

```bash
pip install -r requirements.txt
```

```bash
python run_candidate.py                           # default (Spearhead VTOL), headless
python run_candidate.py candidates.spearhead_vtol
python run_candidate.py candidates.x4_lqg
python run_candidate.py candidates.x4_lqg --show  # interactive plot windows
python run_candidate.py candidates.x4_lqg --set vehicle.M=0.9 --set config.tf=20

python run_sweep.py sweeps/example_x4_dispersion.yaml -j 3   # batch / Monte Carlo
python analyze_candidate.py candidates.x4_lqg                # trim + eigenvalues
```

Runs are **headless by default**: figures are saved as PNGs next to the log
files; pass `--show` to open interactive windows instead. The exit code is
machine-readable: `0` = PASS/COMPLETE, `2` = FAIL, `3` = CRASHED/DEPARTED/DIVERGED.

Each run produces timestamped files in `logs/`:
- `flight_YYYYMMDD_HHMMSS.log` — human-readable header (incl. git commit + full
  config for reproducibility), phase events, envelope events, per-phase
  statistics, and a final `[VERDICT]` block
- `flight_YYYYMMDD_HHMMSS.csv` — time-series at `log_hz` Hz (all state and
  control columns, plus `data_valid` = 0 after any envelope exit)
- `flight_YYYYMMDD_HHMMSS.json` — machine-readable run summary (verdict,
  criteria results, envelope stats, end-state metrics) for automated sweeps

### Tests

```bash
python -m unittest discover -s tests              # fast suite (~10 s)
RUN_SLOW=1 python -m unittest discover -s tests   # + full-length missions
```

Golden-run regression tests pin the end state of short runs of both
candidates; any change that shifts sim behavior fails the suite. After a
*deliberate* behavior change, re-pin with
`python tests/test_golden.py --repin` and commit the new numbers with it.

### Disturbances (toggle-able, off by default)

`SimConfig.wind = None` (default) is bit-identical to a windless build.
Enable constant wind and/or seeded Dryden gusts per run:

```python
wind = {
    'constant_ned': [3.0, 0.0, 0.0],                       # steady, NED m/s
    'dryden': {'V': 5.0, 'sigma': [0.8, 0.8, 0.4],         # gusts (see
               'L': [50.0, 50.0, 20.0]},                   #  sim/wind.py)
    'seed': 42,                                            # required w/ dryden
}
```

or from the CLI: `--set 'config.wind={constant_ned: [3, 0, 0]}'`.
Aero/drag act on air-relative flow; envelope checks use air-relative
alpha/beta; wind history is logged as CSV columns when enabled.

### Sweeps & Monte Carlo

```bash
python run_sweep.py sweeps/example_x4_dispersion.yaml -j 3
```

A YAML/JSON spec expands a parameter grid x seeded Monte Carlo dispersions
(mass, thrust coefficients, ... — see run_sweep.py docstring) into parallel
runs and produces one `summary.csv` (overrides + verdict + final metrics per
run) plus `sweep.json` with the verdict histogram and pass rate. Plants are
perturbed; controllers stay at the design point — the mismatch is what a
dispersion sweep measures. One-off overrides work on single runs too:
`python run_candidate.py candidates.x4_lqg --set vehicle.M=0.9`.

### Trim & linearization

```bash
python analyze_candidate.py candidates.spearhead_vtol
```

For each condition a candidate declares via `trim_specs()`, solves the
equilibrium (bounded least-squares), linearizes about it, and reports the
eigenvalues — answering "is this flight condition an equilibrium, and is it
stable?" in milliseconds instead of a 180 s sim. Notable built-in result: the
Spearhead has **no pure wing-borne level trim at 54 m/s** (wing lift equals
weight with zero margin), which is the root cause of its cruise-phase sink —
the rotor-assisted trim exists but is open-loop unstable (wn ~ 1.3 rad/s).

---

## Core Conventions

These apply to all vehicles and controllers in this platform.

### Coordinate system

**NED (North-East-Down).** z is positive *downward*. Altitude above ground level (AGL) stored as negative z:

```
altitude h AGL  →  z_NED = −h
hover at 10 m   →  z_NED = −10.0
```

### Quaternion ordering

All quaternions use scalar-first ordering: `[qw, qx, qy, qz]`.

At level, north-heading hover the identity quaternion `[1, 0, 0, 0]` is the rest state.

### Time

`t` is wall-clock simulation time in seconds, starting at 0. The runner calls `controller.step(t, X)` at every integration timestep `dt`; controllers are responsible for their own sub-sampling if they run at a lower rate than the integrator.

### Units

| Quantity | Unit |
|---|---|
| Position | m |
| Velocity | m/s |
| Angles | rad (internal); deg acceptable in `info` dict |
| Angular rates | rad/s |
| Time | s |
| Motor commands | vehicle-defined (e.g. PWM 0–800 for X4, normalised 0–1 for Spearhead) |

---

## Interfaces

The runner is duck-typed — no base classes required. Implement the attributes and methods below.

### VehicleDynamics

```python
class MyDynamics:
    state_dim:    int         # number of states
    control_dim:  int         # number of control inputs
    state_names:  list[str]   # len == state_dim, used as CSV column headers
    control_names: list[str]  # len == control_dim

    def initial_state(self, **kwargs) -> np.ndarray:
        """Return X at t = 0."""

    def derivatives(self, t: float, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Return dX/dt (the ODE right-hand side)."""

    def describe(self) -> dict:
        """Key-value pairs written to the log header (mass, gains, etc.)."""

    # --- Optional hooks (the runner never assumes state layout) ---

    def get_position(self, X: np.ndarray) -> np.ndarray:
        """NED position (north, east, down) [m]. Enables position/altitude
        reporting in the [END STATE] block and pass_criteria keys
        north_m / east_m / down_m / alt_agl_m."""

    def apply_constraints(self, X: np.ndarray) -> np.ndarray:
        """Called after every RK4 step. Normalise quaternion, clamp ground
        contact, etc. If absent, nothing is applied — constraints are
        entirely the vehicle's responsibility."""

    def envelope_violations(self, X: np.ndarray) -> list[str]:
        """Non-empty list => the state is outside the model's validated
        envelope (e.g. aero-table alpha/beta range). The runner logs the
        first exit and marks all subsequent CSV rows data_valid=0.
        Everything integrated past the envelope edge is fiction — this hook
        is what keeps it from being mistaken for data."""

    def terminal_condition(self, t: float, X: np.ndarray) -> str | None:
        """Return 'crash' or 'departure' to end the run early (gated by
        config.terminate_on). Divergence (non-finite state) always
        terminates regardless."""

    def set_wind_ned(self, w: np.ndarray):
        """Accept the current wind vector (NED, m/s), called once per
        integration step when config.wind is enabled. Apply it to your
        aero/drag terms as air-relative flow. Vehicles without this hook
        are simply not disturbed (the runner warns)."""
```

### Controller

```python
class MyController:
    def step(self, t: float, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Compute control command for this timestep.

        Returns
        -------
        U    : np.ndarray (control_dim,)  — absolute motor commands
        info : dict  — MUST contain 'phase': str
                       any other scalar values are logged to CSV automatically
        """

    def reset(self):
        """Clear all integrator/filter state. Called before each run."""

    def describe(self) -> dict:
        """Key-value pairs written to the log header."""
```

**The `phase` key is mandatory.** The runner uses it to detect phase transitions, print events to the log, and compute per-phase statistics. Common phase strings: `'ground'`, `'climbing'`, `'hover'`, `'transit'`, `'cruise'`. You can use any strings; what matters is that they change meaningfully during the mission.

**Everything else in `info` is logged automatically.** Fields like `alt_m`, `roll_deg`, `pitch_deg`, `vx_ms` will appear in both the phase-transition event lines and the CSV. Use this to log any quantity you want to track without modifying the runner.

### SimConfig

```python
from sim.config import SimConfig

config = SimConfig(
    dt              = 0.001,    # integration step [s]
    tf              = 120.0,    # total simulation time [s]
    phases          = {         # informational — not enforced by runner
        'ground':   0.0,        # phase_name: start time [s]
        'climbing': 2.0,
        'hover':    10.0,
    },
    references      = {         # passed to build(); document what your controller expects
        'alt_m': -10.0,
        'heading_rad': 0.0,
    },
    vehicle_name    = 'my_craft',
    controller_name = 'my_ctrl',
    log_dir         = 'logs',   # relative to the repo root (or absolute)
    log_hz          = 50.0,     # CSV decimation rate [Hz]
    terminate_on    = {         # which detected events end the run early
        'crash':         True,
        'departure':     True,
        'envelope_exit': False, # exits only mark data invalid by default
    },
    pass_criteria   = {         # mission verdict, checked at end of run
        'alt_m': (9.0, 11.0),   # metric -> (lo, hi); metrics come from the
        'vx_ms': (49.0, 59.0),  # final controller info dict + position keys
    },
    wind            = None,     # None = disturbances OFF (default);
                                # see "Disturbances" above for the schema
)
```

The `phases` dict is *documentation only* — the runner does not trigger anything based on it. Phase transitions are driven entirely by the string returned from `controller.step()`.

**Verdicts.** Every run ends with a `[VERDICT]` block in the log and a JSON
summary: `PASS`/`FAIL` (ran to `tf`, judged against `pass_criteria`),
`COMPLETE` (ran to `tf`, no criteria configured), or `CRASHED`/`DEPARTED`/
`DIVERGED` (ended early). `run_candidate.py` converts the verdict into its
exit code, so sweeps and CI can consume results without parsing logs.

---

## Step-by-Step: Adding a New Candidate

### 1. Create the vehicle package

```
vehicles/
└── my_vehicle/
    ├── __init__.py    (empty)
    └── dynamics.py
```

Minimal `dynamics.py`:

```python
import numpy as np

class MyDynamics:
    state_dim     = 6
    control_dim   = 2
    state_names   = ['x', 'xdot', 'y', 'ydot', 'z', 'zdot']
    control_names = ['thrust', 'torque']

    def initial_state(self):
        return np.zeros(self.state_dim)

    def derivatives(self, t, X, U):
        # write your ODE here
        dX = np.zeros(self.state_dim)
        ...
        return dX

    def get_position(self, X):
        # NED (north, east, down) — enables [END STATE], alt/pos pass criteria
        return np.array([X[0], X[2], X[4]])

    def apply_constraints(self, X):
        # called after every RK4 step — normalise quaternion, clamp to ground, etc.
        return X

    def describe(self):
        return {'mass_kg': 1.0, 'model': 'point mass'}
```

Optional extras when you need them: `envelope_violations(X)`,
`terminal_condition(t, X)`, `set_wind_ned(w)` — see the interface reference
above.

### 2. Create the controller package

```
controllers/
└── my_ctrl/
    ├── __init__.py    (empty)
    └── controller.py
```

Minimal `controller.py`:

```python
import numpy as np

class MyController:
    def __init__(self, gain):
        self.gain   = gain
        self._integ = 0.0

    def reset(self):
        self._integ = 0.0

    def step(self, t, X):
        error        = target - X[0]   # example: position error
        self._integ += error * 0.001   # dt = 0.001 s
        U = np.array([self.gain * error + 0.1 * self._integ, 0.0])

        phase = 'ground' if X[4] > -0.1 else 'hover'
        info  = {'phase': phase, 'alt_m': -X[4], 'error_m': error}
        return U, info

    def describe(self):
        return {'gain': self.gain}
```

### 3. Create the candidate

```
candidates/
└── my_candidate.py
```

```python
import numpy as np
import matplotlib.pyplot as plt

from sim.config import SimConfig
from vehicles.my_vehicle.dynamics import MyDynamics
from controllers.my_ctrl.controller import MyController

def build(overrides=None):
    """overrides: optional {'vehicle': {...}, 'config': {...}} from
    sweeps / --set. Apply what your candidate supports; ignore the rest."""
    ov = overrides or {}
    config = SimConfig(dt=0.001, tf=60.0,
                       vehicle_name='my_vehicle',
                       controller_name='my_ctrl',
                       pass_criteria={'alt_m': (9.0, 11.0)})
    for k, v in ov.get('config', {}).items():
        setattr(config, k, v)
    dynamics   = MyDynamics()          # pass ov.get('vehicle') if it takes params
    controller = MyController(gain=5.0)
    return dynamics, controller, config

def plot(X_hist, U_hist, config, show=True):
    """Build figures and return them; show=False for headless (default CLI)."""
    t = np.arange(X_hist.shape[0]) * config.dt
    fig = plt.figure()
    plt.plot(t, X_hist[:, 4], label='z NED [m]')
    plt.grid(True)
    if show:
        plt.show()
    return [fig]
```

Optionally declare trim conditions for `analyze_candidate.py`:

```python
def trim_specs(dynamics):
    return {'hover': {
        'X0': dynamics.initial_state(), 'U0': np.zeros(2),
        'free_states': ['zdot'], 'free_controls': ['thrust'],
        'residual_states': ['zdot'],
    }}
```

### 4. Run it

```bash
cd sim_platform
python run_candidate.py candidates.my_candidate
```

---

## Shared Utilities

### `sim/pid.py` — PID with integral anti-windup

```python
from sim.pid import PID

pid = PID(Kp=1.0, Ki=0.05, Kd=0.1, dt=0.001, integral_limit=25.0)
output = pid.update(error)      # dt is fixed at construction
pid.reset()
```

| Parameter | Description |
|---|---|
| `Kp, Ki, Kd` | Proportional, integral, derivative gains |
| `dt` | Controller timestep [s] (used by the I and D terms) |
| `integral_limit` | Clamps the raw accumulator to `±integral_limit`, bounding the Ki contribution to `±Ki·integral_limit` (None = no clamp) |

### `sim/quaternion.py` — Quaternion tools

All functions use scalar-first ordering `[qw, qx, qy, qz]`, and `q` is the
**body→NED attitude quaternion** (`v_NED = q ⊗ v_body ⊗ q*`) — consistent
across `quat_kinematics`, `quat_to_euler`, and both rotate helpers. This
invariant is guarded by tests (`tests/test_quaternion.py`); see the module
header for the history of the pre-2026-07 crossed-convention bug.

```python
from sim.quaternion import (
    normalize_quaternion,    # q / |q|
    quat_to_euler,           # [q0..q3] -> (phi, theta, psi) in radians
    quat_kinematics,         # q_dot = 0.5 q (x) [0, omega_body]
    quat_multiply,           # Hamilton product
    rotate_body_to_inertial, # v_NED  = q (x) v_body (x) q*
    rotate_inertial_to_body, # v_body = q* (x) v_NED (x) q
)
```

### `sim/wind.py` — WindModel

Constructed by the runner from `SimConfig.wind`; you normally don't
instantiate it yourself. `WindModel(spec, dt).step()` returns the current
NED wind vector: constant part + first-order Gauss–Markov Dryden surrogate
(exact discretisation, seeded RNG — same seed, same gust history).

### `sim/analysis.py` — trim / linearize / eigenvalues

```python
from sim.analysis import trim, linearize, eig_report, format_mode

res     = trim(dyn, X0, U0, free_states, free_controls, residual_states,
               quat_states=None, bounds=None)   # bounded least-squares
A, B, f0 = linearize(dyn, res.X, res.U)         # central differences
for m in eig_report(A):                          # most unstable first
    print(format_mode(m))
```

---

## Common Pitfalls

### Quaternion normalisation drift

RK4 does not conserve the unit norm constraint on quaternions. After ~1000 steps the quaternion will drift. Provide `apply_constraints()` and call `normalize_quaternion()` there. The runner never touches your state itself — if you omit the hook, nothing normalises your quaternion and it *will* drift.

### Waypoint sequencing with a rate-limited reference

If you rate-limit your reference (recommended to avoid integrator wind-up), initialise the active reference to the *vehicle's starting position*, not the first waypoint. If the active reference already equals the first waypoint on step 0, the completion check fires immediately and the sequencer skips to waypoint 2 before the vehicle has moved.

```python
# BAD — active_ref starts at wp[0]; completion fires on step 0
self._active_ref = waypoints[0].copy()

# GOOD — check actual vehicle position, not the rate-limited ref
veh_pos = np.array([X[0], X[2], X[4]])   # NED: x, y, z
if np.linalg.norm(veh_pos - tgt[:3]) < tol:
    advance_waypoint()
```

### Integral wind-up on large reference steps

Setting a 10 m position step directly as the reference will saturate the integral state within a few timesteps. Always rate-limit reference changes:

```python
delta = target - current_ref
current_ref += np.clip(delta, -max_rate, max_rate)   # per controller tick
```

For a 100 Hz controller, `max_rate = 0.02 m/tick` gives 2 m/s — enough for smooth quadrotor transit without saturating the motors.

### Controller rate vs integrator rate

The runner calls `step(t, X)` at every integration step `dt`. If your controller runs at a lower rate (e.g. 100 Hz vs 1 kHz integration), cache the last command and return it on non-control ticks:

```python
def step(self, t, X):
    if self._t_last is not None and (t - self._t_last) < self.T_ctrl - 1e-9:
        return self._last_U.copy(), self._make_info(X)
    self._t_last = t
    # ... recompute U ...
    self._last_U = U
    return U, info
```

### NED z-sign in the phase string

A common mistake: checking `X[z_idx] > 0` to detect "in the air". In NED, z > 0 means *below ground*. Use altitude AGL:

```python
alt_agl = -X[z_idx]   # positive above ground
phase   = 'ground' if alt_agl < 0.05 else 'hover'
```

### `apply_constraints` ground clamp direction

In NED, z_NED is negative when airborne. Ground is z_NED = 0. Clamp like this:

```python
if X[z_idx] > 0.0:    # vehicle has gone through the ground (NED)
    X[z_idx] = 0.0
    if X[zdot_idx] > 0.0:
        X[zdot_idx] = 0.0
```

---

## Existing Candidates Reference

### Spearhead VTOL (`candidates/spearhead_vtol.py`)

A quadrotor-wing VTOL: four vertical lift rotors (m1–m4), one pusher motor (m5), three aerodynamic surfaces (left elevon, right elevon, rudder).

**State (21):** `[u, v, w, p, q, r, x, y, z, q0, q1, q2, q3, w1..w5, dl, dr, drd]`

**Control (8):** `[m1, m2, m3, m4, m5, servo_le, servo_re, servo_rud]`

**Controller:** 4-loop nested PID. Altitude → vertical velocity → rotor thrust. Angle → angular rate → motor differential (hover) or surface deflection (forward flight). Wing-lift handoff blends rotor authority to zero at 54 m/s.

**Flight phases:** spinup (0–5 s) → hover (5–30 s) → transition (30–90 s) → fwd_flight (90 s →)

**Trim conditions** (`analyze_candidate.py candidates.spearhead_vtol`): `hover` (converges to the coded `W_HOVER = 808`), `cruise` (pure wing-borne at 54 m/s — **infeasible**: wing lift equals weight with zero margin, the solver pins the ±30° α clip), `cruise_assisted` (converges: rear-heavy rotor lift m3/m4 ≈ 410/425 vs m1/m2 ≈ 119/154, pusher ≈ 529, rudder countering pusher torque — but open-loop unstable, ωn ≈ 1.3 and 1.9 rad/s).

**Known issue — mission verdict is `CRASHED` (~t = 99 s).** The wing-lift handoff blends out rotor lift exactly where the `cruise` trim shows the wing alone cannot carry the weight, so the aircraft sinks into the ground during cruise. Root-caused via the trim tools above; fixing it means keeping partial rotor assist in cruise, raising cruise speed, or retuning the blend schedule.

**Performance note:** the full mission at dt = 1 ms takes roughly 2–4 minutes wall time (it terminates at the ~99 s crash). Use `--set config.tf=35` to quickly verify phase transitions.

---

### X4 Quadcopter (`candidates/x4_lqg.py`)

A 0.86 kg symmetric quadrotor with LQR+I full-state feedback.

**State (17):** `[x, ẋ, y, ẏ, z, ż, qw, qx, qy, qz, p, q, r, w1, w2, w3, w4]`

All in NED (z positive down). Motor speeds `w1..w4` in rad/s.

**Control (4):** motor PWM commands 0–800 per motor. Equilibrium ≈ 166.5.

**Controller:** Discrete-time LQR+I gains (Kdt, Kidt, Ldt) loaded from `vehicles/x4/data/` — the plant model and its tuned gains are vendored with the vehicle they describe. Runs at 100 Hz with 1 kHz RK4 integration, conditional-integration anti-windup against the 0–800 command clip. Reference format: `[x_N, y_E, z_NED, psi]`. The controller always uses the *nominal* plant (design point), even when the dynamics are perturbed by a sweep.

**Plant/gain data files** (in `vehicles/x4/data/`):

| File | Shape | Description |
|---|---|---|
| `Adt.txt` | 16×16 | Discrete state matrix |
| `Bdt.txt` | 16×4 | Discrete input matrix |
| `Cdt.txt` | 4×16 | Output matrix (x, y, z, psi) |
| `Ddt.txt` | 4×4 | Feedthrough (zero) |
| `Kdt.txt` | 4×16 | LQR state-feedback gain |
| `Kidt.txt` | 4×4 | Integral gain |
| `Ldt.txt` | 16×4 | Kalman filter gain (unused in full-state mode) |
| `U_e.txt` | 4 | Equilibrium motor command |

**Flight phases:** ground → climbing → hover → transit (waypoint sequencing)

**Mission:** hover at 1.5 m AGL → transit NE to (5 m, 3 m) → climb to 3 m AGL → return to origin. Full mission (120 s, ~30 s wall time) ends **PASS**; robustness demo: 8/8 PASS across ±5 % mass, ±8 % thrust and 3 m/s gusty wind (`sweeps/example_x4_dispersion.yaml`).

**Trim condition** (`analyze_candidate.py candidates.x4_lqg`): `hover` converges exactly to the analytic `W_e`/`U_e`; open-loop modes are the drag poles (−Dxx/M, −Dzz/M), motor poles (−1/Mtau) and neutral kinematic integrators.

---

## Output Files

### Log file (`.log`)

```
[REPRODUCIBILITY]           git commit (+dirty flag), python, numpy versions
[SIMULATION PARAMETERS]     full SimConfig dump (incl. terminate_on,
                            pass_criteria, wind)
[VEHICLE]                   describe() key-value pairs
[CONTROLLER]                describe() key-value pairs
[PHASE EVENTS]              t=..s  ENTER <PHASE>  alt_m=...  roll_deg=...
                            t=..s  ENVELOPE EXIT — data invalid from here on
[PHASE PERFORMANCE SUMMARY] mean / min / max of every numeric info field per phase
[END STATE]                 final time, position (via get_position), altitude
[ENVELOPE]                  first exit time + valid fraction (if vehicle
                            declares an envelope)
[VERDICT]                   PASS / FAIL / COMPLETE / CRASHED / DEPARTED /
                            DIVERGED, with per-criterion results
```

### CSV file (`.csv`)

Columns: `time_s`, `phase`, all `state_names`, all `control_names`, all numeric
`info` keys (in the order first seen on step 0), then `wind_n / wind_e / wind_d`
(only when wind is enabled) and `data_valid` (0 after any envelope exit).
Row rate = `log_hz`. Histories are truncated at early termination.

Load with:
```python
import pandas as pd
df = pd.read_csv('logs/flight_20260315_205137.csv')
```

### JSON summary (`.json`)

One machine-readable document per run: schema id, timestamp, vehicle /
controller names, git metadata, python/numpy versions, the full config,
verdict + reason + per-criterion results, envelope stats, end-state metrics,
and paths to the log/CSV. This is the artifact `run_sweep.py` aggregates
into `summary.csv`.
