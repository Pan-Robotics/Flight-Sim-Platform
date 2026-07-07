# UAV Simulation Platform

A modular, candidate-based simulation framework for testing arbitrary vehicle dynamics and control systems together. Each *candidate* pairs a vehicle dynamics model with a controller and a set of simulation parameters. The infrastructure (integrator, logger, runner) is shared and vehicle-agnostic.

Two candidates are included: the Spearhead VTOL with a nested-PID controller, and the X4 quadcopter with an LQG (LQR+I) controller.

---

## Directory Structure

```
sim_platform/
├── run_candidate.py              Entry point — run any candidate from the command line
│
├── sim/                          Simulation infrastructure (vehicle-agnostic)
│   ├── config.py                 SimConfig dataclass
│   ├── pid.py                    Generic PID with anti-windup (utility, optional)
│   ├── quaternion.py             Quaternion utilities (normalise, Euler, rotate, kinematics)
│   └── runner.py                 SimRunner: RK4 loop + phase tracker + CSV/log writer
│
├── vehicles/                     One sub-package per vehicle
│   ├── spearhead/
│   │   ├── dynamics.py           SpearheadDynamics — 21-state quaternion 6-DOF VTOL
│   │   └── *.txt                 Aerodynamic database tables
│   └── x4/
│       └── dynamics.py           X4Dynamics — 17-state quaternion quadcopter
│
├── controllers/                  One sub-package per controller type
│   ├── spearhead_vtol/
│   │   └── controller.py         SpearheadVTOLController — 4-loop nested PID
│   └── x4_lqg/
│       └── controller.py         X4LQGController — discrete-time LQR+I (full-state)
│
└── candidates/                   Wiring: vehicle + controller + SimConfig
    ├── spearhead_vtol.py          build() + plot() for Spearhead VTOL
    └── x4_lqg.py                 build() + plot() for X4 quadcopter
```

Logs are written to `sim_platform/logs/` (created automatically on first run).

---

## Quick Start

Run from inside `sim_platform/`:

```bash
python run_candidate.py                           # default (Spearhead VTOL), headless
python run_candidate.py candidates.spearhead_vtol
python run_candidate.py candidates.x4_lqg
python run_candidate.py candidates.x4_lqg --show  # interactive plot windows
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
    log_dir         = 'logs',   # relative to sim_platform/sim/
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

    def apply_constraints(self, X):
        # called after every RK4 step — normalise quaternion, clamp to ground, etc.
        return X

    def describe(self):
        return {'mass_kg': 1.0, 'model': 'point mass'}
```

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

def build():
    config     = SimConfig(dt=0.001, tf=60.0,
                           vehicle_name='my_vehicle',
                           controller_name='my_ctrl')
    dynamics   = MyDynamics()
    controller = MyController(gain=5.0)
    return dynamics, controller, config

def plot(X_hist, U_hist, config):
    t = np.arange(X_hist.shape[0]) * config.dt
    plt.figure()
    plt.plot(t, X_hist[:, 4], label='z NED [m]')
    plt.grid(True)
    plt.show()
```

### 4. Run it

```bash
cd sim_platform
python run_candidate.py candidates.my_candidate
```

---

## Shared Utilities

### `sim/pid.py` — PID with anti-windup

```python
from sim.pid import PID

pid = PID(kp=1.0, ki=0.05, kd=0.1, lim=25.0)
output = pid.update(error, dt)
pid.reset()
```

| Parameter | Description |
|---|---|
| `kp, ki, kd` | Proportional, integral, derivative gains |
| `lim` | Symmetric saturation on integral and output `[-lim, +lim]` |

### `sim/quaternion.py` — Quaternion tools

All functions use scalar-first ordering `[qw, qx, qy, qz]`.

```python
from sim.quaternion import (
    normalize_quaternion,    # q / |q|
    quat_to_euler,           # [q0..q3] -> (phi, theta, psi) in radians
    quat_kinematics,         # q_dot from q and body rates (p, q_omega, r)
    quat_multiply,           # Hamilton product
    rotate_body_to_inertial, # rotate 3-vector using quaternion
    rotate_inertial_to_body,
)
```

---

## Common Pitfalls

### Quaternion normalisation drift

RK4 does not conserve the unit norm constraint on quaternions. After ~1000 steps the quaternion will drift. Provide `apply_constraints()` and call `normalize_quaternion()` there. If you omit `apply_constraints`, the runner applies its own normalisation at indices `[9:13]` — wrong if your quaternion lives elsewhere.

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

**Flight phases:** spinup (0–5 s) → hover (5–30 s) → transition (30–90 s) → fwd_flight (90–180 s)

**Performance note:** 180 s of simulation at dt = 1 ms takes approximately 10 minutes wall time on a modern laptop. Use `tf = 35` to quickly verify phase transitions without waiting for the full run.

---

### X4 Quadcopter (`candidates/x4_lqg.py`)

A 0.86 kg symmetric quadrotor with LQR+I full-state feedback.

**State (17):** `[x, ẋ, y, ẏ, z, ż, qw, qx, qy, qz, p, q, r, w1, w2, w3, w4]`

All in NED (z positive down). Motor speeds `w1..w4` in rad/s.

**Control (4):** motor PWM commands 0–800 per motor. Equilibrium ≈ 166.5.

**Controller:** Discrete-time LQG gains (Kdt, Kidt, Ldt) loaded from `x4_quadcopter/data/`. Runs at 100 Hz with 1 kHz RK4 integration. Reference format: `[x_N, y_E, z_NED, psi]`.

**LQG data files** (in `x4_quadcopter/data/`):

| File | Shape | Description |
|---|---|---|
| `Adt.txt` | 16×16 | Discrete state matrix |
| `Bdt.txt` | 16×4 | Discrete input matrix |
| `Cdt.txt` | 4×16 | Output matrix (x, y, z, psi) |
| `Kdt.txt` | 4×16 | LQR state-feedback gain |
| `Kidt.txt` | 4×4 | Integral gain |
| `Ldt.txt` | 16×4 | Kalman filter gain |
| `U_e.txt` | 4 | Equilibrium motor command |

**Flight phases:** ground → climbing → hover → transit (waypoint sequencing)

**Mission:** hover at 1.5 m AGL → transit NE to (5 m, 3 m) → descend to 3 m AGL → return to origin.

---

## Output Files

### Log file (`.log`)

```
[SIMULATION PARAMETERS]   dt, tf, phases, references, log_hz
[VEHICLE]                  describe() key-value pairs
[CONTROLLER]               describe() key-value pairs
[PHASE EVENTS]             t=..s  ENTER <PHASE>  alt_m=...  roll_deg=...
[PHASE PERFORMANCE SUMMARY] mean / min / max of every numeric info field per phase
[END STATE]                final position and altitude
```

### CSV file (`.csv`)

Columns: `time_s`, `phase`, all `state_names`, all `control_names`, all numeric `info` keys (in the order first seen on step 0). Row rate = `log_hz`.

Load with:
```python
import pandas as pd
df = pd.read_csv('logs/flight_20260315_205137.csv')
```
