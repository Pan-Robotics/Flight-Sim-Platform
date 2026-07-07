"""Simulation configuration dataclass."""
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    """
    Holds all sim-level parameters that are independent of the vehicle and controller.

    phases: dict mapping phase_name -> t_end (seconds).  The last phase runs until tf.
      e.g. {'spinup': 5.0, 'hover': 30.0, 'transition': 90.0}
      Phase names are passed to controller.step() each timestep.

    references: open-ended dict of reference signals.  Convention varies by controller;
      typical keys: 'alt' (NED m), 'fwd_vel' (m/s), 'roll' (rad), 'yaw' (rad).

    terminate_on: which detected events end the run early.
      Keys: 'crash', 'departure' (from the vehicle's terminal_condition hook)
      and 'envelope_exit' (from the vehicle's envelope_violations hook).
      Divergence (non-finite state) always terminates — NaNs can't be integrated.

    pass_criteria: mission pass/fail bounds, checked at end of run against the
      final controller info dict plus the runner-computed keys 'north_m',
      'east_m', 'down_m' (from dynamics.get_position).  Maps metric -> (lo, hi).
      Empty dict -> verdict is COMPLETE/CRASHED/... with no PASS/FAIL judgement.
      e.g. {'alt_m': (9.0, 11.0), 'vx_ms': (49.0, 59.0)}

    wind: None (default — disturbances OFF, bit-identical to a windless build)
      or a dict enabling constant wind and/or seeded Dryden gusts; see
      sim/wind.py for the schema.  Requires the vehicle to expose
      set_wind_ned(w); the runner warns and ignores wind otherwise.
    """
    dt:               float = 0.001
    tf:               float = 180.0
    phases:           dict  = field(default_factory=lambda: {
                                'spinup': 5.0, 'hover': 30.0, 'transition': 90.0})
    references:       dict  = field(default_factory=lambda: {
                                'alt': -10.0, 'fwd_vel': 54.0,
                                'roll': 0.0, 'yaw': 0.0})
    vehicle_name:     str   = 'unknown'
    controller_name:  str   = 'unknown'
    log_dir:          str   = 'logs'
    log_hz:           float = 10.0
    terminate_on:     dict  = field(default_factory=lambda: {
                                'crash': True, 'departure': True,
                                'envelope_exit': False})
    pass_criteria:    dict  = field(default_factory=dict)
    wind:             dict | None = None
