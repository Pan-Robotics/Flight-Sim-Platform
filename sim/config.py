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
