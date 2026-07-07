"""
Candidate: Spearhead VTOL with nested-PID controller.

Provides:
  build(overrides=None) -> (dynamics, controller, config)
  plot(X_hist, U_hist, config, show=True) -> list[Figure]
  trim_specs(dynamics)  -> trim conditions for analyze_candidate.py
                           (incl. the infeasible pure-wing 'cruise' — see
                            README "Known issue")
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sim.config import SimConfig
from sim.quaternion import normalize_quaternion, quat_to_euler
from vehicles.spearhead.dynamics import SpearheadDynamics
from controllers.spearhead_vtol.controller import SpearheadVTOLController


def build(overrides=None):
    """Create and return (dynamics, controller, config) for this candidate.

    overrides (optional): {'vehicle': {param: val}, 'config': {field: val}}
    — used by sweeps / Monte Carlo.
    """
    ov = overrides or {}
    config = SimConfig(
        dt             = 0.001,
        tf             = 180.0,
        phases         = {'spinup': 5.0, 'hover': 30.0, 'transition': 90.0},
        references     = {'alt': -10.0, 'fwd_vel': 54.0, 'roll': 0.0, 'yaw': 0.0},
        vehicle_name   = 'Spearhead',
        controller_name = 'VTOL_nested_PID',
        log_dir        = 'logs',
        log_hz         = 10.0,
        # Mission intent: hold 10 m AGL while cruising at 54 m/s
        pass_criteria  = {'alt_m':  (7.0, 13.0),
                          'vx_ms': (49.0, 59.0)},
    )

    for k, v in ov.get('config', {}).items():
        setattr(config, k, v)

    # Plant may be perturbed (Monte Carlo); the controller is built from the
    # NOMINAL class-level params (design point) so plant/controller mismatch
    # is what a dispersion sweep actually measures.
    dynamics   = SpearheadDynamics(params=ov.get('vehicle'))
    controller = SpearheadVTOLController(SpearheadDynamics.params, config)
    return dynamics, controller, config


# ---------------------------------------------------------------------------
# Trim conditions (consumed by analyze_candidate.py)
# ---------------------------------------------------------------------------

def trim_specs(dynamics):
    p = dynamics.params
    X_hover = dynamics.initial_state()
    U_hover = np.zeros(8)
    U_hover[0:4] = p['W_HOVER']

    X_cruise = dynamics.initial_state()
    X_cruise[0] = 54.0                    # body u at cruise speed
    U_cruise = np.zeros(8)

    # Warm start for the assisted condition: partial rotor lift + pusher
    # spinning near its transition operating point.
    X_assist = X_cruise.copy()
    X_assist[13:17] = 3000.0              # vertical rotors ~45% hover speed
    X_assist[17]    = 5000.0              # pusher
    U_assist = np.zeros(8)
    U_assist[0:4] = 3000.0 / (44.2205 * 8.18)   # steady-state cmds for w1..w4
    U_assist[4]   = 5000.0 / (44.2205 * 7.02)   # and w5

    return {
        'hover': {
            'X0': X_hover, 'U0': U_hover,
            'free_states':    ['w1', 'w2', 'w3', 'w4'],
            'free_controls':  ['m1', 'm2', 'm3', 'm4'],
            'residual_states': ['u', 'v', 'w', 'p', 'q', 'r',
                                'w1', 'w2', 'w3', 'w4'],
        },
        'cruise': {
            # Pure wing-borne flight at u = 54 m/s, vertical rotors off:
            # solve pitch attitude, heave, pusher and elevons for equilibrium.
            # (Known result: does NOT converge — at 54 m/s the wing is right
            # at the weight-support boundary, so no in-envelope level trim
            # exists without rotor assist. This is the root cause of the
            # cruise-phase sink/crash seen in the full mission.)
            'X0': X_cruise, 'U0': U_cruise,
            'free_states':    ['w', 'q0', 'q2', 'w5', 'dl', 'dr'],
            'free_controls':  ['m5', 'servo_le', 'servo_re'],
            'residual_states': ['u', 'v', 'w', 'p', 'q', 'r',
                                'w5', 'dl', 'dr'],
            'quat_states':    ['q0', 'q1', 'q2', 'q3'],
        },
        'cruise_assisted': {
            # Same condition with the vertical rotors also free — quantifies
            # how much rotor lift the airframe still needs at 54 m/s.
            # Bounds keep the solve physical (rotors can't spin backwards,
            # heave/pitch inside the aero envelope).
            'X0': X_assist, 'U0': U_assist,
            'free_states':    ['w', 'q0', 'q2',
                               'w1', 'w2', 'w3', 'w4', 'w5',
                               'dl', 'dr', 'drd'],
            'free_controls':  ['m1', 'm2', 'm3', 'm4', 'm5',
                               'servo_le', 'servo_re', 'servo_rud'],
            'residual_states': ['u', 'v', 'w', 'p', 'q', 'r',
                                'w1', 'w2', 'w3', 'w4', 'w5',
                                'dl', 'dr', 'drd'],
            'quat_states':    ['q0', 'q1', 'q2', 'q3'],
            'bounds': {
                'w':  (-15.0, 15.0), 'q0': (0.9, 1.0), 'q2': (-0.25, 0.25),
                'w1': (0.0, 9000.0), 'w2': (0.0, 9000.0),
                'w3': (0.0, 9000.0), 'w4': (0.0, 9000.0),
                'w5': (0.0, 9000.0),
                'm1': (0.0, 1000.0), 'm2': (0.0, 1000.0),
                'm3': (0.0, 1000.0), 'm4': (0.0, 1000.0),
                'm5': (0.0, 1000.0),
                'dl': (-20.0, 20.0), 'dr': (-20.0, 20.0),
                'drd': (-20.0, 20.0),
                'servo_le': (-20.0, 20.0), 'servo_re': (-20.0, 20.0),
                'servo_rud': (-20.0, 20.0),
            },
        },
    }


# ---------------------------------------------------------------------------
# Plotting (Spearhead-specific)
# ---------------------------------------------------------------------------

def plot(X_hist, U_hist, config, show=True):
    """Build figures; returns them. show=False for headless use."""
    dt = config.dt
    tf = config.tf
    time = np.arange(0.0, tf + dt, dt)
    N    = min(len(time), X_hist.shape[0])   # histories may be truncated by
                                             # early termination

    T_SPINUP = config.phases.get('spinup',     5.0)
    T_HOVER  = config.phases.get('hover',      30.0)
    T_TRANS  = config.phases.get('transition', 90.0)
    ref_alt  = config.references.get('alt', -10.0)

    phase_regions = [
        (0,        T_SPINUP, '#e0e0e0', 'Spin-up'),
        (T_SPINUP, T_HOVER,  '#cce5ff', 'Hover'),
        (T_HOVER,  T_TRANS,  '#fff3cd', 'Transition'),
        (T_TRANS,  tf,       '#d4edda', 'Fwd flight'),
    ]

    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    for ax in axs:
        for x0, x1, col, lbl in phase_regions:
            ax.axvspan(x0, x1, alpha=0.25, color=col, label=lbl)
        ax.grid(True, zorder=0)

    axs[0].plot(time[:N], X_hist[:N, 8], 'b', lw=1.5, zorder=3)
    axs[0].axhline(ref_alt, color='k', ls='--', lw=1, label=f'ref={ref_alt} m')
    axs[0].set_title('Altitude (NED, positive = down)')
    axs[0].set_ylabel('Z [m]')
    axs[0].legend(loc='upper right', fontsize=7, ncol=3)

    eul = np.array([quat_to_euler(normalize_quaternion(X_hist[i, 9:13]))
                    for i in range(N)])
    axs[1].plot(time[:N], np.degrees(eul[:, 0]), 'r',  label='Roll φ',  lw=1.2, zorder=3)
    axs[1].plot(time[:N], np.degrees(eul[:, 1]), 'g',  label='Pitch θ', lw=1.2, zorder=3)
    axs[1].plot(time[:N], np.degrees(eul[:, 2]), 'b',  label='Yaw ψ',   lw=1.2, zorder=3)
    axs[1].set_title('Attitude (from Quaternion)')
    axs[1].set_ylabel('Angles [deg]')
    axs[1].legend(loc='upper right', fontsize=7)

    for j, lbl in enumerate(['M1', 'M2', 'M3', 'M4']):
        axs[2].plot(time[:N], U_hist[:N, j], label=lbl, lw=1.2, zorder=3)
    axs[2].plot(time[:N], U_hist[:N, 4], 'm--', label='Pusher', lw=1.2, zorder=3)
    axs[2].set_title('Motor Commands')
    axs[2].set_ylabel('Command')
    axs[2].legend(loc='upper right', fontsize=7)

    for j, lbl in enumerate(['δ_le', 'δ_re', 'δ_rud']):
        axs[3].plot(time[:N], U_hist[:N, 5 + j], label=lbl, lw=1.2, zorder=3)
    axs[3].set_title('Control Surface Deflections [deg]')
    axs[3].set_ylabel('Deflection [deg]')
    axs[3].set_xlabel('Time [s]')
    axs[3].legend(loc='upper right', fontsize=7)

    plt.tight_layout()

    # 3D trajectory
    x_pos = X_hist[:N, 6]
    y_pos = X_hist[:N, 7]
    alt   = -X_hist[:N, 8]

    n_su = int(T_SPINUP / dt)
    n_hv = int(T_HOVER  / dt)
    n_tr = int(T_TRANS  / dt)

    fig3d = plt.figure(figsize=(10, 7))
    ax3d  = fig3d.add_subplot(111, projection='3d')
    ax3d.plot(x_pos[:n_su],       y_pos[:n_su],       alt[:n_su],
              color='gray',       lw=1.5, label=f'Spin-up (0–{T_SPINUP:.0f} s)')
    ax3d.plot(x_pos[n_su:n_hv],   y_pos[n_su:n_hv],   alt[n_su:n_hv],
              color='royalblue',  lw=1.5, label=f'Hover ({T_SPINUP:.0f}–{T_HOVER:.0f} s)')
    ax3d.plot(x_pos[n_hv:n_tr],   y_pos[n_hv:n_tr],   alt[n_hv:n_tr],
              color='orange',     lw=1.5, label=f'Transition ({T_HOVER:.0f}–{T_TRANS:.0f} s)')
    ax3d.plot(x_pos[n_tr:],       y_pos[n_tr:],       alt[n_tr:],
              color='forestgreen', lw=1.5, label=f'Fwd flight ({T_TRANS:.0f}–{tf:.0f} s)')
    ax3d.scatter([x_pos[0]],  [y_pos[0]],  [alt[0]],  c='k', s=60, zorder=5, label='Start')
    ax3d.scatter([x_pos[-1]], [y_pos[-1]], [alt[-1]], c='r', s=60, zorder=5, label='End')

    ref_alt_m = -config.references.get('alt', -10.0)
    xlim = (min(x_pos.min() - 5, -5), max(x_pos.max() + 5, 5))
    ylim = (min(y_pos.min() - 5, -5), max(y_pos.max() + 5, 5))
    xs = np.array([xlim[0], xlim[1], xlim[1], xlim[0]])
    ys = np.array([ylim[0], ylim[0], ylim[1], ylim[1]])
    poly = Poly3DCollection(
        [list(zip(xs, ys, np.full(4, ref_alt_m)))],
        alpha=0.15, facecolor='cyan', edgecolor='teal')
    ax3d.add_collection3d(poly)
    ax3d.set_xlabel('X [m] (North)')
    ax3d.set_ylabel('Y [m] (East)')
    ax3d.set_zlabel('Altitude [m]')
    ax3d.set_title('Spearhead UAV — 3D Trajectory')
    ax3d.legend(fontsize=8)

    if show:
        plt.show()
    return [fig, fig3d]
