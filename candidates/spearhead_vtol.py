"""
Candidate: Spearhead VTOL with nested-PID controller.

Provides:
  build()  -> (dynamics, controller, config)
  plot(X_hist, U_hist, config, log_path)  -> shows figures
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


def build():
    """Create and return (dynamics, controller, config) for this candidate."""
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

    dynamics   = SpearheadDynamics()  # ADB data in vehicles/spearhead/ by default
    controller = SpearheadVTOLController(dynamics.params, config)
    return dynamics, controller, config


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
