"""
Candidate: X4 quadcopter with LQR+I (full-state feedback) controller.

Mission: take off to 1.5 m AGL, hold, then fly a 3-waypoint path:
  wp0: hover  [0, 0, -1.5, 0]
  wp1: [5, 3, -1.5, 0]   North-East transit
  wp2: [5, 3, -3.0, 0]   descend to 3 m AGL
  wp3: [0, 0, -3.0, 0]   return (higher altitude)

Provides:
  build()  -> (dynamics, controller, config)
  plot(X_hist, U_hist, config)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from sim.config import SimConfig
from vehicles.x4.dynamics import X4Dynamics, U_e_eq
from controllers.x4_lqg.controller import X4LQGController


# ---------------------------------------------------------------------------
# Path to the x4 plant model + tuned LQG gains, vendored with the x4 vehicle
# (relative to this file)
# ---------------------------------------------------------------------------
_DATA = os.path.normpath(
    os.path.join(os.path.dirname(__file__),
                 '..', 'vehicles', 'x4', 'data'))


def build():
    Adt  = np.loadtxt(os.path.join(_DATA, 'Adt.txt'),  delimiter=',')
    Bdt  = np.loadtxt(os.path.join(_DATA, 'Bdt.txt'),  delimiter=',')
    Cdt  = np.loadtxt(os.path.join(_DATA, 'Cdt.txt'),  delimiter=',')
    Kdt  = np.loadtxt(os.path.join(_DATA, 'Kdt.txt'),  delimiter=',')
    Kidt = np.loadtxt(os.path.join(_DATA, 'Kidt.txt'), delimiter=',')
    Ldt  = np.loadtxt(os.path.join(_DATA, 'Ldt.txt'),  delimiter=',')
    U_e  = np.loadtxt(os.path.join(_DATA, 'U_e.txt'),  delimiter=',')

    config = SimConfig(
        dt              = 0.001,
        tf              = 120.0,
        phases          = {'ground': 0.0, 'climbing': 2.0,
                           'hover': 30.0, 'transit': 60.0},
        references      = {'x': 0.0, 'y': 0.0, 'z_NED': -1.5, 'psi': 0.0},
        vehicle_name    = 'X4_quadcopter',
        controller_name = 'LQR+I_full_state',
        log_dir         = 'logs',
        log_hz          = 50.0,
    )

    waypoints = [
        np.array([0.0,  0.0, -1.5, 0.0]),   # wp0: hover
        np.array([5.0,  3.0, -1.5, 0.0]),   # wp1: NE transit
        np.array([5.0,  3.0, -3.0, 0.0]),   # wp2: descend
        np.array([0.0,  0.0, -3.0, 0.0]),   # wp3: return
    ]

    dynamics   = X4Dynamics()
    controller = X4LQGController(
        Adt, Bdt, Cdt, Kdt, Kidt, Ldt, U_e,
        ref          = waypoints[0],
        T_ctrl       = 0.01,
        max_ref_rate = 0.02,    # 2 m/s max ref ramp rate — keeps attitudes small
        waypoints    = waypoints,
        wp_tol       = 0.15,
    )
    return dynamics, controller, config


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(X_hist, U_hist, config):
    dt = config.dt
    N  = X_hist.shape[0]
    t  = np.arange(N) * dt

    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axs[0].plot(t, X_hist[:, 0],  label='North x [m]')
    axs[0].plot(t, X_hist[:, 2],  label='East  y [m]')
    axs[0].plot(t, -X_hist[:, 4], label='Alt AGL [m]')
    axs[0].set_ylabel('Position [m]')
    axs[0].legend(fontsize=8, ncol=3)
    axs[0].grid(True)
    axs[0].set_title('X4 Quadcopter — LQR+I waypoint following (sim_platform)')

    # Euler angles from quaternion
    from sim.quaternion import quat_to_euler, normalize_quaternion
    eul = np.zeros((N, 3))
    for i in range(N):
        q = X_hist[i, 6:10]
        n = np.linalg.norm(q)
        if n > 1e-10:
            q = q / n
        phi, theta, psi = quat_to_euler(q)
        eul[i] = [phi, theta, psi]

    axs[1].plot(t, np.degrees(eul[:, 0]), label='Roll φ [°]')
    axs[1].plot(t, np.degrees(eul[:, 1]), label='Pitch θ [°]')
    axs[1].plot(t, np.degrees(eul[:, 2]), label='Yaw ψ [°]')
    axs[1].set_ylabel('Attitude [°]')
    axs[1].legend(fontsize=8)
    axs[1].grid(True)

    for j, lbl in enumerate(['m1', 'm2', 'm3', 'm4']):
        axs[2].plot(t, U_hist[:, j], label=lbl, lw=1.0)
    axs[2].axhline(U_e_eq, color='gray', ls=':', lw=0.8, label=f'U_e={U_e_eq:.1f}')
    axs[2].set_ylabel('Motor cmd')
    axs[2].set_xlabel('Time [s]')
    axs[2].legend(fontsize=8, ncol=5)
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
