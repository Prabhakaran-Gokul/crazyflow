import os

import jax
import numpy as np
from numpy.typing import NDArray

from crazyflow.sim import Sim
from crazyflow.sim.data import SimData

os.environ["SCIPY_ARRAY_API"] = "1"

from scipy.spatial.transform import Rotation as R


WIND_START_TIME = 1.5  # seconds
WIND_FORCE = 0.001      # Newton

def disturbance_fn(data: SimData) -> SimData:
    states = data.states

    wind_force = jax.numpy.zeros_like(states.force)
    wind_force = wind_force.at[..., 0].set(WIND_FORCE)

    states = states.replace(force=states.force + wind_force)

    return data.replace(states=states)


def main(plot: bool = False):
    sim = Sim(control="state")
    control = np.zeros((sim.n_worlds, sim.n_drones, 13))
    control[..., :3] = 0.5

    # First run
    pos, quat = [], []
    sim.reset()
    for _ in range(3 * sim.control_freq):
        sim.state_control(control)
        sim.step(sim.freq // sim.control_freq)
        pos.append(sim.data.states.pos[0, 0])
        quat.append(sim.data.states.quat[0, 0])
        sim.render()

    # Second run
    # We insert the disturbance function into the step pipeline before the integration step. You can
    # inspect the step pipeline with
    # print(sim.step_pipeline)
    # sim.build_step_fn()
    pos_disturbed, quat_disturbed = [], []
    sim.reset()
    disturbance_flag = False
    for _ in range(3 * sim.control_freq):
        sim.state_control(control)
        sim.step(sim.freq // sim.control_freq)
        pos_disturbed.append(sim.data.states.pos[0, 0])
        quat_disturbed.append(sim.data.states.quat[0, 0])
        if sim.time >= 1.5 and not disturbance_flag:
            sim.step_pipeline = sim.step_pipeline[:2] + (disturbance_fn,) + sim.step_pipeline[2:]
            sim.build_step_fn()
            disturbance_flag = True
            print("Setting disturbance")
        sim.render()

    sim.close()
    if plot:
        plot_results(pos, pos_disturbed, quat, quat_disturbed)


def plot_results(
    pos: list[NDArray],
    pos_disturbed: list[NDArray],
    quat: list[NDArray],
    quat_disturbed: list[NDArray],
):
    # Only import if plotting is desired to avoid a dependency on matplotlib
    import matplotlib.pyplot as plt  # noqa: F401

    pos, pos_disturbed = np.array(pos), np.array(pos_disturbed)
    rpy = R.from_quat(quat).as_euler("xyz")
    rpy_disturbed = R.from_quat(quat_disturbed).as_euler("xyz")
    fig, ax = plt.subplots(3, 2, sharex="all", figsize=(10, 6))
    t = np.linspace(0, 3, len(pos))
    # XYZ position
    ax[0, 0].plot(t, pos[:, 0], label="x undisturbed", color="r")
    ax[0, 0].plot(t, pos_disturbed[:, 0], label="x disturbed", color="r", linestyle="--")
    ax[1, 0].plot(t, pos[:, 1], label="y undisturbed", color="g")
    ax[1, 0].plot(t, pos_disturbed[:, 1], label="y perturbed", color="g", linestyle="--")
    ax[2, 0].plot(t, pos[:, 2], label="z undisturbed", color="b")
    ax[2, 0].plot(t, pos_disturbed[:, 2], label="z disturbed", color="b", linestyle="--")
    # RPY angles
    ax[0, 1].plot(t, rpy[:, 0], label="roll undisturbed", color="r")
    ax[0, 1].plot(t, rpy_disturbed[:, 0], label="roll disturbed", color="r", linestyle="--")
    ax[1, 1].plot(t, rpy[:, 1], label="pitch undisturbed", color="g")
    ax[1, 1].plot(t, rpy_disturbed[:, 1], label="pitch disturbed", color="g", linestyle="--")
    ax[2, 1].plot(t, rpy[:, 2], label="yaw undisturbed", color="b")
    ax[2, 1].plot(t, rpy_disturbed[:, 2], label="yaw disturbed", color="b", linestyle="--")
    fig.suptitle("Dynamics with disturbance")
    ax[2, 0].set_xlabel("Time (s)")
    ax[2, 1].set_xlabel("Time (s)")
    for _ax in ax.flatten():
        _ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(plot=True)  # Default is False to disable plotting during testing
