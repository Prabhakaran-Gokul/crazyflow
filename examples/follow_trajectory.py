from collections import deque

import numpy as np
import jax.numpy as jnp

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.visualize import draw_line, draw_trajectory


WIND_START_TIME = 1.5  # seconds
WIND_SCALE = 0.0  # Newton


def wind_field(pos: jnp.ndarray, time: float) -> jnp.ndarray:
    """Return a position-dependent wind force in the world frame.

    Replace this function with your own wind model if you already have a field.
    """
    gust = jnp.stack(
        [
            0.0 * pos[..., 1] + 0.1,
            0.0 * pos[..., 0],
            0.0 * pos[..., 2],
        ],
        axis=-1,
    )

    return WIND_SCALE * gust


def wind_disturbance_fn(data: SimData) -> SimData:
    time = (data.core.steps / data.core.freq)[:, 0][:, None]
    wind = jnp.where(time[..., None] >= WIND_START_TIME, wind_field(data.states.pos, time), 0.0)
    states = data.states.replace(force=wind, torque=jnp.zeros_like(data.states.torque))
    return data.replace(states=states)


def draw_wind_field(
    sim: Sim,
    xlim: tuple[float, float] = (-0.2, 1.7),
    ylim: tuple[float, float] = (-0.2, 1.1),
    z: float = 0.3,
    nx: int = 7,
    ny: int = 5,
    arrow_scale: float = 0.5,
) -> None:
    if sim.viewer is None:
        return

    t = float(np.asarray(sim.time)[0, 0])
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    base = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    points = np.zeros((base.shape[0], 3), dtype=np.float32)
    points[:, :2] = base
    points[:, 2] = z

    pos = jnp.asarray(points)[None, ...]
    time = jnp.array([[t]], dtype=pos.dtype)
    wind = np.asarray(wind_field(pos, time))[0]

    max_mag = np.max(np.linalg.norm(wind, axis=-1))
    for i in range(points.shape[0]):
        start = points[i]
        end = start + arrow_scale * wind[i]
        strength = 0.0 if max_mag < 1e-9 else float(np.linalg.norm(wind[i]) / max_mag)
        rgba = np.array([strength, 0.2, 1.0 - 0.8 * strength, 0.9], dtype=np.float32)
        draw_line(sim, np.stack([start, end]), rgba=rgba, start_size=1.6, end_size=1.6)


def follow_trajectory(sim: Sim, waypoints: np.ndarray, seconds_per_waypoint: float = 1.5) -> None:
    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError("waypoints must have shape (N, 3).")

    cmd = np.zeros((sim.n_worlds, sim.n_drones, 13), dtype=np.float32)
    start_pos = np.array(sim.data.states.pos, dtype=np.float32)
    trail = deque(maxlen=200_000_000)
    trail_stride = 2
    trail_colors = np.tile(np.array([[0.95, 0.2, 0.2, 1.0]], dtype=np.float32), (sim.n_drones, 1))
    frame_idx = 0

    for target_pos in waypoints:
        n_steps = max(1, int(seconds_per_waypoint * sim.control_freq))
        target_pos = np.asarray(target_pos, dtype=np.float32)

        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            cmd[..., :3] = (1.0 - alpha) * start_pos + alpha * target_pos
            sim.state_control(cmd)
            sim.step(sim.freq // sim.control_freq)
            if frame_idx % trail_stride == 0:
                trail.append(np.array(sim.data.states.pos[0, :], dtype=np.float32))

            draw_trajectory(
                sim,
                waypoints,
                line_rgba=np.array([0.2, 0.9, 0.9, 0.8], dtype=np.float32),
                waypoint_rgba=np.array([1.0, 0.6, 0.1, 1.0], dtype=np.float32),
                line_start_size=1.8,
                line_end_size=1.8,
                waypoint_size=0.008,
            )
            if len(trail) > 1:
                lines = np.array(trail)
                for drone_idx in range(sim.n_drones):
                    draw_line(
                        sim,
                        lines[:, drone_idx, :],
                        rgba=trail_colors[drone_idx],
                        start_size=0.3,
                        end_size=2.2,
                    )
            draw_wind_field(sim)
            sim.render()
            frame_idx += 1

        start_pos = np.broadcast_to(target_pos, start_pos.shape).copy()


def main() -> None:
    sim = Sim(control=Control.state)
    sim.step_pipeline = sim.step_pipeline[:2] + (wind_disturbance_fn,) + sim.step_pipeline[2:]
    sim.build_step_fn()
    sim.reset()
    sim.render()

    # Waypoints [x, y, z]. Replace these with your planner output if desired.
    waypoints = np.array(
        [
            [0.0, 0.0, 0.3],
            [0.4, 0.0, 0.3],
            [0.4, 0.4, 0.3],
            [0.0, 0.4, 0.3],
            [0.0, 0.0, 0.3],
        ],
        dtype=np.float32,
    )

    follow_trajectory(sim, waypoints, seconds_per_waypoint=1.5)
    sim.close()


if __name__ == "__main__":
    main()
