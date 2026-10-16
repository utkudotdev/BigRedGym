"""Evaluate max -> min -> max height steps with three parallel initializations.

Usage:
    uv run scripts/test_height_tracking.py --task go2trot --headless --no-show
    uv run scripts/test_height_tracking.py --task go2trot --load_run RUN \
        --checkpoint 1500 --hold-time 2

Height limits default to commands.ranges.base_height. Initial states are sampled
independently from the task's configured reset ranges; pushes are disabled.
All commands except base height (column 3) are held at zero. Trials are never
automatically reset, and first termination times are marked on the plot.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .play import get_play_parser, setup
else:
    from play import get_play_parser, setup


def get_args(argv=None):
    parser = get_play_parser()
    parser.description = __doc__
    parser.set_defaults(num_envs=3, keyboard=False)
    parser.add_argument("--min-height", type=float)
    parser.add_argument("--max-height", type=float)
    parser.add_argument(
        "--hold-time",
        type=float,
        default=2.0,
        help="Hold duration at each extremum [s]",
    )
    parser.add_argument(
        "--reset-mode",
        choices=["reset_to_range", "reset_to_basic"],
        default="reset_to_range",
    )
    parser.add_argument("--output", type=Path, default=Path("height_tracking.png"))
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the plot in addition to saving it",
    )
    args = parser.parse_args(argv)
    if args.num_envs != 3:
        parser.error("height tracking uses exactly three environments")
    if not np.isfinite(args.hold_time) or args.hold_time <= 0:
        parser.error("--hold-time must be finite and positive")
    return args


def height_trajectory(dt, min_height, max_height, hold_time):
    """Hold max, then step directly to min, then directly back to max."""
    if not np.isfinite([min_height, max_height]).all() or min_height >= max_height:
        raise ValueError("Height limits must be finite with min_height < max_height")
    duration = 3 * hold_time
    times = np.arange(int(np.ceil(duration / dt)) + 1) * dt
    heights = np.where(
        (times >= hold_time) & (times < 2 * hold_time), min_height, max_height
    )
    return times, heights


@torch.no_grad()
def evaluate(env, runner, times, target):
    """Run without resets, retaining failures instead of hiding them with new trials."""
    if env.commands.shape[1] != 4 or "base_height" not in env.command_ranges:
        raise ValueError("This test requires a height-command task such as go2trot")
    positions = np.empty((len(times), env.num_envs, 3))
    first_termination = np.full(env.num_envs, np.nan)
    positions[0] = env.root_states[:, :3].detach().cpu().numpy()
    for i in range(len(times) - 1):
        env.commands.zero_()
        env.commands[:, 3] = float(target[i])
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        positions[i + 1] = env.root_states[:, :3].detach().cpu().numpy()
        terminated = env.terminated.detach().cpu().numpy()
        first_termination[terminated & np.isnan(first_termination)] = times[i + 1]
        env.check_exit()
    return positions, first_termination


def plot_results(times, target, positions, first_termination, output, show):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (height_ax, drift_ax) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    height_ax.step(times, target, "k--", where="post", label="Command", linewidth=2)
    for trial in range(positions.shape[1]):
        height = positions[:, trial, 2]
        rmse = np.sqrt(np.mean((height - target) ** 2))
        (line,) = height_ax.plot(
            times, height, label=f"Init {trial + 1} (RMSE {rmse:.3f} m)"
        )
        drift = np.linalg.norm(
            positions[:, trial, :2] - positions[0, trial, :2], axis=1
        )
        drift_ax.plot(times, drift, color=line.get_color(), label=f"Init {trial + 1}")
        failure = first_termination[trial]
        if np.isfinite(failure):
            height_ax.axvline(
                failure,
                color=line.get_color(),
                linestyle=":",
                label=f"Init {trial + 1} terminated",
            )
        print(
            f"Init {trial + 1}: height RMSE={rmse:.4f} m, "
            f"max XY drift={drift.max():.4f} m, first termination={failure:.2f} s"
        )
    height_ax.set(
        title="Stationary height tracking: max → min → max", ylabel="Base height [m]"
    )
    drift_ax.set(xlabel="Time [s]", ylabel="XY drift from initial position [m]")
    for ax in (height_ax, drift_ax):
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved plot to {output}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None):
    args = get_args(argv)
    duration = 3 * args.hold_time
    with torch.no_grad():
        env, runner = setup(
            args, reset_mode=args.reset_mode, episode_length_s=duration + 1
        )
        try:
            if "base_height" not in env.command_ranges:
                raise ValueError(
                    "Task must support commands.ranges.base_height (e.g. go2trot)"
                )
            low, high = env.command_ranges["base_height"]
            times, target = height_trajectory(
                env.dt,
                low if args.min_height is None else args.min_height,
                high if args.max_height is None else args.max_height,
                args.hold_time,
            )
            # Keep resampling beyond even very long user-selected trajectories.
            env.cfg.commands.resampling_time = duration + 2 * env.dt
            positions, failures = evaluate(env, runner, times, target)
        finally:
            env._backend.close()
    plot_results(times, target, positions, failures, args.output, args.show)


if __name__ == "__main__":
    main()
