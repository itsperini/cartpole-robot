from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from cartpole_robot.eval_world_model import valid_rollout_starts
from cartpole_robot.world_model import WorldModelEnsemble, load_transition_data


OBS_LABELS = [
    "cart position",
    "cart velocity",
    "sin(theta)",
    "cos(theta)",
    "pole angular velocity",
]
ERROR_LABELS = [*OBS_LABELS, "reward"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create richer plots for a trained world-model ensemble."
    )
    parser.add_argument("--dataset", type=Path, default=Path("datasets/swingup_random.npz"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("world_models/cartpole_swingup_ensemble.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/world_model"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs/world_model_viz"))
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--trajectory-count", type=int, default=4)
    parser.add_argument("--error-rollout-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def trajectory_truth(
    *,
    obs: np.ndarray,
    next_obs: np.ndarray,
    rewards: np.ndarray,
    start: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    true_states = np.concatenate(
        [
            obs[start : start + 1],
            next_obs[start : start + horizon],
        ],
        axis=0,
    )
    true_rewards = rewards[start : start + horizon]
    return true_states, true_rewards


def rollout_summary(
    ensemble: WorldModelEnsemble,
    initial_obs: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    member_states, member_rewards = ensemble.rollout_members(initial_obs, actions)
    state_mean = member_states.mean(axis=1)
    state_std = member_states.std(axis=1)
    reward_mean = member_rewards.mean(axis=1)
    reward_std = member_rewards.std(axis=1)
    return state_mean, state_std, reward_mean, reward_std


def plot_trajectory(
    *,
    path: Path,
    title: str,
    true_states: np.ndarray,
    true_rewards: np.ndarray,
    pred_states: np.ndarray,
    pred_state_std: np.ndarray,
    pred_rewards: np.ndarray,
    pred_reward_std: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    flat_axes = axes.ravel()
    state_steps = np.arange(len(true_states))
    reward_steps = np.arange(1, len(true_rewards) + 1)

    for obs_index, label in enumerate(OBS_LABELS):
        axis = flat_axes[obs_index]
        axis.plot(state_steps, true_states[:, obs_index], label="true", linewidth=2)
        axis.plot(state_steps, pred_states[:, obs_index], label="predicted", linewidth=1.8)
        axis.fill_between(
            state_steps,
            pred_states[:, obs_index] - pred_state_std[:, obs_index],
            pred_states[:, obs_index] + pred_state_std[:, obs_index],
            alpha=0.2,
            label="ensemble std" if obs_index == 0 else None,
        )
        axis.set_title(label)
        axis.set_xlabel("step")
        axis.grid(alpha=0.25)

    reward_axis = flat_axes[-1]
    reward_axis.plot(reward_steps, true_rewards, label="true", linewidth=2)
    reward_axis.plot(reward_steps, pred_rewards, label="predicted", linewidth=1.8)
    reward_axis.fill_between(
        reward_steps,
        pred_rewards - pred_reward_std,
        pred_rewards + pred_reward_std,
        alpha=0.2,
        label="ensemble std",
    )
    reward_axis.set_title("reward")
    reward_axis.set_xlabel("step")
    reward_axis.grid(alpha=0.25)

    flat_axes[0].legend(loc="best")
    reward_axis.legend(loc="best")
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    return fig


def plot_angle_overlay(
    *,
    path: Path,
    true_states: np.ndarray,
    pred_states: np.ndarray,
    pred_state_std: np.ndarray,
) -> plt.Figure:
    true_angle = np.arctan2(true_states[:, 2], true_states[:, 3])
    pred_angle = np.arctan2(pred_states[:, 2], pred_states[:, 3])
    angle_std = np.sqrt(pred_state_std[:, 2] ** 2 + pred_state_std[:, 3] ** 2)
    steps = np.arange(len(true_states))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

    axes[0].plot(steps, true_angle, label="true", linewidth=2)
    axes[0].plot(steps, pred_angle, label="predicted", linewidth=1.8)
    axes[0].fill_between(
        steps,
        pred_angle - angle_std,
        pred_angle + angle_std,
        alpha=0.2,
        label="ensemble std",
    )
    axes[0].set_title("Pole angle over time")
    axes[0].set_ylabel("angle")
    axes[0].legend(loc="best")

    axes[1].plot(steps, true_states[:, 4], label="true", linewidth=2)
    axes[1].plot(steps, pred_states[:, 4], label="predicted", linewidth=1.8)
    axes[1].fill_between(
        steps,
        pred_states[:, 4] - pred_state_std[:, 4],
        pred_states[:, 4] + pred_state_std[:, 4],
        alpha=0.2,
    )
    axes[1].set_ylabel("angular velocity")

    axes[2].plot(steps, true_states[:, 0], label="true", linewidth=2)
    axes[2].plot(steps, pred_states[:, 0], label="predicted", linewidth=1.8)
    axes[2].fill_between(
        steps,
        pred_states[:, 0] - pred_state_std[:, 0],
        pred_states[:, 0] + pred_state_std[:, 0],
        alpha=0.2,
    )
    axes[2].set_ylabel("cart position")
    axes[2].set_xlabel("step")

    for axis in axes:
        axis.grid(alpha=0.25)

    fig.savefig(path, dpi=160)
    return fig


def plot_error_heatmap(
    *,
    path: Path,
    true_states: np.ndarray,
    true_rewards: np.ndarray,
    pred_states: np.ndarray,
    pred_rewards: np.ndarray,
) -> plt.Figure:
    state_error = np.abs(pred_states[1:] - true_states[1:])
    reward_error = np.abs(pred_rewards - true_rewards).reshape(-1, 1)
    error_matrix = np.concatenate([state_error, reward_error], axis=1).T

    fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    image = axis.imshow(error_matrix, aspect="auto", interpolation="nearest", cmap="magma")
    axis.set_title("Absolute prediction error heatmap")
    axis.set_xlabel("prediction step")
    axis.set_yticks(np.arange(len(ERROR_LABELS)), ERROR_LABELS)
    axis.set_xticks(np.linspace(0, error_matrix.shape[1] - 1, min(8, error_matrix.shape[1]), dtype=int))
    fig.colorbar(image, ax=axis, label="absolute error")
    fig.savefig(path, dpi=160)
    return fig


def plot_uncertainty_error(
    *,
    path: Path,
    true_states: np.ndarray,
    true_rewards: np.ndarray,
    pred_states: np.ndarray,
    pred_state_std: np.ndarray,
    pred_rewards: np.ndarray,
    pred_reward_std: np.ndarray,
) -> plt.Figure:
    state_error = np.sqrt(np.mean((pred_states[1:] - true_states[1:]) ** 2, axis=1))
    reward_error = np.abs(pred_rewards - true_rewards)
    state_disagreement = pred_state_std[1:].mean(axis=1)
    steps = np.arange(1, len(true_states))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

    axes[0].plot(steps, state_error, label="state prediction error", linewidth=2)
    axes[0].plot(steps, state_disagreement, label="ensemble disagreement", linewidth=2)
    axes[0].set_title("Uncertainty vs actual error")
    axes[0].set_ylabel("observation")
    axes[0].legend(loc="best")

    axes[1].plot(steps, reward_error, label="reward error", linewidth=2)
    axes[1].plot(steps, pred_reward_std, label="reward ensemble std", linewidth=2)
    axes[1].set_ylabel("reward")
    axes[1].legend(loc="best")

    axes[2].scatter(state_disagreement, state_error, s=18, alpha=0.7)
    axes[2].set_xlabel("ensemble disagreement")
    axes[2].set_ylabel("state prediction error")
    axes[2].set_title("Calibration scatter")

    for axis in axes:
        axis.grid(alpha=0.25)

    fig.savefig(path, dpi=160)
    return fig


def horizon_errors(
    *,
    ensemble: WorldModelEnsemble,
    obs: np.ndarray,
    actions: np.ndarray,
    next_obs: np.ndarray,
    rewards: np.ndarray,
    starts: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_errors: list[list[np.ndarray]] = [[] for _ in range(horizon)]
    reward_errors: list[list[np.ndarray]] = [[] for _ in range(horizon)]
    disagreements: list[list[np.ndarray]] = [[] for _ in range(horizon)]

    for start in starts:
        true_states, true_rewards = trajectory_truth(
            obs=obs,
            next_obs=next_obs,
            rewards=rewards,
            start=int(start),
            horizon=horizon,
        )
        pred_states, pred_state_std, pred_rewards, _ = rollout_summary(
            ensemble,
            obs[start],
            actions[start : start + horizon],
        )

        for step in range(horizon):
            state_errors[step].append(pred_states[step + 1] - true_states[step + 1])
            reward_errors[step].append(
                np.asarray([pred_rewards[step] - true_rewards[step]], dtype=np.float32)
            )
            disagreements[step].append(np.asarray([pred_state_std[step + 1].mean()]))

    state_rmse = np.asarray(
        [np.sqrt(np.mean(np.asarray(errors) ** 2)) for errors in state_errors],
        dtype=np.float32,
    )
    reward_rmse = np.asarray(
        [np.sqrt(np.mean(np.asarray(errors) ** 2)) for errors in reward_errors],
        dtype=np.float32,
    )
    disagreement = np.asarray(
        [np.mean(np.asarray(values)) for values in disagreements],
        dtype=np.float32,
    )
    return state_rmse, reward_rmse, disagreement


def plot_horizon_errors(
    *,
    path: Path,
    state_rmse: np.ndarray,
    reward_rmse: np.ndarray,
    disagreement: np.ndarray,
) -> plt.Figure:
    steps = np.arange(1, len(state_rmse) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)

    axes[0].plot(steps, state_rmse, linewidth=2)
    axes[0].set_ylabel("obs RMSE")
    axes[0].set_title("Recursive rollout error over horizon")

    axes[1].plot(steps, reward_rmse, linewidth=2)
    axes[1].set_ylabel("reward RMSE")

    axes[2].plot(steps, disagreement, linewidth=2)
    axes[2].set_ylabel("ensemble disagreement")
    axes[2].set_xlabel("prediction horizon step")

    for axis in axes:
        axis.grid(alpha=0.25)

    fig.savefig(path, dpi=160)
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    if not args.model.exists():
        raise SystemExit(f"World model not found: {args.model}")

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    transitions = load_transition_data(args.dataset)
    ensemble = WorldModelEnsemble.load(args.model, device=args.device)
    starts = valid_rollout_starts(
        transitions.episode,
        transitions.step,
        transitions.terminated,
        transitions.truncated,
        args.horizon,
    )

    if not starts:
        raise SystemExit(
            f"No contiguous non-terminal rollouts found for horizon {args.horizon}."
        )

    selected_count = min(args.trajectory_count, len(starts))
    selected_starts = rng.choice(starts, size=selected_count, replace=False)
    error_count = min(args.error_rollout_count, len(starts))
    error_starts = rng.choice(starts, size=error_count, replace=False)

    writer = SummaryWriter(log_dir=str(args.log_dir))
    written_paths: list[Path] = []

    try:
        for plot_index, start in enumerate(selected_starts):
            start = int(start)
            true_states, true_rewards = trajectory_truth(
                obs=transitions.obs,
                next_obs=transitions.next_obs,
                rewards=transitions.rewards,
                start=start,
                horizon=args.horizon,
            )
            pred_states, pred_state_std, pred_rewards, pred_reward_std = rollout_summary(
                ensemble,
                transitions.obs[start],
                transitions.actions[start : start + args.horizon],
            )
            trajectory_path = args.output_dir / f"trajectory_{plot_index + 1}.png"
            fig = plot_trajectory(
                path=trajectory_path,
                title=f"World-model rollout from dataset index {start}",
                true_states=true_states,
                true_rewards=true_rewards,
                pred_states=pred_states,
                pred_state_std=pred_state_std,
                pred_rewards=pred_rewards,
                pred_reward_std=pred_reward_std,
            )
            writer.add_figure(f"trajectory/{plot_index + 1}", fig, global_step=0)
            plt.close(fig)
            written_paths.append(trajectory_path)

            angle_path = args.output_dir / f"angle_overlay_{plot_index + 1}.png"
            fig = plot_angle_overlay(
                path=angle_path,
                true_states=true_states,
                pred_states=pred_states,
                pred_state_std=pred_state_std,
            )
            writer.add_figure(f"angle_overlay/{plot_index + 1}", fig, global_step=0)
            plt.close(fig)
            written_paths.append(angle_path)

            heatmap_path = args.output_dir / f"error_heatmap_{plot_index + 1}.png"
            fig = plot_error_heatmap(
                path=heatmap_path,
                true_states=true_states,
                true_rewards=true_rewards,
                pred_states=pred_states,
                pred_rewards=pred_rewards,
            )
            writer.add_figure(f"error_heatmap/{plot_index + 1}", fig, global_step=0)
            plt.close(fig)
            written_paths.append(heatmap_path)

            uncertainty_path = args.output_dir / f"uncertainty_error_{plot_index + 1}.png"
            fig = plot_uncertainty_error(
                path=uncertainty_path,
                true_states=true_states,
                true_rewards=true_rewards,
                pred_states=pred_states,
                pred_state_std=pred_state_std,
                pred_rewards=pred_rewards,
                pred_reward_std=pred_reward_std,
            )
            writer.add_figure(f"uncertainty_error/{plot_index + 1}", fig, global_step=0)
            plt.close(fig)
            written_paths.append(uncertainty_path)

        state_rmse, reward_rmse, disagreement = horizon_errors(
            ensemble=ensemble,
            obs=transitions.obs,
            actions=transitions.actions,
            next_obs=transitions.next_obs,
            rewards=transitions.rewards,
            starts=np.asarray(error_starts, dtype=np.int32),
            horizon=args.horizon,
        )
        error_path = args.output_dir / "error_over_horizon.png"
        fig = plot_horizon_errors(
            path=error_path,
            state_rmse=state_rmse,
            reward_rmse=reward_rmse,
            disagreement=disagreement,
        )
        writer.add_figure("error_over_horizon", fig, global_step=0)
        plt.close(fig)
        written_paths.append(error_path)

        for step, value in enumerate(state_rmse, start=1):
            writer.add_scalar("horizon/obs_rmse", float(value), step)
        for step, value in enumerate(reward_rmse, start=1):
            writer.add_scalar("horizon/reward_rmse", float(value), step)
        for step, value in enumerate(disagreement, start=1):
            writer.add_scalar("horizon/disagreement", float(value), step)
    finally:
        writer.close()

    print(f"Wrote {len(written_paths)} plots to {args.output_dir}")
    print(f"TensorBoard figures: {args.log_dir}")
    for path in written_paths:
        print(path)
