from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from cartpole_robot.world_model import WorldModelEnsemble, load_transition_data


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MLP ensemble world model."
    )
    parser.add_argument("--dataset", type=Path, default=Path("datasets/swingup_random.npz"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("world_models/cartpole_swingup_ensemble.pt"),
    )
    parser.add_argument("--rollout-horizon", type=int, default=25)
    parser.add_argument("--rollout-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/world_model_eval"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def one_step_metrics(
    ensemble: WorldModelEnsemble,
    obs: np.ndarray,
    actions: np.ndarray,
    next_obs: np.ndarray,
    rewards: np.ndarray,
) -> dict[str, float]:
    predicted_next_obs, predicted_rewards, disagreement = ensemble.predict(obs, actions)
    obs_error = predicted_next_obs - next_obs
    reward_error = predicted_rewards - rewards

    return {
        "one_step/obs_rmse": float(np.sqrt(np.mean(obs_error**2))),
        "one_step/reward_rmse": float(np.sqrt(np.mean(reward_error**2))),
        "one_step/mean_disagreement": float(np.mean(disagreement)),
    }


def valid_rollout_starts(
    episode: np.ndarray,
    step: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    horizon: int,
) -> list[int]:
    starts: list[int] = []

    for index in range(0, len(episode) - horizon):
        end_index = index + horizon
        same_episode = episode[index] == episode[end_index]
        contiguous = step[end_index] - step[index] == horizon
        no_done_inside = not np.any(terminated[index:end_index] | truncated[index:end_index])

        if same_episode and contiguous and no_done_inside:
            starts.append(index)

    return starts


def rollout_metrics(
    ensemble: WorldModelEnsemble,
    *,
    obs: np.ndarray,
    actions: np.ndarray,
    next_obs: np.ndarray,
    rewards: np.ndarray,
    starts: np.ndarray,
    horizon: int,
) -> dict[str, float]:
    if len(starts) == 0:
        return {
            "rollout/obs_rmse": float("nan"),
            "rollout/reward_rmse": float("nan"),
            "rollout/mean_disagreement": float("nan"),
        }

    obs_errors = []
    reward_errors = []
    disagreements = []

    for start in starts:
        predicted_obs = obs[start : start + 1].copy()

        for offset in range(horizon):
            action = actions[start + offset : start + offset + 1]
            predicted_next_obs, predicted_reward, disagreement = ensemble.predict(
                predicted_obs,
                action,
            )

            true_next_obs = next_obs[start + offset : start + offset + 1]
            true_reward = rewards[start + offset : start + offset + 1]

            obs_errors.append(predicted_next_obs - true_next_obs)
            reward_errors.append(predicted_reward - true_reward)
            disagreements.append(disagreement)
            predicted_obs = predicted_next_obs

    obs_error_array = np.concatenate(obs_errors, axis=0)
    reward_error_array = np.concatenate(reward_errors, axis=0)
    disagreement_array = np.concatenate(disagreements, axis=0)

    return {
        "rollout/obs_rmse": float(np.sqrt(np.mean(obs_error_array**2))),
        "rollout/reward_rmse": float(np.sqrt(np.mean(reward_error_array**2))),
        "rollout/mean_disagreement": float(np.mean(disagreement_array)),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    if not args.model.exists():
        raise SystemExit(f"World model not found: {args.model}")

    rng = np.random.default_rng(args.seed)
    transitions = load_transition_data(args.dataset)
    ensemble = WorldModelEnsemble.load(args.model, device=args.device)

    metrics = one_step_metrics(
        ensemble,
        transitions.obs,
        transitions.actions,
        transitions.next_obs,
        transitions.rewards,
    )

    starts = valid_rollout_starts(
        transitions.episode,
        transitions.step,
        transitions.terminated,
        transitions.truncated,
        args.rollout_horizon,
    )
    if len(starts) > args.rollout_count:
        starts = rng.choice(starts, size=args.rollout_count, replace=False).tolist()

    metrics.update(
        rollout_metrics(
            ensemble,
            obs=transitions.obs,
            actions=transitions.actions,
            next_obs=transitions.next_obs,
            rewards=transitions.rewards,
            starts=np.asarray(starts, dtype=np.int32),
            horizon=args.rollout_horizon,
        )
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.log_dir))
    try:
        for key, value in metrics.items():
            writer.add_scalar(key, value, 0)
    finally:
        writer.close()

    print(f"Evaluated {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Rollout starts: {len(starts)}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
