from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cartpole_robot.algorithms import load_policy
from cartpole_robot.registration import ENV_ID, make_env
from cartpole_robot.robustness import wrap_history


UPRIGHT_THRESHOLD_RADIANS = 0.2
STABLE_ANGULAR_VELOCITY = 1.0


@dataclass(frozen=True)
class EpisodeMetrics:
    episode: int
    reward: float
    steps: int
    terminated: bool
    truncated: bool
    success: bool
    upright_fraction: float
    stable_fraction: float
    mean_abs_cart_position: float
    max_abs_cart_position: float
    mean_abs_action: float
    mean_abs_force: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a TD3, SAC, or PPO cart-pole swing-up policy."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "models/best/td3_cartpole_swingup_best_20260628-012620.zip"
        ),
    )
    parser.add_argument(
        "--algo",
        default="auto",
        choices=["auto", "td3", "sac", "ppo"],
        help="Algorithm that created the model. Auto infers from the path when possible.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-fps", type=float, default=25.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--observation-history-steps",
        type=int,
        default=1,
        help="Number of recent observations exposed to the policy.",
    )
    parser.add_argument(
        "--action-history-steps",
        type=int,
        default=0,
        help="Number of previous commanded actions exposed to the policy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args(argv)


def evaluate_episode(
    *,
    env,
    model,
    episode: int,
    seed: int,
    max_steps: int,
    deterministic: bool,
    render_delay: float,
    observation_history_steps: int,
) -> EpisodeMetrics:
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    truncated = False
    upright_steps = 0
    stable_steps = 0
    abs_cart_positions: list[float] = []
    abs_actions: list[float] = []
    abs_forces: list[float] = []

    upright_threshold = float(np.cos(UPRIGHT_THRESHOLD_RADIANS))

    for step in range(1, max_steps + 1):
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        base_offset = (observation_history_steps - 1) * 5
        current_observation = np.asarray(
            observation[base_offset : base_offset + 5],
            dtype=np.float32,
        )
        cart_position = float(current_observation[0])
        pole_cos = float(current_observation[3])
        pole_angular_velocity = float(current_observation[4])
        is_upright = pole_cos > upright_threshold
        is_stable = is_upright and abs(pole_angular_velocity) < STABLE_ANGULAR_VELOCITY

        upright_steps += int(is_upright)
        stable_steps += int(is_stable)
        abs_cart_positions.append(abs(cart_position))
        abs_actions.append(float(np.mean(np.abs(action))))
        abs_forces.append(abs(float(info.get("force", 0.0))))

        if render_delay > 0:
            time.sleep(render_delay)

        if terminated or truncated:
            break

    upright_fraction = upright_steps / step
    stable_fraction = stable_steps / step
    success = (not terminated) and upright_fraction >= 0.25 and stable_fraction >= 0.15

    return EpisodeMetrics(
        episode=episode,
        reward=total_reward,
        steps=step,
        terminated=terminated,
        truncated=truncated,
        success=success,
        upright_fraction=upright_fraction,
        stable_fraction=stable_fraction,
        mean_abs_cart_position=float(np.mean(abs_cart_positions)),
        max_abs_cart_position=float(np.max(abs_cart_positions)),
        mean_abs_action=float(np.mean(abs_actions)),
        mean_abs_force=float(np.mean(abs_forces)),
    )


def summarize(episodes: list[EpisodeMetrics]) -> dict[str, float]:
    keys = (
        "reward",
        "steps",
        "upright_fraction",
        "stable_fraction",
        "mean_abs_cart_position",
        "max_abs_cart_position",
        "mean_abs_action",
        "mean_abs_force",
    )
    summary: dict[str, float] = {
        "episodes": float(len(episodes)),
        "success_rate": float(np.mean([episode.success for episode in episodes])),
        "termination_rate": float(np.mean([episode.terminated for episode in episodes])),
    }

    for key in keys:
        values = np.array([getattr(episode, key) for episode in episodes], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())

    return summary


def print_summary(algorithm: str, model_path: Path, summary: dict[str, float]) -> None:
    print(f"\nEvaluated {algorithm.upper()} policy on {ENV_ID}")
    print(f"Model: {model_path}")
    print(f"Episodes: {int(summary['episodes'])}")
    print(f"Mean reward:       {summary['reward_mean']:8.2f} +/- {summary['reward_std']:.2f}")
    print(f"Mean steps:        {summary['steps_mean']:8.1f} +/- {summary['steps_std']:.1f}")
    print(f"Success rate:      {summary['success_rate'] * 100:8.1f}%")
    print(f"Upright fraction:  {summary['upright_fraction_mean'] * 100:8.1f}%")
    print(f"Stable fraction:   {summary['stable_fraction_mean'] * 100:8.1f}%")
    print(f"Mean |cart x|:     {summary['mean_abs_cart_position_mean']:8.3f}")
    print(f"Max |cart x|:      {summary['max_abs_cart_position_mean']:8.3f}")
    print(f"Mean |action|:     {summary['mean_abs_action_mean']:8.3f}")
    print(f"Mean |force|:      {summary['mean_abs_force_mean']:8.3f}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    render_delay = 1 / args.render_fps if args.render and args.render_fps > 0 else 0.0
    env = make_env(
        render_mode="human" if args.render else None,
        max_episode_steps=args.max_steps,
    )
    env = wrap_history(
        env,
        observation_history_steps=args.observation_history_steps,
        action_history_steps=args.action_history_steps,
    )

    try:
        algorithm, model = load_policy(
            args.model,
            algorithm=args.algo,
            env=env,
            device=args.device,
        )

        episodes = [
            evaluate_episode(
                env=env,
                model=model,
                episode=episode,
                seed=args.seed + episode,
                max_steps=args.max_steps,
                deterministic=not args.stochastic,
                render_delay=render_delay,
                observation_history_steps=args.observation_history_steps,
            )
            for episode in range(1, args.episodes + 1)
        ]
    finally:
        env.close()

    summary = summarize(episodes)
    print_summary(algorithm, args.model, summary)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": algorithm,
            "env_id": ENV_ID,
            "model": str(args.model),
            "episodes": [asdict(episode) for episode in episodes],
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Saved evaluation report to {args.output}")
