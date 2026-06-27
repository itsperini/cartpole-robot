from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass

import gymnasium as gym

from cartpole_robot.registration import ENV_ID, make_env


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    steps: int
    reward: float
    finished: str


def run_episode(
    env: gym.Env,
    episode: int,
    seed: int,
    render_delay: float,
) -> EpisodeResult:
    env.action_space.seed(seed)
    _, _ = env.reset(seed=seed)

    total_reward = 0.0
    finished = "step-limit"

    for step in range(1, env.spec.max_episode_steps + 1):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        if render_delay > 0:
            time.sleep(render_delay)

        if terminated:
            finished = "terminated"
            break

        if truncated:
            finished = "truncated"
            break

    return EpisodeResult(
        episode=episode,
        steps=step,
        reward=total_reward,
        finished=finished,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a random-policy rollout in the MuJoCo cart-pole swing-up environment."
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open a MuJoCo viewer while the environment runs.",
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=25.0,
        help="Approximate frames per second when rendering. Use 0 for uncapped.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    render_delay = 1 / args.render_fps if args.render and args.render_fps > 0 else 0.0
    env = make_env(
        render_mode="human" if args.render else None,
        max_episode_steps=args.max_steps,
    )

    print(f"\n{ENV_ID}")
    print(f"  observation space: {env.observation_space}")
    print(f"  action space:      {env.action_space}  # normalized cart force")

    try:
        for episode in range(1, args.episodes + 1):
            result = run_episode(
                env=env,
                episode=episode,
                seed=args.seed + episode,
                render_delay=render_delay,
            )
            print(
                f"  episode {result.episode}: "
                f"{result.steps:>3} steps, "
                f"reward={result.reward:>8.2f}, "
                f"done={result.finished}"
            )
    finally:
        env.close()
