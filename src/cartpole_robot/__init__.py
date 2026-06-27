from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import gymnasium as gym


ENVIRONMENTS = {
    "cartpole": "CartPole-v1",
    "mujoco": "InvertedPendulum-v5",
}


@dataclass(frozen=True)
class EpisodeResult:
    env_id: str
    episode: int
    steps: int
    reward: float
    finished: str


def run_episode(env_id: str, episode: int, seed: int, max_steps: int, render: bool) -> EpisodeResult:
    env = gym.make(env_id, render_mode="human" if render else None)

    try:
        env.action_space.seed(seed)
        env.reset(seed=seed)

        total_reward = 0.0
        finished = "step-limit"

        for step in range(1, max_steps + 1):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            if terminated:
                finished = "terminated"
                break

            if truncated:
                finished = "truncated"
                break
        else:
            step = max_steps

        return EpisodeResult(
            env_id=env_id,
            episode=episode,
            steps=step,
            reward=total_reward,
            finished=finished,
        )
    finally:
        env.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny random-policy demo for Gymnasium CartPole and MuJoCo."
    )
    parser.add_argument(
        "--env",
        choices=("cartpole", "mujoco", "both"),
        default="both",
        help="Which environment to run.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per environment.")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open a viewer window while the environment runs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    selected_envs = ENVIRONMENTS if args.env == "both" else {args.env: ENVIRONMENTS[args.env]}

    for env_name, env_id in selected_envs.items():
        print(f"\n{env_name}: {env_id}")

        for episode in range(1, args.episodes + 1):
            result = run_episode(
                env_id=env_id,
                episode=episode,
                seed=args.seed + episode,
                max_steps=args.max_steps,
                render=args.render,
            )
            print(
                f"  episode {result.episode}: "
                f"{result.steps:>3} steps, "
                f"reward={result.reward:>8.2f}, "
                f"done={result.finished}"
            )
