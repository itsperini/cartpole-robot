from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from cartpole_robot.algorithms import load_policy
from cartpole_robot.registration import ENV_ID, make_env
from cartpole_robot.robustness import make_robust_env, scenario_names, wrap_history


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a trained TD3, SAC, or PPO policy in the MuJoCo cart-pole swing-up environment."
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
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--render-fps", type=float, default=25.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--scenario",
        choices=scenario_names(),
        default="clean",
        help="Robustness scenario to render.",
    )
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
    return parser.parse_args(argv)


def make_watch_env(args: argparse.Namespace):
    env_kwargs = {
        "render_mode": "human",
        "max_episode_steps": args.max_steps,
    }
    if args.scenario == "clean":
        env = make_env(**env_kwargs)
        return wrap_history(
            env,
            observation_history_steps=args.observation_history_steps,
            action_history_steps=args.action_history_steps,
        )

    return make_robust_env(
        args.scenario,
        seed=args.seed,
        observation_history_steps=args.observation_history_steps,
        action_history_steps=args.action_history_steps,
        **env_kwargs,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    render_delay = 1 / args.render_fps if args.render_fps > 0 else 0.0
    env = make_watch_env(args)
    algorithm, model = load_policy(
        args.model,
        algorithm=args.algo,
        env=env,
        device=args.device,
    )

    print(f"\nWatching {algorithm.upper()} policy on {ENV_ID}")
    print(f"Model: {args.model}")
    print(f"Scenario: {args.scenario}")

    try:
        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0

            for step in range(1, args.max_steps + 1):
                action, _ = model.predict(
                    observation,
                    deterministic=not args.stochastic,
                )
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)

                if render_delay > 0:
                    time.sleep(render_delay)

                if terminated or truncated:
                    break

            print(
                f"  episode {episode}: "
                f"{step:>3} steps, "
                f"reward={total_reward:>8.2f}"
            )
    finally:
        env.close()
