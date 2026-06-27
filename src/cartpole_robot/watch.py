from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from stable_baselines3 import TD3

from cartpole_robot.registration import ENV_ID, make_env


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a trained TD3 policy in the MuJoCo cart-pole swing-up environment."
    )
    parser.add_argument("--model", type=Path, default=Path("models/td3_cartpole_swingup.zip"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--render-fps", type=float, default=25.0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    render_delay = 1 / args.render_fps if args.render_fps > 0 else 0.0
    env = make_env(render_mode="human", max_episode_steps=args.max_steps)
    model = TD3.load(args.model, env=env, device=args.device)

    print(f"\nWatching {args.model} on {ENV_ID}")

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
