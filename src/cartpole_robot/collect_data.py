from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3

from cartpole_robot.registration import ENV_ID, make_env


DEFAULT_TD3_MODEL = Path("models/best/td3_cartpole_swingup_best_20260628-012620.zip")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect transition data from CartPoleSwingUp-v0."
    )
    parser.add_argument("--output", type=Path, default=Path("datasets/swingup_random.npz"))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", choices=("random", "td3"), default="random")
    parser.add_argument("--model", type=Path, default=DEFAULT_TD3_MODEL)
    parser.add_argument(
        "--action-noise",
        type=float,
        default=0.0,
        help="Gaussian noise added to policy actions in normalized action units.",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-fps", type=float, default=25.0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(
        render_mode="human" if args.render else None,
        max_episode_steps=args.max_steps,
    )
    render_delay = 1 / args.render_fps if args.render and args.render_fps > 0 else 0.0
    rng = np.random.default_rng(args.seed)
    model = None

    if args.policy == "td3":
        if not args.model.exists():
            raise SystemExit(f"Model not found: {args.model}")
        model = TD3.load(args.model, env=env, device=args.device)

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []

    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            env.action_space.seed(args.seed + episode)

            for step in range(args.max_steps):
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(observation, deterministic=True)

                if args.action_noise > 0:
                    action = action + rng.normal(
                        loc=0.0,
                        scale=args.action_noise,
                        size=env.action_space.shape,
                    )

                action = np.clip(action, env.action_space.low, env.action_space.high)
                next_observation, reward, terminated, truncated, _ = env.step(action)

                observations.append(np.asarray(observation, dtype=np.float32))
                actions.append(np.asarray(action, dtype=np.float32))
                next_observations.append(np.asarray(next_observation, dtype=np.float32))
                rewards.append(float(reward))
                terminated_flags.append(bool(terminated))
                truncated_flags.append(bool(truncated))
                episode_ids.append(episode)
                step_ids.append(step)

                if render_delay > 0:
                    time.sleep(render_delay)

                observation = next_observation

                if terminated or truncated:
                    break
    finally:
        env.close()

    metadata = {
        "env_id": ENV_ID,
        "policy": args.policy,
        "model": str(args.model) if args.policy == "td3" else None,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "action_noise": args.action_noise,
    }

    np.savez_compressed(
        args.output,
        obs=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        next_obs=np.asarray(next_observations, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated_flags, dtype=np.bool_),
        truncated=np.asarray(truncated_flags, dtype=np.bool_),
        episode=np.asarray(episode_ids, dtype=np.int32),
        step=np.asarray(step_ids, dtype=np.int32),
        metadata=np.asarray(json.dumps(metadata)),
    )

    print(f"Saved {len(rewards):,} transitions to {args.output}")
    print(f"Observation shape: {np.asarray(observations).shape}")
    print(f"Action shape:      {np.asarray(actions).shape}")
