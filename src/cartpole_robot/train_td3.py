from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from cartpole_robot.registration import ENV_ID, make_env


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TD3 on the MuJoCo cart-pole swing-up environment."
    )
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--action-noise", type=float, default=0.3)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(make_env(max_episode_steps=args.max_steps))
    eval_env = Monitor(make_env(max_episode_steps=args.max_steps))

    action_dim = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim),
        sigma=args.action_noise * np.ones(action_dim),
    )

    callbacks = []
    if args.eval_every > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(args.model_dir / "best"),
                log_path=str(args.log_dir / "eval"),
                eval_freq=args.eval_every,
                deterministic=True,
                render=False,
            )
        )

    if args.checkpoint_every > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_every,
                save_path=str(args.model_dir / "checkpoints"),
                name_prefix="td3_cartpole_swingup",
            )
        )

    callback = CallbackList(callbacks) if callbacks else None
    final_model_path = args.model_dir / "td3_cartpole_swingup"

    try:
        if args.resume is not None:
            model = TD3.load(
                args.resume,
                env=env,
                action_noise=action_noise,
                tensorboard_log=str(args.log_dir),
                device=args.device,
            )
        else:
            model = TD3(
                "MlpPolicy",
                env,
                action_noise=action_noise,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                tau=args.tau,
                train_freq=(1, "step"),
                gradient_steps=1,
                policy_delay=2,
                verbose=1,
                tensorboard_log=str(args.log_dir),
                seed=args.seed,
                device=args.device,
            )

        print(f"Training TD3 on {ENV_ID} for {args.timesteps:,} timesteps.")
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name="td3_cartpole_swingup",
            progress_bar=False,
        )
        model.save(final_model_path)
        print(f"Saved model to {final_model_path.with_suffix('.zip')}")
    finally:
        env.close()
        eval_env.close()
