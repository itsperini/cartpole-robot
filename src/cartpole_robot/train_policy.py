from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from cartpole_robot.algorithms import ALGORITHM_CLASSES, normalize_algorithm
from cartpole_robot.registration import ENV_ID, make_env, register_env


COMMON_DEFAULTS: dict[str, Any] = {
    "timesteps": 300_000,
    "seed": 7,
    "max_steps": 500,
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "batch_size": 256,
    "eval_every": 10_000,
    "checkpoint_every": 50_000,
    "device": "auto",
    "n_envs": 1,
}


ALGORITHM_DEFAULTS: dict[str, dict[str, Any]] = {
    "td3": {
        "buffer_size": 1_000_000,
        "learning_starts": 10_000,
        "tau": 0.005,
        "action_noise": 0.3,
    },
    "sac": {
        "buffer_size": 1_000_000,
        "learning_starts": 10_000,
        "tau": 0.005,
        "ent_coef": "auto",
    },
    "ppo": {
        "n_steps": 2048,
        "n_epochs": 10,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TD3, SAC, or PPO on the MuJoCo cart-pole swing-up task."
    )
    parser.add_argument(
        "--algo",
        choices=sorted(ALGORITHM_CLASSES),
        default=None,
        help="Algorithm to train. Defaults to the config value, or td3.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional flat TOML config file.",
    )
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts"),
        help="Root folder for timestamped training runs.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run folder name. Refuses to overwrite existing runs.",
    )
    parser.add_argument("--resume", type=Path, default=None)

    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument(
        "--action-noise",
        type=float,
        default=None,
        help="TD3 exploration noise in normalized action units.",
    )
    parser.add_argument(
        "--ent-coef",
        default=None,
        help="SAC entropy coefficient. Usually 'auto'.",
    )

    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=None)
    parser.add_argument("--vf-coef", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    return parser.parse_args(argv)


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    if not path.exists():
        raise SystemExit(f"Config not found: {path}")

    with path.open("rb") as handle:
        return dict(tomllib.load(handle))


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    file_config = load_config(args.config)
    algorithm = normalize_algorithm(args.algo or file_config.get("algo", "td3"))

    config = {
        "algo": algorithm,
        **COMMON_DEFAULTS,
        **ALGORITHM_DEFAULTS[algorithm],
        **file_config,
    }
    config["algo"] = algorithm

    for key in (
        "timesteps",
        "seed",
        "max_steps",
        "learning_rate",
        "gamma",
        "batch_size",
        "eval_every",
        "checkpoint_every",
        "device",
        "n_envs",
        "buffer_size",
        "learning_starts",
        "tau",
        "action_noise",
        "ent_coef",
        "n_steps",
        "n_epochs",
        "gae_lambda",
        "clip_range",
        "vf_coef",
        "max_grad_norm",
        "target_kl",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    return config


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def make_run_dir(artifact_root: Path, algorithm: str, run_id: str | None) -> Path:
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_dir = artifact_root / "runs" / algorithm / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Run directory already exists and is not empty: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_action_noise(config: dict[str, Any], action_dim: int) -> NormalActionNoise:
    sigma = float(config["action_noise"])
    return NormalActionNoise(
        mean=np.zeros(action_dim),
        sigma=sigma * np.ones(action_dim),
    )


def make_training_env(config: dict[str, Any]):
    n_envs = int(config["n_envs"])
    if n_envs <= 0:
        raise ValueError("n_envs must be positive.")

    if n_envs == 1:
        return Monitor(make_env(max_episode_steps=int(config["max_steps"])))

    register_env()
    return make_vec_env(
        ENV_ID,
        n_envs=n_envs,
        seed=int(config["seed"]),
        env_kwargs={"max_episode_steps": int(config["max_steps"])},
    )


def make_model_kwargs(
    algorithm: str,
    config: dict[str, Any],
    env,
    tensorboard_dir: Path,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": config["learning_rate"],
        "gamma": config["gamma"],
        "batch_size": int(config["batch_size"]),
        "verbose": 1,
        "tensorboard_log": str(tensorboard_dir),
        "seed": int(config["seed"]),
        "device": config["device"],
    }

    if algorithm == "td3":
        action_dim = env.action_space.shape[-1]
        kwargs.update(
            {
                "buffer_size": int(config["buffer_size"]),
                "learning_starts": int(config["learning_starts"]),
                "tau": float(config["tau"]),
                "train_freq": (1, "step"),
                "gradient_steps": 1,
                "policy_delay": 2,
                "action_noise": make_action_noise(config, action_dim),
            }
        )
    elif algorithm == "sac":
        kwargs.update(
            {
                "buffer_size": int(config["buffer_size"]),
                "learning_starts": int(config["learning_starts"]),
                "tau": float(config["tau"]),
                "train_freq": (1, "step"),
                "gradient_steps": 1,
                "ent_coef": config["ent_coef"],
            }
        )
    elif algorithm == "ppo":
        kwargs.update(
            {
                "n_steps": int(config["n_steps"]),
                "n_epochs": int(config["n_epochs"]),
                "gae_lambda": float(config["gae_lambda"]),
                "clip_range": float(config["clip_range"]),
                "ent_coef": float(config["ent_coef"]),
                "vf_coef": float(config["vf_coef"]),
                "max_grad_norm": float(config["max_grad_norm"]),
            }
        )
        if "target_kl" in config:
            kwargs["target_kl"] = float(config["target_kl"])

    return kwargs


def load_resume_kwargs(
    algorithm: str,
    config: dict[str, Any],
    env,
    tensorboard_dir: Path,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "env": env,
        "tensorboard_log": str(tensorboard_dir),
        "device": config["device"],
    }

    if algorithm == "td3":
        action_dim = env.action_space.shape[-1]
        kwargs["action_noise"] = make_action_noise(config, action_dim)

    return kwargs


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = resolved_config(args)
    algorithm = config["algo"]
    run_dir = make_run_dir(args.artifact_root, algorithm, args.run_id)
    tensorboard_dir = run_dir / "tensorboard"
    eval_dir = run_dir / "eval"
    checkpoint_dir = run_dir / "checkpoints"

    save_json(run_dir / "config.json", config)

    env = make_training_env(config)
    eval_env = Monitor(make_env(max_episode_steps=int(config["max_steps"])))
    n_envs = int(config["n_envs"])

    callbacks = []
    if int(config["eval_every"]) > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "best"),
                log_path=str(eval_dir),
                eval_freq=max(int(config["eval_every"]) // n_envs, 1),
                deterministic=True,
                render=False,
            )
        )

    if int(config["checkpoint_every"]) > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(int(config["checkpoint_every"]) // n_envs, 1),
                save_path=str(checkpoint_dir),
                name_prefix=f"{algorithm}_cartpole_swingup",
            )
        )

    callback = CallbackList(callbacks) if callbacks else None
    algorithm_class = ALGORITHM_CLASSES[algorithm]

    try:
        if args.resume is not None:
            if not args.resume.exists():
                raise SystemExit(f"Resume model not found: {args.resume}")
            model = algorithm_class.load(
                args.resume,
                **load_resume_kwargs(algorithm, config, env, tensorboard_dir),
            )
        else:
            model = algorithm_class(
                **make_model_kwargs(algorithm, config, env, tensorboard_dir)
            )

        print(f"Training {algorithm.upper()} on {ENV_ID}.")
        print(f"Run directory: {run_dir}")
        print(f"Timesteps: {int(config['timesteps']):,}")
        model.learn(
            total_timesteps=int(config["timesteps"]),
            callback=callback,
            tb_log_name=f"{algorithm}_cartpole_swingup",
            progress_bar=False,
        )

        final_model_path = run_dir / "final_model"
        model.save(final_model_path)
        metadata = {
            "algorithm": algorithm,
            "env_id": ENV_ID,
            "git_commit": current_git_commit(),
            "final_model": str(final_model_path.with_suffix(".zip")),
            "best_model": str(run_dir / "best" / "best_model.zip"),
            "tensorboard_dir": str(tensorboard_dir),
            "eval_dir": str(eval_dir),
            "checkpoint_dir": str(checkpoint_dir),
        }
        save_json(run_dir / "metadata.json", metadata)

        print(f"Saved final model to {final_model_path.with_suffix('.zip')}")
        print(f"Saved metadata to {run_dir / 'metadata.json'}")
    finally:
        env.close()
        eval_env.close()
