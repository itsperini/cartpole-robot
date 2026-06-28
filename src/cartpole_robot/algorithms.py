from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm


ALGORITHM_CLASSES: dict[str, type[BaseAlgorithm]] = {
    "td3": TD3,
    "sac": SAC,
    "ppo": PPO,
}


def normalize_algorithm(name: str) -> str:
    algorithm = name.lower()
    if algorithm not in ALGORITHM_CLASSES:
        choices = ", ".join(sorted(ALGORITHM_CLASSES))
        raise ValueError(f"Unknown algorithm '{name}'. Choose one of: {choices}.")
    return algorithm


def infer_algorithm_from_path(path: Path) -> str | None:
    path_text = str(path).lower()
    for algorithm in ALGORITHM_CLASSES:
        if algorithm in path_text:
            return algorithm
    return None


def load_policy(
    model_path: Path,
    *,
    algorithm: str = "auto",
    env: gym.Env | None = None,
    device: str = "auto",
    load_kwargs: dict[str, Any] | None = None,
) -> tuple[str, BaseAlgorithm]:
    kwargs = load_kwargs or {}

    if algorithm != "auto":
        normalized = normalize_algorithm(algorithm)
        model = ALGORITHM_CLASSES[normalized].load(
            model_path,
            env=env,
            device=device,
            **kwargs,
        )
        return normalized, model

    inferred = infer_algorithm_from_path(model_path)
    if inferred is not None:
        model = ALGORITHM_CLASSES[inferred].load(
            model_path,
            env=env,
            device=device,
            **kwargs,
        )
        return inferred, model

    errors: list[str] = []
    for candidate, algorithm_class in ALGORITHM_CLASSES.items():
        try:
            model = algorithm_class.load(
                model_path,
                env=env,
                device=device,
                **kwargs,
            )
            return candidate, model
        except Exception as exc:  # noqa: BLE001 - keep auto-detection user friendly.
            errors.append(f"{candidate}: {exc}")

    joined_errors = "\n".join(errors)
    raise ValueError(
        f"Could not infer which algorithm created {model_path}.\n{joined_errors}"
    )
