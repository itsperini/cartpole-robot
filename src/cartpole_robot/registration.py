from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import register, registry


ENV_ID = "CartPoleSwingUp-v0"
DEFAULT_MAX_EPISODE_STEPS = 500


def register_env() -> None:
    if ENV_ID not in registry:
        register(
            id=ENV_ID,
            entry_point="cartpole_robot.swingup_env:CartPoleSwingUpEnv",
            max_episode_steps=DEFAULT_MAX_EPISODE_STEPS,
        )


def make_env(
    *,
    render_mode: str | None = None,
    max_episode_steps: int | None = DEFAULT_MAX_EPISODE_STEPS,
    **kwargs: Any,
) -> gym.Env:
    register_env()
    make_kwargs: dict[str, Any] = {"render_mode": render_mode, **kwargs}

    if max_episode_steps is not None:
        make_kwargs["max_episode_steps"] = max_episode_steps

    return gym.make(ENV_ID, **make_kwargs)
