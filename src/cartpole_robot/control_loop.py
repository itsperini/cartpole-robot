from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class ControlLoopConfig:
    observation_delay_steps: int = 0
    action_delay_steps: int = 0
    observation_jitter_steps: int = 0
    action_jitter_steps: int = 0
    jitter_probability: float = 0.0
    action_slew_rate: float | None = None
    action_deadband: float = 0.0
    observation_lowpass_alpha: float | None = None
    observation_quantization: float | tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILES: dict[str, ControlLoopConfig] = {
    "none": ControlLoopConfig(),
    "policy_25hz_latency": ControlLoopConfig(
        observation_delay_steps=1,
        action_delay_steps=1,
    ),
    "policy_25hz_jitter": ControlLoopConfig(
        observation_delay_steps=1,
        action_delay_steps=1,
        observation_jitter_steps=1,
        action_jitter_steps=1,
        jitter_probability=0.2,
    ),
    "policy_25hz_hardware_safe": ControlLoopConfig(
        observation_delay_steps=1,
        action_delay_steps=1,
        observation_jitter_steps=1,
        action_jitter_steps=1,
        jitter_probability=0.15,
        action_slew_rate=0.25,
        action_deadband=0.015,
        observation_lowpass_alpha=0.75,
        observation_quantization=(0.0005, 0.005, 0.0005, 0.0005, 0.01),
    ),
    "policy_25hz_hardware_delay": ControlLoopConfig(
        observation_delay_steps=2,
        action_delay_steps=2,
        observation_jitter_steps=1,
        action_jitter_steps=1,
        jitter_probability=0.15,
        action_slew_rate=0.25,
        action_deadband=0.015,
        observation_lowpass_alpha=0.75,
        observation_quantization=(0.0005, 0.005, 0.0005, 0.0005, 0.01),
    ),
}


class ControlLoopWrapper(gym.Wrapper):
    """Optional hardware-like timing and command interface wrapper."""

    def __init__(
        self,
        env: gym.Env,
        config: ControlLoopConfig,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(env)
        if not isinstance(self.env.observation_space, gym.spaces.Box):
            raise TypeError("ControlLoopWrapper requires a Box observation space.")
        if not isinstance(self.env.action_space, gym.spaces.Box):
            raise TypeError("ControlLoopWrapper requires a Box action space.")

        self.config = config
        self._rng = np.random.default_rng(seed)
        self._action_history: deque[np.ndarray] = deque(
            maxlen=self._max_action_delay + 1
        )
        self._observation_history: deque[np.ndarray] = deque(
            maxlen=self._max_observation_delay + 1
        )
        self._last_commanded_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._last_filtered_observation: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        observation, info = self.env.reset(seed=seed, options=options)
        self._last_filtered_observation = None
        observation = self._prepare_observation(observation)

        self._last_commanded_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._last_filtered_observation = observation.copy()
        self._action_history.clear()
        self._observation_history.clear()
        for _ in range(self._max_action_delay + 1):
            self._action_history.append(self._last_commanded_action.copy())
        for _ in range(self._max_observation_delay + 1):
            self._observation_history.append(observation.copy())

        info = dict(info)
        info.update(self._info(0, 0, self._last_commanded_action))
        return self._delayed_observation(0), info

    def step(self, action):
        command = self._shape_action(action)
        action_delay = self._sample_delay(
            self.config.action_delay_steps,
            self.config.action_jitter_steps,
        )
        observation_delay = self._sample_delay(
            self.config.observation_delay_steps,
            self.config.observation_jitter_steps,
        )

        self._action_history.append(command.copy())
        applied_action = self._delayed_action(action_delay)
        observation, reward, terminated, truncated, info = self.env.step(applied_action)

        observation = self._prepare_observation(observation)
        self._observation_history.append(observation.copy())
        returned_observation = self._delayed_observation(observation_delay)

        info = dict(info)
        info.update(self._info(action_delay, observation_delay, command, applied_action))
        return returned_observation, reward, terminated, truncated, info

    @property
    def _max_action_delay(self) -> int:
        return max(
            int(self.config.action_delay_steps) + int(self.config.action_jitter_steps),
            0,
        )

    @property
    def _max_observation_delay(self) -> int:
        return max(
            int(self.config.observation_delay_steps)
            + int(self.config.observation_jitter_steps),
            0,
        )

    def _shape_action(self, action: Any) -> np.ndarray:
        command = np.asarray(action, dtype=np.float32)
        command = np.clip(command, self.action_space.low, self.action_space.high)

        if self.config.action_deadband > 0:
            deadband = float(self.config.action_deadband)
            command = np.where(np.abs(command) < deadband, 0.0, command)

        if self.config.action_slew_rate is not None:
            max_delta = float(self.config.action_slew_rate)
            delta = np.clip(
                command - self._last_commanded_action,
                -max_delta,
                max_delta,
            )
            command = self._last_commanded_action + delta

        command = np.clip(command, self.action_space.low, self.action_space.high)
        self._last_commanded_action = command.astype(np.float32)
        return self._last_commanded_action.copy()

    def _delayed_action(self, delay_steps: int) -> np.ndarray:
        return list(self._action_history)[-(delay_steps + 1)].copy()

    def _prepare_observation(self, observation: Any) -> np.ndarray:
        prepared = np.asarray(observation, dtype=np.float32)

        if self.config.observation_lowpass_alpha is not None:
            alpha = float(self.config.observation_lowpass_alpha)
            if not 0.0 < alpha <= 1.0:
                raise ValueError("observation_lowpass_alpha must be in (0, 1].")
            if self._last_filtered_observation is None:
                self._last_filtered_observation = prepared.copy()
            else:
                self._last_filtered_observation = (
                    alpha * prepared + (1.0 - alpha) * self._last_filtered_observation
                ).astype(np.float32)
            prepared = self._last_filtered_observation.copy()

        quantization = self._observation_quantization(prepared.shape)
        if quantization is not None:
            prepared = prepared.copy()
            quantized = quantization > 0.0
            prepared[quantized] = (
                np.round(prepared[quantized] / quantization[quantized])
                * quantization[quantized]
            )

        return prepared.astype(np.float32)

    def _delayed_observation(self, delay_steps: int) -> np.ndarray:
        return list(self._observation_history)[-(delay_steps + 1)].copy()

    def _sample_delay(self, base_steps: int, jitter_steps: int) -> int:
        delay = max(int(base_steps), 0)
        jitter = max(int(jitter_steps), 0)
        probability = float(self.config.jitter_probability)
        if jitter <= 0 or probability <= 0.0:
            return delay
        if self._rng.random() > min(probability, 1.0):
            return delay
        return delay + int(self._rng.integers(1, jitter + 1))

    def _observation_quantization(
        self,
        shape: tuple[int, ...],
    ) -> np.ndarray | None:
        configured = self.config.observation_quantization
        if configured is None:
            return None
        if isinstance(configured, (int, float)):
            value = float(configured)
            if value <= 0.0:
                return None
            return np.full(shape, value, dtype=np.float32)

        quantization = np.asarray(configured, dtype=np.float32)
        if quantization.shape != shape:
            raise ValueError(
                "observation_quantization must be scalar or match observation shape "
                f"{shape}, got {quantization.shape}."
            )
        return quantization.astype(np.float32)

    def _info(
        self,
        action_delay: int,
        observation_delay: int,
        commanded_action: np.ndarray,
        applied_action: np.ndarray | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "control_loop_action_delay_steps": int(action_delay),
            "control_loop_observation_delay_steps": int(observation_delay),
            "control_loop_action_commanded": commanded_action.astype(np.float32),
            "control_loop_config": self.config.to_dict(),
        }
        if applied_action is not None:
            payload["control_loop_action_applied"] = applied_action.astype(np.float32)
        return payload


def control_loop_profile_names() -> list[str]:
    return sorted(PROFILES)


def resolve_control_loop_config(profile: str) -> ControlLoopConfig:
    try:
        return PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(control_loop_profile_names())
        raise ValueError(
            f"Unknown control loop profile '{profile}'. Choices: {choices}."
        ) from exc


def wrap_control_loop(
    env: gym.Env,
    *,
    profile: str = "none",
    seed: int | None = None,
) -> gym.Env:
    config = resolve_control_loop_config(profile)
    if profile == "none":
        return env
    return ControlLoopWrapper(env, config, seed=seed)
