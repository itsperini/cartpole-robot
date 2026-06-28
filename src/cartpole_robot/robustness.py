from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from cartpole_robot.registration import DEFAULT_MAX_EPISODE_STEPS, make_env


Range = tuple[float, float] | None


@dataclass(frozen=True)
class RobustnessConfig:
    observation_noise_std: float | tuple[float, ...] = 0.0
    action_noise_std: float = 0.0
    observation_delay_steps: int = 0
    action_delay_steps: int = 0
    cart_mass_scale_range: Range = None
    pole_mass_scale_range: Range = None
    slider_damping_scale_range: Range = None
    hinge_damping_scale_range: Range = None
    slider_frictionloss_range: Range = None
    hinge_frictionloss_range: Range = None
    force_limit_scale_range: Range = None
    gravity_z_range: Range = None
    push_start_step: int = 75
    push_interval_steps: int = 0
    push_probability: float = 0.0
    cart_velocity_impulse_range: Range = None
    pole_angular_velocity_impulse_range: Range = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SCENARIOS: dict[str, RobustnessConfig] = {
    "clean": RobustnessConfig(),
    "sensor_noise": RobustnessConfig(
        observation_noise_std=(0.01, 0.04, 0.01, 0.01, 0.05),
        action_noise_std=0.02,
    ),
    "delay": RobustnessConfig(
        observation_delay_steps=2,
        action_delay_steps=2,
    ),
    "dynamics": RobustnessConfig(
        cart_mass_scale_range=(0.8, 1.2),
        pole_mass_scale_range=(0.75, 1.25),
        slider_damping_scale_range=(0.5, 2.0),
        hinge_damping_scale_range=(0.5, 2.0),
        slider_frictionloss_range=(0.0, 0.03),
        hinge_frictionloss_range=(0.0, 0.01),
        force_limit_scale_range=(0.8, 1.15),
        gravity_z_range=(-10.2, -9.4),
    ),
    "pushes": RobustnessConfig(
        push_start_step=80,
        push_interval_steps=80,
        cart_velocity_impulse_range=(-0.35, 0.35),
        pole_angular_velocity_impulse_range=(-2.5, 2.5),
    ),
    "combined": RobustnessConfig(
        observation_noise_std=(0.015, 0.05, 0.015, 0.015, 0.08),
        action_noise_std=0.03,
        observation_delay_steps=1,
        action_delay_steps=1,
        cart_mass_scale_range=(0.85, 1.15),
        pole_mass_scale_range=(0.8, 1.2),
        slider_damping_scale_range=(0.7, 1.8),
        hinge_damping_scale_range=(0.7, 1.8),
        slider_frictionloss_range=(0.0, 0.02),
        hinge_frictionloss_range=(0.0, 0.008),
        force_limit_scale_range=(0.85, 1.1),
        gravity_z_range=(-10.05, -9.55),
        push_start_step=100,
        push_interval_steps=100,
        cart_velocity_impulse_range=(-0.25, 0.25),
        pole_angular_velocity_impulse_range=(-1.8, 1.8),
    ),
}


@dataclass(frozen=True)
class _NominalDynamics:
    force_limit: float
    actuator_ctrlrange: np.ndarray
    cart_mass: float
    cart_inertia: np.ndarray
    pole_mass: float
    pole_inertia: np.ndarray
    slider_damping: float
    hinge_damping: float
    slider_frictionloss: float
    hinge_frictionloss: float
    gravity: np.ndarray


class RobustnessWrapper(gym.Wrapper):
    """Adds first-pass sim2real stressors without changing the task API."""

    def __init__(
        self,
        env: gym.Env,
        config: RobustnessConfig,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(env)
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._action_queue: deque[np.ndarray] = deque()
        self._observation_queue: deque[np.ndarray] = deque()
        self._last_dynamics: dict[str, float] = {}

        base = self.env.unwrapped
        self._cart_body_id = int(base.model.body("cart").id)
        self._pole_body_id = int(base.model.body("pole").id)
        self._slider_dof_id = self._joint_dof_id("slider")
        self._hinge_dof_id = self._joint_dof_id("hinge")
        self._nominal = self._capture_nominal_dynamics()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._restore_nominal_dynamics()
        self._last_dynamics = self._sample_and_apply_dynamics()
        self._reset_action_delay()

        observation, info = self.env.reset(seed=seed, options=options)
        true_observation = np.asarray(observation, dtype=np.float32).copy()
        observed = self._apply_observation_noise(true_observation)
        self._reset_observation_delay(observed)

        info = dict(info)
        info["true_observation"] = true_observation
        info["robustness_dynamics"] = dict(self._last_dynamics)

        return self._current_observation(observed), info

    def step(self, action):
        self._step_count += 1
        push_info = self._maybe_apply_push()
        applied_action = self._delayed_action(action)
        applied_action = self._apply_action_noise(applied_action)

        observation, reward, terminated, truncated, info = self.env.step(applied_action)
        true_observation = np.asarray(observation, dtype=np.float32).copy()
        observed = self._apply_observation_noise(true_observation)
        returned_observation = self._current_observation(observed)

        info = dict(info)
        info["true_observation"] = true_observation
        info["robustness_step"] = self._step_count
        info["robustness_action_applied"] = np.asarray(applied_action, dtype=np.float32)
        info["robustness_dynamics"] = dict(self._last_dynamics)
        info.update(push_info)

        return returned_observation, reward, terminated, truncated, info

    def close(self) -> None:
        self._restore_nominal_dynamics()
        super().close()

    def _joint_dof_id(self, joint_name: str) -> int:
        base = self.env.unwrapped
        joint_id = int(base.model.joint(joint_name).id)
        return int(base.model.jnt_dofadr[joint_id])

    def _capture_nominal_dynamics(self) -> _NominalDynamics:
        base = self.env.unwrapped
        model = base.model
        return _NominalDynamics(
            force_limit=float(base.force_limit),
            actuator_ctrlrange=model.actuator_ctrlrange[0].copy(),
            cart_mass=float(model.body_mass[self._cart_body_id]),
            cart_inertia=model.body_inertia[self._cart_body_id].copy(),
            pole_mass=float(model.body_mass[self._pole_body_id]),
            pole_inertia=model.body_inertia[self._pole_body_id].copy(),
            slider_damping=float(model.dof_damping[self._slider_dof_id]),
            hinge_damping=float(model.dof_damping[self._hinge_dof_id]),
            slider_frictionloss=float(model.dof_frictionloss[self._slider_dof_id]),
            hinge_frictionloss=float(model.dof_frictionloss[self._hinge_dof_id]),
            gravity=model.opt.gravity.copy(),
        )

    def _restore_nominal_dynamics(self) -> None:
        base = self.env.unwrapped
        model = base.model
        nominal = self._nominal
        base.force_limit = nominal.force_limit
        model.actuator_ctrlrange[0, :] = nominal.actuator_ctrlrange
        model.body_mass[self._cart_body_id] = nominal.cart_mass
        model.body_inertia[self._cart_body_id] = nominal.cart_inertia
        model.body_mass[self._pole_body_id] = nominal.pole_mass
        model.body_inertia[self._pole_body_id] = nominal.pole_inertia
        model.dof_damping[self._slider_dof_id] = nominal.slider_damping
        model.dof_damping[self._hinge_dof_id] = nominal.hinge_damping
        model.dof_frictionloss[self._slider_dof_id] = nominal.slider_frictionloss
        model.dof_frictionloss[self._hinge_dof_id] = nominal.hinge_frictionloss
        model.opt.gravity[:] = nominal.gravity

    def _sample_and_apply_dynamics(self) -> dict[str, float]:
        base = self.env.unwrapped
        model = base.model
        nominal = self._nominal
        sampled: dict[str, float] = {}

        cart_scale = self._sample_range(self.config.cart_mass_scale_range)
        if cart_scale is not None:
            model.body_mass[self._cart_body_id] = nominal.cart_mass * cart_scale
            model.body_inertia[self._cart_body_id] = nominal.cart_inertia * cart_scale
            sampled["cart_mass_scale"] = cart_scale

        pole_scale = self._sample_range(self.config.pole_mass_scale_range)
        if pole_scale is not None:
            model.body_mass[self._pole_body_id] = nominal.pole_mass * pole_scale
            model.body_inertia[self._pole_body_id] = nominal.pole_inertia * pole_scale
            sampled["pole_mass_scale"] = pole_scale

        slider_damping_scale = self._sample_range(
            self.config.slider_damping_scale_range
        )
        if slider_damping_scale is not None:
            model.dof_damping[self._slider_dof_id] = (
                nominal.slider_damping * slider_damping_scale
            )
            sampled["slider_damping_scale"] = slider_damping_scale

        hinge_damping_scale = self._sample_range(self.config.hinge_damping_scale_range)
        if hinge_damping_scale is not None:
            model.dof_damping[self._hinge_dof_id] = (
                nominal.hinge_damping * hinge_damping_scale
            )
            sampled["hinge_damping_scale"] = hinge_damping_scale

        slider_frictionloss = self._sample_range(self.config.slider_frictionloss_range)
        if slider_frictionloss is not None:
            model.dof_frictionloss[self._slider_dof_id] = slider_frictionloss
            sampled["slider_frictionloss"] = slider_frictionloss

        hinge_frictionloss = self._sample_range(self.config.hinge_frictionloss_range)
        if hinge_frictionloss is not None:
            model.dof_frictionloss[self._hinge_dof_id] = hinge_frictionloss
            sampled["hinge_frictionloss"] = hinge_frictionloss

        force_limit_scale = self._sample_range(self.config.force_limit_scale_range)
        if force_limit_scale is not None:
            force_limit = nominal.force_limit * force_limit_scale
            base.force_limit = float(force_limit)
            model.actuator_ctrlrange[0, :] = [-force_limit, force_limit]
            sampled["force_limit_scale"] = force_limit_scale
            sampled["force_limit"] = float(force_limit)

        gravity_z = self._sample_range(self.config.gravity_z_range)
        if gravity_z is not None:
            model.opt.gravity[2] = gravity_z
            sampled["gravity_z"] = gravity_z

        return sampled

    def _reset_action_delay(self) -> None:
        self._action_queue.clear()
        zero_action = np.zeros(self.action_space.shape, dtype=np.float32)
        for _ in range(max(int(self.config.action_delay_steps), 0)):
            self._action_queue.append(zero_action.copy())

    def _delayed_action(self, action) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float32)
        if int(self.config.action_delay_steps) <= 0:
            return action_array

        self._action_queue.append(action_array.copy())
        return self._action_queue.popleft()

    def _apply_action_noise(self, action: np.ndarray) -> np.ndarray:
        if self.config.action_noise_std <= 0:
            return np.clip(action, self.action_space.low, self.action_space.high)

        noisy = action + self._rng.normal(
            0.0,
            float(self.config.action_noise_std),
            size=action.shape,
        )
        return np.clip(noisy, self.action_space.low, self.action_space.high).astype(
            np.float32
        )

    def _reset_observation_delay(self, observation: np.ndarray) -> None:
        self._observation_queue.clear()
        for _ in range(max(int(self.config.observation_delay_steps), 0)):
            self._observation_queue.append(observation.copy())

    def _current_observation(self, observation: np.ndarray) -> np.ndarray:
        if int(self.config.observation_delay_steps) <= 0:
            return observation.astype(np.float32)

        self._observation_queue.append(observation.copy())
        return self._observation_queue.popleft().astype(np.float32)

    def _apply_observation_noise(self, observation: np.ndarray) -> np.ndarray:
        std = self._observation_noise_std(observation.shape)
        if not np.any(std > 0):
            return observation.astype(np.float32)

        noisy = observation + self._rng.normal(0.0, std, size=observation.shape)
        return noisy.astype(np.float32)

    def _observation_noise_std(self, shape: tuple[int, ...]) -> np.ndarray:
        configured = self.config.observation_noise_std
        if isinstance(configured, (int, float)):
            return np.full(shape, float(configured), dtype=np.float32)

        std = np.asarray(configured, dtype=np.float32)
        if std.shape != shape:
            raise ValueError(
                "observation_noise_std must be scalar or match observation shape "
                f"{shape}, got {std.shape}."
            )
        return std

    def _maybe_apply_push(self) -> dict[str, Any]:
        if self._step_count < int(self.config.push_start_step):
            return {"robustness_push_applied": False}

        interval_hit = (
            int(self.config.push_interval_steps) > 0
            and (self._step_count - int(self.config.push_start_step))
            % int(self.config.push_interval_steps)
            == 0
        )
        probability_hit = (
            float(self.config.push_probability) > 0
            and self._rng.random() < float(self.config.push_probability)
        )
        if not (interval_hit or probability_hit):
            return {"robustness_push_applied": False}

        cart_impulse = self._sample_range(self.config.cart_velocity_impulse_range) or 0.0
        pole_impulse = (
            self._sample_range(self.config.pole_angular_velocity_impulse_range) or 0.0
        )

        base = self.env.unwrapped
        qpos = base.data.qpos.copy()
        qvel = base.data.qvel.copy()
        qvel[0] += cart_impulse
        qvel[1] += pole_impulse
        base.set_state(qpos, qvel)

        return {
            "robustness_push_applied": True,
            "robustness_cart_velocity_impulse": float(cart_impulse),
            "robustness_pole_angular_velocity_impulse": float(pole_impulse),
        }

    def _sample_range(self, value_range: Range) -> float | None:
        if value_range is None:
            return None
        low, high = value_range
        return float(self._rng.uniform(low, high))


def scenario_names() -> list[str]:
    return sorted(SCENARIOS)


def resolve_robustness_config(scenario: str) -> RobustnessConfig:
    try:
        return SCENARIOS[scenario]
    except KeyError as exc:
        choices = ", ".join(scenario_names())
        raise ValueError(f"Unknown robustness scenario '{scenario}'. Choices: {choices}.") from exc


def make_robust_env(
    scenario: str = "clean",
    *,
    render_mode: str | None = None,
    max_episode_steps: int | None = DEFAULT_MAX_EPISODE_STEPS,
    seed: int | None = None,
    **kwargs: Any,
) -> RobustnessWrapper:
    env = make_env(
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        **kwargs,
    )
    return RobustnessWrapper(
        env,
        resolve_robustness_config(scenario),
        seed=seed,
    )
