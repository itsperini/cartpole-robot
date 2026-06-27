from __future__ import annotations

from importlib.resources import files
from typing import Any

import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    "lookat": np.array([0.0, 0.0, 0.35]),
    "elevation": -18.0,
}


def default_xml_path() -> str:
    return str(files("cartpole_robot.assets").joinpath("cartpole_swingup.xml"))


class CartPoleSwingUpEnv(MujocoEnv, utils.EzPickle):
    """Continuous-force MuJoCo cart-pole swing-up task.

    The single action is the horizontal force applied to the cart. The pole starts
    near the downward position, and the reward encourages swinging it upright
    while keeping the cart centered and avoiding excessive control effort.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
        "render_fps": 25,
    }

    def __init__(
        self,
        xml_file: str | None = None,
        frame_skip: int = 2,
        reset_noise_scale: float = 0.08,
        force_limit: float = 10.0,
        max_cart_position: float = 2.4,
        default_camera_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if force_limit <= 0:
            raise ValueError("force_limit must be positive.")

        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            reset_noise_scale,
            force_limit,
            max_cart_position,
            default_camera_config,
            **kwargs,
        )

        self._reset_noise_scale = reset_noise_scale
        self.force_limit = float(force_limit)
        self.max_cart_position = float(max_cart_position)

        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32,
        )

        MujocoEnv.__init__(
            self,
            xml_file or default_xml_path(),
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config or DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.model.actuator_ctrlrange[0, :] = [-self.force_limit, self.force_limit]
        self._set_action_space()

        self.observation_structure = {
            "cart_position": 1,
            "cart_velocity": 1,
            "pole_angle_sin_cos": 2,
            "pole_angular_velocity": 1,
        }

    def step(self, action: np.ndarray):
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        self.do_simulation(clipped_action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._reward(observation, clipped_action)
        terminated = self._is_terminated(observation)

        if self.render_mode == "human":
            self.render()

        info = {
            "force": float(clipped_action[0]),
            **reward_info,
        }

        return observation, reward, terminated, False, info

    def reset_model(self) -> np.ndarray:
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        qpos[0] = self.np_random.uniform(-0.05, 0.05)
        qpos[1] = np.pi + self.np_random.uniform(
            -self._reset_noise_scale,
            self._reset_noise_scale,
        )
        qvel[:] = self.np_random.uniform(
            -self._reset_noise_scale,
            self._reset_noise_scale,
            size=self.model.nv,
        )

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        x_position = self.data.qpos[0]
        pole_angle = self.data.qpos[1]
        x_velocity = self.data.qvel[0]
        pole_angular_velocity = self.data.qvel[1]

        return np.array(
            [
                x_position,
                x_velocity,
                np.sin(pole_angle),
                np.cos(pole_angle),
                pole_angular_velocity,
            ],
            dtype=np.float32,
        )

    def _reward(
        self,
        observation: np.ndarray,
        action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        x_position, x_velocity, _, pole_cos, pole_angular_velocity = observation

        upright_reward = (float(pole_cos) + 1.0) * 0.5
        centered_penalty = 0.1 * float((x_position / self.max_cart_position) ** 2)
        velocity_penalty = 0.01 * float(x_velocity**2) + 0.001 * float(
            pole_angular_velocity**2
        )
        action_penalty = 0.001 * float((action[0] / self.force_limit) ** 2)

        near_upright = float(pole_cos > np.cos(0.2))
        stable_bonus = near_upright * float(abs(pole_angular_velocity) < 1.0)

        reward = (
            2.0 * upright_reward
            + stable_bonus
            - centered_penalty
            - velocity_penalty
            - action_penalty
        )

        return float(reward), {
            "reward_upright": float(2.0 * upright_reward),
            "reward_stable_bonus": float(stable_bonus),
            "penalty_centered": float(centered_penalty),
            "penalty_velocity": float(velocity_penalty),
            "penalty_action": float(action_penalty),
        }

    def _is_terminated(self, observation: np.ndarray) -> bool:
        x_position = float(observation[0])
        return bool(
            not np.isfinite(observation).all()
            or abs(x_position) > self.max_cart_position
        )
