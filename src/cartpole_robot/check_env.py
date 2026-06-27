from __future__ import annotations

from stable_baselines3.common.env_checker import check_env

from cartpole_robot.swingup_env import CartPoleSwingUpEnv


def main() -> None:
    env = CartPoleSwingUpEnv()

    try:
        check_env(env, warn=True, skip_render_check=True)
        print("Environment check passed.")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space:      {env.action_space}  # continuous cart force")
    finally:
        env.close()
