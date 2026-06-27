# cartpole-robot

MuJoCo cart-pole swing-up with a custom Gymnasium environment, normalized continuous actions, and TD3 training scripts.

This is not the standard `CartPole-v1` task. The goal is to start with the pole hanging downward, apply horizontal force to the cart, swing the pole upright, and stabilize it there.

## Why Custom

- `CartPole-v1` has a discrete action space, so it is not suitable for TD3.
- `InvertedPendulum-v5` has continuous actions, but it starts near upright and terminates when the pole falls, so it is a balance task rather than swing-up.
- `CartPoleSwingUp-v0` keeps MuJoCo physics, starts near the downward position, and uses one continuous action for cart force.

## Environment

The policy sees a 5D observation:

```text
[cart_position, cart_velocity, sin(pole_angle), cos(pole_angle), pole_angular_velocity]
```

The policy outputs one normalized action:

```text
action in [-1, 1]
```

The environment maps that to physical cart force:

```text
force = action * 10
```

The reward encourages the pole to be upright and stable while penalizing cart drift, high velocity, and excessive force.

## Setup

```bash
uv sync
```

## Quick Start

Watch the included trained policy:

```bash
uv run cartpole-robot-watch
```

Run a random policy for comparison:

```bash
uv run cartpole-robot --render --render-fps 25
```

Check that the custom environment is compatible with Stable-Baselines3:

```bash
uv run cartpole-robot-check
```

## Training

Start a TD3 run:

```bash
uv run cartpole-robot-train --timesteps 300000
```

Useful options:

```bash
uv run cartpole-robot-train --timesteps 1000000
uv run cartpole-robot-train --action-noise 0.2
uv run cartpole-robot-train --resume models/checkpoints/td3_cartpole_swingup_150000_steps.zip
```

Training writes:

- checkpoints to `models/checkpoints/`
- best evaluated model to `models/best/`
- final model to `models/td3_cartpole_swingup.zip`
- TensorBoard logs to `runs/`

Open TensorBoard while training:

```bash
uv run tensorboard --logdir runs
```

Then open `http://localhost:6006`.

## Included Model

This repo includes a trained policy:

```bash
models/best/td3_cartpole_swingup_best_20260628-012620.zip
```

Watch it:

```bash
uv run cartpole-robot-watch
```

Watch another checkpoint:

```bash
uv run cartpole-robot-watch --model models/checkpoints/td3_cartpole_swingup_150000_steps.zip
```

## Commands

```bash
uv run cartpole-robot-check          # validate the Gymnasium environment
uv run cartpole-robot                # random rollout, headless by default
uv run cartpole-robot --render       # random rollout with MuJoCo viewer
uv run cartpole-robot-train          # train TD3
uv run cartpole-robot-watch          # render the trained best model
```

## Project Layout

```text
src/cartpole_robot/
  swingup_env.py                  custom Gymnasium/MuJoCo environment
  assets/cartpole_swingup.xml     MuJoCo cart, rail, pole, hinge, and motor model
  registration.py                 registers CartPoleSwingUp-v0
  train_td3.py                    TD3 training entry point
  watch.py                        loads and renders a trained TD3 policy
  rollout.py                      random-policy rollout helper
  check_env.py                    Stable-Baselines3 env checker
  __main__.py                     enables python -m cartpole_robot
```

## Model Zip Contents

Stable-Baselines3 model zips contain the policy weights, optimizer states, algorithm metadata, action and observation spaces, version info, and system info. They can be loaded directly with `TD3.load(...)`.

The included best model is about 5.7 MiB.
