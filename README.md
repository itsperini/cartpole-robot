# cartpole-robot

MuJoCo cart-pole swing-up with a custom Gymnasium environment, normalized continuous actions, and model-free RL baselines.

This is not the standard `CartPole-v1` task. The goal is to start with the pole hanging downward, apply horizontal force to the cart, swing the pole upright, and stabilize it there.

For the learning roadmap from classic RL to visual control, sim2real, ROS 2 hardware, and VLA-style systems, open [docs/index.html](docs/index.html).

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

## Training Baselines

Train TD3, SAC, or PPO from config files:

```bash
uv run cartpole-robot-train --config configs/algorithms/td3.toml
uv run cartpole-robot-train --config configs/algorithms/sac.toml
uv run cartpole-robot-train --config configs/algorithms/ppo.toml
```

Useful options:

```bash
uv run cartpole-robot-train --algo td3 --timesteps 1000000
uv run cartpole-robot-train --algo sac --learning-rate 0.0003
uv run cartpole-robot-train --algo ppo --n-steps 1024
uv run cartpole-robot-train --config configs/algorithms/td3.toml --action-noise 0.2
```

New runs are timestamped so older experiments are not overwritten:

```text
artifacts/runs/<algorithm>/<timestamp>/
  best/best_model.zip
  checkpoints/
  config.json
  final_model.zip
  metadata.json
  tensorboard/
```

Open TensorBoard while training:

```bash
uv run tensorboard --logdir artifacts/runs
```

Then open `http://localhost:6006`.

## Evaluation

Evaluate any saved TD3, SAC, or PPO policy:

```bash
uv run cartpole-robot-eval \
  --algo td3 \
  --model artifacts/runs/td3/<run-id>/best/best_model.zip \
  --output artifacts/runs/td3/<run-id>/eval_metrics.json
```

The evaluator reports reward, episode length, success rate, upright time,
stable time, cart travel, action effort, and force effort.

Compare evaluation reports:

```bash
uv run cartpole-robot-compare \
  artifacts/runs/td3/<run-id>/eval_metrics.json \
  artifacts/runs/sac/<run-id>/eval_metrics.json \
  artifacts/runs/ppo/<run-id>/eval_metrics.json
```

This writes a markdown table and plot under `artifacts/comparisons/`.

## Robustness Evaluation

Evaluate a trained policy against first-pass sim2real stress tests:

```bash
uv run cartpole-robot-robust-eval \
  --algo ppo \
  --model models/best/ppo_cartpole_swingup_best_20260629-010531.zip
```

The robust evaluator runs the same policy across these scenarios:

```text
clean          no extra stressors
sensor_noise   encoder-like observation noise and small action noise
delay          observation and action delay
dynamics       randomized mass, damping, friction, gravity, and force limit
pushes         repeated cart/pole velocity impulses during the episode
combined       a moderate mix of noise, delay, dynamics, and pushes
```

Reports are written under `artifacts/robustness/<timestamp>/`:

```text
robustness.json
robustness.md
robustness.png
```

This is evaluation-only for now. The next training step is to train policies with
some of these stressors enabled and compare clean-trained versus robust-trained
controllers.

## Included Models

This repo includes trained policies:

```bash
models/best/td3_cartpole_swingup_best_20260628-012620.zip
models/best/ppo_cartpole_swingup_best_20260629-010531.zip
```

Watch the default TD3 policy:

```bash
uv run cartpole-robot-watch
```

Watch the PPO policy:

```bash
uv run cartpole-robot-watch \
  --algo ppo \
  --model models/best/ppo_cartpole_swingup_best_20260629-010531.zip
```

Watch another checkpoint:

```bash
uv run cartpole-robot-watch --model models/checkpoints/td3_cartpole_swingup_150000_steps.zip
```

Watch a new SAC or PPO run:

```bash
uv run cartpole-robot-watch --algo sac --model artifacts/runs/sac/<run-id>/best/best_model.zip
uv run cartpole-robot-watch --algo ppo --model artifacts/runs/ppo/<run-id>/best/best_model.zip
```

## Commands

```bash
uv run cartpole-robot-check          # validate the Gymnasium environment
uv run cartpole-robot                # random rollout, headless by default
uv run cartpole-robot --render       # random rollout with MuJoCo viewer
uv run cartpole-robot-train          # train TD3 by default, or choose --algo sac/ppo
uv run cartpole-robot-eval           # evaluate TD3/SAC/PPO with robotics metrics
uv run cartpole-robot-robust-eval    # evaluate clean/noisy/delayed/randomized scenarios
uv run cartpole-robot-compare        # compare saved evaluation JSON reports
uv run cartpole-robot-watch          # render a trained TD3/SAC/PPO model
```

## Project Layout

```text
src/cartpole_robot/
  swingup_env.py                  custom Gymnasium/MuJoCo environment
  assets/cartpole_swingup.xml     MuJoCo cart, rail, pole, hinge, and motor model
  registration.py                 registers CartPoleSwingUp-v0
  algorithms.py                   shared TD3/SAC/PPO loading helpers
  train_policy.py                 TD3/SAC/PPO training entry point
  train_td3.py                    compatibility wrapper for the policy trainer
  evaluate_policy.py              reward, stability, and effort metrics
  evaluate_robustness.py          clean/noisy/delayed/randomized robustness reports
  compare_results.py              markdown and plot summaries from eval reports
  robustness.py                   sim2real stress-test wrappers and scenarios
  watch.py                        loads and renders a trained TD3/SAC/PPO policy
  rollout.py                      random-policy rollout helper
  check_env.py                    Stable-Baselines3 env checker
  __main__.py                     enables python -m cartpole_robot
```

## Model Zip Contents

Stable-Baselines3 model zips contain the policy weights, optimizer states, algorithm metadata, action and observation spaces, version info, and system info. They can be loaded directly with `TD3.load(...)`.

The included best model is about 5.7 MiB.
