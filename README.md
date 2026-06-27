# cartpole-robot

A `uv` project for MuJoCo cart-pole swing-up with a continuous cart-force action and TD3 training scripts.

The environment is custom-built instead of using `CartPole-v1` or `InvertedPendulum-v5`:

- `CartPole-v1` uses a discrete left/right action, so it is not a TD3 target.
- `InvertedPendulum-v5` is continuous, but it starts upright and terminates when the pole falls, so it is a balance task rather than swing-up.

This project uses `CartPoleSwingUp-v0`, a MuJoCo environment where the pole starts near the downward position and the single action is continuous force applied to the cart.

## Setup

```bash
uv sync
```

## Check The Environment

Run the Stable-Baselines3 environment checker:

```bash
uv run cartpole-robot-check
```

## Random Rollout

Run a random continuous-force policy:

```bash
uv run cartpole-robot
```

Open the MuJoCo viewer:

```bash
uv run cartpole-robot --render --render-fps 25
```

## Train TD3

Start training:

```bash
uv run cartpole-robot-train --timesteps 300000
```

The final model is saved to:

```bash
models/td3_cartpole_swingup.zip
```

Watch a trained policy:

```bash
uv run cartpole-robot-watch --model models/td3_cartpole_swingup.zip
```

TensorBoard logs go into `runs/`.
