# cartpole-robot

A small `uv` project that demonstrates Gymnasium with:

- `CartPole-v1`, the classic control cart-pole task.
- `InvertedPendulum-v5`, a MuJoCo cart-pole style task.

The example runs a random policy, prints episode lengths and rewards, and stays headless by default.

## Setup

```bash
uv sync
```

## Run

Run both examples:

```bash
uv run cartpole-robot
```

Run only classic CartPole:

```bash
uv run cartpole-robot --env cartpole --episodes 5
```

Run only the MuJoCo environment:

```bash
uv run cartpole-robot --env mujoco
```

Open a viewer window:

```bash
uv run cartpole-robot --env mujoco --render
```

You can also run the package module directly:

```bash
uv run python -m cartpole_robot --env both --seed 42
```
