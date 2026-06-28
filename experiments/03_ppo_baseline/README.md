# 03 PPO Baseline

PPO is the first on-policy comparison. It is popular in robotics and simulation
because it is stable and simple to reason about, but it usually needs more
environment interaction than off-policy methods.

Run:

```bash
uv run cartpole-robot-train --config configs/algorithms/ppo.toml
```

Evaluate a trained run:

```bash
uv run cartpole-robot-eval \
  --algo ppo \
  --model artifacts/runs/ppo/<run-id>/best/best_model.zip \
  --output artifacts/runs/ppo/<run-id>/eval_metrics.json
```

What to study:

- How on-policy learning differs from replay-buffer methods.
- Whether PPO needs more timesteps to swing up reliably.
- How reward shaping affects policy stability.
