# 01 TD3 Baseline

TD3 is the first trained baseline for this project. It is an off-policy,
deterministic actor-critic method for continuous control.

Run:

```bash
uv run cartpole-robot-train --config configs/algorithms/td3.toml
```

Evaluate a trained run:

```bash
uv run cartpole-robot-eval \
  --algo td3 \
  --model artifacts/runs/td3/<run-id>/best/best_model.zip \
  --output artifacts/runs/td3/<run-id>/eval_metrics.json
```

What to study:

- How exploration noise affects swing-up.
- Whether the final policy is smooth or jerky.
- How often it reaches upright and remains stable.
