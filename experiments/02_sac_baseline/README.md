# 02 SAC Baseline

SAC is the natural second baseline. It is off-policy like TD3, but learns a
stochastic policy with an entropy objective. In practice, SAC is often a strong
continuous-control default.

Run:

```bash
uv run cartpole-robot-train --config configs/algorithms/sac.toml
```

Evaluate a trained run:

```bash
uv run cartpole-robot-eval \
  --algo sac \
  --model artifacts/runs/sac/<run-id>/best/best_model.zip \
  --output artifacts/runs/sac/<run-id>/eval_metrics.json
```

What to study:

- Whether entropy helps exploration during swing-up.
- Whether SAC learns a smoother or more robust controller than TD3.
- How sensitive it is to the entropy coefficient.
