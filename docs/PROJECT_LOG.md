# Project Log

This is the running log for `cartpole-robot`.

Rule: every meaningful change, experiment, training run, evaluation sweep, design
decision, or sim2real note should get a new timestamped entry here. Keep entries
append-only when possible so the project history stays readable.

Timezone: Europe/Rome.

## Log Format

```text
## YYYY-MM-DD HH:MM +0200 - short title

Type: code | docs | training | evaluation | decision | artifact

What changed:
- ...

Why:
- ...

Artifacts:
- ...

Next:
- ...
```

## 2026-06-29 13:00 +0200 - Add project log

Type: docs

What changed:
- Added this timestamped project log under `docs/PROJECT_LOG.md`.
- Established the rule that future meaningful work should be documented here.
- Linked the log from `README.md`, `docs/index.html`, and `docs/status.html`.

Why:
- The project is now branching into multiple learning tracks: model-free RL,
  robust simulation, delay handling, actuator realism, and later sim2real.
- A chronological log makes it possible to reconstruct why each change happened,
  which policy/checkpoint was current, and what each experiment taught.

Artifacts:
- `docs/PROJECT_LOG.md`
- `README.md`
- `docs/index.html`
- `docs/status.html`

Next:
- Update it before or after each future experiment/change.

## 2026-06-29 12:24 +0200 - Run control-loop profile evaluation sweep

Type: evaluation

What changed:
- Ran an evaluation-only sweep across promoted policies, scenarios, and new
  control-loop profiles.
- No policies were trained and no model checkpoints were overwritten.
- Matrix:
  - 7 promoted policies.
  - 5 control-loop profiles.
  - 2 scenarios: `clean` and `hardware_mild`.
  - 10 episodes per policy/profile/scenario.
  - 700 total evaluation episodes.

Why:
- Before training a new policy, we wanted evidence that existing policies really
  fail under realistic control-loop effects.
- This tests a sim2real-relevant gap: not only physical variation, but also
  delayed sensing, delayed actuation, jitter, command shaping, filtering, and
  quantization.

Artifacts:
- `artifacts/control_loop_sweeps/20260629-122450/summary.md`
- `artifacts/control_loop_sweeps/20260629-122450/summary.csv`
- `artifacts/control_loop_sweeps/20260629-122450/hardware_mild_success_heatmap.png`
- `artifacts/control_loop_sweeps/20260629-122450/hardware_mild_stable_heatmap.png`
- `artifacts/control_loop_sweeps/20260629-122450/hardware_mild_reward_heatmap.png`

Findings:
- With `control_loop_profile = "none"`, TD3 hardware-mild remains the strongest
  hardware-randomized policy: 100% success on `hardware_mild`, 88% stable time.
- With one-tick latency, TD3 hardware-mild and SAC hardware-mild survive at 80%
  success, but stable time drops sharply.
- With jitter, the TD3 delay-history policy is the best survivor, but only reaches
  50% success on `hardware_mild`.
- With `policy_25hz_hardware_safe`, almost everything breaks. The best result is
  TD3 delay-history at only 10% success on `hardware_mild`.
- With `policy_25hz_hardware_delay`, all promoted policies fail.

Next:
- Treat this as evidence that the next training target should be a policy trained
  directly with realistic control-loop effects and history inputs.
- Before hardware, refine the action interface so the simulator matches the ESP32
  command semantics.

## 2026-06-29 12:17 +0200 - Add modular control-loop profiles

Type: code, docs

Commit: `3c8a495`

What changed:
- Added `src/cartpole_robot/control_loop.py`.
- Added opt-in control-loop profiles:
  - `none`
  - `policy_25hz_latency`
  - `policy_25hz_jitter`
  - `policy_25hz_hardware_safe`
  - `policy_25hz_hardware_delay`
- Wired `--control-loop-profile` into training, evaluation, robustness evaluation,
  recording, and watch commands.
- Added future training configs:
  - `configs/algorithms/td3_realistic_control_loop_history.toml`
  - `configs/algorithms/sac_realistic_control_loop_history.toml`
- Added `docs/status.html`.
- Updated `README.md` and `docs/index.html`.

Why:
- Previous robust scenarios modeled physical variation, but not the command/sensor
  interface around the policy.
- Real hardware has observation delay, action delay, jitter, velocity/acceleration
  limits, deadband, filtering, and encoder quantization.
- The new layer is modular and defaults to `none`, preserving old behavior.

Verification:
- Compiled `src/cartpole_robot`.
- Checked whitespace with `git diff --check`.
- Confirmed default env still returns a 5D observation.
- Confirmed profile plus history returns 23 inputs.
- Ran one-episode smoke evaluations with `none` and `policy_25hz_hardware_safe`.

Next:
- Run an evaluation sweep before training any new realistic-loop policy.

## 2026-06-29 02:54 +0200 - Document latency and delay experiments

Type: docs

Commit: `3a350eb`

What changed:
- Documented what one tick and two ticks mean in the current simulator.
- Added delay/latency explanation to docs.
- Connected delay handling to control-loop frequency.

Why:
- The hardware loop may run faster than 25 Hz, but the current MuJoCo policy acts
  every 40 ms.
- Delay has to be interpreted in control ticks, not only in milliseconds.

Next:
- Keep delay experiments separate from general hardware randomization.

## 2026-06-29 02:48 +0200 - Add delay-aware robustness training

Type: code, training, evaluation

Commit: `1937f2e`

What changed:
- Added delay-aware policies that use observation/action history.
- The main history shape was:
  - 4 recent 5D observations.
  - 3 previous 1D actions.
  - 23 total policy inputs.
- Promoted a TD3 delay-history checkpoint:
  - `models/best/td3_cartpole_swingup_delay_history_obs4_act3_best_20260629-023430.zip`

Why:
- A delayed observation alone is no longer a full Markov state.
- Recent observations and actions help the policy infer what the real current
  state might be.

Artifacts:
- `artifacts/robustness/20260629-023430-td3-delay-history/robustness.md`
- `docs/assets/td3_delay_history_rollout.mp4`
- `docs/assets/td3_delay_history_hardware_delay_rollout.mp4`

Findings:
- TD3 delay-history became strong on the explicit `delay` scenario.
- It was not a universal replacement for the hardware-randomized TD3 policy.

Next:
- Combine delay history with more realistic hardware variation later.

## 2026-06-29 01:52 +0200 - Add hardware randomization training

Type: code, training, evaluation

Commit: `4cf4cb7`

What changed:
- Added hardware-style randomization scenarios.
- Fine-tuned TD3 under `hardware_mild`.
- Promoted:
  - `models/best/td3_cartpole_swingup_robust_hardware_mild_best_20260629-020520.zip`

Why:
- Sim2real needs robustness to friction, damping, mass variation, force limits,
  sensor/action noise, and pushes.
- TD3 was brought back because it performed strongly under hardware randomization.

Artifacts:
- `artifacts/robustness/20260629-020520-td3-hardware-mild/robustness.md`
- `docs/assets/td3_hardware_mild_robustness.png`
- `docs/assets/td3_hardware_mild_rollout.mp4`

Findings:
- TD3 hardware-mild handled clean, friction, dynamics, pushes, and `hardware_mild`
  well.
- It still struggled with harsher delay.

Next:
- Treat delay as a separate policy family rather than expecting one policy to
  solve everything immediately.

## 2026-06-29 01:33 +0200 - Add disturbance rollout videos

Type: artifact, docs

Commit: `6676442`

What changed:
- Recorded disturbance videos for multiple policies under the `pushes` scenario.
- Added those videos to the main HTML docs.

Why:
- Videos make failure modes visible in a way that scalar reward cannot.
- Push recovery is important for the real cart-pole because the user wants the
  pole to autocorrect after being disturbed.

Artifacts:
- `docs/assets/td3_push_disturbance.mp4`
- `docs/assets/sac_push_disturbance.mp4`
- `docs/assets/ppo_clean_push_disturbance.mp4`
- `docs/assets/ppo_robust_push_disturbance.mp4`

Next:
- Use videos as qualitative checks alongside numeric robustness reports.

## 2026-06-29 01:28 +0200 - Add PPO push recovery training

Type: training, evaluation

Commit: `174cdc5`

What changed:
- Fine-tuned PPO against repeated simulated pushes.
- Promoted:
  - `models/best/ppo_cartpole_swingup_robust_pushes_best_20260629-012559.zip`

Why:
- Clean PPO solved swing-up, but push disturbance was a useful failure case.
- Training on pushes tested whether a policy could recover from external impulses.

Artifacts:
- `artifacts/robustness/20260629-012559-ppo-robust-pushes-final/robustness.md`
- `docs/assets/ppo_robust_pushes_robustness.png`

Findings:
- Push-trained PPO improved disturbance recovery while preserving clean swing-up.
- It did not become a general sim2real policy.

Next:
- Add separate tracks for hardware randomization and delay.

## 2026-06-29 01:23 +0200 - Add robust simulation evaluation

Type: code, evaluation

Commit: `e6559ce`

What changed:
- Added robustness wrappers and `cartpole-robot-robust-eval`.
- Scenarios included clean, sensor noise, friction, latency, delay, dynamics,
  pushes, hardware-mild, hardware-delay, and combined stress tests.

Why:
- Clean reward alone was no longer enough.
- We needed a repeatable way to ask which policies survive sim2real-like
  perturbations.

Artifacts:
- `src/cartpole_robot/robustness.py`
- `src/cartpole_robot/evaluate_robustness.py`
- `artifacts/robustness/*/robustness.md`

Next:
- Use robustness reports to decide what to train next.

## 2026-06-29 01:15 +0200 - Add PPO swing-up policy and comparison docs

Type: training, docs

Commit: `6c103da`

What changed:
- Continued PPO training until it solved swing-up.
- Added updated policy videos and comparison docs.
- Promoted:
  - `models/best/ppo_cartpole_swingup_best_20260629-010531.zip`

Why:
- The original PPO run lagged TD3/SAC.
- A fair algorithm comparison needed PPO to be trained long enough to reach the
  task objective.

Next:
- Compare TD3, SAC, and PPO with the same evaluator.

## 2026-06-29 00:41 +0200 - Add RL algorithm comparison workflow

Type: code, training, evaluation

Commit: `fb8db73`

What changed:
- Added TD3/SAC/PPO shared training workflow.
- Added timestamped run directories.
- Added comparison plots and markdown reports.

Why:
- The project was moving from one TD3 example to a structured RL learning path.
- Timestamped artifacts prevent older experiments from being overwritten.

Artifacts:
- `artifacts/comparisons/20260629-td3-sac-ppo/comparison.md`
- `docs/assets/algorithm_metrics_comparison.png`
- `docs/assets/algorithm_effort_tradeoff.png`
- `docs/assets/algorithm_eval_curves.png`

Next:
- Use algorithm comparisons as baselines before robust training.

## 2026-06-29 00:28 +0200 - Remove world model prototype

Type: code, decision

Commit: `b61ca89`

What changed:
- Removed the world-model code path and references from the active repo.

Why:
- The world-model track was useful for learning, but it felt premature for the
  immediate goal.
- The project refocused on model-free RL basics, robustness, delay, and sim2real.

Next:
- Reintroduce world models later only after the state-based RL/sim2real foundation
  is more solid.

## 2026-06-28 01:54 to 09:38 +0200 - Prototype and then park world-model learning

Type: code, docs, decision

Commits:
- `76b39ac` Add MLP ensemble world model pipeline.
- `806d5e2` Add world model visualizations.
- `9690178` Improve world model diagnostics.
- `bfb50b0` Add world model and VLA roadmap docs.
- `7c39f43` Add world model architecture diagram.

What changed:
- Built an MLP ensemble world-model prototype.
- Added dataset collection, training, evaluation, and visual diagnostics.
- Added docs explaining the architecture and future world-model/VLA direction.

Why:
- The project briefly explored model-based learning and future VLA/world-model
  concepts.

Findings:
- The prototype was educational, but it was too early relative to the current
  learning goal.

Next:
- Park this track and return later when the RL and sim2real basics are stronger.

## 2026-06-28 01:30 to 02:46 +0200 - Train and document first TD3 swing-up model

Type: training, artifact, docs

Commits:
- `1a22e3a` Add trained swing-up models.
- `b0b74d2` Rename trained best model artifact.
- `b02322e` Improve project README.
- `7c58105` Add TD3 swing-up video to docs.

What changed:
- Trained the first successful TD3 swing-up policy.
- Renamed the best checkpoint with a timestamped descriptive name.
- Added README explanation and the first policy video.

Artifacts:
- `models/best/td3_cartpole_swingup_best_20260628-012620.zip`
- `docs/assets/td3_swingup.mp4`
- `docs/assets/td3_learned_policy.mp4`

Why:
- This created the first concrete success: swing up from the downward position
  and stabilize using continuous control.

Next:
- Expand from one TD3 model to multiple algorithm baselines.

## 2026-06-28 01:19 to 01:21 +0200 - Create MuJoCo swing-up environment and normalize action

Type: code

Commits:
- `9eda9ad` Add MuJoCo swing-up TD3 setup.
- `703ccb3` Normalize swing-up action space.

What changed:
- Added the custom MuJoCo `CartPoleSwingUp-v0` environment.
- Removed the unsuitable discrete `CartPole-v1` and upright-only inverted
  pendulum path from the active learning target.
- Normalized the policy action to `[-1, 1]`.
- Mapped normalized action to physical cart force in the simulator.

Why:
- TD3/SAC need continuous actions.
- The real cart-pole should eventually use a normalized command interface even if
  the hardware maps that command to velocity, acceleration, or motor steps.

Next:
- Train the first TD3 policy.

## 2026-06-28 01:00 to 01:05 +0200 - Start project and fix rendering dependencies

Type: code

Commits:
- `50d3480` Initial cartpole robot example.
- `f6c8cdf` Fix rendered cartpole examples.

What changed:
- Created the initial `cartpole-robot` project using `uv`.
- Added basic Gymnasium examples.
- Fixed classic-control rendering dependency issue.

Why:
- The project started as a simple Gymnasium/MuJoCo learning scaffold before
  becoming a focused swing-up and sim2real learning track.

Next:
- Replace unsuitable examples with a real continuous-control swing-up task.
