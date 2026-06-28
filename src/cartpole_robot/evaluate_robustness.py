from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cartpole_robot.algorithms import load_policy
from cartpole_robot.evaluate_policy import (
    STABLE_ANGULAR_VELOCITY,
    UPRIGHT_THRESHOLD_RADIANS,
)
from cartpole_robot.registration import ENV_ID
from cartpole_robot.robustness import (
    make_robust_env,
    resolve_robustness_config,
    scenario_names,
)


DEFAULT_SCENARIOS = [
    "clean",
    "sensor_noise",
    "friction",
    "delay",
    "dynamics",
    "pushes",
    "hardware_mild",
    "combined",
]


@dataclass(frozen=True)
class RobustEpisodeMetrics:
    episode: int
    reward: float
    steps: int
    terminated: bool
    truncated: bool
    success: bool
    upright_fraction: float
    stable_fraction: float
    mean_abs_cart_position: float
    max_abs_cart_position: float
    mean_abs_action: float
    mean_abs_force: float
    disturbance_count: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained policy across robust-simulation scenarios."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/best/ppo_cartpole_swingup_best_20260629-010531.zip"),
    )
    parser.add_argument(
        "--algo",
        default="auto",
        choices=["auto", "td3", "sac", "ppo"],
        help="Algorithm that created the model. Auto infers from the path when possible.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        choices=scenario_names(),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for robustness.json, robustness.md, and robustness.png.",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def make_output_dir(path: Path | None) -> Path:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        return path

    output_dir = (
        Path("artifacts")
        / "robustness"
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def evaluate_episode(
    *,
    env,
    model,
    episode: int,
    seed: int,
    max_steps: int,
    deterministic: bool,
) -> RobustEpisodeMetrics:
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    truncated = False
    upright_steps = 0
    stable_steps = 0
    disturbance_count = 0
    abs_cart_positions: list[float] = []
    abs_actions: list[float] = []
    abs_forces: list[float] = []

    upright_threshold = float(np.cos(UPRIGHT_THRESHOLD_RADIANS))

    for step in range(1, max_steps + 1):
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, info = env.step(action)
        true_observation = np.asarray(
            info.get("true_observation", observation),
            dtype=np.float32,
        )

        total_reward += float(reward)
        cart_position = float(true_observation[0])
        pole_cos = float(true_observation[3])
        pole_angular_velocity = float(true_observation[4])
        is_upright = pole_cos > upright_threshold
        is_stable = is_upright and abs(pole_angular_velocity) < STABLE_ANGULAR_VELOCITY

        upright_steps += int(is_upright)
        stable_steps += int(is_stable)
        disturbance_count += int(bool(info.get("robustness_push_applied", False)))
        abs_cart_positions.append(abs(cart_position))
        abs_actions.append(float(np.mean(np.abs(action))))
        abs_forces.append(abs(float(info.get("force", 0.0))))

        if terminated or truncated:
            break

    upright_fraction = upright_steps / step
    stable_fraction = stable_steps / step
    success = (not terminated) and upright_fraction >= 0.25 and stable_fraction >= 0.15

    return RobustEpisodeMetrics(
        episode=episode,
        reward=total_reward,
        steps=step,
        terminated=terminated,
        truncated=truncated,
        success=success,
        upright_fraction=upright_fraction,
        stable_fraction=stable_fraction,
        mean_abs_cart_position=float(np.mean(abs_cart_positions)),
        max_abs_cart_position=float(np.max(abs_cart_positions)),
        mean_abs_action=float(np.mean(abs_actions)),
        mean_abs_force=float(np.mean(abs_forces)),
        disturbance_count=disturbance_count,
    )


def summarize(episodes: list[RobustEpisodeMetrics]) -> dict[str, float]:
    keys = (
        "reward",
        "steps",
        "upright_fraction",
        "stable_fraction",
        "mean_abs_cart_position",
        "max_abs_cart_position",
        "mean_abs_action",
        "mean_abs_force",
        "disturbance_count",
    )
    summary: dict[str, float] = {
        "episodes": float(len(episodes)),
        "success_rate": float(np.mean([episode.success for episode in episodes])),
        "termination_rate": float(np.mean([episode.terminated for episode in episodes])),
    }

    for key in keys:
        values = np.array([getattr(episode, key) for episode in episodes], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())

    return summary


def evaluate_scenario(
    *,
    scenario: str,
    model,
    episodes: int,
    seed: int,
    max_steps: int,
    deterministic: bool,
) -> dict[str, Any]:
    env = make_robust_env(
        scenario,
        max_episode_steps=max_steps,
        seed=seed,
    )
    try:
        episode_metrics = [
            evaluate_episode(
                env=env,
                model=model,
                episode=episode,
                seed=seed + episode,
                max_steps=max_steps,
                deterministic=deterministic,
            )
            for episode in range(1, episodes + 1)
        ]
    finally:
        env.close()

    return {
        "scenario": scenario,
        "config": resolve_robustness_config(scenario).to_dict(),
        "episodes": [asdict(episode) for episode in episode_metrics],
        "summary": summarize(episode_metrics),
    }


def print_table(results: list[dict[str, Any]]) -> None:
    print("\nRobustness evaluation")
    print(
        "Scenario       Reward      Success  Upright  Stable   Terminated  "
        "Mean|force|  Pushes"
    )
    print("-" * 86)
    for row in results:
        summary = row["summary"]
        print(
            f"{row['scenario']:<13} "
            f"{summary['reward_mean']:>8.2f} "
            f"{summary['success_rate'] * 100:>9.1f}% "
            f"{summary['upright_fraction_mean'] * 100:>7.1f}% "
            f"{summary['stable_fraction_mean'] * 100:>6.1f}% "
            f"{summary['termination_rate'] * 100:>10.1f}% "
            f"{summary['mean_abs_force_mean']:>11.3f} "
            f"{summary['disturbance_count_mean']:>7.1f}"
        )


def write_markdown(results: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Robustness Evaluation",
        "",
        "| Scenario | Reward | Success | Upright | Stable | Terminated | Mean abs force | Pushes/episode |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in results:
        summary = row["summary"]
        lines.append(
            "| "
            f"{row['scenario']} | "
            f"{summary['reward_mean']:.2f} | "
            f"{summary['success_rate'] * 100:.1f}% | "
            f"{summary['upright_fraction_mean'] * 100:.1f}% | "
            f"{summary['stable_fraction_mean'] * 100:.1f}% | "
            f"{summary['termination_rate'] * 100:.1f}% | "
            f"{summary['mean_abs_force_mean']:.3f} | "
            f"{summary['disturbance_count_mean']:.1f} |"
        )

    path.write_text("\n".join(lines) + "\n")


def write_plot(results: list[dict[str, Any]], path: Path) -> None:
    labels = [row["scenario"] for row in results]
    reward = [row["summary"]["reward_mean"] for row in results]
    upright = [row["summary"]["upright_fraction_mean"] * 100 for row in results]
    termination = [row["summary"]["termination_rate"] * 100 for row in results]
    force = [row["summary"]["mean_abs_force_mean"] for row in results]

    x = np.arange(len(labels))
    colors = ["#2563eb", "#0f766e", "#b45309", "#be123c", "#7c3aed", "#334155"]
    colors = (colors * max(len(labels), 1))[: len(labels)]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), constrained_layout=True)
    charts = [
        (axes[0, 0], reward, "Mean reward"),
        (axes[0, 1], upright, "Upright time (%)"),
        (axes[1, 0], termination, "Termination rate (%)"),
        (axes[1, 1], force, "Mean |force|"),
    ]

    for axis, values, title in charts:
        axis.bar(x, values, color=colors)
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Cart-Pole Robustness Evaluation", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    output_dir = make_output_dir(args.output_dir)
    algorithm, model = load_policy(
        args.model,
        algorithm=args.algo,
        env=None,
        device=args.device,
    )
    deterministic = not args.stochastic

    print(f"Evaluating {algorithm.upper()} policy on {ENV_ID}")
    print(f"Model: {args.model}")

    results = [
        evaluate_scenario(
            scenario=scenario,
            model=model,
            episodes=args.episodes,
            seed=args.seed + index * 10_000,
            max_steps=args.max_steps,
            deterministic=deterministic,
        )
        for index, scenario in enumerate(args.scenarios)
    ]

    print_table(results)

    payload = {
        "algorithm": algorithm,
        "env_id": ENV_ID,
        "model": str(args.model),
        "episodes_per_scenario": args.episodes,
        "max_steps": args.max_steps,
        "scenarios": results,
    }
    json_path = output_dir / "robustness.json"
    markdown_path = output_dir / "robustness.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(results, markdown_path)
    print(f"\nSaved JSON report to {json_path}")
    print(f"Saved markdown table to {markdown_path}")

    if not args.no_plot:
        plot_path = output_dir / "robustness.png"
        write_plot(results, plot_path)
        print(f"Saved plot to {plot_path}")
