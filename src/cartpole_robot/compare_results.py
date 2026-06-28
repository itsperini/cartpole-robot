from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cart-pole policy evaluation reports."
    )
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="Evaluation JSON files created by cartpole-robot-eval.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for comparison.md and comparison.png.",
    )
    return parser.parse_args(argv)


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Report not found: {path}")
    return json.loads(path.read_text())


def label_for_report(report: dict[str, Any], path: Path) -> str:
    algorithm = str(report.get("algorithm", "policy")).upper()
    model_path = Path(str(report.get("model", path)))
    parts = model_path.parts
    if "runs" in parts:
        runs_index = parts.index("runs")
        run_id = parts[runs_index + 2] if len(parts) > runs_index + 2 else path.stem
    elif model_path.parent.name == "best":
        run_id = model_path.parent.parent.name
    else:
        run_id = path.stem
    return f"{algorithm} {run_id}"


def make_output_dir(path: Path | None) -> Path:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        return path

    output_dir = Path("artifacts") / "comparisons" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Cart-Pole Policy Comparison",
        "",
        "| Policy | Reward | Success | Upright | Stable | Mean abs action | Mean abs cart x |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        summary = row["summary"]
        lines.append(
            "| "
            f"{row['label']} | "
            f"{summary['reward_mean']:.2f} | "
            f"{summary['success_rate'] * 100:.1f}% | "
            f"{summary['upright_fraction_mean'] * 100:.1f}% | "
            f"{summary['stable_fraction_mean'] * 100:.1f}% | "
            f"{summary['mean_abs_action_mean']:.3f} | "
            f"{summary['mean_abs_cart_position_mean']:.3f} |"
        )

    path.write_text("\n".join(lines) + "\n")


def write_plot(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [row["label"] for row in rows]
    reward = [row["summary"]["reward_mean"] for row in rows]
    success = [row["summary"]["success_rate"] * 100 for row in rows]
    upright = [row["summary"]["upright_fraction_mean"] * 100 for row in rows]
    action = [row["summary"]["mean_abs_action_mean"] for row in rows]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    charts = [
        (axes[0, 0], reward, "Mean reward"),
        (axes[0, 1], success, "Success rate (%)"),
        (axes[1, 0], upright, "Upright time (%)"),
        (axes[1, 1], action, "Mean |action|"),
    ]

    colors = ["#2563eb", "#0f766e", "#b45309", "#be123c"] * max(len(labels), 1)
    for axis, values, title in charts:
        axis.bar(x, values, color=colors[: len(labels)])
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Cart-Pole Policy Comparison", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = make_output_dir(args.output_dir)
    rows = [
        {
            "label": label_for_report(report, path),
            "summary": report["summary"],
            "path": str(path),
        }
        for path in args.reports
        for report in [load_report(path)]
    ]

    markdown_path = output_dir / "comparison.md"
    plot_path = output_dir / "comparison.png"
    write_markdown(rows, markdown_path)
    write_plot(rows, plot_path)

    print(f"Saved comparison table to {markdown_path}")
    print(f"Saved comparison plot to {plot_path}")
