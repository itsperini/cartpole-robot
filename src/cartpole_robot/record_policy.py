from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cartpole_robot.algorithms import load_policy
from cartpole_robot.control_loop import control_loop_profile_names, wrap_control_loop
from cartpole_robot.registration import ENV_ID, make_env
from cartpole_robot.robustness import make_robust_env, scenario_names, wrap_history


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a trained TD3, SAC, or PPO cart-pole policy as MP4."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--algo",
        default="auto",
        choices=["auto", "td3", "sac", "ppo"],
        help="Algorithm that created the model. Auto infers from the path when possible.",
    )
    parser.add_argument(
        "--scenario",
        choices=scenario_names(),
        default="clean",
        help="Robustness scenario to record.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poster", type=Path, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--control-loop-profile",
        choices=control_loop_profile_names(),
        default="none",
        help="Optional hardware-like control loop timing profile.",
    )
    parser.add_argument(
        "--observation-history-steps",
        type=int,
        default=1,
        help="Number of recent observations exposed to the policy.",
    )
    parser.add_argument(
        "--action-history-steps",
        type=int,
        default=0,
        help="Number of previous commanded actions exposed to the policy.",
    )
    parser.add_argument(
        "--push-flash-frames",
        type=int,
        default=14,
        help="Number of frames to keep the push overlay visible after each impulse.",
    )
    return parser.parse_args(argv)


def make_recording_env(args: argparse.Namespace):
    env_kwargs = {
        "render_mode": "rgb_array",
        "max_episode_steps": args.max_steps,
        "width": args.width,
        "height": args.height,
    }
    if args.scenario == "clean":
        env = make_env(**env_kwargs)
        env = wrap_control_loop(
            env,
            profile=args.control_loop_profile,
            seed=args.seed,
        )
        return wrap_history(
            env,
            observation_history_steps=args.observation_history_steps,
            action_history_steps=args.action_history_steps,
        )

    return make_robust_env(
        args.scenario,
        seed=args.seed,
        control_loop_profile=args.control_loop_profile,
        observation_history_steps=args.observation_history_steps,
        action_history_steps=args.action_history_steps,
        **env_kwargs,
    )


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_text_with_shadow(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
) -> None:
    x, y = xy
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 180))
    draw.text((x, y), text, font=font, fill=fill)


def overlay_frame(
    frame: np.ndarray,
    *,
    label: str,
    scenario: str,
    step: int,
    reward: float,
    force: float,
    push_flash: int,
    pushes: int,
) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGBA")
    width, _ = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    title_font = load_font(18)
    small_font = load_font(14)

    draw.rectangle((0, 0, width, 58), fill=(17, 24, 39, 180))
    draw_text_with_shadow(
        draw,
        (14, 8),
        label,
        font=title_font,
        fill=(255, 255, 255, 255),
    )
    draw_text_with_shadow(
        draw,
        (14, 34),
        f"{ENV_ID} | scenario: {scenario} | step {step:03d} | reward {reward:7.1f} | force {force:5.2f}",
        font=small_font,
        fill=(226, 232, 240, 255),
    )

    if push_flash > 0:
        draw.rounded_rectangle(
            (width - 185, 10, width - 14, 50),
            radius=7,
            fill=(190, 18, 60, 235),
        )
        draw_text_with_shadow(
            draw,
            (width - 170, 20),
            f"PUSH #{pushes}",
            font=title_font,
            fill=(255, 255, 255, 255),
        )

    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def save_poster(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.poster is not None:
        args.poster.parent.mkdir(parents=True, exist_ok=True)

    env = make_recording_env(args)
    label = args.label or args.model.stem
    poster_frame: np.ndarray | None = None
    last_frame: np.ndarray | None = None

    try:
        algorithm, model = load_policy(
            args.model,
            algorithm=args.algo,
            env=env,
            device=args.device,
        )
        observation, _ = env.reset(seed=args.seed)
        total_reward = 0.0
        push_flash = 0
        pushes = 0

        with imageio.get_writer(
            args.output,
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=16,
        ) as writer:
            for step in range(1, args.max_steps + 1):
                action, _ = model.predict(
                    observation,
                    deterministic=not args.stochastic,
                )
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)

                if bool(info.get("robustness_push_applied", False)):
                    pushes += 1
                    push_flash = max(int(args.push_flash_frames), 1)

                frame = env.render()
                force = float(info.get("force", 0.0))
                frame = overlay_frame(
                    frame,
                    label=f"{label} ({algorithm.upper()})",
                    scenario=args.scenario,
                    step=step,
                    reward=total_reward,
                    force=force,
                    push_flash=push_flash,
                    pushes=pushes,
                )
                writer.append_data(frame)
                last_frame = frame.copy()

                if poster_frame is None and (pushes > 0 or step >= 80):
                    poster_frame = frame.copy()

                if push_flash > 0:
                    push_flash -= 1

                if terminated or truncated:
                    break

    finally:
        env.close()

    if args.poster is not None:
        if poster_frame is None:
            poster_frame = last_frame
        if poster_frame is None:
            raise SystemExit("No video frames were recorded.")
        save_poster(args.poster, poster_frame)

    print(f"Saved {args.output}")
    if args.poster is not None:
        print(f"Saved {args.poster}")


if __name__ == "__main__":
    main()
