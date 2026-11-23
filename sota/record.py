"""Record the trained agent on the shared environment, in the framework's video format.

    python -m sota.record --checkpoint sota/models/pqn_champion.npz \
        --output sota/videos/pqn_demo.mp4 --episodes 1 --seed-start 42

Episodes are recorded consecutively from the demo seeds; nothing is cherry-picked.
``--frame-skip k`` keeps every k-th frame (a k-times fast-forward) and says so on screen.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from flappy_bird_rl.config import EnvConfig, load_env_config
from flappy_bird_rl.env import make_env
from flappy_bird_rl.evaluate import EpisodeResult, run_episode, write_csv, write_metadata
from sota.evaluate import ACTION_SEED_OFFSET
from sota.policy import QPolicy, load_policy


def record(
    policy: QPolicy,
    output: Path,
    env_config: EnvConfig,
    *,
    episodes: int = 1,
    seed_start: int = 42,
    frame_skip: int = 1,
) -> list[EpisodeResult]:
    if output.suffix.lower() != ".mp4":
        raise ValueError("Video output must end in .mp4")
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = make_env(render_mode="rgb_array", config=env_config)
    fps = env.metadata["render_fps"]
    speed = "real time" if frame_skip == 1 else f"{frame_skip}x fast-forward"
    results: list[EpisodeResult] = []
    frames = 0
    preview_saved = False
    try:
        with imageio.get_writer(
            str(output),
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=16,
        ) as writer:
            for episode, seed in enumerate(range(seed_start, seed_start + episodes), start=1):

                def capture(current_env, steps, reward, info, episode=episode, seed=seed):
                    nonlocal frames, preview_saved
                    if steps % frame_skip:
                        return
                    frame = current_env.render()
                    canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + 64), "#102a43")
                    canvas.paste(Image.fromarray(frame), (0, 64))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((10, 6), f"PQN | episode {episode}/{episodes}", fill="white")
                    draw.text(
                        (10, 25),
                        f"seed={seed} | steps={steps} | pipes={info['score']}",
                        fill="white",
                    )
                    draw.text((10, 44), f"reward={reward:.1f} | {speed}", fill="#a7d9ed")
                    writer.append_data(np.asarray(canvas))
                    frames += 1
                    # Preview from mid-flight rather than the empty first frame.
                    if not preview_saved and info["score"] >= 3:
                        canvas.save(output.with_suffix(".png"))
                        preview_saved = True

                results.append(
                    run_episode(
                        env,
                        policy,
                        eval_seed=seed,
                        action_seed=seed + ACTION_SEED_OFFSET,
                        agent="PQN",
                        config_name="pqn",
                        split="demo",
                        train_seed=policy.meta["train_seed"],
                        train_steps=policy.meta["train_steps"],
                        on_frame=capture,
                    )
                )
    finally:
        env.close()
    write_csv(output.with_suffix(".csv"), (asdict(row) for row in results))
    write_metadata(
        output.with_suffix(".json"),
        env_config,
        {
            "agent": "PQN",
            "episodes": episodes,
            "seed_start": seed_start,
            "split": "demo",
            "fps": fps,
            "frame_skip": frame_skip,
            "frames": frames,
            "policy": "greedy argmax over Q-values (deterministic)",
        },
    )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record the PQN agent")
    parser.add_argument("--checkpoint", type=Path, default=Path("sota/models/pqn_champion.npz"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env.json"))
    parser.add_argument("--output", type=Path, default=Path("sota/videos/pqn_demo.mp4"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--frame-skip", type=int, default=1)
    args = parser.parse_args(argv)
    results = record(
        load_policy(args.checkpoint),
        args.output,
        load_env_config(args.env_config),
        episodes=args.episodes,
        seed_start=args.seed_start,
        frame_skip=args.frame_skip,
    )
    for r in results:
        print(
            f"seed={r.eval_seed} steps={r.episode_steps} pipes={r.pipes_passed} "
            f"terminated={r.terminated} truncated={r.truncated}"
        )


if __name__ == "__main__":
    main()
