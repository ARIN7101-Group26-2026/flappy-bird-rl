"""Stream actual RGB frames to an MP4 with a small, factual status overlay."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from flappy_bird_rl.agents.random import RandomAgent
from flappy_bird_rl.config import EnvConfig, RandomConfig
from flappy_bird_rl.env import make_env
from flappy_bird_rl.evaluate import EpisodeResult, run_episode, write_csv, write_metadata


def record_random(
    output: Path,
    config: RandomConfig,
    env_config: EnvConfig,
) -> list[EpisodeResult]:
    """Record all requested episodes, in order, without selecting lucky attempts.

    Frames stream directly to FFmpeg to bound memory even at 5,000 steps.
    Adjacent CSV/JSON files describe every episode and the exact seed schedule.
    """
    if output.suffix.lower() != ".mp4":
        raise ValueError("Video output must end in .mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = make_env(render_mode="rgb_array", config=env_config)
    results = []
    first_frame = True
    fps = env.metadata["render_fps"]
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
            for p in config.flap_probs:
                for episode, seed in enumerate(config.seeds, start=1):

                    def capture(current_env, steps, reward, info, p=p, episode=episode, seed=seed):
                        nonlocal first_frame
                        frame = current_env.render()
                        canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + 64), "#102a43")
                        canvas.paste(Image.fromarray(frame), (0, 64))
                        draw = ImageDraw.Draw(canvas)
                        draw.text(
                            (10, 6),
                            f"Random | p={p:g} | episode {episode}/{config.episodes}",
                            fill="white",
                        )
                        draw.text(
                            (10, 25),
                            f"seed={seed} | steps={steps} | pipes={info['score']}",
                            fill="white",
                        )
                        draw.text(
                            (10, 44), f"reward={reward:.2f} | untrained baseline", fill="#a7d9ed"
                        )
                        writer.append_data(np.asarray(canvas))
                        if first_frame:
                            canvas.save(output.with_suffix(".png"))
                            first_frame = False

                    results.append(
                        run_episode(
                            env,
                            RandomAgent(p),
                            eval_seed=seed,
                            action_seed=seed + config.action_seed_offset,
                            agent="Random",
                            config_name=f"random_p{p:g}",
                            split=config.split,
                            flap_prob=p,
                            on_frame=capture,
                        )
                    )
    finally:
        env.close()
    write_csv(output.with_suffix(".csv"), (asdict(row) for row in results))
    write_metadata(
        output.with_suffix(".json"),
        env_config,
        {**asdict(config), "fps": fps, "frames": sum(r.episode_steps + 1 for r in results)},
    )
    return results
