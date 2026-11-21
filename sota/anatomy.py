"""Record one failure in detail: two agents flying the same pipes, one of them crashing.

    python -m sota.anatomy --failing sota/runs/ablation/no_margin_s3/final.npz \
        --reference sota/models/pqn_champion.npz --seed 10042

Writes ``sota/results/anatomy/trajectory.csv`` (bird position per step and agent),
``pipes.csv`` (when each pipe is level with the bird, and its gap) and ``meta.json``.
The figure is drawn from these files by ``python -m sota.analyze report``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flappy_bird_rl.config import load_env_config
from flappy_bird_rl.env import make_env
from flappy_bird_rl.evaluate import write_csv
from sota.evaluate import variant_of
from sota.policy import QPolicy, load_policy

BIRD_LEFT, BIRD_RIGHT, PIPE_WIDTH, PIPE_HEIGHT = 57, 91, 52, 320


def fly(policy: QPolicy, seed: int, env_config, max_steps: int) -> tuple[list[dict], list[dict]]:
    """One episode from ``seed``: bird track, and the pipes that came level with the bird."""
    env = make_env(config=env_config)
    raw = env.unwrapped
    track, windows = [], []
    open_window: dict[int, dict] = {}  # pipe slot -> the window it is currently adding to
    try:
        obs, _ = env.reset(seed=seed)
        track.append({"step": 0, "bird_y": float(raw._player_y), "flap": 0, "crashed": False})
        for step in range(1, max_steps + 1):
            action = policy.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            track.append(
                {
                    "step": step,
                    "bird_y": float(raw._player_y),
                    "flap": int(action),
                    "crashed": bool(terminated),
                }
            )
            pairs = zip(raw._upper_pipes, raw._lower_pipes, strict=True)
            for slot, (upper, lower) in enumerate(pairs):
                x = int(upper["x"])
                if x < BIRD_RIGHT and x + PIPE_WIDTH > BIRD_LEFT:
                    window = open_window.get(slot)
                    if window is None or window["last_step"] != step - 1:
                        window = {
                            "first_step": step,
                            "last_step": step,
                            "gap_top": float(upper["y"]) + PIPE_HEIGHT,
                            "gap_bottom": float(lower["y"]),
                        }
                        open_window[slot] = window
                        windows.append(window)
                    window["last_step"] = step
            if terminated or truncated:
                break
    finally:
        env.close()
    return track, windows


def find_fatal_seed(policy: QPolicy, env_config, seeds: range) -> int | None:
    for seed in seeds:
        track, _ = fly(policy, seed, env_config, env_config.max_episode_steps)
        if track[-1]["crashed"]:
            return seed
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record one failure for the anatomy figure")
    parser.add_argument("--failing", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, help="Default: first test seed the agent dies on")
    parser.add_argument("--env-config", type=Path, default=Path("configs/env.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("sota/results/anatomy"))
    args = parser.parse_args(argv)
    env_config = load_env_config(args.env_config)
    failing, reference = load_policy(args.failing), load_policy(args.reference)
    seed = args.seed
    if seed is None:
        seed = find_fatal_seed(failing, env_config, range(10000, 10100))
        if seed is None:
            raise SystemExit("The agent survives every test seed; pass --seed explicitly")
    failed, pipes = fly(failing, seed, env_config, env_config.max_episode_steps)
    horizon = failed[-1]["step"] + 25
    good, good_pipes = fly(reference, seed, env_config, horizon)
    rows = [{"agent": "failing", "eval_seed": seed, **r} for r in failed]
    rows += [{"agent": "reference", "eval_seed": seed, **r} for r in good]
    write_csv(args.output_dir / "trajectory.csv", rows)
    write_csv(args.output_dir / "pipes.csv", good_pipes)
    meta = {
        "eval_seed": seed,
        "failing": str(args.failing),
        "failing_variant": variant_of(args.failing),
        "failing_train_seed": failing.meta["train_seed"],
        "reference": str(args.reference),
        "crash_step": failed[-1]["step"],
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"seed {seed}: failing agent crashed at step {failed[-1]['step']}")


if __name__ == "__main__":
    main()
