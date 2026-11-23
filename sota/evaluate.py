"""Evaluate trained checkpoints.

Two experiments, kept in separate files because they answer different questions:

``protocol``  The framework's standard evaluation: validation seeds 1000-1099 and test
              seeds 10000-10099, 5,000-step limit, through the shared ``run_episode``.
              Rows follow the shared CSV contract, so they can be concatenated with the
              Random baseline and any other agent.
``stress``    The same environment with a much longer time limit and fresh seeds, to
              measure how often the agent dies once the 5,000-step cap no longer hides it.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from flappy_bird_rl.config import EnvConfig, load_env_config
from flappy_bird_rl.env import make_env
from flappy_bird_rl.evaluate import run_episode, summarize, write_csv, write_metadata
from sota.policy import load_policy
from sota.rollout import greedy_rollouts

SPLIT_SEEDS = {"validation": 1000, "test": 10000, "demo": 42}
ACTION_SEED_OFFSET = 1_000_000


def variant_of(checkpoint: Path) -> str:
    """``.../<variant>_s<seed>/final.npz`` -> ``<variant>``; anything else -> ``pqn``."""
    name, _, seed = Path(checkpoint).parent.name.rpartition("_s")
    return name if name and seed.isdigit() else "pqn"


def _protocol_job(job: tuple) -> list[dict]:
    checkpoint, split, episodes, env_config, config_name = job
    config_name = config_name or variant_of(checkpoint)
    policy = load_policy(checkpoint)
    env = make_env(config=env_config)
    rows = []
    try:
        for seed in range(SPLIT_SEEDS[split], SPLIT_SEEDS[split] + episodes):
            result = run_episode(
                env,
                policy,
                eval_seed=seed,
                action_seed=seed + ACTION_SEED_OFFSET,
                agent="PQN",
                config_name=config_name,
                split=split,
                train_seed=policy.meta["train_seed"],
                train_steps=policy.meta["train_steps"],
            )
            rows.append(asdict(result))
    finally:
        env.close()
    return rows


def run_protocol(
    checkpoints: list[Path],
    output_dir: Path,
    env_config: EnvConfig,
    *,
    splits: tuple[str, ...] = ("validation", "test"),
    episodes: int = 100,
    config_name: str | None = None,
    workers: int = 1,
) -> list[dict]:
    jobs = [(c, s, episodes, env_config, config_name) for c in checkpoints for s in splits]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for chunk in pool.map(_protocol_job, jobs) for row in chunk]
    write_csv(output_dir / "pqn_results.csv", rows)
    from flappy_bird_rl.evaluate import EpisodeResult

    write_csv(output_dir / "pqn_summary.csv", summarize([EpisodeResult(**r) for r in rows]))
    write_metadata(
        output_dir / "metadata.json",
        env_config,
        {
            "agent": "PQN",
            "config": config_name or "variant name taken from each run directory",
            "splits": {s: [SPLIT_SEEDS[s], SPLIT_SEEDS[s] + episodes - 1] for s in splits},
            "episodes_per_split": episodes,
            "action_seed_offset": ACTION_SEED_OFFSET,
            "policy": "greedy argmax over Q-values (deterministic)",
            "checkpoints": [str(c) for c in checkpoints],
        },
    )
    return rows


def _stress_job(job: tuple) -> list[dict]:
    checkpoint, seeds, env_config = job
    policy = load_policy(checkpoint)
    result = greedy_rollouts(policy.act, seeds, env_config)
    return [
        {
            "checkpoint": str(checkpoint),
            "train_seed": policy.meta["train_seed"],
            "train_steps": policy.meta["train_steps"],
            "eval_seed": int(seed),
            "max_episode_steps": env_config.max_episode_steps,
            "pipe_gap": env_config.pipe_gap,
            "episode_steps": int(steps),
            "pipes_passed": int(pipes),
            "terminated": bool(term),
            "truncated": bool(trunc),
        }
        for seed, steps, pipes, term, trunc in zip(
            result.seeds,
            result.steps,
            result.pipes,
            result.terminated,
            result.truncated,
            strict=True,
        )
    ]


def run_stress(
    checkpoints: list[Path],
    output: Path,
    env_config: EnvConfig,
    *,
    episodes: int,
    seed_start: int,
    workers: int = 1,
    chunk: int = 4,
) -> list[dict]:
    seeds = list(range(seed_start, seed_start + episodes))
    jobs = [
        (c, seeds[i : i + chunk], env_config)
        for c in checkpoints
        for i in range(0, episodes, chunk)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for part in pool.map(_stress_job, jobs) for row in part]
    write_csv(output, rows)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate PQN checkpoints")
    sub = parser.add_subparsers(dest="command", required=True)
    workers = max(1, (os.cpu_count() or 2) - 2)

    protocol = sub.add_parser("protocol", help="Standard validation/test evaluation")
    protocol.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    protocol.add_argument("--output-dir", type=Path, required=True)
    protocol.add_argument("--splits", nargs="+", default=["validation", "test"])
    protocol.add_argument("--episodes", type=int, default=100)
    protocol.add_argument("--config-name", help="Default: the variant in the run directory")

    stress = sub.add_parser("stress", help="Long-horizon survival on fresh seeds")
    stress.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    stress.add_argument("--output", type=Path, required=True)
    stress.add_argument("--max-steps", type=int, default=1_000_000)
    stress.add_argument("--episodes", type=int, default=64)
    stress.add_argument("--seed-start", type=int, default=20000)
    stress.add_argument("--pipe-gap", type=int)

    for p in (protocol, stress):
        p.add_argument("--env-config", type=Path, default=Path("configs/env.json"))
        p.add_argument("--workers", type=int, default=workers)
    args = parser.parse_args(argv)
    env_config = load_env_config(args.env_config)

    if args.command == "protocol":
        rows = run_protocol(
            args.checkpoints,
            args.output_dir,
            env_config,
            splits=tuple(args.splits),
            episodes=args.episodes,
            config_name=args.config_name,
            workers=args.workers,
        )
        pipes = np.array([r["pipes_passed"] for r in rows])
        died = np.array([r["terminated"] for r in rows])
        print(
            json.dumps(
                {
                    "episodes": len(rows),
                    "mean_pipes": float(pipes.mean()),
                    "min_pipes": int(pipes.min()),
                    "deaths": int(died.sum()),
                }
            )
        )
    else:
        env_config = replace(env_config, max_episode_steps=args.max_steps)
        if args.pipe_gap is not None:
            env_config = replace(env_config, pipe_gap=args.pipe_gap)
        rows = run_stress(
            args.checkpoints,
            args.output,
            env_config,
            episodes=args.episodes,
            seed_start=args.seed_start,
            workers=args.workers,
        )
        pipes = sum(r["pipes_passed"] for r in rows)
        deaths = sum(r["terminated"] for r in rows)
        print(
            json.dumps(
                {
                    "episodes": len(rows),
                    "total_pipes": pipes,
                    "deaths": deaths,
                    "total_steps": sum(r["episode_steps"] for r in rows),
                }
            )
        )


if __name__ == "__main__":
    main()
