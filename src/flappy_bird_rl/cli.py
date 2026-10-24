"""CLI: environment smoke check, random sweep, plots, and recorded/live demos."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from flappy_bird_rl.config import EnvConfig, RandomConfig, load_env_config, load_random_config
from flappy_bird_rl.env import make_env
from flappy_bird_rl.evaluate import evaluate_random, run_episode, save_results, summarize
from flappy_bird_rl.random_agent import RandomAgent


def check_environment(config: EnvConfig) -> dict:
    env = make_env(config=config)
    try:
        observation, info = env.reset(seed=42)
        assert observation.shape == (12,)
        assert env.observation_space.contains(observation)
        assert env.action_space.n == 2
        for action in (0, 1):
            env.reset(seed=42)
            observation, reward, terminated, truncated, info = env.step(action)
            assert observation.shape == (12,) and np.isfinite(observation).all()
            assert env.observation_space.contains(observation)
            assert np.isfinite(reward) and "score" in info
    finally:
        env.close()
    env = make_env(config=replace(config, max_episode_steps=1))
    try:
        env.reset(seed=42)
        _, _, terminated, truncated, _ = env.step(0)
        assert truncated and not terminated
    finally:
        env.close()
    env = make_env(render_mode="rgb_array", config=config)
    try:
        env.reset(seed=42)
        frame = env.render()
        assert frame.shape == (512, 288, 3) and frame.dtype == np.uint8
    finally:
        env.close()
    return {
        "status": "ok",
        "observation_shape": [12],
        "actions": [0, 1],
        "rgb_shape": [512, 288, 3],
        "time_limit_checked": True,
        "environment": asdict(config),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flappy Bird RL baseline toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="Verify reset, both actions, time limit, and RGB")
    check.add_argument("--env-config", type=Path)

    random = subparsers.add_parser(
        "random", help="Evaluate random policies and export CSV + figures"
    )
    random.add_argument("--env-config", type=Path)
    random.add_argument("--config", type=Path)
    random.add_argument("--episodes", type=int)
    random.add_argument("--flap-probs", nargs="+", type=float)
    random.add_argument("--seed-start", type=int)
    random.add_argument("--split", choices=["validation", "test", "demo"])
    random.add_argument("--output-dir", type=Path, default=Path("results/random_baseline"))
    random.add_argument("--no-plots", action="store_true")

    demo = subparsers.add_parser("demo", help="Record MP4 or watch a random policy live")
    demo.add_argument("--env-config", type=Path)
    demo.add_argument("--flap-probs", nargs="+", type=float, default=[0.5])
    demo.add_argument("--episodes", type=int, default=3, help="Episodes per probability")
    demo.add_argument("--seed-start", type=int, default=42)
    demo.add_argument("--output", type=Path, default=Path("videos/random_demo.mp4"))
    demo.add_argument("--human", action="store_true", help="Use a desktop window instead of MP4")

    plot = subparsers.add_parser("plot", help="Regenerate figures from measured CSV")
    plot.add_argument(
        "--input", type=Path, default=Path("results/random_baseline/random_results.csv")
    )
    plot.add_argument("--output-dir", type=Path, default=Path("results/random_baseline/figures"))
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plot":
            from flappy_bird_rl.plot import plot_results

            for path in plot_results(args.input, args.output_dir):
                print(path)
            return

        env_config = load_env_config(args.env_config)
        if args.command == "check":
            print(json.dumps(check_environment(env_config), indent=2))
        elif args.command == "random":
            config = load_random_config(args.config)
            overrides = {
                name: getattr(args, name)
                for name in ("episodes", "seed_start", "split")
                if getattr(args, name) is not None
            }
            if args.flap_probs is not None:
                overrides["flap_probs"] = tuple(args.flap_probs)
            if args.split is not None and args.split != config.split and args.seed_start is None:
                overrides["seed_start"] = {"validation": 1000, "test": 10000, "demo": 42}[
                    args.split
                ]
            config = replace(config, **overrides)
            results = evaluate_random(config, env_config)
            save_results(args.output_dir, results, config, env_config)
            if not args.no_plots:
                from flappy_bird_rl.plot import plot_results

                plot_results(args.output_dir / "random_results.csv", args.output_dir / "figures")
            print(json.dumps(summarize(results), indent=2))
            print(f"Saved {len(results)} episodes to {args.output_dir}")
        elif args.command == "demo":
            config = RandomConfig(
                flap_probs=tuple(args.flap_probs),
                episodes=args.episodes,
                seed_start=args.seed_start,
                split="demo",
            )
            if args.human:
                env = make_env(render_mode="human", config=env_config)
                try:
                    for p in config.flap_probs:
                        for seed in config.seeds:
                            result = run_episode(
                                env,
                                RandomAgent(p),
                                eval_seed=seed,
                                action_seed=seed + config.action_seed_offset,
                                agent="Random",
                                config_name=f"random_p{p:g}",
                                split="demo",
                                flap_prob=p,
                            )
                            print(json.dumps(asdict(result)))
                finally:
                    env.close()
            else:
                from flappy_bird_rl.demo import record_random

                results = record_random(args.output, config, env_config)
                print(f"Recorded {len(results)} episodes to {args.output}")
    except (ValueError, TypeError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
