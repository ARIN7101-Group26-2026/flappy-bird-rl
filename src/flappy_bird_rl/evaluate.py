"""Policy-agnostic episode runner and the shared CSV contract."""

from __future__ import annotations

import csv
import json
import platform
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

import gymnasium as gym
import numpy as np

from flappy_bird_rl.config import EnvConfig, RandomConfig
from flappy_bird_rl.env import make_env
from flappy_bird_rl.random_agent import RandomAgent


class Policy(Protocol):
    def reset(self, seed: int) -> None: ...

    def predict(self, observation: np.ndarray) -> int: ...


@dataclass(frozen=True)
class EpisodeResult:
    agent: str
    config: str
    train_seed: int | None
    eval_seed: int
    action_seed: int
    train_steps: int
    episode_steps: int
    pipes_passed: int
    episode_reward: float
    terminated: bool
    truncated: bool
    flap_prob: float | None
    split: str


# Called after reset (step=0) and after each transition. Rendering stays optional.
FrameCallback = Callable[[gym.Env, int, float, dict], None]


def run_episode(
    env: gym.Env,
    policy: Policy,
    *,
    eval_seed: int,
    action_seed: int,
    agent: str,
    config_name: str,
    split: str,
    train_seed: int | None = None,
    train_steps: int = 0,
    flap_prob: float | None = None,
    on_frame: FrameCallback | None = None,
) -> EpisodeResult:
    """Evaluate on an original-reward env; stop on either end-of-episode flag."""
    observation, info = env.reset(seed=eval_seed)
    env.action_space.seed(action_seed)
    policy.reset(action_seed)
    steps, total_reward = 0, 0.0
    if on_frame:
        on_frame(env, steps, total_reward, info)
    while True:
        action = policy.predict(observation)
        if not env.action_space.contains(action):
            raise ValueError(f"Policy returned invalid action: {action!r}")
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += float(reward)
        if on_frame:
            on_frame(env, steps, total_reward, info)
        if terminated or truncated:
            return EpisodeResult(
                agent=agent,
                config=config_name,
                train_seed=train_seed,
                eval_seed=eval_seed,
                action_seed=action_seed,
                train_steps=train_steps,
                episode_steps=steps,
                pipes_passed=int(info["score"]),
                episode_reward=total_reward,
                terminated=bool(terminated),
                truncated=bool(truncated),
                flap_prob=flap_prob,
                split=split,
            )


def evaluate_random(
    config: RandomConfig, env_config: EnvConfig | None = None
) -> list[EpisodeResult]:
    results = []
    env = make_env(config=env_config)
    try:
        for p in config.flap_probs:
            policy = RandomAgent(p)
            for seed in config.seeds:
                results.append(
                    run_episode(
                        env,
                        policy,
                        eval_seed=seed,
                        action_seed=seed + config.action_seed_offset,
                        agent="Random",
                        config_name=f"random_p{p:g}",
                        split=config.split,
                        flap_prob=p,
                    )
                )
    finally:
        env.close()
    return results


def summarize(results: Sequence[EpisodeResult]) -> list[dict]:
    groups = defaultdict(list)
    for result in results:
        groups[(result.agent, result.config, result.train_seed, result.split)].append(result)
    rows = []
    for (agent, config, train_seed, split), episodes in groups.items():
        scores = np.array([e.pipes_passed for e in episodes], dtype=float)
        steps = np.array([e.episode_steps for e in episodes], dtype=float)
        rewards = np.array([e.episode_reward for e in episodes])
        rows.append(
            {
                "agent": agent,
                "config": config,
                "train_seed": train_seed,
                "split": split,
                "flap_prob": episodes[0].flap_prob,
                "episodes": len(episodes),
                "mean_pipes": float(scores.mean()),
                "std_pipes": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                "median_pipes": float(np.median(scores)),
                "max_pipes": int(scores.max()),
                "mean_steps": float(steps.mean()),
                "std_steps": float(steps.std(ddof=1)) if len(steps) > 1 else 0.0,
                "mean_reward": float(rewards.mean()),
                "std_reward": float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0,
                "terminated_episodes": sum(e.terminated for e in episodes),
                "truncated_episodes": sum(e.truncated for e in episodes),
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path: Path, env_config: EnvConfig, experiment: dict) -> None:
    data = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in (
                "flappy-bird-rl",
                "flappy-bird-gymnasium",
                "gymnasium",
                "numpy",
                "pygame",
                "matplotlib",
                "imageio",
                "imageio-ffmpeg",
                "pillow",
            )
        },
        "environment": asdict(env_config),
        "experiment": experiment,
        "reward": "original environment reward (no shaping)",
        "std_definition": "sample standard deviation (ddof=1); 0 for one episode",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_results(
    output_dir: Path,
    results: Sequence[EpisodeResult],
    config: RandomConfig,
    env_config: EnvConfig,
) -> None:
    write_csv(output_dir / "random_results.csv", (asdict(row) for row in results))
    write_csv(output_dir / "random_summary.csv", summarize(results))
    write_metadata(output_dir / "metadata.json", env_config, asdict(config))
