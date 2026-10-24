"""Small, validated JSON configs shared by scripts and Python callers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


def positive_int(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def nonnegative_int(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def probability(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("flap probability must be a number in [0, 1]")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("flap probability must be finite and in [0, 1]")


@dataclass(frozen=True)
class EnvConfig:
    env_id: str = "FlappyBird-v0"
    use_lidar: bool = False
    normalize_obs: bool = True
    max_episode_steps: int = 5000
    pipe_gap: int = 100

    def __post_init__(self) -> None:
        if self.env_id != "FlappyBird-v0":
            raise ValueError("Only FlappyBird-v0 is supported")
        if self.use_lidar is not False or self.normalize_obs is not True:
            raise ValueError("This baseline requires normalized 12-D features (no LIDAR)")
        positive_int(self.max_episode_steps, "max_episode_steps")
        if type(self.pipe_gap) is not int or not 40 <= self.pipe_gap <= 200:
            raise ValueError("pipe_gap must be an integer between 40 and 200")


@dataclass(frozen=True)
class RandomConfig:
    flap_probs: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05)
    episodes: int = 100
    split: str = "validation"
    seed_start: int = 1000
    action_seed_offset: int = 1000000

    def __post_init__(self) -> None:
        positive_int(self.episodes, "episodes")
        nonnegative_int(self.seed_start, "seed_start")
        positive_int(self.action_seed_offset, "action_seed_offset")
        if self.split not in {"validation", "test", "demo"}:
            raise ValueError("split must be validation, test, or demo")
        if not self.flap_probs:
            raise ValueError("flap_probs cannot be empty")
        for p in self.flap_probs:
            probability(p)
        if len(set(self.flap_probs)) != len(self.flap_probs):
            raise ValueError("flap_probs must be unique")

    @property
    def seeds(self) -> range:
        return range(self.seed_start, self.seed_start + self.episodes)


def load_env_config(path: Path | None) -> EnvConfig:
    return EnvConfig(**json.loads(path.read_text(encoding="utf-8"))) if path else EnvConfig()


def load_random_config(path: Path | None) -> RandomConfig:
    if path is None:
        return RandomConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    if "flap_probs" in data:
        data["flap_probs"] = tuple(data["flap_probs"])
    return RandomConfig(**data)
