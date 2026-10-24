"""The one environment factory all agents must use."""

from __future__ import annotations

import os

import gymnasium as gym

from flappy_bird_rl.config import EnvConfig


def make_env(render_mode: str | None = None, *, config: EnvConfig | None = None) -> gym.Env:
    """Create a normalized 12-feature environment with the original rewards.

    Seeds belong to reset(seed=...), so every evaluation episode can be replayed.
    Gymnasium installs exactly one TimeLimit via max_episode_steps.
    """
    if render_mode not in {None, "human", "rgb_array"}:
        raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
    config = config or EnvConfig()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import flappy_bird_gymnasium  # noqa: F401 -- registers FlappyBird-v0

    return gym.make(
        config.env_id,
        use_lidar=config.use_lidar,
        normalize_obs=config.normalize_obs,
        audio_on=False,
        pipe_gap=config.pipe_gap,
        max_episode_steps=config.max_episode_steps,
        render_mode=render_mode,
    )
