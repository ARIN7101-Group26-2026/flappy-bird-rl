from dataclasses import replace

import numpy as np
from gymnasium.utils.env_checker import check_env

from flappy_bird_rl.config import EnvConfig
from flappy_bird_rl.env import make_env


def test_gymnasium_contract_and_determinism():
    env = make_env()
    try:
        check_env(env.unwrapped, skip_render_check=True)
        obs, info = env.reset(seed=42)
        assert obs.shape == (12,)
        assert env.observation_space.contains(obs)
        assert info["score"] == 0
        assert env.action_space.n == 2
        assert env.spec.max_episode_steps == 5000
        for action in (0, 1):
            env.reset(seed=42)
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            assert np.isfinite(reward)
            assert not terminated and not truncated
    finally:
        env.close()


def test_real_crash_and_time_limit_have_different_flags():
    env = make_env()
    try:
        env.reset(seed=42)
        for _ in range(5000):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        assert terminated and not truncated
    finally:
        env.close()
    env = make_env(config=replace(EnvConfig(), max_episode_steps=1))
    try:
        env.reset(seed=42)
        _, _, terminated, truncated, _ = env.step(0)
        assert truncated and not terminated
    finally:
        env.close()


def test_rgb_without_a_display():
    env = make_env(render_mode="rgb_array")
    try:
        env.reset(seed=42)
        frame = env.render()
        assert frame.shape == (512, 288, 3)
        assert frame.dtype == np.uint8
        assert frame.std() > 0
    finally:
        env.close()
