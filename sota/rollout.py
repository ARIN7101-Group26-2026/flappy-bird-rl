"""Batched greedy rollouts on the shared environment.

Used for validation during training and for long-horizon stress tests. The official
validation/test tables go through the framework's own ``run_episode`` instead; a unit
test checks that both paths agree episode for episode.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from flappy_bird_rl.config import EnvConfig
from flappy_bird_rl.env import make_env


@dataclass(frozen=True)
class Rollouts:
    seeds: np.ndarray
    steps: np.ndarray
    pipes: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    @property
    def survival_rate(self) -> float:
        """Share of episodes that reached the time limit without dying."""
        return float((~self.terminated).mean())


def greedy_rollouts(
    act: Callable[[np.ndarray], np.ndarray],
    seeds: Sequence[int],
    env_config: EnvConfig | None = None,
) -> Rollouts:
    """Play one episode per seed with ``act`` (batch of observations -> batch of actions)."""
    seeds = np.asarray(list(seeds), dtype=np.int64)
    n = len(seeds)
    envs = [make_env(config=env_config) for _ in range(n)]
    try:
        obs = np.stack([env.reset(seed=int(s))[0] for env, s in zip(envs, seeds, strict=True)])
        steps = np.zeros(n, dtype=np.int64)
        pipes = np.zeros(n, dtype=np.int64)
        rewards = np.zeros(n)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        active = np.arange(n)
        while active.size:
            actions = act(obs[active])
            still_running = []
            for action, i in zip(actions, active, strict=True):
                o, r, term, trunc, info = envs[i].step(int(action))
                obs[i] = o
                steps[i] += 1
                rewards[i] += r
                if term or trunc:
                    pipes[i] = info["score"]
                    terminated[i], truncated[i] = term, trunc
                else:
                    still_running.append(i)
            active = np.asarray(still_running, dtype=np.int64)
    finally:
        for env in envs:
            env.close()
    return Rollouts(seeds, steps, pipes, rewards, terminated, truncated)
