"""In-process batch of shared-factory environments with explicit truncation bookkeeping."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from flappy_bird_rl.config import EnvConfig
from flappy_bird_rl.env import make_env


@dataclass
class VecStep:
    """One batched transition.

    ``obs`` is what the policy acts on next (already reset where an episode ended).
    ``next_obs`` is the true successor state, which differs from ``obs`` exactly where
    ``done`` is set. Bootstrapping must use ``next_obs`` and must stop only on
    ``terminated``: a time-limit truncation is not a death.
    """

    obs: np.ndarray
    next_obs: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    @property
    def done(self) -> np.ndarray:
        return self.terminated | self.truncated


class HardPipeSampler:
    """Training-only pipe generator that over-samples large gap changes.

    The environment draws each gap height uniformly from eight values, so the tightest
    transition (highest gap to lowest, or the reverse) follows a given pipe only once in
    64 times. With probability ``hard_prob`` this sampler instead picks the extreme value
    farthest from the previous gap. It replaces ``env.np_random`` in training
    environments; evaluation environments keep the original uniform generator.
    """

    def __init__(self, rng: np.random.Generator, hard_prob: float) -> None:
        self._rng = rng
        self._hard_prob = float(hard_prob)
        self._last: int | None = None

    def integers(self, low: int, high: int) -> int:
        if self._last is not None and self._rng.random() < self._hard_prob:
            index = high - 1 if self._last < (low + high) / 2 else low
        else:
            index = int(self._rng.integers(low, high))
        self._last = index
        return index


def too_close(raw, margin_y: int, margin_x: int) -> bool:
    """True if the bird is within the given margins of a pipe or the ground.

    Same geometry as the environment's crash test, with the bird's box grown by
    ``margin_y`` pixels up and down and ``margin_x`` pixels left and right.
    """
    y = raw._player_y
    if y + 24 + margin_y >= raw._ground["y"] - 1:
        return True
    for upper, lower in zip(raw._upper_pipes, raw._lower_pipes, strict=True):
        x = int(upper["x"])
        if x < 91 + margin_x and x + 52 > 57 - margin_x:
            if y - margin_y < upper["y"] + 320 or y + 24 + margin_y > lower["y"]:
                return True
    return False


class VecFlappy:
    """Steps ``num_envs`` copies of ``make_env()`` sequentially and auto-resets them.

    The simulator runs at roughly 140k steps/s on one core, so a single process is
    enough for training; spare cores are better spent on independent seeds.
    """

    def __init__(
        self,
        num_envs: int,
        seed: int,
        env_config: EnvConfig | None = None,
        restart_prob: float = 0.0,
        hard_pipe_prob: float = 0.0,
        safety_margin: Sequence[int] = (0, 0),
    ) -> None:
        """Three training-only options, all off by default. None of them changes what
        the agent observes, and none is used in evaluation.

        ``restart_prob`` truncates each running episode with this probability per step.
        A strong agent otherwise plays 5,000-step episodes, so the first seconds of an
        episode nearly vanish from the data and the agent forgets how to start. The
        cut is reported as a truncation, so value targets still bootstrap across it.

        ``hard_pipe_prob`` over-samples large gap changes (see ``HardPipeSampler``).

        ``safety_margin=(vertical, horizontal)`` ends a training episode as a crash when
        the bird comes within that many pixels of a pipe or the ground. The agent learns
        to keep that clearance from the edges it sees; in evaluation, where only real
        contact counts, the clearance is spare room.
        """
        self.num_envs = num_envs
        self._restart_prob = float(restart_prob)
        self._margin_y, self._margin_x = (int(v) for v in safety_margin)
        self._rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5EED]))
        self.envs = [make_env(config=env_config) for _ in range(num_envs)]
        seeds = np.random.SeedSequence(seed).generate_state(num_envs)
        self.obs = np.stack(
            [env.reset(seed=int(s))[0] for env, s in zip(self.envs, seeds, strict=True)]
        ).astype(np.float32)
        if hard_pipe_prob:
            # The first episode of each env was drawn uniformly; later ones are biased.
            for env in self.envs:
                env.unwrapped._np_random = HardPipeSampler(self._rng, hard_pipe_prob)
        self._returns = np.zeros(num_envs)
        self._lengths = np.zeros(num_envs, dtype=np.int64)
        # Finished training episodes: (return, steps, pipes, terminated).
        self.finished: deque[tuple[float, int, int, bool]] = deque(maxlen=1000)
        self.episodes_finished = 0

    def step(self, actions: np.ndarray) -> VecStep:
        n = self.num_envs
        next_obs = np.empty((n, self.obs.shape[1]), dtype=np.float32)
        reward = np.empty(n, dtype=np.float32)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        obs = np.empty_like(next_obs)
        restart = self._rng.random(n) < self._restart_prob if self._restart_prob else None
        margin = self._margin_y > 0 or self._margin_x > 0
        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.step(int(actions[i]))
            if margin and not term and too_close(env.unwrapped, self._margin_y, self._margin_x):
                term, r = True, -1.0
            if restart is not None and restart[i] and not term:
                trunc = True
            next_obs[i] = o
            reward[i] = r
            self._returns[i] += r
            self._lengths[i] += 1
            if term or trunc:
                terminated[i], truncated[i] = term, trunc
                self.finished.append(
                    (float(self._returns[i]), int(self._lengths[i]), int(info["score"]), bool(term))
                )
                self.episodes_finished += 1
                self._returns[i] = 0.0
                self._lengths[i] = 0
                o, _ = env.reset()
            obs[i] = o
        self.obs = obs
        return VecStep(obs, next_obs, reward, terminated, truncated)

    def close(self) -> None:
        for env in self.envs:
            env.close()
