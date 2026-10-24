"""An observation-independent Bernoulli random policy; no training occurs."""

from __future__ import annotations

import numpy as np

from flappy_bird_rl.config import nonnegative_int, probability


class RandomAgent:
    def __init__(self, flap_prob: float = 0.5) -> None:
        probability(flap_prob)
        self.flap_prob = float(flap_prob)
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int) -> None:
        nonnegative_int(seed, "action_seed")
        self._rng = np.random.default_rng(seed)

    def predict(self, observation: np.ndarray) -> int:
        """Return 1 with probability p, otherwise 0; ignore the observation."""
        if self._rng is None:
            raise RuntimeError("Call agent.reset(action_seed) before predicting")
        return int(self._rng.random() < self.flap_prob)
