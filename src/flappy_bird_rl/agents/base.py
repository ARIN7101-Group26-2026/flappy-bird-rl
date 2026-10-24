"""Policy interface shared by Random, DQN, and PPO evaluation."""

from typing import Protocol

import numpy as np


class Policy(Protocol):
    def reset(self, seed: int) -> None: ...

    def predict(self, observation: np.ndarray) -> int: ...
