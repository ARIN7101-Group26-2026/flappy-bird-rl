"""Bootstrapped return targets (NumPy only, so they are testable without PyTorch)."""

from __future__ import annotations

import numpy as np


def lambda_returns(
    rewards: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    done: np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Peng's Q(lambda) targets over a rollout of shape ``(T, N)``.

    ``next_values[t]`` is max_a Q(s_{t+1}, a) for the true successor state. The recursion
    is cut at every episode boundary and at the end of the rollout. The successor value
    is dropped only on termination, so time-limit truncations still bootstrap.
    """
    steps = rewards.shape[0]
    returns = np.empty_like(rewards)
    for t in reversed(range(steps)):
        if t == steps - 1:
            bootstrap = next_values[t]
        else:
            mixed = (1.0 - lam) * next_values[t] + lam * returns[t + 1]
            bootstrap = np.where(done[t], next_values[t], mixed)
        returns[t] = rewards[t] + gamma * np.where(terminated[t], 0.0, bootstrap)
    return returns
