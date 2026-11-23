"""NumPy-only inference for the trained Q-network.

Training needs PyTorch; playing does not. Checkpoints are plain ``.npz`` archives, so the
evaluation path, the tests, and CI run with the base dependencies of the framework.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LAYER_NORM_EPS = 1e-5

# Geometry of the shared environment, used only by the optional "relative" feature map.
_SCREEN_HEIGHT = 512.0
_PLAYER_HEIGHT = 24.0


def feature_map(observation: np.ndarray, kind: str) -> np.ndarray:
    """Fixed, parameter-free re-expression of the 12 shared features.

    ``raw`` passes the observation through. ``relative`` appends, for each of the three
    pipes, the bird's clearance to the upper and lower gap edge. No information is
    added: every extra column is a difference of two existing columns.
    """
    if kind == "raw":
        return observation
    if kind != "relative":
        raise ValueError(f"Unknown feature map: {kind!r}")
    y = observation[..., 9:10]
    tops = observation[..., [1, 4, 7]]
    bottoms = observation[..., [2, 5, 8]]
    above = y - tops
    below = bottoms - y - _PLAYER_HEIGHT / _SCREEN_HEIGHT
    return np.concatenate([observation, above, below], axis=-1)


def feature_dim(kind: str) -> int:
    return {"raw": 12, "relative": 18}[kind]


class QPolicy:
    """Greedy policy over a small MLP Q-function; satisfies the shared ``Policy`` protocol."""

    def __init__(self, weights: dict[str, np.ndarray], meta: dict) -> None:
        self.meta = meta
        self.weights = dict(weights)
        self.features = meta["features"]
        self.layer_norm = bool(meta["layer_norm"])
        depth = len(meta["hidden"])
        self._hidden = [
            (
                weights[f"w{i}"].astype(np.float64),
                weights[f"b{i}"].astype(np.float64),
                weights[f"g{i}"].astype(np.float64) if self.layer_norm else None,
                weights[f"s{i}"].astype(np.float64) if self.layer_norm else None,
            )
            for i in range(depth)
        ]
        self._w_out = weights["w_out"].astype(np.float64)
        self._b_out = weights["b_out"].astype(np.float64)

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        """Q-values for one observation ``(12,)`` or a batch ``(n, 12)``."""
        x = feature_map(np.asarray(observation, dtype=np.float64), self.features)
        for w, b, gain, shift in self._hidden:
            x = x @ w.T + b
            if self.layer_norm:
                x = x - x.mean(axis=-1, keepdims=True)
                x = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + LAYER_NORM_EPS)
                x = x * gain + shift
            x = np.maximum(x, 0.0)
        return x @ self._w_out.T + self._b_out

    def reset(self, seed: int) -> None:
        """Deterministic and stateless: nothing to reset."""

    def predict(self, observation: np.ndarray) -> int:
        q = self.q_values(observation)
        return int(q[1] > q[0])

    def act(self, observations: np.ndarray) -> np.ndarray:
        q = self.q_values(observations)
        return (q[:, 1] > q[:, 0]).astype(np.int64)


def save_checkpoint(path: Path, weights: dict[str, np.ndarray], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(value, dtype=np.float32) for name, value in weights.items()}
    np.savez(path, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)


def load_policy(path: Path | str) -> QPolicy:
    with np.load(Path(path), allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"]))
        weights = {name: archive[name] for name in archive.files if name != "meta"}
    return QPolicy(weights, meta)
