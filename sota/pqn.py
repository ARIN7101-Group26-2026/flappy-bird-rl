"""PQN (Parallelised Q-Network) trained on the shared Flappy Bird environment.

The algorithm follows Gallici et al., "Simplifying Deep Temporal Difference Learning"
(ICLR 2025): online Q-learning on many parallel environments, with LayerNorm in the
Q-network instead of a target network, and lambda-returns over short rollouts instead
of a replay buffer. This file is an independent PyTorch implementation.

Run from the repository root:

    python -m sota.pqn --config sota/configs/pqn.json --seed 0 --run-dir sota/runs/pqn_s0
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flappy_bird_rl.config import EnvConfig, load_env_config
from sota.policy import LAYER_NORM_EPS, QPolicy, feature_dim, save_checkpoint
from sota.rollout import greedy_rollouts
from sota.targets import lambda_returns
from sota.vec_env import VecFlappy


@dataclass(frozen=True)
class PQNConfig:
    total_steps: int = 10_000_000
    num_envs: int = 256
    rollout_steps: int = 32
    epochs: int = 2
    minibatches: int = 16
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    q_lambda: float = 0.65
    max_grad_norm: float = 10.0
    hidden: tuple[int, ...] = (256, 256)
    layer_norm: bool = True
    features: str = "raw"
    # Exploration: epsilon decays linearly; a random action flaps with explore_flap_prob.
    eps_start: float = 1.0
    eps_end: float = 0.005
    eps_fraction: float = 0.2
    explore_flap_prob: float = 0.1
    # Training-only random restarts: probability per step of cutting an episode short
    # (a truncation, so targets bootstrap across it). Keeps episode starts in the data.
    train_restart_prob: float = 0.0
    # Training-only: probability that a new pipe is the extreme gap farthest from the
    # previous one. Over-samples the tightest transitions, which are otherwise 1 in 64.
    train_hard_pipe_prob: float = 0.0
    # Training-only safety margin (vertical px, horizontal px): coming this close to a
    # pipe or the ground counts as a crash while training. Observations are unchanged.
    train_safety_margin: tuple[int, int] = (0, 0)
    # Validation during training (greedy, shared validation seeds).
    eval_interval: int = 500_000
    eval_episodes: int = 32
    eval_seed_start: int = 1000

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden", tuple(self.hidden))
        object.__setattr__(self, "train_safety_margin", tuple(self.train_safety_margin))


def load_pqn_config(path: Path | None, **overrides) -> PQNConfig:
    data = json.loads(path.read_text(encoding="utf-8")) if path else {}
    known = {f.name for f in fields(PQNConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown PQN config keys: {sorted(unknown)}")
    return replace(PQNConfig(**data), **{k: v for k, v in overrides.items() if v is not None})


class QNetwork(nn.Module):
    """MLP with LayerNorm before each ReLU; mirrors ``sota.policy.QPolicy`` exactly."""

    def __init__(self, config: PQNConfig) -> None:
        super().__init__()
        self.features = config.features
        self.use_layer_norm = config.layer_norm
        sizes = [feature_dim(config.features), *config.hidden]
        self.linears = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:], strict=True)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(b, eps=LAYER_NORM_EPS) if config.layer_norm else nn.Identity()
            for b in sizes[1:]
        )
        self.head = nn.Linear(sizes[-1], 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        if self.features == "relative":
            y = obs[..., 9:10]
            above = y - obs[..., [1, 4, 7]]
            below = obs[..., [2, 5, 8]] - y - 24.0 / 512.0
            x = torch.cat([obs, above, below], dim=-1)
        for linear, norm in zip(self.linears, self.norms, strict=True):
            x = torch.relu(norm(linear(x)))
        return self.head(x)

    def export(self) -> dict[str, np.ndarray]:
        weights = {}
        for i, (linear, norm) in enumerate(zip(self.linears, self.norms, strict=True)):
            weights[f"w{i}"] = linear.weight.detach().numpy().copy()
            weights[f"b{i}"] = linear.bias.detach().numpy().copy()
            if self.use_layer_norm:
                weights[f"g{i}"] = norm.weight.detach().numpy().copy()
                weights[f"s{i}"] = norm.bias.detach().numpy().copy()
        weights["w_out"] = self.head.weight.detach().numpy().copy()
        weights["b_out"] = self.head.bias.detach().numpy().copy()
        return weights

    def load(self, weights: dict[str, np.ndarray]) -> None:
        """Inverse of ``export``: continue training from a saved checkpoint."""

        def assign(parameter: nn.Parameter, value: np.ndarray) -> None:
            with torch.no_grad():
                parameter.copy_(torch.as_tensor(np.asarray(value), dtype=parameter.dtype))

        for i, (linear, norm) in enumerate(zip(self.linears, self.norms, strict=True)):
            assign(linear.weight, weights[f"w{i}"])
            assign(linear.bias, weights[f"b{i}"])
            if self.use_layer_norm:
                assign(norm.weight, weights[f"g{i}"])
                assign(norm.bias, weights[f"s{i}"])
        assign(self.head.weight, weights["w_out"])
        assign(self.head.bias, weights["b_out"])


class Learner:
    """Network, optimiser, and parallel environments; one call = one rollout + update."""

    def __init__(
        self,
        config: PQNConfig,
        seed: int,
        env_config: EnvConfig,
        weights: dict[str, np.ndarray] | None = None,
    ) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.config = config
        self.env_config = env_config
        self.rng = np.random.default_rng(seed)
        self.vec = VecFlappy(
            config.num_envs,
            seed,
            env_config,
            config.train_restart_prob,
            config.train_hard_pipe_prob,
            config.train_safety_margin,
        )
        self.net = QNetwork(config)
        if weights is not None:
            self.net.load(weights)
        self.optimizer = torch.optim.RAdam(self.net.parameters(), lr=config.learning_rate, eps=1e-5)
        self.meta = {
            "algorithm": "PQN",
            "train_seed": seed,
            "hidden": list(config.hidden),
            "layer_norm": config.layer_norm,
            "features": config.features,
            "config": asdict(config),
            "environment": asdict(env_config),
        }
        n, steps = config.num_envs, config.rollout_steps
        self.batch = n * steps
        self.global_step = 0
        self._obs = self.vec.obs
        self._obs_buf = np.empty((steps, n, 12), dtype=np.float32)
        self._next_obs_buf = np.empty((steps, n, 12), dtype=np.float32)
        self._act_buf = np.empty((steps, n), dtype=np.int64)
        self._rew_buf = np.empty((steps, n), dtype=np.float32)
        self._term_buf = np.empty((steps, n), dtype=bool)
        self._done_buf = np.empty((steps, n), dtype=bool)

    def policy(self) -> QPolicy:
        return QPolicy(self.net.export(), {**self.meta, "train_steps": self.global_step})

    def save(self, path: Path, **extra_meta) -> None:
        meta = {**self.meta, "train_steps": self.global_step, **extra_meta}
        save_checkpoint(path, self.net.export(), meta)

    def update(self, epsilon: float, lr: float) -> float:
        """Collect one rollout with epsilon-greedy actions, then fit the lambda-returns."""
        config, net, rng = self.config, self.net, self.rng
        n, steps, batch = config.num_envs, config.rollout_steps, self.batch
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        obs = self._obs
        for t in range(steps):
            with torch.no_grad():
                greedy = net(torch.from_numpy(obs)).argmax(dim=1).numpy()
            explore = rng.random(n) < epsilon
            random_actions = (rng.random(n) < config.explore_flap_prob).astype(np.int64)
            actions = np.where(explore, random_actions, greedy)
            step = self.vec.step(actions)
            self._obs_buf[t], self._act_buf[t], self._rew_buf[t] = obs, actions, step.reward
            self._next_obs_buf[t] = step.next_obs
            self._term_buf[t], self._done_buf[t] = step.terminated, step.done
            obs = step.obs
        self._obs = obs
        self.global_step += batch

        flat_obs = torch.from_numpy(self._obs_buf.reshape(batch, 12))
        flat_next = torch.from_numpy(self._next_obs_buf.reshape(batch, 12))
        flat_act = torch.from_numpy(self._act_buf.reshape(batch, 1))
        minibatch = batch // config.minibatches
        losses = []
        for _ in range(config.epochs):
            with torch.no_grad():
                next_values = net(flat_next).max(dim=1).values.numpy().reshape(steps, n)
            targets = torch.from_numpy(
                lambda_returns(
                    self._rew_buf,
                    next_values,
                    self._term_buf,
                    self._done_buf,
                    config.gamma,
                    config.q_lambda,
                ).reshape(batch)
            )
            order = torch.from_numpy(rng.permutation(batch))
            for start in range(0, batch, minibatch):
                index = order[start : start + minibatch]
                q = net(flat_obs[index]).gather(1, flat_act[index]).squeeze(1)
                loss = 0.5 * (q - targets[index]).pow(2).mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
                self.optimizer.step()
                losses.append(loss.item())
        return float(np.mean(losses))

    def close(self) -> None:
        self.vec.close()


def train(
    config: PQNConfig,
    seed: int,
    run_dir: Path,
    env_config: EnvConfig | None = None,
    *,
    quiet: bool = False,
) -> dict:
    """Train one agent; write ``progress.csv``, ``final.npz``, and ``summary.json``.

    The agent of a run is always its last checkpoint. There is no checkpoint selection:
    the validation scores logged along the way are for learning curves only.
    """
    env_config = env_config or EnvConfig()
    run_dir.mkdir(parents=True, exist_ok=True)
    learner = Learner(config, seed, env_config)
    vec = learner.vec
    updates = config.total_steps // learner.batch
    eval_seeds = range(config.eval_seed_start, config.eval_seed_start + config.eval_episodes)

    rows: list[dict] = []
    next_eval = config.eval_interval
    started = time.perf_counter()

    def evaluate() -> dict:
        result = greedy_rollouts(learner.policy().act, eval_seeds, env_config)
        return {
            "eval_mean_pipes": float(result.pipes.mean()),
            "eval_min_pipes": int(result.pipes.min()),
            "eval_survival": result.survival_rate,
            "eval_mean_steps": float(result.steps.mean()),
        }

    for update in range(1, updates + 1):
        progress = (update - 1) / updates
        epsilon = config.eps_start + (config.eps_end - config.eps_start) * min(
            1.0, progress / max(config.eps_fraction, 1e-9)
        )
        lr = config.learning_rate * (1.0 - progress)  # linear decay to zero
        loss = learner.update(epsilon, lr)
        global_step = learner.global_step
        row = {
            "step": global_step,
            "seconds": round(time.perf_counter() - started, 2),
            "epsilon": round(epsilon, 5),
            "lr": lr,
            "loss": loss,
            "train_episodes": vec.episodes_finished,
            "train_mean_pipes": float(np.mean([e[2] for e in vec.finished]))
            if vec.finished
            else 0.0,
            "train_mean_steps": float(np.mean([e[1] for e in vec.finished]))
            if vec.finished
            else 0.0,
        }
        if global_step >= next_eval or update == updates:
            next_eval += config.eval_interval
            row.update(evaluate())
            if not quiet:
                print(json.dumps({"seed": seed, **row}), flush=True)
        rows.append(row)

    learner.close()
    learner.save(run_dir / "final.npz")
    global_step = learner.global_step
    columns = list(rows[0]) + [
        "eval_mean_pipes",
        "eval_min_pipes",
        "eval_survival",
        "eval_mean_steps",
    ]
    columns = list(dict.fromkeys(columns))
    with (run_dir / "progress.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "train_seed": seed,
        "train_steps": global_step,
        "seconds": round(time.perf_counter() - started, 1),
        "steps_per_second": round(global_step / (time.perf_counter() - started)),
        "final_eval": {k: v for k, v in rows[-1].items() if k.startswith("eval_")},
        "config": asdict(config),
        "environment": asdict(env_config),
        "torch": torch.__version__,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a PQN agent on the shared environment")
    parser.add_argument("--config", type=Path, default=Path("sota/configs/pqn.json"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override keys")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    overrides = {"total_steps": args.total_steps}
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key] = json.loads(value)
    config = load_pqn_config(args.config if args.config.exists() else None, **overrides)
    summary = train(
        config, args.seed, args.run_dir, load_env_config(args.env_config), quiet=args.quiet
    )
    print(json.dumps({k: summary[k] for k in list(summary)[:5]}, indent=2))


if __name__ == "__main__":
    main()
