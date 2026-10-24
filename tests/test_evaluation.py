import csv
import json
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import pytest

from flappy_bird_rl.agents.random import RandomAgent
from flappy_bird_rl.config import EnvConfig, RandomConfig
from flappy_bird_rl.evaluate import evaluate_random, run_episode, save_results, summarize


@pytest.mark.parametrize("p,expected", [(0.0, 0), (1.0, 1)])
def test_probability_boundaries_and_observation_independence(p, expected):
    agent = RandomAgent(p)
    agent.reset(99)
    assert all(agent.predict(np.ones(12) * i) == expected for i in range(20))


def test_random_rng_is_local_repeatable_and_ignores_observation():
    a, b = RandomAgent(0.2), RandomAgent(0.2)
    a.reset(42)
    b.reset(42)
    left = [a.predict(np.zeros(12)) for _ in range(100)]
    np.random.seed(999)
    np.random.random(100)
    right = [b.predict(np.ones(12)) for _ in range(100)]
    assert left == right
    assert set(left) == {0, 1}


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probabilities(p):
    with pytest.raises(ValueError):
        RandomAgent(p)


class ScriptedEnv(gym.Env):
    action_space = gym.spaces.Discrete(2)

    def __init__(self, ending):
        self.ending = ending

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(12), {"score": 0}

    def step(self, action):
        self.steps += 1
        assert self.steps <= 3, "runner stepped after episode ended"
        end = self.steps == 3
        return (
            np.zeros(12),
            [0.1, 1.0, -1.0][self.steps - 1],
            end and self.ending in {"terminated", "both"},
            end and self.ending in {"truncated", "both"},
            {"score": 7},
        )


@pytest.mark.parametrize("ending", ["terminated", "truncated", "both"])
def test_runner_preserves_end_flags_and_score_independent_of_reward(ending):
    result = run_episode(
        ScriptedEnv(ending),
        RandomAgent(),
        eval_seed=1,
        action_seed=1000001,
        agent="Random",
        config_name="test",
        split="validation",
    )
    assert result.episode_steps == 3
    assert result.pipes_passed == 7
    assert result.episode_reward == pytest.approx(0.1)
    assert result.terminated == (ending in {"terminated", "both"})
    assert result.truncated == (ending in {"truncated", "both"})


def test_sweep_replays_exactly_and_exports_shared_seed_schedule(tmp_path):
    config = RandomConfig(episodes=5)
    first = evaluate_random(config)
    assert first == evaluate_random(config)
    assert len(first) == 20
    for p in config.flap_probs:
        group = [row for row in first if row.flap_prob == p]
        assert [row.eval_seed for row in group] == list(config.seeds)
        assert all(row.action_seed == row.eval_seed + 1000000 for row in group)
    assert all(row.train_seed is None and row.train_steps == 0 for row in first)
    save_results(tmp_path, first, config, EnvConfig())
    with (tmp_path / "random_results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert set(rows[0]) == set(asdict(first[0]))
    assert rows[0]["train_seed"] == ""
    assert len(summarize(first)) == 4
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["environment"]["use_lidar"] is False
    assert metadata["experiment"]["seed_start"] == 1000


def test_aggregation_uses_sample_standard_deviation():
    from dataclasses import replace

    original = evaluate_random(RandomConfig(episodes=1, flap_probs=(0.1,)))[0]
    rows = [replace(original, pipes_passed=score) for score in (0, 1, 2)]
    summary = summarize(rows)[0]
    assert summary["mean_pipes"] == 1
    assert summary["std_pipes"] == 1
    assert summary["median_pipes"] == 1
    assert summary["max_pipes"] == 2
