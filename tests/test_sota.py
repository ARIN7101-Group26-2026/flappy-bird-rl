"""Tests for the standalone PQN track. Training tests are skipped without PyTorch."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flappy_bird_rl.config import EnvConfig
from flappy_bird_rl.env import make_env
from flappy_bird_rl.evaluate import run_episode
from sota import stats
from sota.policy import QPolicy, feature_dim, feature_map, load_policy, save_checkpoint
from sota.rollout import greedy_rollouts
from sota.targets import lambda_returns
from sota.vec_env import HardPipeSampler, VecFlappy, too_close
from sota.verify import exact_survival, scoring_steps, search, verify

CHAMPION = Path(__file__).resolve().parents[1] / "sota" / "models" / "pqn_champion.npz"
MAX_PIPES_IN_5000_STEPS = 132


def random_policy(seed: int = 0, hidden=(16, 16), features="raw", layer_norm=True) -> QPolicy:
    rng = np.random.default_rng(seed)
    sizes = [feature_dim(features), *hidden]
    weights = {}
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        weights[f"w{i}"] = rng.normal(size=(b, a))
        weights[f"b{i}"] = rng.normal(size=b)
        weights[f"g{i}"] = rng.uniform(0.5, 1.5, size=b)
        weights[f"s{i}"] = rng.normal(size=b) * 0.1
    weights["w_out"] = rng.normal(size=(2, sizes[-1]))
    weights["b_out"] = np.array([0.3, 0.0])
    meta = {
        "hidden": list(hidden),
        "layer_norm": layer_norm,
        "features": features,
        "train_seed": seed,
        "train_steps": 0,
    }
    return QPolicy(weights, meta)


# --- statistics -----------------------------------------------------------------------


def test_iqm_is_the_mean_of_the_middle_half():
    assert stats.iqm(range(1, 9)) == pytest.approx(4.5)
    assert stats.iqm([1, 2, 3, 4, 100]) == pytest.approx(3.0)
    assert stats.iqm([0, 0, 10, 10, 10, 10, 10, 100]) == pytest.approx(10.0)


def test_clopper_pearson_matches_reference_values():
    assert stats.clopper_pearson(1, 10) == pytest.approx((0.0025286, 0.4450161), abs=1e-6)
    assert stats.clopper_pearson(0, 100) == pytest.approx((0.0, 0.0362167), abs=1e-6)
    assert stats.clopper_pearson(10, 10) == pytest.approx((0.6915029, 1.0), abs=1e-6)
    low, high = stats.clopper_pearson(5, 1_000_000)
    assert low == pytest.approx(1.6235e-6, rel=1e-3) and high == pytest.approx(1.1668e-5, rel=1e-3)


def test_zero_failure_bound_is_the_rule_of_three():
    assert stats.zero_failure_upper_bound(1_000_000) == pytest.approx(3e-6, rel=2e-3)
    assert stats.zero_failure_upper_bound(100, level=0.975) == pytest.approx(
        stats.clopper_pearson(0, 100)[1]
    )


def test_kaplan_meier_handles_right_censoring():
    km = stats.kaplan_meier(
        [1, 2, 2, 3, 4, 5, 5, 5], [True, True, False, True, False, True, False, False]
    )
    assert km["time"].tolist() == [1, 2, 3, 5]
    assert km["survival"] == pytest.approx([0.875, 0.75, 0.6, 0.4])
    assert (km["lower"] <= km["survival"]).all() and (km["survival"] <= km["upper"]).all()


def test_kaplan_meier_without_deaths_is_empty():
    km = stats.kaplan_meier([5000, 5000], [False, False])
    assert km["time"].size == 0


def test_exact_mann_whitney():
    # Complete separation of 3 vs 3: only the two extreme splits are as extreme, of 20.
    result = stats.mann_whitney_exact([1, 2, 3], [4, 5, 6])
    assert result["probability_x_smaller"] == pytest.approx(1.0)
    assert result["p_value"] == pytest.approx(2 / 20)
    # Identical samples: nothing to tell apart.
    same = stats.mann_whitney_exact([1, 2, 3, 4], [1, 2, 3, 4])
    assert same["probability_x_smaller"] == pytest.approx(0.5)
    assert same["p_value"] == pytest.approx(1.0)
    # Known table value: n = m = 4, U = 1 has two-sided p = 4/70.
    table = stats.mann_whitney_exact([1, 2, 3, 5], [4, 6, 7, 8])
    assert table["p_value"] == pytest.approx(4 / 70)


def test_bootstrap_intervals_cover_the_point_estimate():
    low, high = stats.bootstrap_ci(np.arange(100), resamples=2000)
    assert low < 49.5 < high
    low, high = stats.hierarchical_bootstrap_ci(
        [np.arange(10), np.arange(10) + 5, np.arange(10) + 2], resamples=500
    )
    assert low < 6.83 < high


# --- targets ----------------------------------------------------------------------------


def test_lambda_returns_bootstrap_on_truncation_but_not_on_termination():
    rewards = np.ones((3, 3), dtype=np.float32)
    next_values = np.full((3, 3), 10.0, dtype=np.float32)
    terminated = np.zeros((3, 3), dtype=bool)
    done = np.zeros((3, 3), dtype=bool)
    terminated[1, 0] = done[1, 0] = True  # env 0 dies at t=1
    done[1, 1] = True  # env 1 hits the time limit at t=1
    out = lambda_returns(rewards, next_values, terminated, done, gamma=0.5, lam=1.0)
    assert out[1, 0] == pytest.approx(1.0)  # death: no successor value
    assert out[1, 1] == pytest.approx(1.0 + 0.5 * 10.0)  # truncation: bootstrap
    assert out[0, 0] == pytest.approx(1.0 + 0.5 * 1.0)  # recursion stops at the death
    # No boundary, lambda=1: full 3-step return plus the bootstrap at the rollout end.
    assert out[0, 2] == pytest.approx(1 + 0.5 * (1 + 0.5 * (1 + 0.5 * 10)))


def test_lambda_zero_is_one_step_q_learning():
    rng = np.random.default_rng(0)
    rewards, values = rng.normal(size=(4, 2)), rng.normal(size=(4, 2))
    none = np.zeros((4, 2), dtype=bool)
    out = lambda_returns(rewards, values, none, none, gamma=0.9, lam=0.0)
    assert out == pytest.approx(rewards + 0.9 * values)


# --- policy and rollouts ------------------------------------------------------------------


def test_relative_features_only_recombine_existing_columns():
    obs = np.random.default_rng(1).uniform(-1, 1, size=(5, 12))
    out = feature_map(obs, "relative")
    assert out.shape == (5, 18)
    assert out[:, :12] == pytest.approx(obs)
    assert out[:, 12] == pytest.approx(obs[:, 9] - obs[:, 1])
    assert out[:, 15] == pytest.approx(obs[:, 2] - obs[:, 9] - 24 / 512)


def test_checkpoint_roundtrip_and_batch_consistency(tmp_path):
    policy = random_policy(features="relative")
    obs = np.random.default_rng(2).uniform(-1, 1, size=(7, 12))
    save_checkpoint(tmp_path / "p.npz", policy.weights, policy.meta)
    loaded = load_policy(tmp_path / "p.npz")
    assert loaded.meta == policy.meta
    assert loaded.q_values(obs) == pytest.approx(policy.q_values(obs), abs=1e-5)
    assert [loaded.predict(o) for o in obs] == loaded.act(obs).tolist()


def test_batched_rollouts_match_the_shared_episode_runner():
    policy = random_policy(seed=3)
    seeds = [1000, 1001, 1002]
    batched = greedy_rollouts(policy.act, seeds)
    env = make_env()
    try:
        for i, seed in enumerate(seeds):
            single = run_episode(
                env,
                policy,
                eval_seed=seed,
                action_seed=seed,
                agent="PQN",
                config_name="test",
                split="validation",
            )
            assert single.episode_steps == batched.steps[i]
            assert single.pipes_passed == batched.pipes[i]
            assert single.episode_reward == pytest.approx(batched.rewards[i])
            assert single.terminated == batched.terminated[i]
    finally:
        env.close()


def test_vec_env_reports_true_successor_and_truncation():
    vec = VecFlappy(2, seed=0, env_config=EnvConfig(max_episode_steps=3))
    try:
        for _ in range(2):
            step = vec.step(np.zeros(2, dtype=np.int64))
            assert not step.done.any()
            assert step.obs == pytest.approx(step.next_obs)
        step = vec.step(np.zeros(2, dtype=np.int64))
        assert step.truncated.all() and not step.terminated.any()
        # After a boundary the policy sees the reset state, not the final one.
        assert not np.allclose(step.obs, step.next_obs)
        assert vec.episodes_finished == 2
    finally:
        vec.close()


# --- exhaustive verification -----------------------------------------------------------------


def constant_policy(flap: bool) -> QPolicy:
    weights = {
        "w0": np.zeros((4, 12)),
        "b0": np.zeros(4),
        "g0": np.ones(4),
        "s0": np.zeros(4),
        "w_out": np.zeros((2, 4)),
        "b_out": np.array([0.0, 1.0] if flap else [1.0, 0.0]),
    }
    meta = {"hidden": [4], "layer_norm": True, "features": "raw", "train_seed": None}
    return QPolicy(weights, meta)


def test_verifier_finds_every_death_of_a_policy_that_never_flaps():
    result = verify(constant_policy(flap=False))
    # Eight openings (one per first-pipe gap); each falls to the ground at step 31.
    assert result["search_complete"] and not result["verified_never_dies"]
    assert result["openings"] == 8
    assert result["fatal_transitions"] == 8
    assert result["fatal_by_kind"] == {"ground": 8}
    assert result["first_fatal_step"] == 31
    assert result["reachable_states"] == 8 * 31


def test_verifier_agrees_with_the_random_baseline_for_constant_flapping():
    # The repository's Random study measured 50 steps per episode at flap probability 0.5,
    # because the bird sits on the ceiling and hits the first pipe. Always flapping does
    # the same, in every one of the 8 x 8 first-two-pipe combinations.
    result = verify(constant_policy(flap=True))
    assert result["first_fatal_step"] == 50
    assert result["fatal_transitions"] == 64
    assert result["bird_y_range"][0] < 0


def test_verifier_states_match_real_episodes():
    """Every state of a real seeded episode must appear in the search's own bookkeeping."""
    from sota.verify import _Game

    policy = random_policy(seed=5)
    game = _Game(EnvConfig())
    env = make_env()
    try:
        openings = {state for state, _ in game.openings()}
        for seed in range(8):
            obs, _ = env.reset(seed=seed)
            raw = env.unwrapped
            pipes = sorted((p["x"], p["y"] if p["x"] <= 288 else -220) for p in raw._upper_pipes)
            start = (raw._player_y, raw._player_vel_y, raw._player_rot, *sum(pipes, ()))
            assert start in openings
            # Loading the canonical state reproduces the observation of the real reset.
            game.set(start)
            assert game.raw._get_observation()[0] == pytest.approx(obs)
            assert policy.predict(obs) in (0, 1)
    finally:
        env.close()
        game.close()


def test_exact_survival_of_constant_policies():
    idle = exact_survival(search(constant_policy(flap=False), keep_edges=True), horizon=100)
    assert idle["alive_by_step"][30] == pytest.approx(1.0)
    assert idle["alive_by_step"][31] == pytest.approx(0.0)
    assert idle["expected_pipes"] == pytest.approx(0.0)
    flap = exact_survival(search(constant_policy(flap=True), keep_edges=True), horizon=100)
    assert flap["alive_by_step"][49] == pytest.approx(1.0)
    assert flap["alive_by_step"][50] == pytest.approx(0.0)


class AimAtGapPolicy:
    """Hand-written rule: flap when the bird's centre is below the centre of pipe slot 0.

    Not a good player, which is the point: it dies at many different times, so the
    enumeration and sampled episodes can be compared along a whole survival curve.
    """

    meta = {"train_seed": None, "train_steps": 0}

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        obs = np.atleast_2d(observation)
        target = (obs[:, 1] + obs[:, 2]) / 2.0
        flap = obs[:, 9] + 12 / 512 > target + 0.03
        return np.stack([np.zeros(len(obs)), np.where(flap, 1.0, -1.0)], axis=1)

    def act(self, observations: np.ndarray) -> np.ndarray:
        q = self.q_values(observations)
        return (q[:, 1] > q[:, 0]).astype(np.int64)


def test_exact_survival_matches_sampled_episodes():
    """The enumeration and plain seeded episodes must describe the same random process."""
    policy = AimAtGapPolicy()
    found = search(policy, keep_edges=True, max_depth=200)
    exact = exact_survival(found, horizon=200)["alive_by_step"]
    episodes = 400
    sampled = greedy_rollouts(policy.act, range(5000, 5000 + episodes), EnvConfig())
    informative = 0
    for step in (60, 100, 140, 180, 200):
        p = float(exact[step])
        observed = float((sampled.steps > step).mean())
        tolerance = 4.0 * np.sqrt(max(p * (1 - p), 1e-4) / episodes)
        assert abs(observed - p) <= tolerance, (step, p, observed)
        informative += 0.02 < p < 0.98
    assert informative >= 2, "the test policy should die at varied times"


def test_scoring_steps_give_the_score_ceiling():
    assert len(scoring_steps(5000)) == MAX_PIPES_IN_5000_STEPS
    assert scoring_steps(200)[:3] == [61, 97, 133]


# --- training-only environment options ------------------------------------------------------


def test_hard_pipe_sampler_alternates_between_extremes():
    sampler = HardPipeSampler(np.random.default_rng(0), hard_prob=1.0)
    first = sampler.integers(0, 8)
    following = [sampler.integers(0, 8) for _ in range(4)]
    expected = [7, 0, 7, 0] if first < 4 else [0, 7, 0, 7]
    assert following == expected
    uniform = HardPipeSampler(np.random.default_rng(0), hard_prob=0.0)
    assert len({uniform.integers(0, 8) for _ in range(200)}) == 8


def test_safety_margin_only_tightens_the_real_crash_test():
    env = make_env()
    try:
        env.reset(seed=0)
        raw = env.unwrapped
        assert not too_close(raw, 0, 0) and not too_close(raw, 6, 4)
        pipe, lower = raw._upper_pipes[0], raw._lower_pipes[0]
        pipe["x"] = lower["x"] = 60  # level with the bird
        gap_top = pipe["y"] + 320
        raw._player_y = gap_top + 4  # 4 px below the upper pipe: no contact
        assert not raw._check_crash()
        assert not too_close(raw, 3, 0)
        assert too_close(raw, 6, 0)
        pipe["x"] = lower["x"] = 93  # 2 px ahead of the bird's nose
        raw._player_y = gap_top - 50  # would hit the pipe face once it arrives
        assert not raw._check_crash()
        assert not too_close(raw, 0, 0) and too_close(raw, 0, 4)
    finally:
        env.close()


def test_safety_margin_ends_training_episodes_early():
    plain = VecFlappy(1, seed=0)
    strict = VecFlappy(1, seed=0, safety_margin=(40, 40))
    try:
        lengths = []
        for vec in (plain, strict):
            while not vec.finished:
                vec.step(np.zeros(1, dtype=np.int64))
            lengths.append(vec.finished[0][1])
    finally:
        plain.close()
        strict.close()
    assert lengths[1] < lengths[0]  # the same fall "crashes" sooner with a 40 px margin


# --- the committed champion ---------------------------------------------------------------


@pytest.mark.skipif(not CHAMPION.exists(), reason="champion checkpoint not present")
def test_champion_reaches_the_time_limit_with_the_maximum_score():
    policy = load_policy(CHAMPION)
    env = make_env()
    try:
        result = run_episode(
            env,
            policy,
            eval_seed=10000,
            action_seed=1010000,
            agent="PQN",
            config_name="pqn",
            split="test",
        )
    finally:
        env.close()
    assert result.truncated and not result.terminated
    assert result.episode_steps == 5000
    assert result.pipes_passed == MAX_PIPES_IN_5000_STEPS


# --- training (needs PyTorch) -------------------------------------------------------------


def test_torch_network_matches_numpy_policy():
    torch = pytest.importorskip("torch")
    from sota.pqn import PQNConfig, QNetwork

    for features in ("raw", "relative"):
        for layer_norm in (True, False):
            config = PQNConfig(hidden=(32, 16), features=features, layer_norm=layer_norm)
            net = QNetwork(config)
            meta = {"hidden": [32, 16], "layer_norm": layer_norm, "features": features}
            policy = QPolicy(net.export(), meta)
            obs = np.random.default_rng(4).uniform(-1, 1, size=(9, 12)).astype(np.float32)
            with torch.no_grad():
                expected = net(torch.from_numpy(obs)).numpy()
            assert policy.q_values(obs) == pytest.approx(expected, abs=1e-5)


def test_short_training_run_writes_artifacts(tmp_path):
    pytest.importorskip("torch")
    from sota.pqn import PQNConfig, train

    config = replace(
        PQNConfig(),
        total_steps=2048,
        num_envs=8,
        rollout_steps=16,
        minibatches=2,
        hidden=(16,),
        eval_interval=1024,
        eval_episodes=2,
        train_restart_prob=0.01,
        train_hard_pipe_prob=0.5,
        train_safety_margin=(6, 4),
    )
    summary = train(config, seed=0, run_dir=tmp_path, quiet=True)
    assert summary["train_steps"] == 2048
    for name in ("progress.csv", "final.npz", "summary.json"):
        assert (tmp_path / name).exists()
    policy = load_policy(tmp_path / "final.npz")
    assert policy.meta["train_steps"] == 2048
    assert policy.predict(np.zeros(12)) in (0, 1)
