"""Exhaustive check of a greedy policy: can any pipe sequence kill it?

Sampling episodes can only bound a death rate. This module settles the question
instead. The game is deterministic except for the gap height of each new pipe, which
takes one of eight values. Under a fixed deterministic policy the set of reachable game
states is therefore finite, and it can be enumerated: start from every possible opening,
follow the policy, and branch eight ways whenever a new pipe comes into view. If no
reachable state is a crash, the policy survives every pipe sequence the game can
produce, forever.

The transition function is the real environment's own ``step``; nothing is
re-implemented. Only its random generator is replaced, so the search chooses each gap.

One reduction keeps the search small. The environment draws a pipe's gap when the pipe
is recycled, about 28 steps before it enters the screen. Until then the pipe is masked
in the observation and too far away to collide with, so its gap cannot affect anything.
The search therefore leaves the gap undecided while the pipe is off-screen and branches
at the step where it becomes visible. This is the same process with the same
distribution, with eight times fewer states to visit.

    python -m sota.verify --checkpoints sota/models/pqn_champion.npz --exact-horizon 5000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from flappy_bird_rl.config import EnvConfig, load_env_config
from flappy_bird_rl.env import make_env
from sota.game_state import (
    GAP_TOPS,
    PIPE_HEIGHT,
    SCREEN_WIDTH,
    UNDECIDED,
    UPPER_YS,
    State,
    entering_pipe,
    read_state,
    write_state,
)
from sota.policy import QPolicy, load_policy


class _FirstChoice:
    """Replaces the environment's RNG. Every draw returns index 0 (= ``UNDECIDED``);
    the search overwrites the gap itself when the pipe comes into view."""

    def integers(self, low: int, high: int) -> int:
        return 0


class _Game:
    """The real environment with its random generator taken over by the search."""

    def __init__(self, env_config: EnvConfig) -> None:
        self._env = make_env(config=env_config)
        self.raw = self._env.unwrapped
        self.raw.reset(seed=0)
        self.raw._np_random = _FirstChoice()
        if self.raw._get_random_pipe()[0]["y"] != UNDECIDED:
            raise RuntimeError("Environment gap table differs from the one assumed here")

    def close(self) -> None:
        self._env.close()

    def get(self) -> State:
        return read_state(self.raw)

    def set(self, state: State, reveal: tuple[int, int] | None = None) -> None:
        write_state(self.raw, state, reveal=reveal)

    def openings(self) -> list[tuple[State, np.ndarray]]:
        """Every distinct start. Only the first pipe is on screen at reset."""
        out = []
        for upper_y in UPPER_YS:
            self.raw.reset()
            start = self.get()
            if not (start[3] <= SCREEN_WIDTH < start[5]):
                raise RuntimeError("Unexpected opening layout")
            self.set(start, reveal=(0, upper_y))
            obs, _ = self.raw._get_observation()
            out.append((self.get(), obs))
        return out


def _describe_crash(raw) -> tuple[str, dict]:
    """Classify the crash the environment has just produced."""
    y = raw._player_y
    pipes = sorted((p["x"], p["y"] + PIPE_HEIGHT) for p in raw._upper_pipes)
    hit = next((i for i, (x, _) in enumerate(pipes) if int(x) < 91 and int(x) + 52 > 57), None)
    detail = {"bird_y": float(y), "bird_velocity_before": None, "pipe_x": None, "gap_top": None}
    if hit is None or y + 24 >= raw._ground["y"] - 1:
        return "ground", detail
    x, top = pipes[hit]
    detail.update(pipe_x=float(x), gap_top=int(top))
    return ("upper pipe" if y < top else "lower pipe"), detail


def _trace(parents: dict, state: State, action: int, steps: int) -> list[dict]:
    """The last ``steps`` (state, action) pairs on one shortest path to ``state``."""
    path = [(state, action)]
    while len(path) < steps and parents.get(path[-1][0]) is not None:
        path.append(parents[path[-1][0]])
    return [
        {
            "bird_y": float(s[0]),
            "bird_velocity": int(s[1]),
            "action": int(a),
            "pipes_x_and_gap_top": [
                [float(s[3 + 2 * i]), int(s[4 + 2 * i]) + PIPE_HEIGHT]
                for i in range(3)
                if s[3 + 2 * i] <= SCREEN_WIDTH
            ],
        }
        for s, a in reversed(path)
    ]


@dataclass
class Search:
    """Everything the enumeration found; ``summary`` is the JSON-serialisable part."""

    summary: dict
    index: dict[State, int]  # every reachable state -> its row number
    parents: dict[State, tuple[State, int] | None]
    fatal_states: list[State]
    # The policy's Markov chain, as parallel arrays (only with ``keep_edges``):
    # an edge src -> dst taken with probability prob, and fatal edges src -> crash.
    edges: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    fatal_edges: tuple[np.ndarray, np.ndarray] | None = None


def search(
    policy: QPolicy,
    env_config: EnvConfig | None = None,
    *,
    max_states: int = 50_000_000,
    max_depth: int | None = None,
    max_examples: int = 10,
    progress_every: int = 0,
    keep_parents: bool = False,
    keep_edges: bool = False,
    trace_steps: int = 0,
) -> Search:
    """Enumerate every state the policy can reach. See the module docstring.

    ``keep_parents`` stores one predecessor per state (more memory), which allows
    ``trace_steps`` of history on each listed fatal example.
    ``keep_edges`` stores the full transition structure for ``exact_survival``.
    ``max_depth`` stops after that many steps from the start; the search is then
    incomplete, but everything up to that step is exact.
    """
    env_config = env_config or EnvConfig()
    keep_parents = keep_parents or trace_steps > 0
    game = _Game(env_config)
    started = time.perf_counter()
    try:
        frontier = game.openings()
        index = {state: i for i, (state, _) in enumerate(frontier)}
        parents: dict[State, tuple[State, int] | None] = (
            dict.fromkeys(index) if keep_parents else {}
        )
        edge_src, edge_dst, edge_prob = array("l"), array("l"), array("d")
        fatal_src, fatal_prob = array("l"), array("d")
        transitions = fatal = depth = flaps = 0
        fatal_kinds: Counter[str] = Counter()
        fatal_states: list[State] = []
        examples: list[dict] = []
        first_fatal_step = None
        min_margin = float("inf")
        y_low, y_high = float("inf"), float("-inf")
        complete = True
        while frontier:
            depth += 1
            q = policy.q_values(np.stack([obs for _, obs in frontier]))
            margins = q[:, 1] - q[:, 0]
            min_margin = min(min_margin, float(np.abs(margins).min()))
            actions = (margins > 0).astype(np.int64)
            flaps += int(actions.sum())
            next_frontier = []
            for (state, _), action in zip(frontier, actions, strict=True):
                y_low, y_high = min(y_low, state[0]), max(y_high, state[0])
                slot = entering_pipe(state)
                reveals = [None] if slot is None else [(slot, upper_y) for upper_y in UPPER_YS]
                probability = 1.0 / len(reveals)
                source = index[state]
                state_is_fatal = False
                for reveal in reveals:
                    game.set(state, reveal)
                    obs, _, terminated, _, _ = game.raw.step(int(action))
                    transitions += 1
                    if terminated:
                        fatal += 1
                        state_is_fatal = True
                        if keep_edges:
                            fatal_src.append(source)
                            fatal_prob.append(probability)
                        first_fatal_step = first_fatal_step or depth
                        kind, detail = _describe_crash(game.raw)
                        fatal_kinds[kind] += 1
                        if len(examples) < max_examples:
                            detail["bird_velocity_before"] = int(state[1])
                            example = {"step": depth, "kind": kind, **detail}
                            if trace_steps:
                                example["trace"] = _trace(parents, state, int(action), trace_steps)
                            examples.append(example)
                        continue
                    successor = game.get()
                    target = index.get(successor)
                    if target is None:
                        target = index[successor] = len(index)
                        next_frontier.append((successor, obs))
                        if keep_parents:
                            parents[successor] = (state, int(action))
                    if keep_edges:
                        edge_src.append(source)
                        edge_dst.append(target)
                        edge_prob.append(probability)
                if state_is_fatal:
                    fatal_states.append(state)
            frontier = next_frontier
            if progress_every and depth % progress_every == 0:
                print(
                    f"step {depth}: {len(index):,} states, frontier {len(frontier):,}, "
                    f"{fatal} fatal, {time.perf_counter() - started:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
            if frontier and (len(index) > max_states or depth == max_depth):
                complete = False
                break
    finally:
        game.close()
    summary = {
        "verified_never_dies": bool(complete and fatal == 0),
        "search_complete": complete,
        "reachable_states": len(index),
        "transitions_checked": transitions,
        "fatal_transitions": fatal,
        "fatal_states": len(fatal_states),
        "fatal_by_kind": dict(fatal_kinds.most_common()),
        "first_fatal_step": first_fatal_step,
        "fatal_examples": examples,
        "openings": len(GAP_TOPS),
        "steps_until_no_new_state": depth,
        "bird_y_range": [float(y_low), float(y_high)],
        "flap_share_of_states": flaps / max(len(index), 1),
        "smallest_decision_margin": min_margin,
        "seconds": round(time.perf_counter() - started, 1),
        "environment": asdict(env_config),
        "train_seed": policy.meta.get("train_seed"),
        "train_steps": policy.meta.get("train_steps"),
    }
    found = Search(summary, index, parents, fatal_states)
    if keep_edges:
        found.edges = (
            np.frombuffer(edge_src, dtype=np.int64).copy(),
            np.frombuffer(edge_dst, dtype=np.int64).copy(),
            np.frombuffer(edge_prob, dtype=np.float64).copy(),
        )
        found.fatal_edges = (
            np.frombuffer(fatal_src, dtype=np.int64).copy(),
            np.frombuffer(fatal_prob, dtype=np.float64).copy(),
        )
    return found


def scoring_steps(horizon: int) -> list[int]:
    """Steps at which a surviving bird scores. Pipe x-positions do not depend on chance."""
    xs = [288.0, 432.0, 576.0]
    steps = []
    for step in range(1, horizon + 1):
        steps.extend(step for x in xs if x + 26 <= 74 < x + 30)
        xs = [x - 4 for x in xs]
        xs = [SCREEN_WIDTH + 52 + SCREEN_WIDTH * 0.2 if x < -52 else x for x in xs]
    return steps


def exact_survival(found: Search, horizon: int = 5000) -> dict:
    """Exact survival probabilities of the policy, with no sampling.

    Under a fixed policy the game is a finite Markov chain whose only randomness is the
    eight-way draw of each new gap. Starting from the eight equally likely openings, the
    state distribution is pushed forward one step at a time. The result is the exact
    probability of being alive after every step, and from it the exact expected score
    under a time limit. Needs ``search(keep_edges=True)``.
    """
    if found.edges is None or found.fatal_edges is None:
        raise ValueError("search(..., keep_edges=True) is required")
    explored = found.summary["steps_until_no_new_state"]
    if not found.summary["search_complete"] and horizon > explored:
        raise ValueError(f"the search stopped after {explored} steps; horizon is {horizon}")
    src, dst, prob = found.edges
    fatal_src, fatal_prob = found.fatal_edges
    n = len(found.index)
    mass = np.zeros(n)
    mass[: len(GAP_TOPS)] = 1.0 / len(GAP_TOPS)
    alive = np.ones(horizon + 1)
    for step in range(1, horizon + 1):
        died = float((mass[fatal_src] * fatal_prob).sum()) if len(fatal_src) else 0.0
        mass = np.bincount(dst, weights=mass[src] * prob, minlength=n)
        alive[step] = alive[step - 1] - died
    scores = scoring_steps(horizon)
    # A pipe counts if the bird is alive when the scoring step begins.
    expected_pipes = float(sum(alive[step - 1] for step in scores))
    # Late-episode death rate: over the last ten pipe cycles (113 steps each), the
    # probability of dying in a step given that the bird was alive when it began.
    first = max(1, horizon - 1129)
    before, after = alive[first - 1 : horizon], alive[first : horizon + 1]
    alive_then = before > 0
    hazard = (
        float(np.mean((before[alive_then] - after[alive_then]) / before[alive_then]))
        if alive_then.any()
        else None
    )
    steps_per_pipe = horizon / max(len(scores), 1)
    return {
        "horizon_steps": horizon,
        "probability_alive_at_horizon": float(alive[horizon]),
        "expected_pipes": expected_pipes,
        "max_pipes": len(scores),
        "death_probability_per_step_late": hazard,
        "death_probability_per_pipe_late": None if hazard is None else hazard * steps_per_pipe,
        "alive_by_step": alive,
    }


def verify(policy: QPolicy, env_config: EnvConfig | None = None, **options) -> dict:
    """Run ``search`` and return its summary."""
    return search(policy, env_config, **options).summary


def dump_report(results: list[dict]) -> str:
    """Indented JSON, with every list of plain numbers kept on a single line."""
    text = json.dumps(results, indent=2)
    number_list = re.compile(r"\[\s*(-?[\d.eE+-]+(?:,\s*-?[\d.eE+-]+)*)\s*\]")
    return number_list.sub(lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", text) + "\n"


def _job(job: tuple) -> dict:
    checkpoint, env_config, max_states, progress_every, exact_horizon = job
    found = search(
        load_policy(checkpoint),
        env_config,
        max_states=max_states,
        progress_every=progress_every,
        keep_edges=exact_horizon > 0,
    )
    result = {"checkpoint": str(checkpoint), **found.summary}
    if exact_horizon > 0 and found.summary["search_complete"]:
        exact = exact_survival(found, exact_horizon)
        alive = exact.pop("alive_by_step")
        # Keep the curve compact: the value after every 10th step.
        exact["alive_every_10_steps"] = [float(v) for v in alive[::10]]
        result["exact"] = exact
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Exhaustively verify greedy policies")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--env-config", type=Path, default=Path("configs/env.json"))
    parser.add_argument("--output", type=Path, help="JSON file: a list with one entry each")
    parser.add_argument("--max-states", type=int, default=50_000_000)
    parser.add_argument("--progress-every", type=int, default=0, metavar="STEPS")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument(
        "--exact-horizon",
        type=int,
        default=0,
        metavar="STEPS",
        help="Also compute exact survival probabilities up to this many steps",
    )
    args = parser.parse_args(argv)
    env_config = load_env_config(args.env_config)
    jobs = [
        (c, env_config, args.max_states, args.progress_every, args.exact_horizon)
        for c in args.checkpoints
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        results = list(pool.map(_job, jobs))
    for r in results:
        verdict = "NEVER DIES" if r["verified_never_dies"] else f"{r['fatal_transitions']} fatal"
        exact = ""
        if "exact" in r:
            e = r["exact"]
            exact = (
                f"; P(alive at {e['horizon_steps']}) = {e['probability_alive_at_horizon']:.6f}, "
                f"E[pipes] = {e['expected_pipes']:.3f}"
            )
        print(
            f"{r['checkpoint']}: {verdict}; {r['reachable_states']:,} reachable states, "
            f"{r['seconds']:.0f}s{exact}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(dump_report(results), encoding="utf-8")


if __name__ == "__main__":
    main()
