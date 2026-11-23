"""Turn raw outputs into the committed tables and figures.

    python -m sota.analyze collect --group main --runs sota/runs/main
    python -m sota.analyze report

``collect`` condenses training logs into ``sota/results/curves/<group>.csv``.
``report`` reads only files under ``sota/results`` and writes ``summary.json`` and the
figures, so the analysis can be re-run without retraining.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from sota import stats

RESULTS = Path("sota/results")
MAX_PIPES = 132  # pipes that fit into the 5,000-step limit; see RESEARCH.md
RANDOM_BEST_MEAN = 0.07  # results/random_baseline/random_summary.csv, flap probability 0.1

# Ablation variants in reporting order: display name of what was changed.
VARIANTS = {
    "full": "Full method",
    "no_margin_no_hard_pipes": "Neither margin nor over-sampling",
    "no_margin": "No safety margin",
    "no_hard_pipes": "No over-sampling of large gap changes",
    "no_restarts": "No random restarts",
    "raw_features": "Raw 12 features (no clearances)",
    "lambda0": "One-step targets (λ = 0)",
    "uniform_explore": "Uniform random exploration",
    "no_layernorm": "No LayerNorm",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_name(path: str) -> tuple[str, int]:
    """``.../<variant>_s<seed>/final.npz`` -> (variant, seed)."""
    variant, seed = Path(path).parent.name.rsplit("_s", 1)
    return variant, int(seed)


def is_true(value: str) -> bool:
    return value == "True"


# --- collect --------------------------------------------------------------------------


def collect(group: str, runs: Path) -> Path:
    rows = []
    for run in sorted(runs.iterdir()):
        progress = run / "progress.csv"
        if not progress.exists():
            continue
        variant, seed = run.name.rsplit("_s", 1)
        for row in read_csv(progress):
            if row["eval_mean_pipes"] == "":
                continue
            rows.append(
                {
                    "variant": variant,
                    "train_seed": int(seed),
                    "step": int(row["step"]),
                    "seconds": float(row["seconds"]),
                    "train_mean_pipes": round(float(row["train_mean_pipes"]), 3),
                    "eval_mean_pipes": round(float(row["eval_mean_pipes"]), 4),
                    "eval_min_pipes": int(row["eval_min_pipes"]),
                    "eval_survival": round(float(row["eval_survival"]), 4),
                }
            )
    output = RESULTS / "curves" / f"{group}.csv"
    write_rows(output, rows)
    return output


# --- tables ---------------------------------------------------------------------------


def curve_matrix(rows: list[dict], variant: str, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Evaluation curve per training seed: (steps, values[seed, step])."""
    by_seed = defaultdict(dict)
    for row in rows:
        if row["variant"] == variant:
            by_seed[int(row["train_seed"])][int(row["step"])] = float(row[column])
    steps = sorted(set.intersection(*(set(v) for v in by_seed.values())))
    values = np.array([[by_seed[s][t] for t in steps] for s in sorted(by_seed)])
    return np.asarray(steps), values


def load_verification(path: Path) -> dict[tuple[str, int], dict]:
    """Exhaustive-verification reports keyed by (variant, training seed)."""
    out = {}
    for report in json.loads(path.read_text(encoding="utf-8")):
        exact = report.get("exact", {})
        out[run_name(report["checkpoint"])] = {
            "verified_never_dies": report["verified_never_dies"],
            "reachable_states": report["reachable_states"],
            "fatal_transitions": report["fatal_transitions"],
            "death_probability": 1.0 - exact["probability_alive_at_horizon"] if exact else None,
            "expected_pipes": exact.get("expected_pipes"),
            "late_death_per_pipe": exact.get("death_probability_per_pipe_late"),
            "alive_every_10_steps": exact.get("alive_every_10_steps"),
            "smallest_decision_margin": report["smallest_decision_margin"],
        }
    return out


def test_deaths(rows: list[dict]) -> dict[tuple[str, int], tuple[int, int]]:
    """(deaths, episodes) on the test split per (variant, training seed)."""
    counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if row["split"] != "test":
            continue
        key = (row["config"], int(row["train_seed"]))
        counts[key][0] += is_true(row["terminated"])
        counts[key][1] += 1
    return {key: (d, n) for key, (d, n) in counts.items()}


def protocol_summary(rows: list[dict]) -> dict:
    out = {}
    for split in ("validation", "test"):
        part = [r for r in rows if r["split"] == split]
        if not part:
            continue
        per_seed = defaultdict(list)
        for r in part:
            per_seed[int(r["train_seed"])].append(int(r["pipes_passed"]))
        means = {seed: float(np.mean(v)) for seed, v in sorted(per_seed.items())}
        deaths = sum(is_true(r["terminated"]) for r in part)
        low, high = stats.clopper_pearson(deaths, len(part))
        out[split] = {
            "training_seeds": len(per_seed),
            "episodes": len(part),
            "mean_pipes_per_seed": means,
            "iqm_over_seeds": stats.iqm(list(means.values())),
            "iqm_ci95": stats.hierarchical_bootstrap_ci(list(per_seed.values()), stats.iqm),
            "mean_over_seeds": float(np.mean(list(means.values()))),
            "min_episode_pipes": min(int(r["pipes_passed"]) for r in part),
            "deaths": deaths,
            "episodes_at_ceiling": sum(int(r["pipes_passed"]) == MAX_PIPES for r in part),
            "episode_death_rate_ci95": [low, high]
            if deaths
            else [0.0, stats.zero_failure_upper_bound(len(part))],
        }
    return out


def exposure_summary(rows: list[dict]) -> dict:
    """Sample-based reliability of one agent: counts and exact binomial bounds."""
    deaths = sum(is_true(r["terminated"]) for r in rows)
    pipes = sum(int(r["pipes_passed"]) for r in rows)
    steps = sum(int(r["episode_steps"]) for r in rows)
    attempts = pipes + deaths
    episodes = len(rows)
    out = {
        "episodes": episodes,
        "max_episode_steps": int(rows[0]["max_episode_steps"]),
        "total_steps": steps,
        "total_pipes": pipes,
        "deaths": deaths,
        "longest_episode_pipes": max(int(r["pipes_passed"]) for r in rows),
        "hours_of_play_at_30fps": steps / 30 / 3600,
    }
    if deaths:
        out["death_rate_per_pipe"] = deaths / attempts
        out["death_rate_per_pipe_ci95"] = list(stats.clopper_pearson(deaths, attempts))
        out["death_rate_per_episode_ci95"] = list(stats.clopper_pearson(deaths, episodes))
    else:
        out["death_rate_per_pipe_upper95"] = stats.zero_failure_upper_bound(attempts)
        out["death_rate_per_episode_upper95"] = stats.zero_failure_upper_bound(episodes)
    return out


# --- figures ----------------------------------------------------------------------------

PROBABILITY_FLOOR = 1e-13  # where an exact zero is drawn on a log axis


def _power_of_ten(value: float) -> str:
    """0.00031 -> '3.1 × 10⁻⁴'; values of 0.1 and above are printed plainly."""
    if value >= 0.1 or value == 0:
        return f"{value:.2g}"
    exponent = int(np.floor(np.log10(value)))
    digits = str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{value / 10**exponent:.1f} × 10{digits}"


def _probability_axis(ax, floor: float = PROBABILITY_FLOOR) -> None:
    ax.set_xscale("log")
    ax.set_xlim(floor / 3, 3)
    ticks = [floor, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "10⁻¹⁰", "10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻²", "1"])
    ax.minorticks_off()


def figure_learning_curve(curves: list[dict], path: Path, variant: str = "full") -> None:
    import matplotlib.pyplot as plt

    from sota import plot_style as style

    style.apply_style()
    steps, values = curve_matrix(curves, variant, "eval_mean_pipes")
    x = steps / 1e6
    center = np.array([stats.iqm(values[:, j]) for j in range(values.shape[1])])
    bands = np.array(
        [stats.bootstrap_ci(values[:, j], stats.iqm, resamples=4000) for j in range(len(x))]
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for run in values:
        ax.plot(x, run, color=style.CONTEXT, linewidth=1.0, zorder=2)
    ax.fill_between(x, bands[:, 0], bands[:, 1], color=style.SERIES[0], alpha=0.14, lw=0)
    ax.plot(x, center, color=style.SERIES[0], linewidth=2.2, zorder=4)
    ax.axhline(MAX_PIPES, color=style.AXIS, linewidth=1.0, zorder=1)
    ax.text(
        x[-1],
        MAX_PIPES + 2.5,
        f"Ceiling of the 5,000-step limit: {MAX_PIPES} pipes",
        ha="right",
        va="bottom",
        color=style.INK_SECONDARY,
        fontsize=10,
    )
    ax.text(
        x[-1],
        3.0,
        f"Random policy: {RANDOM_BEST_MEAN:.2f} pipes",
        ha="right",
        va="bottom",
        color=style.INK_SECONDARY,
        fontsize=10,
    )
    ax.set_ylim(0, MAX_PIPES + 14)
    ax.set_xlim(0, x[-1])
    ax.set_yticks([0, 25, 50, 75, 100, 125])
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Pipes passed per validation episode")
    at_ceiling = int((values[:, -1] >= MAX_PIPES).sum())
    ax.set_title(f"{at_ceiling} of {len(values)} training seeds end at the score ceiling", pad=26)
    style.subtitle(
        ax,
        f"Blue: interquartile mean over {len(values)} training seeds, 95% bootstrap band. "
        "Grey: individual seeds.",
    )
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def figure_ablation(verification: dict[tuple[str, int], dict], path: Path) -> None:
    """Exact probability of dying within one standard episode, per variant and seed."""
    import matplotlib.pyplot as plt

    from sota import plot_style as style

    style.apply_style()
    by_variant = defaultdict(list)
    for (variant, _), report in verification.items():
        if variant in VARIANTS and report["death_probability"] is not None:
            by_variant[variant].append(report["death_probability"])
    order = sorted(by_variant, key=lambda v: (v != "full", float(np.median(by_variant[v])), v))
    fig, ax = plt.subplots(figsize=(8.8, 0.56 * len(order) + 2.0))
    for row, variant in enumerate(order):
        y = len(order) - 1 - row
        values = np.maximum(np.asarray(by_variant[variant]), PROBABILITY_FLOOR)
        jitter = np.linspace(-0.17, 0.17, len(values)) if len(values) > 1 else [0.0]
        ax.scatter(
            values,
            y + np.asarray(jitter),
            s=60,
            color=style.SERIES[0],
            zorder=3,
            edgecolors=style.SURFACE,
            linewidths=1.2,
        )
        zeros = int((np.asarray(by_variant[variant]) == 0).sum())
        note = f"median {_power_of_ten(float(np.median(by_variant[variant])))}"
        if zeros:
            note = f"{zeros} of {len(values)} seeds exactly 0"
        ax.text(5, y, note, va="center", ha="left", color=style.INK, fontsize=10)
    _probability_axis(ax)
    ax.set_xlim(PROBABILITY_FLOOR / 3, 3e6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([VARIANTS[v] for v in reversed(order)])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Exact probability of dying within a 5,000-step episode (log scale)")
    ax.set_title("What each ingredient contributes", pad=26)
    style.subtitle(
        ax, "One dot per training seed, computed by enumeration, not sampled. Lower is better."
    )
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def figure_sampled_vs_exact(
    verification: dict[tuple[str, int], dict],
    deaths: dict[tuple[str, int], tuple[int, int]],
    path: Path,
) -> dict:
    """Each agent: 95% interval from its 100 test episodes against its exact value."""
    import matplotlib.pyplot as plt

    from sota import plot_style as style

    style.apply_style()
    agents = []
    for key, report in verification.items():
        if key in deaths and report["death_probability"] is not None:
            d, n = deaths[key]
            low, high = stats.clopper_pearson(d, n)
            agents.append((report["death_probability"], d, n, low, high))
    agents.sort()
    covered = sum(low <= p <= high for p, _, _, low, high in agents)
    floor = 1e-7
    fig, ax = plt.subplots(figsize=(8.4, 0.2 * len(agents) + 2.2))
    for row, (p, _, _, low, high) in enumerate(agents):
        ax.plot(
            [max(low, floor), high],
            [row, row],
            color=style.CONTEXT,
            linewidth=2.0,
            solid_capstyle="butt",
            zorder=2,
        )
        inside = low <= p <= high
        ax.scatter(
            [max(p, floor)],
            [row],
            s=42,
            zorder=3,
            linewidths=1.2,
            color=style.SERIES[0] if inside else style.SERIES[1],
            edgecolors=style.SURFACE,
        )
    ax.scatter([], [], s=42, color=style.SERIES[0], label="Exact value, inside the interval")
    if covered < len(agents):
        ax.scatter([], [], s=42, color=style.SERIES[1], label="Exact value, outside the interval")
    ax.plot([], [], color=style.CONTEXT, linewidth=2.0, label="95% interval from 100 test episodes")
    ax.set_xscale("log")
    ax.set_xlim(floor / 2, 2)
    ax.set_xticks([floor, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_xticklabels(["≤10⁻⁷", "10⁻⁶", "10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "1"])
    ax.minorticks_off()
    ax.set_yticks([])
    ax.set_ylim(-1, len(agents) + 3.5)
    ax.set_xlabel("Probability of dying within a 5,000-step episode (log scale)")
    ax.set_ylabel(f"{len(agents)} trained agents, sorted by exact value")
    ax.set_title(
        f"The interval covers the exact value for {covered} of {len(agents)} agents", pad=26
    )
    style.subtitle(
        ax,
        "Zero deaths in 100 episodes still leaves an interval that reaches 3.6%.",
    )
    ax.legend(loc="upper left", ncols=1, handlelength=1.6)
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)
    return {"agents": len(agents), "intervals_covering_exact_value": covered}


def figure_survival(
    alive_every_10: list[float],
    episodes: list[tuple[int, bool]],
    path: Path,
    label: str,
) -> None:
    """Exact survival curve of one imperfect agent against the Kaplan-Meier estimate."""
    import matplotlib.pyplot as plt

    from sota import plot_style as style

    style.apply_style()
    exact = np.asarray(alive_every_10)
    steps = np.arange(len(exact)) * 10
    km = stats.kaplan_meier([s for s, _ in episodes], [d for _, d in episodes])
    horizon = steps[-1]
    t = np.concatenate([[0], km["time"], [horizon]])
    s = np.concatenate([[1.0], km["survival"], km["survival"][-1:] if len(km["time"]) else [1.0]])
    low = np.concatenate([[1.0], km["lower"], km["lower"][-1:] if len(km["time"]) else [1.0]])
    high = np.concatenate([[1.0], km["upper"], km["upper"][-1:] if len(km["time"]) else [1.0]])
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.fill_between(t, low, high, step="post", color=style.SERIES[1], alpha=0.14, lw=0)
    ax.step(
        t,
        s,
        where="post",
        color=style.SERIES[1],
        linewidth=2.0,
        label=f"Kaplan–Meier estimate from {len(episodes)} test episodes (95% band)",
    )
    ax.plot(steps, exact, color=style.SERIES[0], linewidth=2.0, label="Exact, by enumeration")
    bottom = min(float(exact.min()), float(low.min()))
    ax.set_ylim(max(0.0, bottom - 0.02), 1.004)
    ax.set_xlim(0, horizon)
    ax.set_xlabel("Step of the episode (the time limit is 5,000)")
    ax.set_ylabel("Probability of still being alive")
    ax.set_title("One imperfect agent: exact survival against the sampled estimate", pad=26)
    style.subtitle(ax, label)
    ax.legend(loc="lower right")
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def figure_anatomy(
    track: list[dict], pipes: list[dict], path: Path, failing_variant: str | None = None
) -> dict:
    """Two agents on the same pipes; one crashes. Drawn in screen pixels, up is up."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from sota import plot_style as style

    style.apply_style()
    series = {}
    for agent in ("reference", "failing"):
        rows = [r for r in track if r["agent"] == agent]
        series[agent] = (
            np.array([int(r["step"]) for r in rows]),
            np.array([float(r["bird_y"]) for r in rows]),
        )
    crash_step = int(series["failing"][0][-1])
    crash_y = float(series["failing"][1][-1])
    start, end = max(0, crash_step - 75), crash_step + 22
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    hit = None
    for pipe in pipes:
        first, last = int(pipe["first_step"]), int(pipe["last_step"])
        top, bottom = float(pipe["gap_top"]), float(pipe["gap_bottom"])
        if last < start or first > end:
            continue
        width = last - first + 1
        for y0, height in ((0, top), (bottom, 404 - bottom)):
            ax.add_patch(
                Rectangle((first - 0.5, y0), width, height, color=style.CONTEXT, alpha=0.55, lw=0)
            )
        if first <= crash_step <= last:
            hit = (top, bottom)
    colors = {"reference": style.SERIES[0], "failing": style.SERIES[1]}
    labels = {
        "reference": "Full method (champion)",
        "failing": VARIANTS.get(failing_variant, "Agent that crashes"),
    }
    for agent, (steps, y) in series.items():
        keep = (steps >= start) & (steps <= end)
        ax.fill_between(steps[keep], y[keep], y[keep] + 24, color=colors[agent], alpha=0.22, lw=0)
        ax.plot(steps[keep], y[keep] + 12, color=colors[agent], label=labels[agent])
    ax.scatter(
        [crash_step],
        [crash_y + 12],
        marker="X",
        s=130,
        color=style.SERIES[1],
        zorder=5,
        edgecolors=style.SURFACE,
        linewidths=1.5,
    )
    detail = {"crash_step": crash_step, "eval_seed": int(track[0]["eval_seed"])}
    if hit:
        short = hit[0] - crash_y if crash_y < hit[0] else crash_y + 24 - hit[1]
        where = "above the gap" if crash_y < hit[0] else "below the gap"
        detail.update(pixels_outside_gap=float(short), side=where)
        ax.annotate(
            f"Crash: {short:.0f} px {where}",
            xy=(crash_step, crash_y + 12),
            xytext=(crash_step - 30, crash_y - 52 if crash_y < hit[0] else crash_y + 70),
            color=style.INK,
            fontsize=10,
            arrowprops={"arrowstyle": "-", "color": style.INK_MUTED, "lw": 1.0},
        )
    ax.set_xlim(start, end)
    ax.set_ylim(404, 40)
    ax.set_xlabel("Step of the episode")
    ax.set_ylabel("Height on screen (pixels from the top)")
    ax.set_title("Anatomy of a failure: the same pipes, two agents", pad=26)
    style.subtitle(
        ax,
        "Grey: pipes while level with the bird. Bands: the bird's 24 px body. "
        f"Test seed {detail['eval_seed']}.",
    )
    ax.legend(loc="lower left")
    ax.grid(visible=False)
    fig.savefig(path)
    plt.close(fig)
    return detail


# --- report -----------------------------------------------------------------------------


def report() -> dict:
    figures = RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary: dict = {"max_pipes_in_5000_steps": MAX_PIPES}

    main_curves = RESULTS / "curves" / "main.csv"
    if main_curves.exists():
        curves = read_csv(main_curves)
        figure_learning_curve(curves, figures / "learning_curve.png")
        steps, survival = curve_matrix(curves, "full", "eval_survival")
        summary["training"] = {
            "training_seeds": int(survival.shape[0]),
            "steps": int(steps[-1]),
            "first_step_with_all_validation_episodes_at_ceiling": [
                int(steps[np.argmax(run >= 1.0)]) if (run >= 1.0).any() else None
                for run in survival
            ],
            "final_validation_survival_per_seed": survival[:, -1].tolist(),
        }

    verification: dict[tuple[str, int], dict] = {}
    for name in ("main", "ablation"):
        path = RESULTS / "verification" / f"{name}.json"
        if path.exists():
            verification.update(load_verification(path))
    if verification:
        table = defaultdict(dict)
        for (variant, seed), entry in sorted(verification.items()):
            table[variant][seed] = {k: v for k, v in entry.items() if k != "alive_every_10_steps"}
        summary["verification"] = table
        full = [entry for (variant, _), entry in verification.items() if variant == "full"]
        summary["verification_full_method"] = {
            "training_seeds": len(full),
            "seeds_verified_never_dying": sum(e["verified_never_dies"] for e in full),
            "death_probability_per_seed": [e["death_probability"] for e in full],
            "median_death_probability": float(np.median([e["death_probability"] for e in full])),
        }
        if any(variant != "full" for variant, _ in verification):
            figure_ablation(verification, figures / "ablation.png")
            reference = [e["death_probability"] for e in full]
            tests = {}
            for variant in VARIANTS:
                values = [
                    e["death_probability"] for (v, _), e in verification.items() if v == variant
                ]
                if variant == "full" or not values:
                    continue
                test = stats.mann_whitney_exact(reference, values)
                tests[variant] = {
                    "seeds": len(values),
                    "median_death_probability": float(np.median(values)),
                    "ratio_to_full_median": float(np.median(values) / np.median(reference)),
                    "probability_full_seed_is_safer": test["probability_x_smaller"],
                    "p_value_exact_mann_whitney": test["p_value"],
                }
            summary["ablation_tests"] = tests

    protocol = RESULTS / "protocol" / "pqn_results.csv"
    protocol_rows = read_csv(protocol) if protocol.exists() else []
    if protocol_rows:
        summary["protocol"] = protocol_summary([r for r in protocol_rows if r["config"] == "full"])
    extra = RESULTS / "protocol_ablation" / "pqn_results.csv"
    all_rows = protocol_rows + (read_csv(extra) if extra.exists() else [])
    if verification and all_rows:
        deaths = test_deaths(all_rows)
        summary["sampled_vs_exact"] = figure_sampled_vs_exact(
            verification, deaths, figures / "sampled_vs_exact.png"
        )
        # Survival figure: the agent with the most test deaths, where there is a curve to see.
        candidates = [
            (deaths[key][0], key)
            for key, entry in verification.items()
            if key in deaths
            and entry["alive_every_10_steps"]
            and deaths[key][0] >= 2
            and entry["death_probability"] < 0.5
        ]
        if candidates:
            _, key = max(candidates)
            episodes = [
                (int(r["episode_steps"]), is_true(r["terminated"]))
                for r in all_rows
                if r["split"] == "test" and (r["config"], int(r["train_seed"])) == key
            ]
            entry = verification[key]
            name = VARIANTS.get(key[0], key[0])
            label = (
                f"Variant '{name}', training seed {key[1]}: "
                f"exact death probability {entry['death_probability']:.3f}, "
                f"{deaths[key][0]} deaths in {deaths[key][1]} test episodes."
            )
            figure_survival(
                entry["alive_every_10_steps"], episodes, figures / "survival.png", label
            )
            summary["survival_figure_agent"] = {"variant": key[0], "train_seed": key[1]}

    anatomy = RESULTS / "anatomy"
    if (anatomy / "trajectory.csv").exists():
        meta = json.loads((anatomy / "meta.json").read_text(encoding="utf-8"))
        summary["anatomy"] = figure_anatomy(
            read_csv(anatomy / "trajectory.csv"),
            read_csv(anatomy / "pipes.csv"),
            figures / "anatomy.png",
            meta["failing_variant"],
        )
        summary["anatomy"].update(meta)

    for name in ("large_sample", "endurance"):
        path = RESULTS / name / f"{name}.csv"
        if path.exists():
            summary[name] = exposure_summary(read_csv(path))
    champion = RESULTS / "champion.json"
    if champion.exists():
        summary["champion"] = json.loads(champion.read_text(encoding="utf-8"))
        certificate = RESULTS / "verification" / "champion.json"
        if certificate.exists():
            report_ = json.loads(certificate.read_text(encoding="utf-8"))[0]
            exact = report_["exact"]
            per_pipe = exact["death_probability_per_pipe_late"]
            summary["champion"].update(
                reachable_states=report_["reachable_states"],
                fatal_transitions=report_["fatal_transitions"],
                verified_never_dies=report_["verified_never_dies"],
                exact_horizon_steps=exact["horizon_steps"],
                exact_death_probability=1.0 - exact["probability_alive_at_horizon"],
                exact_expected_pipes=exact["expected_pipes"],
                late_death_probability_per_pipe=per_pipe,
                smallest_decision_margin=report_["smallest_decision_margin"],
            )

    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8"
    )
    return summary


def choose_champion(destination: Path) -> dict:
    """Copy the main agent with the lowest exact death probability to ``destination``.

    The choice uses the exhaustive verification only. No sampled validation or test
    episode is involved, so the test split stays untouched by model selection.
    """
    reports = json.loads((RESULTS / "verification" / "main.json").read_text(encoding="utf-8"))
    best = min(
        reports,
        key=lambda r: (
            1.0 - r["exact"]["probability_alive_at_horizon"],
            r["exact"]["death_probability_per_pipe_late"] or 0.0,
            r["fatal_transitions"],
        ),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best["checkpoint"], destination)
    choice = {
        "checkpoint": best["checkpoint"],
        "train_seed": best["train_seed"],
        "death_probability": 1.0 - best["exact"]["probability_alive_at_horizon"],
        "selected_by": "lowest exact death probability within a 5,000-step episode",
        "candidates": len(reports),
    }
    (RESULTS / "champion.json").write_text(json.dumps(choice, indent=2) + "\n", encoding="utf-8")
    return choice


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect results and build the report")
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--group", required=True)
    collect_parser.add_argument("--runs", type=Path, required=True)
    champion_parser = sub.add_parser("champion")
    champion_parser.add_argument(
        "--destination", type=Path, default=Path("sota/models/pqn_champion.npz")
    )
    sub.add_parser("report")
    args = parser.parse_args(argv)
    if args.command == "collect":
        print(collect(args.group, args.runs))
    elif args.command == "champion":
        print(json.dumps(choose_champion(args.destination), indent=2))
    else:
        summary = report()
        summary.pop("verification", None)
        print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
