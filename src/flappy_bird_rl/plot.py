"""Plot measured random-baseline results without requiring a display server."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_results(csv_path: Path, output_dir: Path) -> list[Path]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Results CSV has no episodes")
    if {row["agent"] for row in rows} != {"Random"}:
        raise ValueError("This plot command expects Random results")
    if len({row["split"] for row in rows}) != 1:
        raise ValueError("Plot validation and test results separately")
    groups = {}
    for row in rows:
        groups.setdefault(float(row["flap_prob"]), []).append(int(row["pipes_passed"]))
    probs = sorted(groups, reverse=True)
    values = [np.array(groups[p]) for p in probs]
    labels = [f"p = {p:g}\nn = {len(groups[p])}" for p in probs]
    means = [float(v.mean()) for v in values]
    stds = [float(v.std(ddof=1)) if len(v) > 1 else 0.0 for v in values]
    split = rows[0]["split"]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / "mean_pipes.png", output_dir / "score_distribution.png"]
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        bars = ax.bar(labels, means, yerr=stds, capsize=5, color="#3b82a0", width=0.6)
        ax.bar_label(bars, fmt="%.3f", padding=5)
        ax.set(title=f"Random baseline | {split}", ylabel="Mean pipes passed")
        upper = max(m + s for m, s in zip(means, stds, strict=True))
        ax.set_ylim(bottom=0, top=max(0.2, upper * 1.35))
        ax.set_xlabel("Flap probability | error bars: sample standard deviation")
        ax.grid(axis="y", alpha=0.2)
        fig.savefig(paths[0], dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(
            1, len(probs), figsize=(10, 3.8), sharey=True, layout="constrained", squeeze=False
        )
        maximum = max(int(v.max()) for v in values)
        bins = np.arange(-0.5, maximum + 1.5, 1)
        for ax, p, v in zip(axes[0], probs, values, strict=True):
            ax.hist(v, bins=bins, color="#3b82a0", edgecolor="white", rwidth=0.8)
            ax.set(title=f"p = {p:g}", xlabel="Pipes passed", xticks=range(maximum + 1))
            ax.grid(axis="y", alpha=0.2)
        axes[0, 0].set_ylabel("Episode count")
        fig.suptitle(f"Episode score distribution | {split} | actual measured episodes")
        fig.savefig(paths[1], dpi=160)
        plt.close(fig)
    return paths
