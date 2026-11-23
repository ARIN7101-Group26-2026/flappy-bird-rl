"""Statistics for the report: robust aggregates, bootstrap intervals, and survival analysis.

Everything is implemented with NumPy and the standard library so the analysis can be
re-run with the framework's base dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from itertools import combinations
from math import comb
from statistics import NormalDist

import numpy as np


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: the mean of the middle 50% (Agarwal et al., 2021)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0:
        raise ValueError("iqm of an empty sample")
    # Weighted trimmed mean so that n not divisible by 4 is handled exactly.
    low, high = 0.25 * n, 0.75 * n
    weights = np.clip(np.minimum(np.arange(1, n + 1), high) - np.maximum(np.arange(n), low), 0, 1)
    return float((x * weights).sum() / weights.sum())


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    level: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``statistic`` over independent units."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(x), size=(resamples, len(x)))
    stats = np.array([statistic(x[row]) for row in index])
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))


def hierarchical_bootstrap_ci(
    groups: Sequence[Sequence[float]],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    level: float = 0.95,
    resamples: int = 5_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Two-level bootstrap: resample training runs, then episodes within each run.

    Episodes from one trained model share that model's quirks, so treating all
    episodes as independent would understate the uncertainty about the *method*.
    ``statistic`` is applied to the per-run means of each resample.
    """
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(g, dtype=float) for g in groups]
    stats = np.empty(resamples)
    for b in range(resamples):
        chosen = rng.integers(0, len(arrays), size=len(arrays))
        means = [
            arrays[i][rng.integers(0, len(arrays[i]), size=len(arrays[i]))].mean() for i in chosen
        ]
        stats[b] = statistic(np.asarray(means))
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * log_p
        + (n - i) * log_q
        for i in range(k + 1)
    ]
    peak = max(terms)
    return min(1.0, math.exp(peak) * sum(math.exp(t - peak) for t in terms))


def _bisect(function: Callable[[float], float], low: float, high: float) -> float:
    """Root of a monotone ``function`` on [low, high]; geometric steps for tiny roots."""
    f_low = function(low)
    for _ in range(200):
        mid = math.sqrt(low * high) if low > 0 and high / low > 4 else 0.5 * (low + high)
        f_mid = function(mid)
        if (f_mid > 0) == (f_low > 0):
            low, f_low = mid, f_mid
        else:
            high = mid
        if high - low <= 1e-12 * max(high, 1e-300):
            break
    return 0.5 * (low + high)


def clopper_pearson(failures: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial interval for a failure probability."""
    if not 0 <= failures <= trials or trials <= 0:
        raise ValueError("need 0 <= failures <= trials and trials > 0")
    alpha = 1.0 - level
    tiny = 1e-300
    lower = 0.0
    if failures > 0:
        # Smallest p with P(X >= failures) = alpha/2.
        lower = _bisect(
            lambda p: (1.0 - _binomial_cdf(failures - 1, trials, p)) - alpha / 2, tiny, 1.0
        )
    upper = 1.0
    if failures < trials:
        # Largest p with P(X <= failures) = alpha/2.
        upper = _bisect(lambda p: alpha / 2 - _binomial_cdf(failures, trials, p), tiny, 1.0)
    return lower, upper


def zero_failure_upper_bound(trials: int, level: float = 0.95) -> float:
    """One-sided upper bound on a failure probability after ``trials`` clean trials.

    Solves (1 - p)^trials = 1 - level. For 95% this is the "rule of three": p < 3/trials.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    return 1.0 - (1.0 - level) ** (1.0 / trials)


def kaplan_meier(
    durations: Sequence[float], observed: Sequence[bool], level: float = 0.95
) -> dict[str, np.ndarray]:
    """Product-limit survival estimate with a log-log (Kalbfleisch-Prentice) band.

    ``observed[i]`` is True when episode ``i`` ended in a death at ``durations[i]`` and
    False when it was cut off there by the time limit (right-censored).
    """
    t = np.asarray(durations, dtype=float)
    d = np.asarray(observed, dtype=bool)
    times = np.unique(t[d])
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    survival, lower, upper = [], [], []
    s, greenwood = 1.0, 0.0
    for time in times:
        at_risk = int((t >= time).sum())
        deaths = int(((t == time) & d).sum())
        s *= 1.0 - deaths / at_risk
        if at_risk > deaths:
            greenwood += deaths / (at_risk * (at_risk - deaths))
        if 0.0 < s < 1.0 and greenwood > 0.0:
            spread = z * math.sqrt(greenwood) / abs(math.log(s))
            lower.append(s ** math.exp(spread))
            upper.append(s ** math.exp(-spread))
        else:
            lower.append(s)
            upper.append(s)
        survival.append(s)
    return {
        "time": times,
        "survival": np.asarray(survival),
        "lower": np.asarray(lower),
        "upper": np.asarray(upper),
    }


def mann_whitney_exact(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    """Exact two-sided Mann-Whitney test of "x and y come from the same distribution".

    Returns the probability that a value from ``x`` is smaller than one from ``y`` (ties
    count half) and the exact p-value, obtained by enumerating every way of choosing
    which pooled values belong to the smaller sample. No normal approximation is used,
    so the test is valid for the handful of training seeds available here.
    """
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    if comb(len(a) + len(b), len(small)) > 5_000_000:
        raise ValueError("samples too large for exact enumeration")
    pooled = np.concatenate([small, large])
    order = np.argsort(pooled, kind="mergesort")
    ranks = np.empty(len(pooled))
    ranks[order] = np.arange(1, len(pooled) + 1)
    for value in np.unique(pooled):  # midranks for ties
        tied = pooled == value
        ranks[tied] = ranks[tied].mean()
    k, total = len(small), len(pooled)
    centre = k * (total + 1) / 2.0
    observed = abs(ranks[:k].sum() - centre)
    sums = np.fromiter(
        (ranks[list(c)].sum() for c in combinations(range(total), k)),
        dtype=float,
        count=comb(total, k),
    )
    p_value = float((np.abs(sums - centre) >= observed - 1e-9).mean())
    wins = (a[:, None] < b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
    return {"probability_x_smaller": float(wins / (len(a) * len(b))), "p_value": p_value}
