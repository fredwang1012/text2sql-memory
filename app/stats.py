"""Paired comparison of two modes on the same questions: exact McNemar test and a bootstrap interval on the EX gap.

With a few dozen questions a gain of a few points is a handful of questions, so every reported gap comes with both.
"""
from math import comb

import numpy as np

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0  # fixed, so a report is reproducible from its per-question results


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs: b questions only A gets right, c only B."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_ci(a: list[bool], b: list[bool], level: float = 0.95, samples: int = BOOTSTRAP_SAMPLES,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile interval for mean(a) - mean(b), resampling questions (pairs) with replacement."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(samples, diff.size))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return float(lo), float(hi)


def compare(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Mode A against mode B over the questions both answered (matched by question_id)."""
    b_by_id = {r["question_id"]: r["correct"] for r in rows_b}
    pairs = [(r["correct"], b_by_id[r["question_id"]]) for r in rows_a if r["question_id"] in b_by_id]
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    only_a = sum(x and not y for x, y in pairs)
    only_b = sum(y and not x for x, y in pairs)
    lo, hi = bootstrap_ci(a, b)
    n = len(pairs)
    return {
        "n": n,
        "ex_diff": (sum(a) - sum(b)) / n if n else 0.0,
        "only_a": only_a,
        "only_b": only_b,
        "p_mcnemar": mcnemar_exact(only_a, only_b),
        "ci95": [lo, hi],
    }
