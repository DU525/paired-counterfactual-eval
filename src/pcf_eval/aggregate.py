"""Aggregation across origins, with origin-cluster bootstrap intervals.

Statistical unit
----------------
The independent unit is the **origin** (one independently sampled start
state). Candidate branches, model seeds and horizons are repeated measurements
inside an origin, not independent samples. Resampling must therefore happen
over origins; anything else inflates the effective sample size and shrinks
intervals for no reason.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass

__all__ = ["Aggregate", "BootstrapCI", "aggregate", "bootstrap_ci", "classify"]


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    low: float
    high: float
    n: int
    resamples: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Aggregate:
    """Summary over origins."""

    n_origins: int
    n_candidates: int
    opportunity: BootstrapCI
    ranking_regret: BootstrapCI
    value_gain: BootstrapCI
    top1_rate: BootstrapCI
    random_top1_rate: float
    mean_pairwise_concordance: float
    verdict: str
    mechanism: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def bootstrap_ci(
    values: Sequence[float],
    resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap over the supplied independent units."""
    data = [float(v) for v in values]
    if not data:
        raise ValueError("no values to bootstrap")
    if len(data) == 1:
        v = data[0]
        return BootstrapCI(mean=v, low=v, high=v, n=1, resamples=0)
    rng = random.Random(seed)
    n = len(data)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += data[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo_index = max(0, math.floor((alpha / 2.0) * resamples) - 1)
    hi_index = min(resamples - 1, math.ceil((1.0 - alpha / 2.0) * resamples) - 1)
    return BootstrapCI(
        mean=sum(data) / n,
        low=means[lo_index],
        high=means[hi_index],
        n=n,
        resamples=resamples,
    )


def classify(gain: BootstrapCI, opportunity: BootstrapCI, regret: BootstrapCI) -> tuple[str, str]:
    """Return ``(verdict, mechanism)``.

    verdict
        ``value_gain_positive``   -- the 95% interval of the net gain excludes 0 and is positive
        ``value_gain_negative``   -- the 95% interval excludes 0 and is negative
        ``inconclusive``          -- the interval covers 0

    mechanism
        ``coverage_limited``      -- no opportunity existed: even the best candidate does not beat the baseline
        ``ranking_limited``       -- opportunity existed but the model failed to rank it first
        ``aligned``               -- the model captured the available opportunity
    """
    if gain.low > 0.0:
        verdict = "value_gain_positive"
    elif gain.high < 0.0:
        verdict = "value_gain_negative"
    else:
        verdict = "inconclusive"

    if opportunity.mean <= 0.0:
        mechanism = "coverage_limited"
    elif regret.mean > opportunity.mean:
        mechanism = "ranking_limited"
    else:
        mechanism = "aligned"
    return verdict, mechanism


def aggregate(results: Sequence[object], resamples: int = 10_000, seed: int = 0) -> Aggregate:
    """Aggregate per-origin results into an :class:`Aggregate`."""
    if not results:
        raise ValueError("no results to aggregate")
    opportunities = [r.coverage.opportunity for r in results]  # type: ignore[attr-defined]
    regrets = [r.coverage.ranking_regret for r in results]  # type: ignore[attr-defined]
    gains = [r.coverage.value_gain for r in results]  # type: ignore[attr-defined]
    hits = [1.0 if r.coverage.top1_hit else 0.0 for r in results]  # type: ignore[attr-defined]
    concordances = [
        r.ranking.pairwise_concordance
        for r in results  # type: ignore[attr-defined]
        if r.ranking.pairwise_concordance == r.ranking.pairwise_concordance  # type: ignore[attr-defined]
    ]
    opportunity_ci = bootstrap_ci(opportunities, resamples=resamples, seed=seed)
    regret_ci = bootstrap_ci(regrets, resamples=resamples, seed=seed + 1)
    gain_ci = bootstrap_ci(gains, resamples=resamples, seed=seed + 2)
    hit_ci = bootstrap_ci(hits, resamples=resamples, seed=seed + 3)
    verdict, mechanism = classify(gain_ci, opportunity_ci, regret_ci)
    return Aggregate(
        n_origins=len(results),
        n_candidates=results[0].coverage.n_candidates,  # type: ignore[attr-defined]
        opportunity=opportunity_ci,
        ranking_regret=regret_ci,
        value_gain=gain_ci,
        top1_rate=hit_ci,
        random_top1_rate=results[0].coverage.random_top1_rate,  # type: ignore[attr-defined]
        mean_pairwise_concordance=(sum(concordances) / len(concordances))
        if concordances
        else float("nan"),
        verdict=verdict,
        mechanism=mechanism,
    )
