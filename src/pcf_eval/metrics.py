"""Metrics for paired-counterfactual evaluation.

Notation
--------
For one origin ``s``, let the candidate set be ``A = {a_0, ..., a_{K-1}}`` and
let ``Y(a)`` be the measured value of executing candidate ``a`` as the *first*
action from the same restored state, with the frozen baseline policy taking
over afterwards. Let ``b`` be the index of the baseline policy's own first
action and ``w`` the index the world model selects.

    opportunity  A = max_i Y(a_i) - Y(a_b)
    regret       L = max_i Y(a_i) - Y(a_w)
    value gain   G = Y(a_w)     - Y(a_b)

Then, identically,

    G = A - L                                                   (*)

``(*)`` is an algebraic identity, not a theorem and not a discovery. Its
practical use is diagnostic: a positive ``A`` with a positive ``L`` means the
candidate set *does* contain a better action while the world model fails to
rank it first. A non-positive ``A`` means there was nothing to gain in the
first place, and no amount of re-ranking can help.

Sign convention: all quantities are "higher is better for the world model".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

__all__ = [
    "CoverageReport",
    "RankReport",
    "CostGate",
    "decompose",
    "oracle_coverage",
    "ranking_regret",
    "value_gain",
    "rank_report",
    "cost_gate",
]


@dataclass(frozen=True)
class CoverageReport:
    """Per-origin decomposition of the world model's contribution."""

    n_candidates: int
    baseline_index: int
    selected_index: int
    best_index: int
    baseline_value: float
    selected_value: float
    best_value: float
    opportunity: float
    ranking_regret: float
    value_gain: float
    identity_residual: float
    top1_hit: bool
    random_top1_rate: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RankReport:
    """How well the world model ranks candidates, per origin."""

    n_candidates: int
    top1_hit: bool
    random_top1_rate: float
    pairwise_concordance: float
    best_index: int
    selected_index: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CostGate:
    """Value of extra compute, in units of the task metric per unit of cost.

    ``net = value_gain - lam * extra_cost``

    ``lam`` is a *declared* trade-off, not something to be tuned after seeing
    results. Report a Pareto front over a fixed ``lam`` grid instead of a
    single hand-picked point.
    """

    value_gain: float
    extra_cost: float
    lam: float
    net: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _check(values: Sequence[float]) -> List[float]:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        raise ValueError("need at least two candidates")
    for v in vals:
        if not math.isfinite(v):
            raise ValueError("candidate values must be finite")
    return vals


def oracle_coverage(values: Sequence[float]) -> float:
    """``max_i Y(a_i)`` -- the best achievable value in the candidate set."""
    return max(_check(values))


def ranking_regret(values: Sequence[float], selected_index: int) -> float:
    """``max_i Y(a_i) - Y(a_w)`` -- value lost to imperfect ranking."""
    vals = _check(values)
    if not 0 <= selected_index < len(vals):
        raise IndexError("selected_index out of range")
    return max(vals) - vals[selected_index]


def value_gain(values: Sequence[float], selected_index: int, baseline_index: int) -> float:
    """``Y(a_w) - Y(a_b)`` -- net gain of the world model over the baseline."""
    vals = _check(values)
    if not 0 <= selected_index < len(vals):
        raise IndexError("selected_index out of range")
    if not 0 <= baseline_index < len(vals):
        raise IndexError("baseline_index out of range")
    return vals[selected_index] - vals[baseline_index]


def decompose(
    values: Sequence[float],
    selected_index: int,
    baseline_index: int,
    tolerance: float = 1e-9,
) -> CoverageReport:
    """Split the world model's net effect into opportunity and ranking regret.

    The identity ``G == A - L`` is verified numerically and the residual is
    returned, so a caller never has to take the identity on faith.
    """
    vals = _check(values)
    k = len(vals)
    if not 0 <= selected_index < k:
        raise IndexError("selected_index out of range")
    if not 0 <= baseline_index < k:
        raise IndexError("baseline_index out of range")

    best_value = max(vals)
    best_index = vals.index(best_value)
    opportunity = best_value - vals[baseline_index]
    regret = best_value - vals[selected_index]
    gain = vals[selected_index] - vals[baseline_index]
    residual = abs(gain - (opportunity - regret))
    if residual > tolerance:
        raise ArithmeticError(
            f"decomposition identity violated: residual={residual:.3e} > {tolerance:.3e}"
        )
    return CoverageReport(
        n_candidates=k,
        baseline_index=baseline_index,
        selected_index=selected_index,
        best_index=best_index,
        baseline_value=vals[baseline_index],
        selected_value=vals[selected_index],
        best_value=best_value,
        opportunity=opportunity,
        ranking_regret=regret,
        value_gain=gain,
        identity_residual=residual,
        top1_hit=selected_index == best_index,
        random_top1_rate=1.0 / k,
    )


def rank_report(
    values: Sequence[float],
    selected_index: int,
    predicted_scores: Sequence[float] | None = None,
) -> RankReport:
    """Summarise ranking quality for one origin.

    ``pairwise_concordance`` is the fraction of candidate pairs whose measured
    ordering agrees with the model's score ordering. With ``K`` candidates it
    is defined over the ``K(K-1)/2`` unordered pairs and is 0.5 for chance.
    """
    vals = _check(values)
    k = len(vals)
    if not 0 <= selected_index < k:
        raise IndexError("selected_index out of range")
    best_index = vals.index(max(vals))

    concordance = float("nan")
    if predicted_scores is not None:
        scores = [float(s) for s in predicted_scores]
        if len(scores) != k:
            raise ValueError("predicted_scores length must match values")
        agree = 0
        total = 0
        for i in range(k):
            for j in range(i + 1, k):
                if vals[i] == vals[j] or scores[i] == scores[j]:
                    continue
                total += 1
                if (vals[i] > vals[j]) == (scores[i] > scores[j]):
                    agree += 1
        if total:
            concordance = agree / total

    return RankReport(
        n_candidates=k,
        top1_hit=selected_index == best_index,
        random_top1_rate=1.0 / k,
        pairwise_concordance=concordance,
        best_index=best_index,
        selected_index=selected_index,
    )


def cost_gate(value_gain_value: float, extra_cost: float, lam: float) -> CostGate:
    """Apply a *declared* compute/utility trade-off."""
    if lam < 0:
        raise ValueError("lam must be non-negative")
    if extra_cost < 0:
        raise ValueError("extra_cost must be non-negative")
    return CostGate(
        value_gain=value_gain_value,
        extra_cost=extra_cost,
        lam=lam,
        net=value_gain_value - lam * extra_cost,
    )