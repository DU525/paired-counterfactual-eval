"""The decomposition identity must hold exactly, and the signs must be sane."""

from __future__ import annotations

import math

import pytest

from pcf_eval.metrics import (
    cost_gate,
    decompose,
    oracle_coverage,
    rank_report,
    ranking_regret,
    value_gain,
)


def test_identity_holds_on_random_data():
    import random

    rng = random.Random(7)
    for _ in range(2000):
        k = rng.randrange(2, 8)
        values = [rng.uniform(-1.0, 1.0) for _ in range(k)]
        w = rng.randrange(k)
        b = rng.randrange(k)
        report = decompose(values, w, b)
        assert report.identity_residual == 0.0
        assert math.isclose(
            report.value_gain,
            report.opportunity - report.ranking_regret,
            rel_tol=0.0,
            abs_tol=1e-12,
        )


def test_coverage_positive_ranking_limited_is_detected():
    # A better candidate exists, but the model picks one worse than the baseline.
    values = [0.40, 0.50, 0.20, 0.10]
    report = decompose(values, selected_index=2, baseline_index=0)
    assert report.opportunity > 0.0
    assert report.ranking_regret > report.opportunity
    assert report.value_gain < 0.0
    assert report.top1_hit is False


def test_aligned_case():
    values = [0.10, 0.50, 0.20, 0.20]
    report = decompose(values, selected_index=1, baseline_index=0)
    assert report.opportunity == pytest.approx(0.40)
    assert report.ranking_regret == pytest.approx(0.0)
    assert report.value_gain == pytest.approx(0.40)
    assert report.top1_hit is True
    assert report.random_top1_rate == pytest.approx(0.25)


def test_coverage_limited_case():
    # no candidate beats the baseline
    values = [0.50, 0.40, 0.30, 0.20]
    report = decompose(values, selected_index=3, baseline_index=0)
    assert report.opportunity == pytest.approx(0.0)
    assert report.value_gain < 0.0


def test_helpers():
    values = [0.1, 0.9, 0.3]
    assert oracle_coverage(values) == pytest.approx(0.9)
    assert ranking_regret(values, 0) == pytest.approx(0.8)
    assert value_gain(values, selected_index=0, baseline_index=0) == pytest.approx(0.0)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        decompose([1.0], 0, 0)
    with pytest.raises(IndexError):
        decompose([0.1, 0.2], 5, 0)
    with pytest.raises(ValueError):
        decompose([float("nan"), 0.2], 0, 1)


def test_rank_report_pairwise():
    report = rank_report([0.1, 0.9, 0.5], 1, predicted_scores=[0.2, 0.8, 0.4])
    assert report.top1_hit is True
    assert report.pairwise_concordance == pytest.approx(1.0)

    anti = rank_report([0.1, 0.9, 0.5], 1, predicted_scores=[0.8, 0.2, 0.4])
    assert anti.pairwise_concordance == pytest.approx(0.0)


def test_cost_gate():
    gate = cost_gate(value_gain_value=0.01, extra_cost=2.0, lam=0.001)
    assert gate.net == pytest.approx(0.008)
    with pytest.raises(ValueError):
        cost_gate(0.01, 2.0, lam=-1.0)
