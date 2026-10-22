"""End-to-end protocol behaviour on the toy environment."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcf_eval.aggregate import aggregate, bootstrap_ci
from pcf_eval.protocol import evaluate_origin, load_backend, load_spec, spec_digest
from pcf_eval.toy import ToyBaseline, ToyCandidateModel, ToyConfig, ToyRingEnv

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _build(name: str):
    spec = load_spec(CONFIG_DIR / name)
    backend = load_backend(CONFIG_DIR / name)
    toy = dict(backend.get("toy", {}))
    cfg = ToyConfig(
        n_states=int(toy.get("n_states", 12)),
        n_actions=int(toy.get("n_actions", 4)),
        horizon=int(toy.get("horizon", 8)),
        n_goals=int(toy.get("n_goals", 2)),
        model_noise=float(toy.get("model_noise", 0.0)),
        model_score_jitter=float(toy.get("model_score_jitter", 0.0)),
    )
    env = ToyRingEnv(cfg)
    model = ToyCandidateModel(cfg, baseline_action=int(toy.get("baseline_action", 0)))
    baseline = ToyBaseline(int(toy.get("baseline_action", 0)))
    return spec, env, model, baseline


def test_spec_digest_is_stable():
    spec = load_spec(CONFIG_DIR / "toy.yaml")
    assert spec_digest(spec) == spec_digest(load_spec(CONFIG_DIR / "toy.yaml"))
    assert len(spec_digest(spec)) == 64


def test_aligned_config_resolves_to_aligned():
    spec, env, model, baseline = _build("toy.yaml")
    results = [evaluate_origin(env, model, baseline, spec, int(s)) for s in spec.seeds]
    summary = aggregate(results, resamples=2000)
    assert summary.mechanism == "aligned"
    assert summary.verdict == "value_gain_positive"
    assert summary.value_gain.mean > 0.0
    assert summary.opportunity.mean > 0.0
    assert summary.ranking_regret.mean == pytest.approx(0.0)


def test_noisy_config_resolves_to_ranking_limited():
    spec, env, model, baseline = _build("toy_ranking_limited.yaml")
    results = [evaluate_origin(env, model, baseline, spec, int(s)) for s in spec.seeds]
    summary = aggregate(results, resamples=2000)
    assert summary.mechanism == "ranking_limited"
    assert summary.verdict == "value_gain_negative"
    assert summary.opportunity.mean > 0.0
    assert summary.ranking_regret.mean > summary.opportunity.mean


def test_branches_are_paired():
    """Two candidates with the same effect must yield identical values, which
    only holds if every branch starts from the same restored state."""
    spec, env, model, baseline = _build("toy.yaml")
    result = evaluate_origin(env, model, baseline, spec, 3)
    assert result.coverage.identity_residual == 0.0
    assert len(result.values) == result.n_candidates
    assert all(v >= 0.0 for v in result.values)


def test_bootstrap_is_deterministic():
    a = bootstrap_ci([0.1, 0.2, 0.3, 0.4], resamples=200, seed=5)
    b = bootstrap_ci([0.1, 0.2, 0.3, 0.4], resamples=200, seed=5)
    assert a.low == b.low and a.high == b.high


def test_bootstrap_resamples_over_units_not_branches():
    # A single unit must not produce a spurious interval.
    only = bootstrap_ci([0.5], resamples=100)
    assert only.low == only.high == 0.5


def test_inconclusive_config_is_inconclusive():
    spec, env, model, baseline = _build("toy_inconclusive.yaml")
    results = [evaluate_origin(env, model, baseline, spec, int(s)) for s in spec.seeds]
    summary = aggregate(results, resamples=500)
    assert summary.verdict == "inconclusive"
    assert summary.value_gain.low <= 0.0 <= summary.value_gain.high
