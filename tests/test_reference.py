"""Drift detection against the committed reference results.

These tests are deliberately *not* a snapshot of every number. They assert the
two things a reader relies on: the verdict/mechanism of each bundled
configuration, and the fact that the decomposition identity is exact. If either
moves, either the code changed or the reference is stale, and both need an
explanation in the pull request.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcf_eval.aggregate import aggregate
from pcf_eval.protocol import evaluate_origin, load_backend, load_spec
from pcf_eval.toy import ToyBaseline, ToyCandidateModel, ToyConfig, ToyRingEnv

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference" / "toy_verdicts.json"


def _load_reference() -> dict:
    assert REFERENCE.is_file(), (
        "reference/toy_verdicts.json is missing; run `python scripts/make_reference.py`"
    )
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _run(config_name: str, resamples: int = 1000):
    path = ROOT / "configs" / config_name
    spec = load_spec(path)
    toy = dict(load_backend(path).get("toy", {}))
    cfg = ToyConfig(
        n_states=int(toy.get("n_states", 12)),
        n_actions=int(toy.get("n_actions", 4)),
        horizon=int(toy.get("horizon", 4)),
        n_goals=int(toy.get("n_goals", 1)),
        steps=tuple(toy.get("steps", (1, 2, -1, 3))),
        goal_offset=int(toy.get("goal_offset", 5)),
        model_noise=float(toy.get("model_noise", 0.0)),
        model_score_jitter=float(toy.get("model_score_jitter", 0.0)),
        seed=int(toy.get("seed", 0)),
    )
    baseline_action = int(toy.get("baseline_action", 0))
    env = ToyRingEnv(cfg)
    model = ToyCandidateModel(cfg, baseline_action)
    baseline = ToyBaseline(baseline_action)
    results = [evaluate_origin(env, model, baseline, spec, int(s)) for s in spec.seeds]
    return results, aggregate(results, resamples=resamples)


@pytest.mark.parametrize(
    "config_name",
    ["toy.yaml", "toy_inconclusive.yaml", "toy_ranking_limited.yaml"],
)
def test_reference_verdict_is_reproduced(config_name: str):
    reference = _load_reference()
    expected = reference["configs"][config_name]
    results, summary = _run(config_name)

    assert summary.verdict == expected["verdict"], (
        f"verdict drifted for {config_name}: {summary.verdict!r} != {expected['verdict']!r}"
    )
    assert summary.mechanism == expected["mechanism"]
    assert summary.n_origins == expected["n_origins"]
    assert summary.n_candidates == expected["n_candidates"]
    # The identity must be exact, not merely small.
    assert max(r.coverage.identity_residual for r in results) == 0.0


@pytest.mark.parametrize(
    "config_name",
    ["toy.yaml", "toy_inconclusive.yaml", "toy_ranking_limited.yaml"],
)
def test_reference_point_estimate_is_reproduced(config_name: str):
    """Point estimates are deterministic, so they must match exactly."""
    reference = _load_reference()
    expected = reference["configs"][config_name]
    _, summary = _run(config_name)

    assert summary.opportunity.mean == pytest.approx(expected["opportunity"]["mean"], abs=1e-6)
    assert summary.ranking_regret.mean == pytest.approx(
        expected["ranking_regret"]["mean"], abs=1e-6
    )
    assert summary.value_gain.mean == pytest.approx(expected["value_gain"]["mean"], abs=1e-6)
    assert summary.top1_rate.mean == pytest.approx(expected["top1_rate"], abs=1e-6)


def test_reference_covers_every_bundled_config():
    reference = _load_reference()
    committed = set(reference["configs"])
    on_disk = {p.name for p in (ROOT / "configs").glob("*.yaml")}
    assert on_disk <= committed, (
        f"configs without a reference entry: {sorted(on_disk - committed)}. "
        "Run `python scripts/make_reference.py` and commit the result."
    )
