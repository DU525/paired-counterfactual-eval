"""Evaluation protocol: what to run, and how to record it.

A run is defined by a frozen :class:`EvalSpec`. The spec is hashed and stored
with the results, so a reader can check that the reported numbers were produced
under the declared conditions.

Post-terminal convention
------------------------
If a branch terminates early, the remaining steps of the horizon contribute
**zero** reward, and the reported value is ``sum(rewards) / horizon`` -- not
``sum(rewards) / steps_survived``. Dividing by the surviving steps would make
early termination look like *higher* per-step value. Both denominators are
reported so the choice is visible.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pcf_eval.metrics import CoverageReport, RankReport, decompose, rank_report

__all__ = [
    "EvalSpec",
    "OriginResult",
    "evaluate_origin",
    "load_backend",
    "load_spec",
    "spec_digest",
]


@dataclass(frozen=True)
class EvalSpec:
    """Frozen description of one evaluation."""

    name: str
    horizon: int = 40
    baseline_index: int = 0
    seeds: Sequence[int] = field(default_factory=tuple)
    lam_grid: Sequence[float] = field(default_factory=tuple)
    extra_cost: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2")
        if self.baseline_index < 0:
            raise ValueError("baseline_index must be non-negative")
        if not self.seeds:
            raise ValueError("at least one seed is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return spec_digest(self)


def spec_digest(spec: EvalSpec) -> str:
    """SHA256 over the canonical JSON form of the spec."""
    payload = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class OriginResult:
    """Result for one origin (one independently sampled start state)."""

    seed: int
    n_candidates: int
    values: list[float]
    steps_survived: list[int]
    selected_index: int
    baseline_index: int
    predicted_scores: list[float] | None = None
    coverage: CoverageReport = None  # type: ignore[assignment]
    ranking: RankReport = None  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "n_candidates": self.n_candidates,
            "values": self.values,
            "steps_survived": self.steps_survived,
            "selected_index": self.selected_index,
            "baseline_index": self.baseline_index,
            "predicted_scores": self.predicted_scores,
            "coverage": self.coverage.to_dict() if self.coverage else None,
            "ranking": self.ranking.to_dict() if self.ranking else None,
        }


def evaluate_origin(
    env: Any,
    model: Any,
    baseline: Any,
    spec: EvalSpec,
    seed: int,
) -> OriginResult:
    """Run every candidate from the *same* restored state and measure the value.

    The returned values are per-candidate averages over ``spec.horizon`` control
    steps. Steps 1..H-1 always follow the baseline policy in closed loop, so
    the only thing that differs between branches is the first action.
    """
    handle = env.reset(seed)
    observation = env.observe()

    candidates = list(model.candidates(observation))
    if len(candidates) < 2:
        raise ValueError("need at least two candidates")
    selected_index = int(model.select(observation))
    if not 0 <= selected_index < len(candidates):
        raise IndexError("model selected an out-of-range candidate")
    if spec.baseline_index >= len(candidates):
        raise IndexError("baseline_index outside the candidate set")

    values: list[float] = []
    survived: list[int] = []

    for action in candidates:
        env.restore(handle)
        total = 0.0
        steps = 0
        result = env.step(action)
        total += float(result.reward)
        steps += 1
        for _ in range(spec.horizon - 1):
            if getattr(result, "terminal", False):
                break
            act = baseline.act(result.observation)
            result = env.step(act)
            total += float(result.reward)
            steps += 1
        values.append(total / float(spec.horizon))
        survived.append(steps)

    coverage = decompose(values, selected_index, spec.baseline_index)
    scores = None
    try:
        scores = [float(s) for s in model.score(observation, candidates)]
    except Exception:
        scores = None
    ranking = rank_report(values, selected_index, scores)

    return OriginResult(
        seed=seed,
        n_candidates=len(candidates),
        values=values,
        steps_survived=survived,
        selected_index=selected_index,
        baseline_index=spec.baseline_index,
        predicted_scores=scores,
        coverage=coverage,
        ranking=ranking,
    )


def load_spec(path: str | Path) -> EvalSpec:
    """Load a frozen spec from YAML or JSON.

    The file may contain two top-level blocks:

    ``spec``    the frozen evaluation spec (hashed into the results)
    ``backend`` how to realise the environment (not part of the spec digest)

    A file with no ``spec`` block is treated as a bare spec, which keeps
    single-purpose files simple.
    """
    import yaml

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if path.suffix.lower() in (".yaml", ".yml") else json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("spec file must contain a mapping")
    if "spec" in data:
        data = data["spec"]
        if not isinstance(data, dict):
            raise ValueError("the `spec` block must be a mapping")
    allowed = {f for f in EvalSpec.__dataclass_fields__}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown spec fields: {sorted(unknown)}")
    if "seeds" in data:
        data["seeds"] = list(data["seeds"])
    if "lam_grid" in data:
        data["lam_grid"] = list(data["lam_grid"])
    return EvalSpec(**data)


def load_backend(path: str | Path) -> dict[str, object]:
    """Load the backend block of a configuration file."""
    import yaml

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if path.suffix.lower() in (".yaml", ".yml") else json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("configuration file must contain a mapping")
    backend = data.get("backend", {})
    if not isinstance(backend, dict):
        raise ValueError("the `backend` block must be a mapping")
    return backend
