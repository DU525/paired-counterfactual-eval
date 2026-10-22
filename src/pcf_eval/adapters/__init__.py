"""Adapter interfaces.

The evaluation protocol is simulator-agnostic. A user implements three small
protocols and the rest of the package -- metrics, bookkeeping, manifests --
works unchanged.

Why protocols instead of abstract base classes
----------------------------------------------
Adapters are written by people who think in terms of their own simulator
(``dm_control``, ``Isaac``, a proprietary stack). Structural typing keeps the
integration surface small and does not force a particular inheritance model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

__all__ = ["BaselinePolicy", "CandidateModel", "CostModel", "ForkableEnv", "StepResult"]


class StepResult(Protocol):
    """The subset of a simulator step result the protocol needs."""

    reward: float
    terminal: bool
    observation: Any


@runtime_checkable
class ForkableEnv(Protocol):
    """An environment whose *complete* state can be saved and restored.

    "Complete" means: integrator state, task RNG, controller history and any
    episode counters. Restoring must be bit-exact, otherwise a paired
    comparison across branches is not a causal comparison.
    """

    def snapshot(self) -> Any:
        """Capture a restorable handle for the current state."""

    def restore(self, handle: Any) -> None:
        """Restore a previously captured handle exactly."""

    def reset(self, seed: int) -> Any:
        """Reset to a deterministic start state and return its handle."""

    def step(self, action: Any) -> Any:
        """Advance one control step and return a result with reward/terminal."""


@runtime_checkable
class CandidateModel(Protocol):
    """A frozen world model (or any candidate proposer) over a fixed state."""

    def candidates(self, observation: Any) -> Sequence[Any]:
        """Return the candidate actions for this state, in a fixed order."""

    def score(self, observation: Any, actions: Sequence[Any]) -> list[float]:
        """Score candidates; higher is better."""

    def select(self, observation: Any) -> int:
        """Return the index of the selected candidate."""


@runtime_checkable
class BaselinePolicy(Protocol):
    """The strong reference policy whose first action is the control."""

    def act(self, observation: Any) -> Any:
        """Return the action for the current observation."""


@runtime_checkable
class CostModel(Protocol):
    """Measures the cost of using the world model at all."""

    def cost_of_scoring(self, n_candidates: int) -> float:
        """Cost of scoring ``n_candidates`` candidates, in a declared unit."""
