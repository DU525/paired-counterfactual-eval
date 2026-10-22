"""A deterministic toy environment with an exact oracle.

Purpose
-------
The CI pipeline must be able to *verify the evaluation protocol itself* in
milliseconds, without a simulator. This module provides:

* a state whose complete contents can be snapshotted and restored exactly
  (position, step counter, RNG state);
* a deterministic, action-dependent transition with a known closed form;
* a baseline policy and a candidate model whose bias can be dialled, so both
  the "coverage-limited" and "ranking-limited" regimes are reachable.

The toy is a *qualification* environment: it proves the arithmetic and the
bookkeeping, it is never evidence about real world models.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

__all__ = ["ToyRingEnv", "ToyBaseline", "ToyCandidateModel", "ToyConfig"]


@dataclass(frozen=True)
class ToyConfig:
    """A single goal reached by stepping onto one ring position.

    The defaults are chosen so that

    * the best candidate **differs between origins** (so "coverage" varies),
    * candidate values are **not tied** (so a ranking error actually costs
      something -- a wide goal window with a long horizon would make every
      candidate worth the same and hide ranking failures entirely),
    * the baseline action is sometimes already optimal (so "coverage limited"
      origins exist alongside "ranking limited" ones).
    """

    n_states: int = 12
    n_actions: int = 4
    horizon: int = 4
    n_goals: int = 1
    steps: Tuple[int, ...] = (1, 2, -1, 3)
    goal_offset: int = 5
    model_noise: float = 0.0
    model_score_jitter: float = 0.0
    seed: int = 0


class ToyRingEnv:
    """Ring of positions; an action shifts the position and earns reward on goals.

    State contents: ``position``, ``t`` (step counter) and the RNG state.
    """

    def __init__(self, config: ToyConfig) -> None:
        self.config = config
        self._position = 0
        self._t = 0
        self._rng = random.Random(config.seed)
        self._goals = {
            (config.goal_offset + k) % config.n_states for k in range(config.n_goals)
        }

    # -- state ------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "position": self._position,
            "t": self._t,
            "rng": self._rng.getstate(),
        }

    def restore(self, handle: Dict[str, Any]) -> None:
        self._position = handle["position"]
        self._t = handle["t"]
        self._rng = random.Random()
        self._rng.setstate(handle["rng"])

    def reset(self, seed: int) -> Dict[str, Any]:
        self._rng = random.Random(seed)
        self._position = self._rng.randrange(self.config.n_states)
        self._t = 0
        return self.snapshot()

    def observe(self) -> Dict[str, int]:
        return {"position": self._position, "t": self._t}

    # -- dynamics ---------------------------------------------------------
    def step(self, action: int) -> Any:
        if not 0 <= action < self.config.n_actions:
            raise ValueError("action out of range")
        self._position = (self._position + self.config.steps[action]) % self.config.n_states
        self._t += 1
        reward = 1.0 if self._position in self._goals else 0.0
        terminal = self._t >= self.config.horizon
        return _ToyStep(reward=reward, terminal=terminal, observation=self.observe())


@dataclass(frozen=True)
class _ToyStep:
    reward: float
    terminal: bool
    observation: Dict[str, int]


class ToyBaseline:
    """The reference policy: always take the same action."""

    def __init__(self, action: int = 0) -> None:
        self.action = action

    def act(self, observation: Dict[str, int]) -> int:
        return self.action


class ToyCandidateModel:
    """A candidate model with a controllable, *deterministic* ranking bias.

    ``model_noise`` is the probability that the model ignores the true best
    candidate. ``model_score_jitter`` perturbs the reported scores, which is
    what a real world model's imperfect value estimates look like.

    The randomness is a **pure function of the observed state** (via a
    SHA-256 derived hash), not a shared RNG stream. That matters: with a
    shared stream the classification of a run would depend on how many
    origins were evaluated, and the toy could not be used as a protocol
    regression test.
    """

    def __init__(self, config: ToyConfig, baseline_action: int = 0) -> None:
        self.config = config
        self.baseline_action = baseline_action

    def candidates(self, observation: Dict[str, int]) -> List[int]:
        return list(range(self.config.n_actions))

    def _hash_unit(self, observation: Dict[str, int], salt: str) -> float:
        key = f"{self.config.seed}:{salt}:{observation['position']}:{observation['t']}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) / float(1 << 64)

    def _true_values(self, observation: Dict[str, int]) -> List[float]:
        """Closed-form value of starting with each action (goal hits in H steps)."""
        position = observation["position"]
        goals = {
            (self.config.goal_offset + k) % self.config.n_states
            for k in range(self.config.n_goals)
        }
        values = []
        for action in range(self.config.n_actions):
            pos = position
            hits = 0
            for step in range(self.config.horizon):
                if step == 0:
                    pos = (pos + self.config.steps[action]) % self.config.n_states
                else:
                    pos = (pos + self.config.steps[self.baseline_action]) % self.config.n_states
                if pos in goals:
                    hits += 1
            values.append(hits / self.config.horizon)
        return values

    def score(self, observation: Dict[str, int], actions: Sequence[int]) -> List[float]:
        values = self._true_values(observation)
        jitter = self.config.model_score_jitter
        return [
            value + jitter * (2.0 * self._hash_unit(observation, f"score{i}") - 1.0)
            for i, value in enumerate(values)
        ]

    def select(self, observation: Dict[str, int]) -> int:
        values = self._true_values(observation)
        if self.config.model_noise > 0.0 and self._hash_unit(observation, "gate") < self.config.model_noise:
            choice = int(self._hash_unit(observation, "choice") * len(values)) % len(values)
            return choice
        return max(range(len(values)), key=lambda i: values[i])