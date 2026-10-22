"""Paired-counterfactual evaluation for world models.

The question this package answers:

    Given a frozen world model and a strong baseline policy, does the world
    model create *measurable* additional value -- and if it does not, is the
    bottleneck that no good candidate exists (coverage) or that the model
    fails to rank the good candidate first (ranking)?

Both are answered by executing the *same physical state* under several
candidate first actions and measuring what actually happens.
"""

from pcf_eval.metrics import (
    CoverageReport,
    RankReport,
    decompose,
    oracle_coverage,
    ranking_regret,
    value_gain,
)
from pcf_eval.protocol import (
    EvalSpec,
    OriginResult,
    evaluate_origin,
    load_spec,
)

__version__ = "0.1.0"

__all__ = [
    "CoverageReport",
    "RankReport",
    "EvalSpec",
    "OriginResult",
    "decompose",
    "oracle_coverage",
    "ranking_regret",
    "value_gain",
    "load_spec",
    "evaluate_origin",
]