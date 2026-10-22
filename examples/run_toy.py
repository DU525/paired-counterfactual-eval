#!/usr/bin/env python3
"""End-to-end example: run the toy evaluation and print the verdict.

    python examples/run_toy.py --out outputs/toy
    python examples/run_toy.py --out outputs/toy --assert-verdict

The example is deliberately tiny and deterministic so it can run in CI. It
demonstrates the full loop: fork the same state, execute every candidate,
decompose the outcome, aggregate over origins, and commit a hashed bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcf_eval.aggregate import aggregate  # noqa: E402
from pcf_eval.core.transaction import BundleTransaction, verify_committed_bundle  # noqa: E402
from pcf_eval.protocol import evaluate_origin, load_backend, load_spec, spec_digest  # noqa: E402
from pcf_eval.toy import ToyBaseline, ToyCandidateModel, ToyRingEnv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "configs" / "toy.yaml")
    ap.add_argument("--out", type=Path, default=Path("outputs/toy"))
    ap.add_argument("--assert-verdict", action="store_true",
                    help="fail unless the run reproduces the expected verdict")
    args = ap.parse_args()

    spec = load_spec(args.config)
    toy = dict(load_backend(args.config).get("toy", {}))

    from pcf_eval.toy import ToyConfig

    cfg = ToyConfig(
        n_states=int(toy.get("n_states", 12)),
        n_actions=int(toy.get("n_actions", 4)),
        horizon=int(toy.get("horizon", 8)),
        n_goals=int(toy.get("n_goals", 2)),
        steps=tuple(toy.get("steps", (1, 2, -1, 3))),
        goal_offset=int(toy.get("goal_offset", 0)),
        model_noise=float(toy.get("model_noise", 0.0)),
        model_score_jitter=float(toy.get("model_score_jitter", 0.0)),
        seed=int(toy.get("seed", 0)),
    )
    baseline_action = int(toy.get("baseline_action", 0))
    env = ToyRingEnv(cfg)
    model = ToyCandidateModel(cfg, baseline_action=baseline_action)
    baseline = ToyBaseline(baseline_action)

    results = [evaluate_origin(env, model, baseline, spec, int(s)) for s in spec.seeds]
    summary = aggregate(results, resamples=2000, seed=0)

    # The identity must hold exactly on every origin, not just in aggregate.
    max_residual = max(r.coverage.identity_residual for r in results)
    assert max_residual == 0.0, f"decomposition identity residual: {max_residual}"

    payload = {
        "spec": spec.to_dict(),
        "spec_sha256": spec_digest(spec),
        "summary": summary.to_dict(),
        "origins": [r.to_dict() for r in results],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    run_id = args.out.name
    tx = BundleTransaction(args.out, run_id)
    tx.write_bytes("results.json", json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"))
    tx.commit()
    verify_committed_bundle(tx.root)

    print(json.dumps(summary.to_dict(), indent=2))
    print(f"\nbundle committed and verified: {tx.root}")

    if args.assert_verdict:
        expected = {"toy-ring-aligned": "aligned", "toy-ring-ranking-limited": "ranking_limited"}
        want = expected.get(spec.name)
        if want is None:
            print(f"no expected mechanism registered for {spec.name!r}", file=sys.stderr)
            return 2
        if summary.mechanism != want:
            print(f"expected mechanism {want!r}, got {summary.mechanism!r}", file=sys.stderr)
            return 1
        print(f"mechanism assertion passed: {summary.mechanism}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())