"""Command line interface.

    pcf-eval run --config configs/toy.yaml --out outputs/toy
    pcf-eval verify --bundle outputs/toy/<run_id>

The CLI never touches a simulator by itself; it drives whatever adapter the
selected backend provides. The ``toy`` backend is bundled so the protocol can
be smoke-tested anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from pcf_eval.aggregate import aggregate
from pcf_eval.core.transaction import BundleTransaction, verify_committed_bundle
from pcf_eval.protocol import EvalSpec, evaluate_origin, load_backend, load_spec, spec_digest
from pcf_eval.toy import ToyBaseline, ToyCandidateModel, ToyConfig, ToyRingEnv


def _build_toy(config: Dict[str, Any]) -> "tuple[ToyRingEnv, ToyCandidateModel, ToyBaseline]":
    toy_cfg = ToyConfig(
        n_states=int(config.get("n_states", 12)),
        n_actions=int(config.get("n_actions", 4)),
        horizon=int(config.get("horizon", 8)),
        n_goals=int(config.get("n_goals", 2)),
        steps=tuple(config.get("steps", (1, 2, -1, 3))),
        goal_offset=int(config.get("goal_offset", 0)),
        model_noise=float(config.get("model_noise", 0.0)),
        model_score_jitter=float(config.get("model_score_jitter", 0.0)),
        seed=int(config.get("seed", 0)),
    )
    baseline_action = int(config.get("baseline_action", 0))
    env = ToyRingEnv(toy_cfg)
    model = ToyCandidateModel(toy_cfg, baseline_action=baseline_action)
    baseline = ToyBaseline(baseline_action)
    return env, model, baseline


def _run(args: argparse.Namespace) -> int:
    spec = load_spec(args.config)
    backend = args.backend
    if backend == "toy":
        env, model, baseline = _build_toy(dict(load_backend(args.config).get("toy", {})))
    else:  # pragma: no cover - adapters are supplied by the user
        print(f"error: unknown backend {backend!r}", file=sys.stderr)
        return 2

    results: List[Any] = []
    for seed in spec.seeds:
        results.append(evaluate_origin(env, model, baseline, spec, int(seed)))

    summary = aggregate(results, resamples=args.resamples, seed=args.seed)
    payload = {
        "spec": spec.to_dict(),
        "spec_sha256": spec_digest(spec),
        "backend": backend,
        "summary": summary.to_dict(),
        "origins": [r.to_dict() for r in results],
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name
    tx = BundleTransaction(out_dir, run_id)
    tx.write_bytes("results.json", json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"))
    tx.write_bytes(
        "summary.md",
        _render_markdown(spec, summary).encode("utf-8"),
    )
    tx.commit()

    print(f"run_id        : {run_id}")
    print(f"origins      : {summary.n_origins}  candidates/origin: {summary.n_candidates}")
    print(f"opportunity  : {summary.opportunity.mean:+.6f}  "
          f"[{summary.opportunity.low:+.6f}, {summary.opportunity.high:+.6f}]")
    print(f"ranking regret:{summary.ranking_regret.mean:+.6f}  "
          f"[{summary.ranking_regret.low:+.6f}, {summary.ranking_regret.high:+.6f}]")
    print(f"value gain   : {summary.value_gain.mean:+.6f}  "
          f"[{summary.value_gain.low:+.6f}, {summary.value_gain.high:+.6f}]")
    print(f"top-1 rate   : {summary.top1_rate.mean:.3f} (random {summary.random_top1_rate:.3f})")
    print(f"verdict      : {summary.verdict}  mechanism: {summary.mechanism}")
    print(f"bundle       : {tx.root}")
    return 0


def _render_markdown(spec: EvalSpec, summary: Any) -> str:
    lines = [
        f"# {spec.name}",
        "",
        f"- spec sha256: `{spec_digest(spec)}`",
        f"- origins: {summary.n_origins}; candidates per origin: {summary.n_candidates}",
        f"- horizon: {spec.horizon}; baseline index: {spec.baseline_index}",
        "",
        "| metric | mean | 95% CI |",
        "|---|---:|---|",
        f"| opportunity | {summary.opportunity.mean:+.6f} | "
        f"[{summary.opportunity.low:+.6f}, {summary.opportunity.high:+.6f}] |",
        f"| ranking regret | {summary.ranking_regret.mean:+.6f} | "
        f"[{summary.ranking_regret.low:+.6f}, {summary.ranking_regret.high:+.6f}] |",
        f"| value gain | {summary.value_gain.mean:+.6f} | "
        f"[{summary.value_gain.low:+.6f}, {summary.value_gain.high:+.6f}] |",
        f"| top-1 rate | {summary.top1_rate.mean:.3f} | "
        f"[{summary.top1_rate.low:.3f}, {summary.top1_rate.high:.3f}] |",
        "",
        f"**verdict**: {summary.verdict} — **mechanism**: {summary.mechanism}",
        "",
        "> Intervals are percentile bootstrap over origins (origin-cluster resampling).",
        "> A negative value gain is not evidence that world models never help; it is evidence",
        "> about this task, this baseline and this model under the frozen protocol.",
        "",
    ]
    return "\n".join(lines)


def _verify(args: argparse.Namespace) -> int:
    manifest = verify_committed_bundle(args.bundle)
    print(f"bundle OK: {len(manifest['artifacts'])} artifacts verified")
    for item in manifest["artifacts"]:
        print(f"  {item['sha256'][:16]}  {item['bytes']:>10} B  {item['name']}")
    return 0


def main(argv: "List[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(prog="pcf-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an evaluation")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--backend", default="toy", choices=["toy"])
    run.add_argument("--resamples", type=int, default=10_000)
    run.add_argument("--seed", type=int, default=0)
    run.set_defaults(func=_run)

    verify = sub.add_parser("verify", help="verify a committed bundle")
    verify.add_argument("--bundle", required=True, type=Path)
    verify.set_defaults(func=_verify)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())