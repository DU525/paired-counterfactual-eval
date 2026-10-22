# paired-counterfactual-eval

**Does a world model actually create value over a strong baseline — and if it does not, is the bottleneck the candidate set or the model's ranking?**

[![CI](https://github.com/DU525/paired-counterfactual-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/DU525/paired-counterfactual-eval/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![DOI](https://img.shields.io/badge/version-0.1.0-informational.svg)](CHANGELOG.md)

> A paired-counterfactual evaluation harness for world models.
> Instead of asking "what is the success rate", it asks *"given the same physical
> state, what happens if the world model picks a different first action?"* —
> and it answers with a causal, replayable, independently verifiable record.

---

## Table of Contents

- [News](#news)
- [The problem](#the-problem)
- [What it measures](#what-it-measures)
- [Quick start](#quick-start)
- [Requirements](#requirements)
- [Why the bookkeeping is strict](#why-the-bookkeeping-is-strict)
- [Repository layout](#repository-layout)
- [The bundled toy](#the-bundled-toy)
- [Reference results](#reference-results)
- [Known pitfalls](#known-pitfalls)
- [Reproducibility](#reproducibility)
- [Scope and non-goals](#scope-and-non-goals)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## News

- **2026-10-22** — `v0.1.0`: initial release. Paired-counterfactual protocol,
  metric decomposition with a numerically verified identity, origin-cluster
  bootstrap, crash-safe artefact bundles with an offline verifier, deterministic
  toy environment covering all three verdicts, and repository hygiene tooling.
  See [CHANGELOG.md](CHANGELOG.md).

---

## The problem

By 2026, embodied teams report success rates for world-model-based policies,
and the question has moved on:

> **Is the world model adding value, or is the policy succeeding on its own?**

Reporting a success rate cannot answer that. Three things routinely go
unnoticed:

1. **The candidate set may contain nothing better.** The model is blamed for a
   failure that was never its fault.
2. **The candidate set may contain something better, and the model may miss
   it.** Here the fix is ranking, not more training.
3. **Visual quality is not decision quality.** A video that looks right can
   still encode wrong action consequences, and pixel metrics such as FVD can
   hide physical and causal errors.

This package turns "does the world model help?" into a **paired counterfactual
experiment**: restore the *same* complete state, execute several candidate first
actions for real, and compare what actually happened.

---

## What it measures

For one origin `s` with candidate set `A = {a_0, ..., a_{K-1}}`, let `Y(a)` be
the measured value of executing candidate `a` as the **first** action from the
same restored state (the baseline policy takes over afterwards). Let `b` be the
baseline policy's own first action and `w` the candidate the world model
selects.

| quantity | definition | reading |
|---|---|---|
| **Opportunity** `A` | `max_i Y(a_i) - Y(a_b)` | a better action **existed** |
| **Ranking regret** `L` | `max_i Y(a_i) - Y(a_w)` | value lost to imperfect **ranking** |
| **Value gain** `G` | `Y(a_w) - Y(a_b)` | the world model's **net** contribution |

and identically

```
G  =  A  -  L
```

This identity is an algebraic fact, not a theorem and not a discovery. Its
use is **diagnostic**:

- `A > 0` and `L > A` → **ranking limited**. A better action existed and the
  model did not pick it. Fix the ranker, not the model size.
- `A <= 0` → **coverage limited**. Nothing beat the baseline. More training
  on this candidate generator cannot help.
- `L = 0` → **aligned**. The model captured the available opportunity.

Every number is accompanied by an interval computed by **bootstrap over
origins** (never over branches, checkpoints or steps), and the harness reports
`inconclusive` whenever the interval covers zero. The identity is verified
numerically on every origin; the residual is reported, not assumed.

Full definitions: [`docs/metrics.md`](docs/metrics.md).

---

## Quick start

```bash
git clone https://github.com/DU525/paired-counterfactual-eval.git
cd paired-counterfactual-eval
python -m pip install -e ".[dev]"

# run the bundled toy evaluation (seconds, no simulator required)
make example

# run the test suite
make test
```

Evaluate your own model by implementing three small protocols
(`ForkableEnv`, `CandidateModel`, `BaselinePolicy`) and calling
`evaluate_origin`:

```python
from pcf_eval import evaluate_origin, load_spec

spec = load_spec("configs/my_eval.yaml")
result = evaluate_origin(env=env, model=model, baseline=baseline, spec=spec, seed=0)

print(result.coverage.opportunity)        # was there a better action?
print(result.coverage.ranking_regret)     # did the model miss it?
print(result.coverage.value_gain)         # net effect
print(result.coverage.identity_residual)  # always 0.0
```

---

## Requirements

| | |
|---|---|
| Python | 3.9, 3.11, 3.12 (all three are exercised in CI) |
| OS | Linux, macOS, Windows. The core is pure Python; directory fsync is POSIX-only and degrades gracefully elsewhere |
| Runtime dependencies | `numpy`, `pyyaml` — nothing else |
| Optional extras | `[sim]` pulls `gymnasium` / `dm-control` when driving a real simulator; `[dev]` pulls `pytest`, `ruff`, `mypy` |
| GPU | **not required** for the protocol, the tests or the toy example |
| Simulator | **not required** for the tests or the toy example. A real evaluation needs one, through an adapter |

System support note: artifacts are written with write-temp → `fsync` →
atomic-replace everywhere, but POSIX **directory** `fsync` (which makes the
rename itself durable) is only available on POSIX. On Windows the file
contents are still fsynced; the durability guarantee for the rename is weaker.

---

## Why the bookkeeping is strict

A negative result is only useful if it cannot be an artefact of a sloppy
pipeline. This package makes the following guarantees **mechanically**, not by
convention:

| guarantee | mechanism |
|---|---|
| Branches really start from the same state | the adapter must expose `snapshot()` / `restore()`; an A/A check must produce bit-identical trajectories |
| A crashed run cannot look like a finished one | artifacts are written temp→fsync→atomic-replace; `RUN_STATE.json` becomes `COMPLETE` in a single atomic write, and that write is the only commit point |
| Results cannot be silently edited | every artifact's size and SHA256 are recorded in `MANIFEST.json` and re-verified on read |
| No step happens after the budget is spent | the step meter is called *before* `env.step` and charges the intent first; an unresolved intent blocks resume |
| The protocol is known before the result is seen | the spec is hashed (`spec_sha256`) and stored next to the results |
| The conclusion is reproducible | the harness re-derives the numbers from raw traces and compares them to the reported ones |

The same discipline applies to this repository itself: `scripts/scan_repo.py`
refuses secrets and oversized files, and the `.gitignore` is layered by reason
rather than appended by accident. See [`docs/protocol.md`](docs/protocol.md).

---

## Repository layout

```
paired-counterfactual-eval/
├── src/pcf_eval/
│   ├── core/          crash-safe artifacts, step metering (no simulator imports)
│   ├── metrics.py     opportunity / ranking regret / value gain, bootstrap, cost gate
│   ├── aggregate.py   origin-cluster statistics and verdict classification
│   ├── protocol.py    frozen spec, per-origin evaluation, post-terminal convention
│   ├── adapters/      the three protocols a user implements
│   ├── toy.py         deterministic qualification environment with an exact oracle
│   └── cli.py         pcf-eval run / verify
├── configs/           frozen evaluation specs (hashed into the results)
├── docs/              metric definitions, protocol, reproducibility notes
├── examples/          runnable end-to-end examples
├── reference/         committed reference results (small, for drift detection)
├── scripts/           data preparation and repository hygiene
└── tests/             unit tests; no simulator, no GPU, runs in seconds
```

**Layers.** Code lives in `src/`; configuration in `configs/`; explanation in
`docs/`; runnable examples in `examples/`; small reference numbers in
`reference/`; everything produced at runtime (`outputs/`, `results/`, `runs/`,
`data/`, `logs/`) is git-ignored and never committed. See
[`docs/protocol.md`](docs/protocol.md) for the full layering rules.

---

## The bundled toy

`pcf_eval.toy` is a deterministic ring environment with an exact closed-form
oracle. It exists so that **the protocol itself can be regression-tested
without a simulator**, and so that all three verdicts are reachable:

```bash
python examples/run_toy.py --config configs/toy.yaml --out outputs/toy
python examples/run_toy.py --config configs/toy_inconclusive.yaml --out outputs/inc
python examples/run_toy.py --config configs/toy_ranking_limited.yaml --out outputs/lim
```

| config | ranker behaviour | verdict | mechanism |
|---|---|---|---|
| `toy.yaml` | always picks the best candidate | `value_gain_positive` | `aligned` |
| `toy_inconclusive.yaml` | right about half the time | `inconclusive` | `aligned` |
| `toy_ranking_limited.yaml` | ignores the best 90% of the time | `value_gain_negative` | `ranking_limited` |

The toy is a **qualification environment**. It proves the arithmetic and the
bookkeeping. It is never evidence about real world models.

## Reference results

`reference/toy_verdicts.json` contains the numbers produced by the three
configs above, committed so that a metric regression is visible in a diff
rather than only in a failing test. `make reference-check` re-runs the toy and
compares.

This follows the convention of shipping **small** reference numbers in the
repository while keeping datasets, weights and raw traces out of it.

---

## Known pitfalls

Things that have actually gone wrong, or that will silently invalidate a run:

1. **Resampling over the wrong unit.** Branches, checkpoints and steps are
   repeated measurements inside one origin. Bootstrapping over them shrinks the
   interval for no reason. Resample over **origins**.
2. **An incomplete `snapshot()`.** If restore does not bring back the task RNG
   or the controller history, branches are not paired and nothing downstream
   means anything. Run an A/A check: same candidate twice, bit-identical
   trajectory, or the run is `BLOCKED`.
3. **Dividing by surviving steps.** With a per-survived-step denominator, early
   termination looks like *higher* value. This package divides by the full
   horizon and reports both denominators.
4. **Ties masquerading as ranking failures.** If candidate values are equal, a
   wrong pick costs nothing. Check `top1_hit` against `random_top1_rate`: with
   many ties the two can look identical while carrying no information.
5. **Reading a point estimate as a conclusion.** If the interval covers zero,
   the verdict is `inconclusive`. That is a result, not a failure to find one.
6. **Substituting a seed.** Replacing an origin whose state was inconvenient is
   a selection effect. Report assigned / completed / failed counts separately.
7. **A shared RNG stream in a test fixture.** It makes the verdict depend on how
   many origins were evaluated; the toy in this repository derives every random
   choice from a hash of the observed state for exactly this reason.

---

## Reproducibility

- The environment is pinned by `pyproject.toml`; the exact versions used in CI
  are recorded in the workflow.
- Every run writes a self-describing bundle: `results.json`, `summary.md`,
  `MANIFEST.json` (per-artifact SHA256) and `RUN_STATE.json` (commit marker).
- `pcf-eval verify --bundle <dir>` re-checks a bundle offline.
- Intervals are percentile bootstrap over origins with a fixed, recorded seed.

See [`docs/reproducibility.md`](docs/reproducibility.md).

---

## Scope and non-goals

This package **measures**; it does not train. It deliberately does not:

- compute a policy gradient or a world-model loss;
- assume that a world model is a latent state-space model (RSSM), a video
  generator, or anything in particular — the adapter interface is agnostic;
- claim that a null result generalises beyond the task, baseline, model and
  protocol it was measured under;
- replace the six-dimensional perceptual benchmarks used in the literature. It
  complements them by adding the **functional-utility** axis: whether the model
  changes the outcome, and why.

---

## Citation

If you use this package in your work, please cite it — GitHub reads
[`CITATION.cff`](CITATION.cff) directly ("Cite this repository"):

```bibtex
@software{du2026pcfeval,
  author  = {Du, Jinbo},
  title   = {paired-counterfactual-eval: causal evaluation of world models
             against strong baselines},
  year    = {2026},
  url     = {https://github.com/DU525/paired-counterfactual-eval},
  license = {MIT}
}
```

For a paper that uses the protocol, cite the method rather than this
repository; see [`docs/metrics.md`](docs/metrics.md) for the definitions to
quote.

---

## Contributing

Issues and pull requests are welcome — see
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the workflow, the commit convention
and the review criteria. By participating you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE) © 2026 Jinbo Du