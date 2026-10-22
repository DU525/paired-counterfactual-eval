## What this changes

<!-- One or two sentences. What is different after this pull request? -->

## Why

<!-- The problem this solves, or the question it answers. Link the issue. -->

Closes #

## How it was verified

- [ ] `make test` passes
- [ ] `make lint` passes
- [ ] the toy example still produces the expected verdict
      (`python examples/run_toy.py --config configs/toy.yaml --out outputs/x --assert-verdict`)
- [ ] if a metric definition changed, `docs/metrics.md` was updated in the same commit
- [ ] if a published number changed, the pull request says so explicitly
- [ ] `python scripts/scan_repo.py .` reports no findings
- [ ] no data, weights, traces or logs are committed

## Checklist

- [ ] the change is focused and does not mix unrelated refactors
- [ ] new behaviour has a test
- [ ] public functions have type hints
- [ ] `src/pcf_eval/core/` still imports no simulator, framework or CUDA