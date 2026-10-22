# Contributing

Thanks for considering a contribution. This project is small on purpose, and
the bar for adding code is correspondingly explicit.

## Getting set up

```bash
git clone https://github.com/DU525/paired-counterfactual-eval.git
cd paired-counterfactual-eval
python -m pip install -e ".[dev]"
make test
```

## The workflow

1. Open an issue first for anything larger than a bug fix, so the design can be
   discussed before the code is written.
2. Branch from `main`.
3. Keep the change focused; unrelated refactors make review harder.
4. Run `make test lint` before pushing.
5. Open a pull request that explains **what changed and why**, and links the
   issue.

## Commit convention

[Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add a cost gate over a pre-registered lambda grid
fix: reject artifacts whose path escapes the transaction root
docs: explain the post-terminal convention
test: cover the crash-before-commit case
chore: bump ruff
```

## What a good pull request looks like

- **Tests.** New behaviour comes with a test. A pull request that changes the
  metric definitions must also update `docs/metrics.md` in the same commit.
- **No behaviour change without a note.** If a number changes, say so
  explicitly, including whether any previously reported result is invalidated.
- **Protocol changes are protocol changes.** Anything that affects what is
  measured, or how origins are sampled, or the statistical unit, needs a
  version bump of the spec format and a note in `CHANGELOG.md`.
- **No large files.** Data, weights and traces are not committed. Use a
  download script with checksums, or a GitHub release.
- **No secrets.** Run `python scripts/scan_repo.py .` before pushing.

## Code standards

- Python 3.9+, type hints on public functions, `ruff` for lint and format.
- The `src/pcf_eval/core/` package must not import a simulator, a deep
  learning framework or CUDA. That constraint is what keeps the protocol
  auditable and the tests fast; a change that breaks it needs a very good
  reason and a design note.
- The `src/pcf_eval/adapters/` protocols are the only intended extension
  point. Do not add simulator-specific code to the core.

## Reporting a security issue

Please do not open a public issue. See [SECURITY.md](SECURITY.md).

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).