# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-22

### Added
- Paired-counterfactual evaluation core: `evaluate_origin`, metric
  decomposition (`opportunity`, `ranking_regret`, `value_gain`) with a
  numerically verified identity.
- Origin-cluster percentile bootstrap with a recorded seed, plus a
  three-way verdict (`value_gain_positive` / `value_gain_negative` /
  `inconclusive`) and a three-way mechanism classification (`aligned` /
  `coverage_limited` / `ranking_limited`).
- Crash-safe artefact bundles: atomic replace, per-artefact SHA256 manifest,
  single-write commit marker, offline verifier, injected-fault tests.
- Adapter protocols (`ForkableEnv`, `CandidateModel`, `BaselinePolicy`,
  `CostModel`) and a frozen-spec loader with a spec digest.
- Deterministic toy environment with an exact closed-form oracle, and three
  frozen configurations covering all three verdicts.
- `pcf-eval` CLI (`run`, `verify`).
- Repository hygiene tooling: a secret/oversized-file scanner with
  false-positive-resistant value classification, a layered `.gitignore`
  baseline, and a repository-layering reference.