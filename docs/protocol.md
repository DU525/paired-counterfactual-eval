# Protocol

## 1. Freeze before you look

Write the configuration file, hash it, and only then run anything. The spec
hash (`spec_sha256`) is stored next to the results, so a reader can check that
the reported numbers were produced under the declared conditions.

```yaml
spec:
  name: my-eval
  horizon: 40
  baseline_index: 0
  seeds: [ ... ]          # assigned in advance, never substituted
  lam_grid: [ ... ]       # pre-registered trade-off grid
backend:
  type: my-simulator
```

## 2. Sample origins independently

Each origin gets its own assigned seed. **Never** replace a seed whose state
turned out to be inconvenient, and never choose origins by looking at what
the world model predicted. Report the assigned / completed / failed counts
separately so a selection effect is visible.

## 3. Restore the same state for every candidate

`ForkableEnv.snapshot()` must capture the **complete** state: integrator
state, task RNG, controller history, episode counters. Restore must be exact.
If a branch can start from a different state than its siblings, the
comparison is not paired and none of the metrics below are meaningful.

Verify this once per environment with an A/A check: run the same candidate
twice and require bit-identical trajectories. If A/A fails, the run is
`BLOCKED` — it is not a negative result.

## 4. Charge the step budget before the step

The step meter records an **intent** before `env.step` is called and a
**completion** after it returns. If a process dies between the two, the intent
is unresolved and the run may not be resumed automatically: a human must
reconcile it. This is what keeps "we only ran 8,120 steps" true.

## 5. Execute every candidate

For each candidate, from the same restored state: apply the candidate as the
first action, then let the baseline policy act in closed loop for the
remaining `H - 1` steps. Record the per-step rewards, the terminal flag and
the number of steps survived.

## 6. Commit atomically

Artifacts are written temp → fsync → atomic-replace. `RUN_STATE.json`
becomes `COMPLETE` in one atomic write; that write is the commit point. Until
it happens, the bundle is `INCOMPLETE` and must be rejected by any analysis.

## 7. Recompute independently

Re-derive the reported numbers from the raw traces with a separate
implementation and compare. A pipeline that can only report what it already
computed cannot detect its own bugs.

## 8. Report the negative result

If the world model did not help, say so, and say which link broke. The three
verdicts (`value_gain_positive`, `value_gain_negative`, `inconclusive`) and
the three mechanisms (`aligned`, `coverage_limited`, `ranking_limited`) exist
so that a null result is still informative.

## 9. Publish the artefacts, not the dump

Commit the code, the frozen spec, the summary and the hashes. Do not commit
raw traces, model weights, datasets, logs or environment dumps — publish them
through a download script with checksums, or through a release.