# Reproducibility notes

## What is recorded

| artefact | content |
|---|---|
| `results.json` | the frozen spec, its SHA256, the per-origin records and the summary |
| `summary.md` | the human-readable table with intervals and the verdict |
| `MANIFEST.json` | size and SHA256 of every artefact |
| `RUN_STATE.json` | the commit marker; `COMPLETE` means the bundle is admissible |

## Verifying a bundle

```bash
pcf-eval verify --bundle outputs/<run_id>/<run_id>
```

The verifier re-reads every artefact, re-checks its digest against the
manifest, and rejects the bundle if the manifest itself does not match the
digest recorded in the commit marker.

## Determinism

- The bootstrap uses a fixed, recorded seed.
- The toy environment derives all of its randomness from a SHA-256 hash of the
  observed state, so its behaviour does **not** depend on how many origins
  were evaluated. (An earlier version used a shared RNG stream, which made
  the mechanism classification drift with the number of origins; that is
  exactly the kind of artefact a protocol test must not have.)
- Real simulators are not deterministic in general. Pin the simulator version,
  record the seed, and treat an A/A mismatch as a blocked run rather than
  noise.

## Environment

```bash
python -m pip install -e ".[dev]"
python -m pytest          # unit tests: no simulator, no GPU, seconds
make example              # end-to-end toy run
```

The CI workflow records the exact interpreter and dependency versions used,
and runs the tests plus the toy example on every push.

## Dependencies

The core package depends only on `numpy` and `pyyaml`; `torch`, `gymnasium`
and `dm-control` are optional extras, needed only to drive a real simulator.
The protocol layer imports no simulator at all, which is what keeps it
auditable.