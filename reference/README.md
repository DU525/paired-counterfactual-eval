# Reference results

Small, committed numbers so that a metric regression is visible in a diff.

| file | produced by | why it is committed |
|---|---|---|
| `toy_verdicts.json` | `python scripts/make_reference.py` | the three toy configurations produce known verdicts; if a change to the metrics or the aggregation moves them, the diff says so before CI does |

**Why the file is small.** Datasets, weights, raw traces and full result
bundles are *not* committed — they are reproduced or downloaded. What is
committed is the minimum needed to detect drift: the verdict, the mechanism and
the point estimate plus interval for each configuration.

**Regenerating.** Deliberately, and in the same commit as the change that moved
the numbers:

```bash
python scripts/make_reference.py
```

A pull request that regenerates this file without explaining which number moved
and why should not be merged.

```
$ cat reference/toy_verdicts.json
```

See [`../docs/metrics.md`](../docs/metrics.md) for what each field means.