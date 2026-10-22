# Metric definitions

This file is the normative reference. If the code and this file disagree, the
code is wrong.

## Setting

One **origin** is one independently sampled start state. For an origin `s`:

- `A = {a_0, ..., a_{K-1}}` is the candidate set, in a **fixed order** decided
  before the run;
- `Y(a)` is the measured value of executing `a` as the **first** action from
  the same restored state, after which the frozen baseline policy acts in
  closed loop for the remaining steps of the horizon `H`;
- `b` is the index of the baseline policy's own first action (it must be one
  of the candidates);
- `w` is the index the world model selects.

`Y` is a **per-step average over the horizon**, not a per-step average over
surviving steps.

## Post-terminal convention

If a branch terminates before the horizon, the remaining steps contribute
**zero** reward:

```
Y(a) = (sum of collected rewards) / H
```

Dividing by the number of *survived* steps would make early termination look
like **higher** per-step value, which is backwards. Both denominators are
reported (`values` and `steps_survived`) so the choice is visible.

## The three quantities

```
Opportunity      A = max_i Y(a_i) - Y(a_b)
Ranking regret   L = max_i Y(a_i) - Y(a_w)
Value gain       G = Y(a_w)     - Y(a_b)
```

Hence, identically:

```
G = A - L
```

`pcf_eval.metrics.decompose` verifies this numerically on every origin and
returns `identity_residual`. A non-zero residual is raised as an
`ArithmeticError`, not tolerated.

### Reading the decomposition

| A | L | interpretation | what to do |
|---|---|---|---|
| `A <= 0` | any | **coverage limited** — no candidate beat the baseline | change the candidate generator or the task; more training cannot help |
| `A > 0` | `L > A` | **ranking limited** — a better action existed and was missed | improve the value estimates or the ranker |
| `A > 0` | `L = 0` | **aligned** — the model captured the opportunity | scale up |
| `A > 0` | `0 < L < A` | partially captured | quantify the residual |

## Ranking quality

- `top1_hit` — the selected candidate is the arg-max of the measured value.
- `random_top1_rate` — `1/K`, the rate a uniform random selector achieves.
- `pairwise_concordance` — over the `K(K-1)/2` unordered candidate pairs, the
  fraction whose measured ordering agrees with the model's score ordering.
  Ties are excluded from the denominator. `0.5` corresponds to chance.

## Statistics

The independent unit is the **origin**. Candidate branches, model seeds and
horizons are repeated measurements inside an origin.

- Intervals are **percentile bootstrap over origins** (default 10,000
  resamples, fixed seed).
- Resampling over branches, checkpoints or steps would inflate the effective
  sample size and shrink the interval for no reason.
- A verdict of `inconclusive` is returned whenever the 95% interval of `G`
  covers zero. A point estimate alone is never a conclusion.

## Cost gate

Extra compute is only worth paying if it buys value:

```
net = G - lam * extra_cost
```

`lam` is a **declared** trade-off, in units of the task metric per unit of
cost. It is not tuned after seeing results: report a Pareto front over a
pre-registered `lam` grid instead of a single hand-picked point.

## What these metrics do not say

A negative value gain is evidence about **this task, this baseline, this
candidate set, this model and this frozen protocol**. It is not evidence that
world models do not help in general, and it is not a substitute for a
perceptual benchmark. Its value is that it is *specific*: it names which link
in the chain is broken.