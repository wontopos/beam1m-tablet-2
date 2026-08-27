# LongMemEval-S · tablet-2

The eight runs behind the LongMemEval-S numbers in the paper. This is a different
campaign from the BEAM-1M one in the parent directory: earlier, a different
corpus, and different retrieval settings. The two are not comparable and the
paper says so where it reports them.

LongMemEval is Wu et al. ([arXiv:2410.10813](https://arxiv.org/abs/2410.10813)).
The S variant is 500 questions over multi-session dialogue histories. Scoring is
one yes-or-no verdict per question from a `gpt-4o` judge at temperature 0, which
is the benchmark's own default and we did not change it.

## The runs

| file | reader | effort, re-ask | graded | score |
|---|---|---|---|---|
| `t2_opus1` | Claude Opus 5 | max, 3 | 500 | 95.2 |
| `t2_opus_r2` | Claude Opus 5 | max, 3 | 500 | 96.0 |
| `t2_opus_r3` | Claude Opus 5 | max, 3 | 500 | 96.0 |
| `t2_u1` | GPT-5.6-sol | ultra, 3 | 499 | 93.2 |
| `t2_u2` | GPT-5.6-sol | ultra, 3 | 499 | 94.0 |
| `t2_u3` | GPT-5.6-sol | ultra, 3 | 499 | 93.8 |
| `t2_opus_med` | Claude Opus 5 | medium, 3 | 500 | 95.0 |
| `t2_opus_med_h0` | Claude Opus 5 | medium, 0 | 500 | 93.8 |

Engine, corpus, retrieval settings and judge are identical across all eight. The
first two groups differ only in the reader, which is what makes the 2.0-point
reader effect measurable.

## Reading a record

Each file is a JSON list of 500 objects, one per question:

```
question_id   the benchmark's id
type          question category, e.g. single-session-user
q, gt         the benchmark's question and ground truth
ans           what the reader answered
ok            the judge's verdict: true, false, or null
hops          re-ask passes actually used, 0-3
n_mems        memories delivered
gold_in_pool  whether the supporting evidence was in what we retrieved
```

## Two things to know before recomputing the score

**`ok: null` is not `false`.** Each GPT-5.6-sol run has exactly one question the
judge returned no verdict on. An ungraded question leaves the denominator; it
does not count as wrong. Scoring a grader failure as an engine failure turns an
outage into a result. That is why those three rows say 499 and the rest say 500.
Counting the nulls as wrong lowers each of them by 0.2 points, so the convention
is worth stating rather than assuming.

**`gold_in_pool` is the retrieval measure, `ok` is the pair's.** `ok` grades
retrieval and reader together. `gold_in_pool` asks only whether the evidence was
delivered, which is the closest thing here to a score for the engine alone.

## Runs not listed

Three runs from this campaign are not here, because they measure nothing. In two
of them the reader hit a provider limit and returned the same refusal message
for all 500 questions. In the third the reader answered all 500 questions having
received no memories at all, which grades as a language model working from its
own priors rather than as a retrieval result.

Both failures are worth naming because neither announced itself: every run
finished, exited cleanly, and produced a plausible number. The harness now
refuses to grade a run whose answers look like provider errors, and checks the
retrieved count before grading. A run with no valid answers is not a result we
withheld, it is an absence of measurement.

## Licence

The runs are published as measurement records. The LongMemEval benchmark, its
questions and its ground truths belong to its authors.
