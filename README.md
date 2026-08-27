# BEAM-1M · tablet-2

The harness, the scoring scripts, and the per-question records behind every
configuration reported in
[arXiv:2608.23920](https://arxiv.org/abs/2608.23920).

**67.5%, 95% question-sampling interval [64.8, 70.2]**, five runs over 700
questions, 74,630 turns ingested as 2,212,504 memories. Engine `tablet-2`,
binary SHA-256 prefix `cfee6b9d`.

The five runs agree to within 0.22 points, which is how little re-running the
same questions moves the score and not how precisely the score is known. The
interval above is the one that answers the second question, and it is an order
of magnitude wider.

BEAM is Tavakoli et al., *Beyond a Million Tokens* ([arXiv:2510.27246](https://arxiv.org/abs/2510.27246)).
This repository holds our measurement of it, not the benchmark itself.

---

## What is here

```
scripts/wos_bench.py       the harness: retrieve, answer, record, grade
scripts/build_tasks.py     build a task file from a benchmark's corpus
scripts/beam_ingest.py     load a corpus through the SDK, with dates and speakers
scripts/verify_corpus.py   read it back before trusting that it stored
runs/                      BEAM-1M, per question: answer, verdict, timing, size
grades/                    BEAM-1M scores with the rubric-item counts behind them
lme/                       LongMemEval-S, the same for eight runs
MANIFEST.json              machine-readable summary of all of it
```

Records are keyed by each benchmark's own `question_id`. The questions, ground
truths and rubrics are not reproduced here: they belong to the benchmark
authors, and you obtain them from the benchmarks themselves and join.

## Reproducing the number

Obtain the benchmark corpus from its authors, then:

```bash
export WOS_API_KEY=...
export BEAM_PARQUET=/path/to/1M.parquet     # the benchmark's own corpus

python3 scripts/build_tasks.py              # -> beam_tasks_v2.json, rubrics included
python3 scripts/beam_ingest.py              # ingest, once
python3 scripts/verify_corpus.py            # read it back before trusting it

python3 scripts/wos_bench.py run \
  --tasks beam_tasks_v2.json --out run1.json \
  --model tablet-2 --limit 20 --verify 3 --lanes both \
  --no-judge --reader-cmd 'your-llm-cli --model YOUR_MODEL'

export JUDGE_KEY=...
python3 scripts/wos_bench.py grade run1.json \
  --protocol rubric --tasks beam_tasks_v2.json \
  --ordering-type event_ordering \
  --judge-url https://api.openai.com/v1/chat/completions \
  --judge-model gpt-4.1-mini --judge-key-env JUDGE_KEY

python3 scripts/wos_bench.py report run1.json
```

How the corpus is loaded decides the score as much as retrieval does, which is
why the ingest script is here rather than described. Attach the wrong dates and
every temporal question silently becomes unanswerable.

Every call these scripts make to the engine goes through the published SDK
(`pip install wontopos`), using the same public methods any account has. There
is no private endpoint and no internal header anywhere in this repository. The
reader and judge are whatever OpenAI-compatible endpoint you point them at, and
you pay for those.

### `--protocol rubric` is not optional here

BEAM does not score one yes-or-no verdict per question. Each question carries
rubric items, each judged 1.0 / 0.5 / 0.0, and the question's score is their
mean. Ordering questions are scored by rank correlation instead, which is why
`--ordering-type event_ordering` is passed.

The harness will happily grade this benchmark with `--protocol binary`, and it
will print a number. That number is not comparable with anything anyone has
published. Scoring `event_ordering` by rubric mean rather than rank correlation
moves the total by roughly four points on its own.

### The judge

`gpt-4.1-mini` at temperature 0, with the benchmark's own
`unified_llm_judge_base_prompt` used verbatim. It is the default in the
benchmark's scoring code and we did not change it, on the principle that
choosing a better judge after seeing scores is not measurement.

The judge is called once per rubric item, an average of 3.45 times per question.
Five runs came to 38,662 calls and about \$12, which is more than the question
count suggests.

## What the runs contain

Every run file records, per question: the answer, the verdict, the retrieved
context length, the number of memories, the retrieval time, and any error.
Grade files record the per-question score with the number of rubric items behind
it and how many were graded, so a score can be checked against its denominator.

Four of them carry more. `r2`, `r3`, `r4` and `r5` keep the judge's own reason
for every item it scored, 2,415 of them per run, so "why was this a 0.5" can be
answered without paying the judge again. The other three do not; recovering
their reasoning means re-grading, which the harness does from a run file without
repeating retrieval.

What those items do **not** carry is the rubric text itself. The rubrics are the
benchmark's, not ours, so each item is our judge's score and our judge's
reasoning, joined to the benchmark's own rubric by position.

| run | configuration | score |
|---|---|---|
| `beam700_v3`, `r2`, `r3`, `r4`, `r5` | re-ask 3 | 67.8 / 67.7 / 67.3 / 67.1 / 67.5 |
| `beam700_v0` | re-ask 0 | 58.6 |
| `beam700_C` | re-ask 3 + expansion | 70.3 |

These are the configurations reported in the paper: the headline measurement,
and the two ablations either side of it. Each is here with its per-question
record, so any figure quoted from them can be recomputed rather than taken.

## Things worth knowing before you trust a number

- **`beam700_C` scored 681 questions, not 700.** Nineteen errored mid-run and are
  out of its denominator. `beam700_v0` lost two the same way. The counts are in
  `MANIFEST.json`.
- **An unreadable verdict is not a zero.** It leaves the item out of its
  question's denominator. Counting judge failures as wrong answers turns a
  grader outage into an engine result.
- **The rubric judge needs room to answer.** It returns a JSON object, and a
  token cap sized for yes/no truncates it after `{"score":`. That parses as no
  score and silently shrinks the run rather than failing. The harness sizes the
  cap by protocol; if you rebuild this yourself, do the same.
- **Ingest defects, ours.** 211 turns longer than 8,000 characters were truncated
  by our ingest script, and one turn of 74,630 failed to store.
- **`event_ordering` sits at 23.6%**, our worst type, and we do not think it
  measures ordering. The rubric expects ten topics; our answers give ten
  individual events. Both are ordered and both have ten items, but the matcher
  cannot pair across that difference, so unpaired items sink to the bottom and
  the correlation goes negative. Reshaping the answers to the rubric's
  granularity would raise the score and would be answering to the answer key.

## Licence

Scripts under MIT. The runs and grades are published as measurement records.
The BEAM benchmark, its questions and its rubrics belong to its authors.
