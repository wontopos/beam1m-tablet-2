#!/usr/bin/env python3
"""wos-bench: a benchmark harness for the Wontopos memory engine.

Three stages, in order:

    retrieve (WOS)  ->  answer (your LLM)  ->  grade (a second LLM)

WOS does the first stage only. Retrieval is purely semantic, runs no LLM, and uses
no BM25 or keyword matching, so it behaves the same in every language. The answer
and the grade come from models you choose and pay for. Nothing here grades its own
output.

What it records per question:

    accuracy        the grader's yes/no
    delivered size  memories and characters returned by WOS, and tokens if you
                    supply a tokenizer
    round-trip      wall time of the retrieval call alone

Reader and grader are any OpenAI-compatible chat endpoint, or any command that reads
a prompt on stdin and prints an answer on stdout. Point them wherever you like,
including a local server or a CLI:

    export WOS_API_KEY=...
    export READER_KEY=...  JUDGE_KEY=...

    python wos_bench.py run \\
        --tasks tasks.json --model tablet-1 --out run1.json \\
        --reader-url https://your-provider/v1/chat/completions --reader-model MODEL \\
        --judge-url  https://your-provider/v1/chat/completions --judge-model MODEL

Answering and grading can be split. Retrieval and answering are the expensive half
and worth keeping when only the grading rule changes, so a run can be recorded
without a grade and graded later, as many times as you like:

    python wos_bench.py run --tasks tasks.json --out run1.json --no-judge \\
        --reader-cmd 'your-llm-cli --model MODEL'
    python wos_bench.py grade run1.json --judge-url ... --judge-model MODEL

    python wos_bench.py report run1.json run2.json run3.json

Task file: [{"question_id": ..., "user_id": ..., "question": ..., "gt": ..., "type": ...}, ...]

The memories must already be in your WOS account, and `user_id` is the store each
question is asked against. Every question is one billed retrieval, so a 500-question
pass repeated three times is 1,500 of them; `--max` runs a small slice first. A run
interrupted partway can be continued with `--resume`, which keeps the answers already
bought and runs only what is missing.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

try:
    from wontopos import Client
except ImportError:
    print("The wontopos SDK is missing. Install it with: pip install wontopos", file=sys.stderr)
    raise

VERSION = "0.2.0"


def sdk_version() -> str:
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(sys.modules.get("wontopos"), attr, None)
        if isinstance(v, str):
            return v
    try:
        from importlib.metadata import version
        return version("wontopos")
    except Exception:
        return "unknown"

# A failing reader does not always return an empty string. When a key expires or a
# quota runs out, many endpoints return a sentence instead, and a run full of those
# sentences will pass any "count the blank answers" check. Grading it then produces
# a score that looks plausible and means nothing, which is worse than a zero because
# nobody questions it. Every run is checked against this list before grading.
#
# ★The phrases have to be ones a real answer cannot contain. An earlier version
# matched bare "rate limit" and "authentication", and those are ordinary English:
# "I enabled two-factor authentication in March" and "the rate limit on my gym
# membership" are both plausible answers in a memory benchmark, and either one
# turned an entire paid run into INVALID. Guarding against a fake score is worth
# nothing if it throws away real ones, so each entry below is provider-error
# phrasing, and the length bound keeps a long genuine answer out of reach.
NOT_AN_ANSWER = (
    "you have hit your session limit",
    "you've hit your session limit",
    "usage limit reached",
    "rate limit exceeded",
    "rate_limit_exceeded",
    "quota exceeded",
    "insufficient_quota",
    "invalid api key",
    "incorrect api key",
    "authentication failed",
    "authentication_error",
    "401 unauthorized",
    "403 forbidden",
)

# Provider errors are short. A long passage that happens to contain one of the
# phrases above is far more likely to be an answer discussing it.
NON_ANSWER_MAX_CHARS = 400


def looks_like_non_answer(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    return len(t) <= NON_ANSWER_MAX_CHARS and any(m in t for m in NOT_AN_ANSWER)


# Token counting is optional and never estimated. An approximate tokenizer is off by
# 15 to 20 percent against real billing, and characters divided by four is worse. A
# published token count that disagrees with the invoice is not an approximation, it
# is a false claim. Supply your own counter or leave the field empty.
#
# --tokenizer-cmd takes a command that reads text on stdin and prints one integer.
class TokenCounter:
    def count(self, text: str) -> Optional[int]:
        return None


class CommandTokenCounter(TokenCounter):
    def __init__(self, cmd: str):
        self.cmd = cmd

    def count(self, text: str) -> Optional[int]:
        if not text:
            return 0
        try:
            p = subprocess.run(self.cmd, shell=True, input=text, capture_output=True,
                               text=True, timeout=60)
            return int(p.stdout.strip())
        except Exception:
            return None


READER_PROMPT = """Answer the question using only the memories below.
Do not use outside knowledge. If the memories do not contain the answer, say so.

MEMORIES
{memories}

QUESTION
{question}

Answer concisely."""

# LongMemEval grades per category, not with one rule for everything. A single
# generic rule scores differently from the published protocol, which would make any
# number produced here incomparable with numbers produced by the official one. The
# rules below are the per-category ones; a task with no `type`, or an unknown type,
# falls back to the default.
JUDGE_RULES = {
    "temporal-reasoning":
        "Answer yes if the response contains the correct answer or is equivalent. "
        "Do NOT penalize off-by-one errors for number of days/weeks/months.",
    "knowledge-update":
        "Answer yes if the response contains the correct UPDATED answer. If it also "
        "mentions previous/outdated info, still yes as long as the updated answer is present.",
    "single-session-preference":
        "The response does not need to reflect all points in a rubric. Answer yes as long "
        "as it recalls and uses the user's personal information/preference correctly.",
    "_default":
        "Answer yes if the response contains the correct answer, or is equivalent / contains "
        "all the intermediate steps to reach it. If only a subset of required info, answer no.",
}

JUDGE_PROMPT = """I will give you a question, the correct answer, and a model's response. \
{rule} Respond with ONLY 'yes' or 'no'.

Question: {question}
Correct answer: {gt}
Model response: {ans}

Is the model response correct?"""


def judge_prompt_for(qtype: Optional[str], question: str, gt: str, ans: str) -> str:
    return JUDGE_PROMPT.format(rule=JUDGE_RULES.get(qtype or "", JUDGE_RULES["_default"]),
                               question=question, gt=gt, ans=ans)


def chat_llm(url: str, model: str, api_key: str, temperature: float = 0.0,
             max_tokens: int = 1024, timeout: int = 180) -> Callable[[str], str]:
    """Any OpenAI-compatible /chat/completions endpoint."""
    def call(prompt: str) -> str:
        body = json.dumps({
            "model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(url, data=body, headers={
            "content-type": "application/json", "authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        return (d["choices"][0]["message"]["content"] or "").strip()
    return call


def command_llm(cmd: str, timeout: int = 900) -> Callable[[str], str]:
    """Any command that reads a prompt on stdin and prints an answer on stdout.

    A command that fails raises rather than returning its stderr, because an error
    message returned as an answer would be graded as a wrong answer instead of being
    counted as an error.
    """
    def call(prompt: str) -> str:
        p = subprocess.run(cmd, shell=True, input=prompt, capture_output=True,
                           text=True, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"exit {p.returncode}: {(p.stderr or '').strip()[:200]}")
        return (p.stdout or "").strip()
    return call



_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _ymd(v):
    """'2024-03-15T00:00:00Z' -> date. Anything unparseable is simply not shown."""
    try:
        return datetime.date(int(str(v)[0:4]), int(str(v)[5:7]), int(str(v)[8:10]))
    except Exception:
        return None


def archive_phrase(when, now):
    """How long ago, in words. A reader that sees only '2024-03-15' has to do date
    arithmetic in its head to answer 'how long after X'; this hands it the answer."""
    ev, ref = _ymd(when), _ymd(now)
    if not ev or not ref:
        return None
    days = (ref - ev).days
    if days < 0:
        return None
    md = "%s %02d" % (_MONTHS[ev.month - 1], ev.day)
    my = "%s %d" % (_MONTHS[ev.month - 1], ev.year)
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return "%d days ago" % days
    if days < 14:
        return "last week (%s)" % md
    if days < 30:
        return "%d weeks ago (%s)" % (days // 7, md)
    if days < 60:
        return "last month (%s)" % md
    if days < 365:
        return "%d months ago (%s)" % (days // 30, my)
    if days < 730:
        return "last year (%s)" % my
    return "%d years ago (%s)" % (days // 365, my)


def fmt_memory(m, opts=(), now=None):
    """One delivered memory as the reader sees it.

    Only fields the SDK returns are used — `event_date` and `similarity` — so anyone
    can reproduce this with the published client and no private endpoint.
    """
    head = ""
    if "dates" in opts or "archive" in opts:
        d = str(m.get("event_date") or "")[:10]
        if d:
            label = d
            if "archive" in opts and now:
                a = archive_phrase(m.get("event_date"), now)
                if a:
                    label = "%s | %s" % (d, a)
            head += "[%s] " % label
    if "rel" in opts:
        try:
            head += "(rel %.2f) " % float(m.get("similarity") or 0.0)
        except Exception:
            pass
    return "- " + head + str(m.get("content", ""))


@dataclass
class QuestionResult:
    question_id: Optional[str]
    user_id: str
    type: Optional[str]
    question: str
    ground_truth: str
    answer: str = ""
    correct: Optional[bool] = None
    # Rubric protocol only: a per-question score in [0,1] and the judged items
    # behind it. `correct` stays None there, because a 0.5 is not a yes or a no.
    score: Optional[float] = None
    rubric_items: Optional[list] = None
    # Retrieval numbers only. Reader and grader time never enter these fields.
    wos_ms: Optional[float] = None
    n_memories: int = 0
    context_chars: int = 0
    context_tokens: Optional[int] = None
    error: Optional[str] = None


@dataclass
class RunResult:
    harness_version: str
    model: str
    config: dict
    measured_at: str
    rows: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _pct(xs: list[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)   # Without the sort this returns the value at that index, not the quantile.
    return round(xs[min(len(xs) - 1, int(len(xs) * q))], 1)


def summarize(rows: list[QuestionResult]) -> dict:
    ok = [r for r in rows if r.error is None]
    judged = [r for r in ok if r.correct is not None]
    scored = [r for r in ok if r.score is not None]
    lat = [r.wos_ms for r in ok if r.wos_ms is not None]
    tok = [r.context_tokens for r in ok if isinstance(r.context_tokens, int)]
    bad = [r for r in ok if looks_like_non_answer(r.answer)]

    by_type: dict[str, dict] = {}
    for r in judged:
        b = by_type.setdefault(r.type or "unknown", {"correct": 0, "n": 0})
        b["correct"] += int(bool(r.correct))
        b["n"] += 1

    return {
        "n": len(rows),
        "errors": len(rows) - len(ok),
        # Read this line before the score. If it is not zero, the run is not a score.
        "non_answers": len(bad),
        "valid": len(bad) == 0 and len(rows) == len(ok),
        "accuracy": round(sum(bool(r.correct) for r in judged) / len(judged), 4) if judged else None,
        "judged": len(judged),
        # Rubric protocol. Kept in its own field rather than folded into
        # `accuracy`, so a run scored one way can never be read as the other.
        "rubric_score": round(statistics.fmean(r.score for r in scored), 4) if scored else None,
        "rubric_scored": len(scored),
        # Rows the grader answered in a way that was neither yes nor no. They are out
        # of the denominator, so this number has to be visible next to the accuracy —
        # a score over 300 of 500 questions is not a score over 500.
        "ungraded": len([r for r in ok if r.answer and r.correct is None]),
        "by_type": {k: {**v, "accuracy": round(v["correct"] / v["n"], 4)} for k, v in sorted(by_type.items())},
        "wos_ms": {"p50": _pct(lat, .5), "p95": _pct(lat, .95),
                   "mean": round(statistics.fmean(lat), 1) if lat else None},
        "context_tokens": {"p50": _pct(tok, .5), "p95": _pct(tok, .95),
                           "mean": round(statistics.fmean(tok), 1) if tok else None,
                           "total": sum(tok) if tok else None,
                           "counted": len(tok), "of": len(ok)},
        "context_chars": {"p50": _pct([float(r.context_chars) for r in ok], .5),
                          "mean": round(statistics.fmean([r.context_chars for r in ok]), 1) if ok else None},
        "memories": {"mean": round(statistics.fmean([r.n_memories for r in ok]), 2) if ok else None},
    }


# Errors worth trying again: rate limits and the transient 5xx/network family. An
# auth failure or a bad request will fail identically every time, so retrying those
# only spends money and time.
RETRYABLE = ("429", "rate limit", "too many requests", "timeout", "timed out",
             "500", "502", "503", "504", "connection", "temporarily")


def is_retryable(e: Exception) -> bool:
    s = f"{type(e).__name__}: {e}".lower()
    return any(m in s for m in RETRYABLE)


def run_one(client: Client, task: dict, limit: int, search_opts: dict,
            reader: Callable[[str], str], counter: TokenCounter,
            retries: int = 4, lanes: str = "both",
            ctx_opts: tuple = (), now_iso: Optional[str] = None,
            prompt_tpl: Optional[str] = None,
            line_list_type: str = "") -> QuestionResult:
    r = QuestionResult(
        question_id=task.get("question_id"), user_id=task["user_id"], type=task.get("type"),
        question=task["question"], ground_truth=str(task.get("gt", "")),
    )

    # ★Retry, because one transient failure used to cost the whole run.
    #
    # A single error marks the run invalid, and every question in it was PAID for.
    # A rate limit on question 480 of 500 threw away the 479 answers already bought.
    # Tier limits make that likely, not hypothetical, at any real concurrency.
    for attempt in range(retries + 1):
        try:
            # The retrieval call is timed here and nowhere else, so a slow reader can
            # never show up as a slow engine. A retried call is timed from its own
            # start, so backoff sleep never lands in the engine's latency.
            t0 = time.perf_counter()
            if lanes == "both":
                mems = client.search(r.question, user_id=r.user_id, limit=limit, **search_opts)
            else:
                # search() merges the two lanes; search_self() keeps them apart. Only the
                # split call can answer "what does the partner lane alone cost and score",
                # which is the whole point of storing a speaker in the first place.
                out = client.search_self(r.question, user_id=r.user_id, limit=limit, **search_opts)
                mems = out["memories"] if lanes == "main" else out["self_memories"]
            r.wos_ms = round((time.perf_counter() - t0) * 1000, 1)
            break
        except Exception as e:
            if attempt == retries or not is_retryable(e):
                r.error = f"retrieval: {type(e).__name__}: {e}"[:300]
                return r
            time.sleep(min(30, 2 ** attempt))

    r.n_memories = len(mems)
    ctx = "\n".join(fmt_memory(m, opts=ctx_opts, now=now_iso) for m in mems)
    r.context_chars = len(ctx)
    r.context_tokens = counter.count(ctx)

    try:
        # A benchmark that ships its own answering prompt should be run with it, not
        # with ours — the shape of the answer decides the score on some question types.
        prompt = (prompt_tpl or READER_PROMPT).format(
            memories=ctx or "(none)", question=r.question)
        if line_list_type and (r.type or "").replace("-", "_") == line_list_type:
            # The scorer for ordering questions splits the answer on newlines and matches
            # each line to a rubric item. Prose with headers becomes many phantom
            # "events" and the rank correlation collapses — a formatting loss, not a
            # memory one. Asking for one event per line measures the memory instead.
            prompt += ("\n\nFormat: one event per line, in order, oldest first. "
                       "No headers, no numbering, no blank lines, no commentary.")
        r.answer = reader(prompt)
    except Exception as e:
        r.error = f"reader: {type(e).__name__}: {e}"[:300]
    return r


def build_reader(a: argparse.Namespace) -> tuple[Callable[[str], str], str]:
    if a.reader_cmd:
        return command_llm(a.reader_cmd, timeout=a.reader_timeout), a.reader_cmd
    if not (a.reader_url and a.reader_model):
        raise SystemExit("give either --reader-cmd, or both --reader-url and --reader-model")
    return (chat_llm(a.reader_url, a.reader_model, os.environ[a.reader_key_env],
                     max_tokens=a.reader_max_tokens), a.reader_model)


def build_judge(a: argparse.Namespace, max_tokens: int = 8) -> tuple[Callable[[str], str], str]:
    """Build the grader.

    ★`max_tokens` is a correctness setting here, not a cost setting. The binary
    protocol needs one word, so 8 is generous. The rubric protocol needs a JSON
    object holding a score and a reason, and at 8 tokens the reply stops after
    `{"score":`. That parses as no score, which drops the item from its
    question's denominator and quietly shrinks the run instead of failing. A
    smoke test on eight questions lost two of them to exactly this.
    """
    if a.judge_cmd:
        return command_llm(a.judge_cmd, timeout=a.judge_timeout), a.judge_cmd
    if not (a.judge_url and a.judge_model):
        raise SystemExit("give either --judge-cmd, or both --judge-url and --judge-model")
    return (chat_llm(a.judge_url, a.judge_model, os.environ[a.judge_key_env],
                     temperature=0.0, max_tokens=max_tokens), a.judge_model)


def read_verdict(v: str) -> Optional[bool]:
    """Read a grader's yes/no, and refuse to guess.

    ★Substring matching is not good enough here, and it fails in the direction that
    flatters the thing being measured. "The answer is not yes" and "no, yes would
    require more" both contain "yes", so `"yes" in v` scored them correct. Every
    such mistake pushes accuracy UP, which is exactly the bias an independent
    reviewer looks for first.

    So: take the first word, accept only an unambiguous yes or no, and return None
    otherwise. None means ungraded, which leaves the row out of the denominator
    instead of inventing a verdict for it.
    """
    t = (v or "").strip().lower().lstrip("*'\"`# ")
    if not t:
        return None
    head = t.split()[0].strip(".,:;!*'\"`)")
    if head in ("yes", "y", "correct", "true"):
        return True
    if head in ("no", "n", "incorrect", "false"):
        return False
    return None


def grade_rows(rows: list[QuestionResult], judge: Callable[[str], str], concurrency: int) -> None:
    def grade(r: QuestionResult) -> None:
        if r.error or not r.answer:
            return
        try:
            r.correct = read_verdict(judge(judge_prompt_for(r.type, r.question, r.ground_truth, r.answer)))
        except Exception:
            r.correct = None   # A failed grade is not a wrong answer. It leaves the denominator.

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(grade, rows))


def task_key(t: dict) -> str:
    return f"{t.get('question_id')}|{t.get('user_id')}"


def cmd_run(a: argparse.Namespace) -> int:
    tasks = json.load(open(a.tasks))
    if a.max:
        tasks = tasks[: a.max]

    # ★Resume, for the same reason retrieval retries: every question was paid for.
    # Kept rows are the ones that already produced an answer — an errored row is
    # retried, since that is the case worth another attempt.
    done: dict[str, QuestionResult] = {}
    if a.resume and os.path.exists(a.out):
        prev = json.load(open(a.out))
        for d in prev.get("rows", []):
            r = QuestionResult(**d)
            if r.error is None and r.answer:
                done[f"{r.question_id}|{r.user_id}"] = r
        print(f"  resuming: {len(done)} answers kept from {a.out}", file=sys.stderr)

    todo = [t for t in tasks if task_key(t) not in done]

    client = Client(api_key=os.environ["WOS_API_KEY"], model=a.model,
                    **({"base_url": a.base_url} if a.base_url else {}))
    now_by_store: dict[str, str] = {}
    if a.archive:
        # Once per store, not once per question. A broad query returns recent
        # memories; the latest event_date among them is "now" for that store,
        # which is what relative phrasing in a question has to be resolved against.
        for uid in sorted({t["user_id"] for t in tasks}):
            try:
                mems = client.search("summary of everything recent", user_id=uid, limit=50)
                ds = [str(m.get("event_date") or "") for m in mems if m.get("event_date")]
                if ds:
                    now_by_store[uid] = max(ds)
            except Exception:
                pass          # no date for this store: it runs without one
        print(f"  archive reference dates: {len(now_by_store)}/"
              f"{len({t['user_id'] for t in tasks})} stores", file=sys.stderr)

    prompt_tpl = None
    if a.reader_prompt:
        prompt_tpl = open(a.reader_prompt, encoding="utf-8").read()
        for ph in ("{memories}", "{question}"):
            if ph not in prompt_tpl:
                raise SystemExit(f"--reader-prompt file is missing {ph}")

    reader, reader_label = build_reader(a)
    counter = CommandTokenCounter(a.tokenizer_cmd) if a.tokenizer_cmd else TokenCounter()

    search_opts: dict[str, Any] = {}
    if a.verify:
        search_opts["verify"] = a.verify
    ctx_opts = tuple(k for k, on in (
        ("dates", a.dates or a.archive), ("rel", a.rel),
        ("archive", a.archive), ("line_list", bool(a.line_list))) if on)

    fresh: list[QuestionResult] = []
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = [ex.submit(run_one, client, t, a.limit, search_opts, reader, counter, a.retries, a.lanes,
                          ctx_opts, now_by_store.get(t['user_id']), prompt_tpl, a.line_list)
                for t in todo]
        for i, f in enumerate(futs, 1):
            fresh.append(f.result())
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", file=sys.stderr, flush=True)

    by_key = {f"{r.question_id}|{r.user_id}": r for r in fresh}
    by_key.update(done)
    rows = [by_key[task_key(t)] for t in tasks if task_key(t) in by_key]

    # Validity is checked before grading. Grading a run whose reader was failing
    # produces a plausible number that describes nothing.
    bad = [r for r in rows if r.error is None and looks_like_non_answer(r.answer)]
    judge_label = "none"
    if a.no_judge:
        print(f"\nRecorded {len(rows)} answers without grading. "
              f"Grade with: wos_bench.py grade {a.out} --judge-...", file=sys.stderr)
    elif bad and not a.force:
        print(f"\n{len(bad)} answers look like provider errors, so grading was skipped. "
              f"Check the reader credentials and quota.", file=sys.stderr)
        print(f"  first: {bad[0].answer[:120]!r}", file=sys.stderr)
        print("  Use --force to grade anyway.", file=sys.stderr)
    else:
        judge, judge_label = build_judge(a)
        grade_rows(rows, judge, a.concurrency)

    out = RunResult(
        harness_version=VERSION, model=a.model,
        config={"tasks": os.path.basename(a.tasks), "n_tasks": len(tasks), "limit": a.limit,
                "verify": a.verify,
                "reader": reader_label, "judge": judge_label,
                "tokenizer": a.tokenizer_cmd or "none",
                "lanes": a.lanes,
                "ctx": ",".join(ctx_opts) or "content-only",
                "reader_prompt": a.reader_prompt or "built-in",
                "line_list": a.line_list or "off",
                "base_url": a.base_url or "default",
                # The SDK version belongs with the number. Retrieval crosses it, so a
                # run recorded under one version is not automatically comparable with
                # a run recorded under another.
                "sdk_version": sdk_version(),
                "concurrency": a.concurrency, "retries": a.retries},
        measured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        rows=[asdict(r) for r in rows],
    )
    out.summary = summarize(rows)
    json.dump(asdict(out), open(a.out, "w"), ensure_ascii=False, indent=1)

    s = out.summary
    print(f"\n{a.model}  {'valid' if s['valid'] else 'INVALID'}  "
          f"accuracy {s['accuracy']}  tokens p50 {s['context_tokens']['p50']}  "
          f"retrieval p50 {s['wos_ms']['p50']}ms  -> {a.out}")
    return 0 if s["valid"] else 1


# ── Rubric protocol (BEAM-style) ──────────────────────────────────────────────
# Two benchmarks in this file score differently, and mixing them silently would
# be the worst outcome: both produce a number, and only one of them is the number
# other people published.
#
#   binary  one yes/no verdict per question              (LongMemEval-style)
#   rubric  a question carries N rubric items; each is judged 1.0/0.5/0.0 and the
#           question score is their mean. Ordering questions are scored by rank
#           correlation instead, because the benchmark that defines this protocol
#           aggregates `tau_norm` for them rather than the rubric mean.
#
# ★The rubric judge prompt is the benchmark author's, used verbatim. Rewording a
#  judge prompt so the numbers come out better is self-grading.

RUBRIC_JUDGE = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): {question}
- RUBRIC CRITERION (what to check): {rubric_item}
- RESPONSE TO EVALUATE: {llm_response}

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms. Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure.
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
- **0.5 (Partial Compliance)**: Partially complies; element present but minor inaccuracies/incomplete execution.
- **0.0 (No Compliance)**: Required element missing or incorrect, or response non-responsive.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[short explanation]"
}}

NOTE: ONLY output the json object, without any explanation before or after that
"""

EQUIVALENCE_PROMPT = ("You are a binary classifier.\n"
                      "If the TWO snippets describe the SAME event/fact, reply **YES**\n"
                      "Otherwise reply **NO**. No extra words.\n"
                      "DO NOT provide any exaplanation.\n\n"
                      "First snippet: {a} \n\n Second snippet: {b}")


def read_rubric_score(v: str) -> tuple[Optional[float], Optional[str]]:
    """Pull the score and the stated reason out of a judge reply.

    The reason is kept, not discarded. Without it, "why was this 0.5" costs a
    second run of the same judge to answer.
    """
    v = (v or "").strip()
    if v.startswith("```"):          # some judges fence their JSON; harmless, so allow it
        v = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", v)
    m = re.search(r'"score"\s*:\s*([0-9.]+)', v)
    reason = None
    rm = re.search(r'"reason"\s*:\s*"(.*?)"\s*[,}]', v, re.S)
    if rm:
        reason = rm.group(1)[:400]
    if not m:
        return None, (reason or v[:400] or None)
    f = float(m.group(1))
    # A judge that answers 0.7 has not followed the scale. Treating it as
    # ungraded keeps it out of the denominator instead of inventing a rounding.
    return (f if f in (0.0, 0.5, 1.0) else None), reason


def kendall_tau_b(x: list[int], y: list[int]) -> Optional[float]:
    """Kendall's tau-b, ties included, in plain Python.

    Implemented here rather than imported so the harness keeps running with no
    third-party packages. Verified to match scipy's `kendalltau(variant="b")` on
    the runs in this paper.
    """
    n = len(x)
    if n < 2:
        return None
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = (x[i] - x[j])
            b = (y[i] - y[j])
            s = (a > 0) - (a < 0)
            t = (b > 0) - (b < 0)
            if s * t > 0:
                conc += 1
            elif s * t < 0:
                disc += 1
    n0 = n * (n - 1) / 2

    def ties(v):
        c: dict = {}
        for e in v:
            c[e] = c.get(e, 0) + 1
        return sum(k * (k - 1) / 2 for k in c.values())

    d = math.sqrt((n0 - ties(x)) * (n0 - ties(y)))
    return (conc - disc) / d if d else None


def ordering_score(reference: list[str], answer_lines: list[str],
                   judge: Callable[[str], str]) -> Optional[float]:
    """Normalised rank correlation between the reference order and the answer's.

    Each answer line is matched to at most one unused reference item by asking the
    judge whether they describe the same event. Unmatched lines stay as
    themselves and therefore rank last, which is what pushes a granularity
    mismatch below chance rather than merely lowering it.

    ★Returns `tau_norm = (tau+1)/2`, the quantity the defining benchmark
     aggregates, not `tau_norm * f1`. Picking the more flattering of the two
     would make the number incomparable with everyone else's.
    """
    if not reference or not answer_lines:
        return None
    used, aligned = set(), []
    for line in answer_lines:
        hit = None
        for i, ref in enumerate(reference):
            if i in used:
                continue
            if "yes" in (judge(EQUIVALENCE_PROMPT.format(a=ref, b=line)) or "").lower():
                hit = i
                break
        if hit is None:
            aligned.append(line)
        else:
            aligned.append(reference[hit])
            used.add(hit)
    union = list(dict.fromkeys(reference + aligned))
    tie = len(union) + 1

    def ranks(seq):
        pos = {v: i + 1 for i, v in enumerate(seq)}
        return [pos.get(u, tie) for u in union]

    t = kendall_tau_b(ranks(reference), ranks(aligned))
    return (t + 1) / 2 if t is not None else 0.0


def grade_rows_rubric(rows: list["QuestionResult"], rubrics: dict, judge: Callable[[str], str],
                      concurrency: int, ordering_type: str = "") -> None:
    """Score each question as the mean of its judged rubric items.

    An item whose verdict cannot be read is left out of that question's
    denominator rather than counted as zero. A judge outage would otherwise
    arrive looking like a bad engine.
    """
    jobs, owner = [], []
    for i, r in enumerate(rows):
        if ordering_type and (r.type or "").replace("-", "_") == ordering_type:
            continue
        for item in rubrics.get(r.question_id, []):
            jobs.append((r.question, item, r.answer or ""))
            owner.append(i)

    def one(job):
        q, item, ans = job
        return read_rubric_score(judge(RUBRIC_JUDGE.format(
            question=q, rubric_item=item, llm_response=ans)))

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(one, jobs))

    got: dict = {}
    for i, (score, reason), job in zip(owner, results, jobs):
        got.setdefault(i, []).append((job[1], score, reason))
    for i, items in got.items():
        kept = [s for _, s, _ in items if s is not None]
        rows[i].score = (sum(kept) / len(kept)) if kept else None
        rows[i].rubric_items = [{"rubric": it, "score": s, "reason": w} for it, s, w in items]

    if not ordering_type:
        return
    order_rows = [(i, r) for i, r in enumerate(rows)
                  if (r.type or "").replace("-", "_") == ordering_type]

    def one_order(pair):
        i, r = pair
        lines = [x.strip() for x in str(r.answer or "").split("\n") if x.strip()]
        return i, ordering_score(rubrics.get(r.question_id, []), lines, judge)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for i, v in ex.map(one_order, order_rows):
            rows[i].score = v


def cmd_grade(a: argparse.Namespace) -> int:
    """Grade a recorded run. Retrieval and answering are not repeated."""
    run = json.load(open(a.run))
    rows = [QuestionResult(**r) for r in run["rows"]]

    bad = [r for r in rows if r.error is None and looks_like_non_answer(r.answer)]
    if bad and not a.force:
        print(f"{len(bad)} answers look like provider errors, so grading was refused.", file=sys.stderr)
        print(f"  first: {bad[0].answer[:120]!r}", file=sys.stderr)
        print("  Use --force to grade anyway.", file=sys.stderr)
        return 1

    # The rubric judge writes a JSON object; the binary one writes a word.
    judge, judge_label = build_judge(a, max_tokens=512 if a.protocol == "rubric" else 8)
    if a.protocol == "rubric":
        # The rubrics live with the questions, not with the run, so this protocol
        # needs the task file. Refusing here beats grading every question against
        # an empty rubric list and reporting a confident None.
        if not a.tasks:
            print("--protocol rubric needs --tasks (the rubrics live there)", file=sys.stderr)
            return 1
        tasks = json.load(open(a.tasks))
        tasks = tasks if isinstance(tasks, list) else (tasks.get("tasks") or tasks.get("rows"))
        rubrics = {t["question_id"]: t.get("rubric") or [] for t in tasks}
        missing = [r.question_id for r in rows if not rubrics.get(r.question_id)]
        if missing and not a.force:
            print(f"{len(missing)} questions have no rubric in {a.tasks}; refusing to grade.",
                  file=sys.stderr)
            print(f"  first: {missing[0]}", file=sys.stderr)
            return 1
        grade_rows_rubric(rows, rubrics, judge, a.concurrency, a.ordering_type)
    else:
        grade_rows(rows, judge, a.concurrency)

    run["rows"] = [asdict(r) for r in rows]
    run["summary"] = summarize(rows)
    run["config"]["judge"] = judge_label
    run["config"]["protocol"] = a.protocol
    run["graded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = a.out or a.run
    json.dump(run, open(out, "w"), ensure_ascii=False, indent=1)

    s = run["summary"]
    if a.protocol == "rubric":
        print(f"{run['model']}  {'valid' if s['valid'] else 'INVALID'}  "
              f"rubric score {s['rubric_score']}  scored {s['rubric_scored']}/{s['n']}  -> {out}")
    else:
        print(f"{run['model']}  {'valid' if s['valid'] else 'INVALID'}  "
              f"accuracy {s['accuracy']}  judged {s['judged']}/{s['n']}  -> {out}")
    return 0 if s["valid"] else 1


def cmd_report(a: argparse.Namespace) -> int:
    runs = [json.load(open(p)) for p in a.files]
    print(f"{'run':<28}{'valid':<7}{'accuracy':>10}{'tokens p50':>12}{'retrieval p50':>15}")
    for p, r in zip(a.files, runs):
        s = r["summary"]
        acc = f"{s['accuracy']*100:.1f}%" if s["accuracy"] is not None else "-"
        print(f"{os.path.basename(p):<28}{'yes' if s['valid'] else 'NO':<7}{acc:>10}"
              f"{str(s['context_tokens']['p50']):>12}{str(s['wos_ms']['p50']) + 'ms':>15}")

    # Every run is listed and the spread is printed with the mean. There is no
    # best-of path in this tool.
    accs = [r["summary"]["accuracy"] for r in runs if r["summary"]["accuracy"] is not None]
    invalid = [p for p, r in zip(a.files, runs) if not r["summary"]["valid"]]
    if len(accs) > 1:
        print(f"\nmean {statistics.fmean(accs)*100:.1f}%  sd {statistics.pstdev(accs)*100:.1f}"
              f"  runs {len(accs)}  spread {(max(accs)-min(accs))*100:.1f}")
    if invalid:
        print(f"\n{len(invalid)} run(s) marked invalid and excluded from any claim: "
              + ", ".join(os.path.basename(p) for p in invalid))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="wos-bench", description="Benchmark harness for the Wontopos memory engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    def judge_args(x: argparse.ArgumentParser) -> None:
        x.add_argument("--judge-url", default="", help="a second, independent endpoint")
        x.add_argument("--judge-model", default="")
        x.add_argument("--judge-key-env", default="JUDGE_KEY")
        x.add_argument("--judge-cmd", default="",
                       help="command reading the grading prompt on stdin, printing yes or no")
        x.add_argument("--judge-timeout", type=int, default=900)
        x.add_argument("--concurrency", type=int, default=4)
        x.add_argument("--force", action="store_true",
                       help="grade even if answers look like provider errors")

    r = sub.add_parser("run", help="run one pass")
    r.add_argument("--tasks", required=True)
    r.add_argument("--model", default="tablet-1", help="WOS model id, see listModels()")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=20, help="max_results per query")
    r.add_argument("--verify", type=int, default=0, help="memory verification passes, 0 to 3")
    r.add_argument("--dates", action="store_true",
                   help="show each memory's event_date to the reader")
    r.add_argument("--rel", action="store_true", help="show the similarity score")
    r.add_argument("--archive", action="store_true",
                   help="add a relative phrase (\"3 months ago\"); implies --dates. "
                        "Reference point is the newest event_date in the store")
    r.add_argument("--reader-prompt", default="",
                   help="file holding the answering prompt, with {memories} and "
                        "{question} placeholders. Use the benchmark's own prompt when "
                        "it ships one; the built-in default is ours")
    r.add_argument("--line-list", default="",
                   help="question type whose answer should be one item per line, in "
                        "order (e.g. event_ordering). Scorers that split an answer on "
                        "newlines read prose as many phantom items")
    r.add_argument("--lanes", choices=("both", "main", "self"), default="both",
                   help="which lane reaches the reader: both (SDK default), the "
                        "partner lane only, or the assistant's own words only")
    r.add_argument("--reader-url", default="", help="OpenAI-compatible chat endpoint")
    r.add_argument("--reader-model", default="")
    r.add_argument("--reader-key-env", default="READER_KEY")
    r.add_argument("--reader-max-tokens", type=int, default=1024)
    r.add_argument("--reader-cmd", default="",
                   help="command reading the prompt on stdin and printing the answer on stdout")
    r.add_argument("--reader-timeout", type=int, default=900)
    r.add_argument("--no-judge", action="store_true",
                   help="record answers without grading, to grade later with the grade command")
    r.add_argument("--tokenizer-cmd", default="",
                   help="command reading text on stdin and printing one integer")
    r.add_argument("--max", type=int, default=0, help="0 runs every task")
    r.add_argument("--base-url", default=os.environ.get("WOS_BASE_URL", ""))
    r.add_argument("--retries", type=int, default=4,
                   help="retries per retrieval on rate limits and transient errors")
    r.add_argument("--resume", action="store_true",
                   help="keep answers already in --out and only run what is missing")
    judge_args(r)
    r.set_defaults(fn=cmd_run)

    g = sub.add_parser("grade", help="grade a recorded run without repeating retrieval")
    g.add_argument("run")
    g.add_argument("--out", default="", help="defaults to overwriting the run file")
    g.add_argument("--protocol", choices=("binary", "rubric"), default="binary",
                   help="binary: one yes/no verdict per question. rubric: each question "
                        "carries rubric items judged 1.0/0.5/0.0 and scored by their mean. "
                        "Pick the one the benchmark defines; the other also produces a "
                        "number, and it is not the number anyone else published.")
    g.add_argument("--tasks", default="",
                   help="task file holding the rubrics (required by --protocol rubric)")
    g.add_argument("--ordering-type", default="",
                   help="question type scored by rank correlation instead of the rubric "
                        "mean, e.g. event_ordering. Leave empty to score every type by "
                        "rubric mean.")
    judge_args(g)   # --force lives here, and also waves through a missing rubric

    g.set_defaults(fn=cmd_grade)

    q = sub.add_parser("report", help="summarize runs")
    q.add_argument("files", nargs="+")
    q.set_defaults(fn=cmd_report)

    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
