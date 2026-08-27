#!/usr/bin/env python3
"""Build the task file from the benchmark's own corpus.

Ground truth is stored under a different field name per question type, so each
known name is tried in order. If none matches, the build fails rather than
dropping the question: a task file with empty ground truth still loads, runs and
grades, and the judge then scores every answer against nothing.

The rubric is the benchmark's own scoring criteria, carried through so the judge
can score item by item.

Usage:
    export BEAM_PARQUET=/path/to/1M.parquet
    python3 scripts/build_tasks.py
"""
import os
import ast, json, sys

import pandas as pd

PARQUET = os.environ.get("BEAM_PARQUET", "1M.parquet")
OUT = os.environ.get("OUT", "beam_tasks_v2.json")

# Tried in order; the first one present wins. Extend this list rather than
# special-casing a type, and keep the failure below loud.
ANSWER_KEYS = ("ideal_response", "ideal_answer", "answer",
               "expected_compliance", "ideal_summary")

df = pd.read_parquet(PARQUET)
tasks, missing = [], []

for _, row in df.iterrows():
    cid = str(row["conversation_id"])
    probes = ast.literal_eval(row["probing_questions"])
    for typ, items in probes.items():
        for i, it in enumerate(items):
            gt = next((it[k] for k in ANSWER_KEYS if it.get(k)), None)
            qid = "%s_%s_%d" % (cid, typ, i)
            if gt is None:
                missing.append((qid, sorted(it.keys())))
                continue
            rubric = it.get("rubric") or []
            tasks.append({
                "question_id": qid,
                "user_id": "beam1m-%s" % cid,
                "question": it["question"],
                "gt": str(gt),
                "rubric": [str(x) for x in rubric] if isinstance(rubric, list) else [str(rubric)],
                "type": typ,
                "difficulty": it.get("difficulty"),
            })

if missing:
    print("%d questions have no ground truth under any known field name.\n"
          "Add the missing name to ANSWER_KEYS rather than skipping them." % len(missing))
    for qid, ks in missing[:5]:
        print("  %s  fields present: %s" % (qid, ks))
    sys.exit(1)

json.dump(tasks, open(OUT, "w"), ensure_ascii=False)

import collections
print("%d questions across %d stores -> %s" % (
    len(tasks), len({t["user_id"] for t in tasks}), OUT))
print("with ground truth: %d   with a rubric: %d" % (
    sum(1 for t in tasks if t["gt"].strip()),
    sum(1 for t in tasks if t["rubric"])))
print("by type:", dict(collections.Counter(t["type"] for t in tasks)))
print()
t = tasks[70]
print("sample:", t["question_id"])
print("  question:", t["question"][:130])
print("  truth   :", t["gt"][:160])
print("  rubric  :", (t["rubric"][0][:130] if t["rubric"] else "(none)"))
