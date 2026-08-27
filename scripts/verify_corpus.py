#!/usr/bin/env python3
"""Read the corpus back before scoring on it.

An ingest that exits 0 has reported what it sent, not what can be retrieved.
This checks the second thing, by reading: every store exists, returns results,
and carries event dates. It exits non-zero if any of that fails, because a
corpus that loaded but cannot be read produces a score describing nothing.

Usage:
    export WOS_API_KEY=...
    python3 scripts/verify_corpus.py
"""
import os
from concurrent.futures import ThreadPoolExecutor

try:
    from wontopos import Client
except ImportError:
    raise SystemExit("The wontopos SDK is missing. Install it with: pip install wontopos")

STORES = ["beam1m-%d" % i for i in range(1, 36)]
PROBE = "what did we work on and when"

client = Client(api_key=os.environ["WOS_API_KEY"], model="tablet-2")


def check(store):
    row = {"store": store, "count": None, "returned": None,
           "dated": None, "earliest": None, "error": None}
    try:
        row["count"] = client.stats(user_id=store).get("total_memories")
        mems = client.search(PROBE, user_id=store, limit=20)
        row["returned"] = len(mems)
        dates = sorted(m["event_date"] for m in mems if m.get("event_date"))
        row["dated"] = "%d/%d" % (len(dates), len(mems))
        row["earliest"] = dates[0][:10] if dates else None
    except Exception as e:
        row["error"] = str(e)[:60]
    return row


with ThreadPoolExecutor(max_workers=8) as ex:
    rows = list(ex.map(check, STORES))

print("%-12s%10s%10s%10s%13s" % ("store", "memories", "returned", "dated", "earliest"))
for r in rows:
    print("%-12s%10s%10s%10s%13s" % (
        r["store"],
        format(r["count"], ",") if isinstance(r["count"], int) else (r["error"] or "-"),
        r["returned"] if r["returned"] is not None else "-",
        r["dated"] or "-",
        r["earliest"] or "-"))

# Anything listed here means the corpus is not ready to be scored on. Reporting
# it as a warning and continuing would hand the next stage a number that
# describes an ingest failure.
bad = []
for r in rows:
    if r["error"] or not isinstance(r["count"], int) or r["count"] == 0:
        bad.append("%s: nothing stored" % r["store"])
    elif not r["returned"]:
        bad.append("%s: stored but nothing retrievable" % r["store"])
    elif r["earliest"] is None:
        bad.append("%s: no event dates, time questions cannot be answered" % r["store"])

total = sum(r["count"] for r in rows if isinstance(r["count"], int))
print("\n%d stores, %s memories" % (len(rows), format(total, ",")))
if bad:
    print("\nnot ready:")
    for b in bad:
        print("  " + b)
    raise SystemExit(1)
print("no problems found")
