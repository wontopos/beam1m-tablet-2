#!/usr/bin/env python3
"""Load the BEAM corpus through the SDK, one store per conversation.

Each turn is written with `speaker` and `event_date`. Neither can be added
afterwards without re-ingesting everything.

`event_date` is the one to get right. `time_anchor` appears only on the turn
that opens a session and applies to the rest of it, so the value is carried
forward here. Drop it and every memory lands with the ingest date: the corpus
still loads, search still returns sensible text, and every temporal and ordering
question becomes unanswerable without anything reporting an error.

Usage:
    export WOS_API_KEY=...
    export BEAM_PARQUET=/path/to/1M.parquet
    export CONV=35                      # conversations to load
    python3 scripts/beam_ingest.py
"""
import os, time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

try:
    from wontopos import Client
except ImportError:
    raise SystemExit("The wontopos SDK is missing. Install it with: pip install wontopos")

PARQUET = os.environ.get("BEAM_PARQUET", "1M.parquet")   # the benchmark's own corpus file
BASE = os.environ.get("WOS_BASE_URL")                    # unset: the SDK's default endpoint
CONV = int(os.environ.get("CONV", "5"))                  # conversations to load
CONC = int(os.environ.get("CONC", "8"))                  # stores written in parallel
MAX_CHARS = 8000                                         # see the note at the end

MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"])}


def rfc3339(anchor):
    """'March-01-2024' -> '2024-03-01T00:00:00Z'. Unparseable input returns
    None, and the caller keeps the previous date rather than inventing one."""
    try:
        mon, day, year = str(anchor).split("-")
        return f"{int(year):04d}-{MONTHS[mon]:02d}-{int(day):02d}T00:00:00Z"
    except Exception:
        return None


client = Client(api_key=os.environ["WOS_API_KEY"], model="tablet-2", timeout=300,
                **({"base_url": BASE} if BASE else {}))

df = pd.read_parquet(PARQUET)
jobs, truncated = [], 0
for _, row in df.head(CONV).iterrows():
    store = f"beam1m-{row['conversation_id']}"
    cat = str(dict(row["conversation_seed"]).get("category") or "general").lower()
    cur = None
    for sess in row["chat"]:
        for t in sess:
            d = dict(t)
            if d.get("time_anchor"):
                cur = rfc3339(d["time_anchor"]) or cur
            content = str(d.get("content") or "").strip()
            if not content:
                continue
            md = {"category": cat}
            if d.get("role") == "assistant":
                md["speaker"] = "me"
            if cur:
                md["event_date"] = cur
            if len(content) > MAX_CHARS:
                truncated += 1
            jobs.append((store, content, md))

stores = sorted({s for s, _, _ in jobs})
print(f"{CONV} conversations - {len(stores)} stores - {len(jobs):,} turns "
      f"- {CONC} stores in parallel", flush=True)
if truncated:
    print(f"  {truncated} turns are longer than {MAX_CHARS:,} characters and will "
          f"be truncated on the way in", flush=True)

# Stores are explicit: a store must exist before anything can be written to it.
for s in stores:
    try:
        client.create_store(s)
    except Exception as e:
        print(f"  store {s}: {str(e)[:70]}", flush=True)

done = [0, 0]
t0 = time.time()


def put(j):
    store, content, md = j
    try:
        client.add(content[:MAX_CHARS], store, metadata=md)
        done[0] += 1
    except Exception as e:
        done[1] += 1
        if done[1] <= 3:
            print(f"  failed: {str(e)[:90]}", flush=True)
    n = done[0] + done[1]
    if n % 200 == 0:
        el = time.time() - t0
        rate = done[0] / max(el, .01)
        print(f"  {n:,}/{len(jobs):,} - {rate:.1f}/s - "
              f"{(len(jobs)-n)/max(rate,.01)/60:.0f} min left", flush=True)


# Sequential within a store, parallel across stores. Concurrent writes to one
# store are rejected, and since a conversation is a store, parallelising turns
# would send every worker at the same store and most would be refused. Grouping
# first costs nothing and removes the failure entirely.
by_store = {}
for j in jobs:
    by_store.setdefault(j[0], []).append(j)


def put_store(items):
    for j in items:
        put(j)


with ThreadPoolExecutor(max_workers=min(CONC, len(by_store))) as ex:
    list(ex.map(put_store, by_store.values()))

el = time.time() - t0
print(f"\nstored {done[0]:,}, failed {done[1]:,} - {el/60:.1f} min "
      f"- {done[0]/max(el,.01):.1f}/s")
print(f"stores: {', '.join(stores)}")
print("\nRun scripts/verify_corpus.py before scoring on this. An ingest that "
      "exits cleanly has reported what it sent, not what can be read back.")
