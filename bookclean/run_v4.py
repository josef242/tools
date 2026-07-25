#!/usr/bin/env python3
"""Produce book.v4.jsonl: mid-book noise removal (transcriber notes,
illustration/sidenote tags), corpus-wide, parallel shards."""
import json, os, sys
from multiprocessing import Pool
sys.path.insert(0, '.')
from midbook_clean import clean_midbook

SRC = os.path.expanduser('~/data/book.v3.jsonl')
DST = os.path.expanduser('~/data/book.v4.jsonl')
SHARD_DIR = os.path.expanduser('~/data/.v4_shards')

def work(task):
    wid, start, end = task
    shard = f"{SHARD_DIR}/shard_{wid:03d}.jsonl"
    led = f"{SHARD_DIR}/ledger_{wid:03d}.jsonl"
    from collections import Counter
    stats = Counter()
    with open(SRC, 'rb') as f, open(shard, 'wb') as out, open(led, 'w') as lg:
        f.seek(start)
        if start: f.readline()
        pos = f.tell()
        while pos < end:
            raw = f.readline()
            if not raw: break
            pos += len(raw)
            stats['books'] += 1
            if b'llustration' not in raw and b'ranscriber' not in raw and \
               b'idenote' not in raw:
                out.write(raw); continue  # fast path
            rec = json.loads(raw)
            t2, counts = clean_midbook(rec['text'])
            if counts:
                stats['touched'] += 1
                for k, v in counts.items(): stats[k] += v
                lg.write(json.dumps({'pos': pos, **counts}) + '\n')
                rec['text'] = t2
                out.write(json.dumps(rec, ensure_ascii=True).encode() + b'\n')
            else:
                out.write(raw)
    return dict(stats)

def main():
    os.makedirs(SHARD_DIR, exist_ok=True)
    size = os.path.getsize(SRC)
    n = 48
    bounds = [size * i // n for i in range(n + 1)]
    from collections import Counter
    total = Counter()
    with Pool(12) as pool:
        for i, st in enumerate(pool.imap_unordered(work, [(i, bounds[i], bounds[i+1]) for i in range(n)])):
            total.update(st)
            if (i+1) % 8 == 0: print(f"[{i+1}/{n}] {dict(total)}", flush=True)
    with open(DST, 'wb') as out:
        for i in range(n):
            with open(f"{SHARD_DIR}/shard_{i:03d}.jsonl", 'rb') as sh:
                while True:
                    chunk = sh.read(64_000_000)
                    if not chunk: break
                    out.write(chunk)
            os.remove(f"{SHARD_DIR}/shard_{i:03d}.jsonl")
    with open('reports/midbook_clean_ledger.jsonl', 'w') as L:
        for i in range(n):
            L.write(open(f"{SHARD_DIR}/ledger_{i:03d}.jsonl").read())
            os.remove(f"{SHARD_DIR}/ledger_{i:03d}.jsonl")
    print(f"DONE {dict(total)} -> {DST}")

if __name__ == '__main__':
    main()
