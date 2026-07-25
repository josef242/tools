#!/usr/bin/env python3
"""Produce book.v3.jsonl: books3 front/back matter cut (ledgered), pg19
passthrough. Parallel over byte ranges; shards concatenated in order."""
import json, os, sys
from multiprocessing import Pool
sys.path.insert(0, '.')
from books3_clean import clean_books3

SRC = os.path.expanduser('~/data/book.v2.jsonl')
DST = os.path.expanduser('~/data/book.v3.jsonl')
SHARD_DIR = os.path.expanduser('~/data/.v3_shards')

def work(task):
    wid, start, end = task
    shard = f"{SHARD_DIR}/shard_{wid:03d}.jsonl"
    led = f"{SHARD_DIR}/ledger_{wid:03d}.jsonl"
    stats = {'books': 0, 'head_cut': 0, 'tail_cut': 0, 'flagged': 0, 'chars': 0}
    with open(SRC, 'rb') as f, open(shard, 'wb') as out, open(led, 'w') as lg:
        f.seek(start)
        if start: f.readline()
        pos = f.tell()
        while pos < end:
            raw = f.readline()
            if not raw: break
            pos += len(raw)
            stats['books'] += 1
            if b'"short_book_title"' in raw[:300]:
                out.write(raw); continue
            rec = json.loads(raw)
            text = rec['text']
            t2, entry = clean_books3(text)
            if t2 is not text and len(t2) != len(text):
                stats['head_cut'] += entry.get('head', {}).get('action') == 'cut'
                stats['tail_cut'] += entry.get('tail', {}).get('action') == 'cut'
                stats['chars'] += len(text) - len(t2)
                rec['text'] = t2
                out.write(json.dumps(rec, ensure_ascii=True).encode() + b'\n')
            else:
                out.write(raw)
            if entry:
                stats['flagged'] += any(v.get('action') == 'flagged' for v in entry.values())
                lg.write(json.dumps({'pos': pos, **entry}) + '\n')
    return stats

def main():
    os.makedirs(SHARD_DIR, exist_ok=True)
    size = os.path.getsize(SRC)
    n = 48
    bounds = [size * i // n for i in range(n + 1)]
    tasks = [(i, bounds[i], bounds[i+1]) for i in range(n)]
    total = {'books': 0, 'head_cut': 0, 'tail_cut': 0, 'flagged': 0, 'chars': 0}
    with Pool(12) as pool:
        for i, st in enumerate(pool.imap_unordered(work, tasks)):
            for k in total: total[k] += st[k]
            print(f"[{i+1}/{n}] {total}", flush=True)
    # concat shards in order
    with open(DST, 'wb') as out:
        for i in range(n):
            with open(f"{SHARD_DIR}/shard_{i:03d}.jsonl", 'rb') as sh:
                while True:
                    chunk = sh.read(64_000_000)
                    if not chunk: break
                    out.write(chunk)
            os.remove(f"{SHARD_DIR}/shard_{i:03d}.jsonl")
    with open('reports/books3_clean_ledger.jsonl', 'w') as L:
        for i in range(n):
            L.write(open(f"{SHARD_DIR}/ledger_{i:03d}.jsonl").read())
            os.remove(f"{SHARD_DIR}/ledger_{i:03d}.jsonl")
    print(f"DONE {total} -> {DST}")

if __name__ == '__main__':
    main()
