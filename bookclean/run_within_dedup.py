#!/usr/bin/env python3
"""Produce book.v11.jsonl = v10 + within-book furniture/degenerate-line dedup.
Parallel shards (mirrors run_dedup.py). Tracks removed chars/lines, furniture vs
degenerate split, flagged books, per-book removal-rate, and top removed lines
(within-book harm-proxy)."""
import json, os, sys
from multiprocessing import Pool
from collections import Counter
sys.path.insert(0, '.')
SRC = os.path.expanduser('~/data/book.v10.jsonl')
DST = os.path.expanduser('~/data/book.v11.jsonl')
SH  = os.path.expanduser('~/data/.v11w_shards')

def work(task):
    from within_dedup_filter import within_dedup_book, norm
    wid, start, end = task
    st = Counter(); topline = Counter()
    with open(SRC, 'rb') as f, \
         open(f"{SH}/s{wid:03d}.jsonl", 'wb') as out, \
         open(f"{SH}/l{wid:03d}.jsonl", 'w') as lg:
        f.seek(start)
        if start:
            f.readline()
        pos = f.tell()
        while pos < end:
            raw = f.readline()
            if not raw:
                break
            pos += len(raw); st['books'] += 1
            rec = json.loads(raw)
            t2, info = within_dedup_book(rec['text'])
            if info.get('flagged'):
                st['flagged'] += 1
                lg.write(json.dumps({'flag': info['reason']}) + '\n')
                out.write(raw)
            elif info['removed']:
                st['touched'] += 1; st['lines'] += info['removed']
                st['chars'] += len(rec['text']) - len(t2)
                for kind, n in info.get('reasons', {}).items():
                    st['r_' + kind] += n
                # harm-proxy: which normalized lines got removed most
                kept = set(t2.split('\n'))
                for l in rec['text'].split('\n'):
                    if l and l not in kept:
                        topline[norm(l)[:60]] += 1
                rec['text'] = t2
                out.write(json.dumps(rec, ensure_ascii=True).encode() + b'\n')
            else:
                out.write(raw)
    # persist this shard's top removed lines
    with open(f"{SH}/t{wid:03d}.json", 'w') as tf:
        json.dump(topline.most_common(200), tf)
    return dict(st)

def main():
    os.makedirs(SH, exist_ok=True)
    size = os.path.getsize(SRC); n = 24
    bounds = [size * i // n for i in range(n + 1)]
    tot = Counter()
    with Pool(10) as pool:
        for i, s in enumerate(pool.imap_unordered(
                work, [(i, bounds[i], bounds[i + 1]) for i in range(n)])):
            tot.update(s)
            if (i + 1) % 6 == 0:
                print(f"[{i+1}/{n}] {dict(tot)}", flush=True)
    # stitch shards in order
    with open(DST, 'wb') as out:
        for i in range(n):
            with open(f"{SH}/s{i:03d}.jsonl", 'rb') as sh:
                while True:
                    c = sh.read(64_000_000)
                    if not c:
                        break
                    out.write(c)
            os.remove(f"{SH}/s{i:03d}.jsonl")
    # merge ledgers + harm-proxy
    with open('reports/within_dedup_ledger.jsonl', 'w') as L:
        for i in range(n):
            L.write(open(f"{SH}/l{i:03d}.jsonl").read()); os.remove(f"{SH}/l{i:03d}.jsonl")
    top = Counter()
    for i in range(n):
        for ln, c in json.load(open(f"{SH}/t{i:03d}.json")):
            top[ln] += c
        os.remove(f"{SH}/t{i:03d}.json")
    json.dump(top.most_common(300), open('reports/within_dedup_topremoved.json', 'w'), ensure_ascii=True)
    print(f"DONE {dict(tot)} -> {DST}")
    print("TOP REMOVED within-book lines:")
    for ln, c in top.most_common(30):
        print(f"  x{c:<7} {ln!r}")

if __name__ == '__main__':
    main()
