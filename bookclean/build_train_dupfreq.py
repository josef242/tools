#!/usr/bin/env python3
"""Line-frequency over the ACTUAL 18k training docs (book.v9.shuf.18k.jsonl = arm A's data).
Counts distinct-DOC occurrences per normalized line -> the memorization-relevant dup count
(how many training docs repeat each junk line). Used to stratify the junk eval into
{2-4, 5-20, 21-100, 100+} buckets for the capacity action-trigger. Output: reports/train_dupfreq.pkl"""
import json, re, hashlib, os, pickle
from multiprocessing import Pool
from collections import Counter
SRC = os.path.expanduser('~/data/book.v9.shuf.18k.jsonl')
def norm(l):
    l = re.sub(r'\s+', ' ', l.strip().lower()); return re.sub(r'\d', '#', l)
def hh(l): return hashlib.blake2b(norm(l).encode(), digest_size=8).digest()
def scan(task):
    start, end = task; f = Counter()
    with open(SRC, 'rb') as fh:
        fh.seek(start)
        if start: fh.readline()
        pos = fh.tell()
        while pos < end:
            raw = fh.readline()
            if not raw: break
            pos += len(raw)
            try: t = json.loads(raw)['text']
            except Exception: continue
            bl = set()
            for l in t.split('\n'):
                s = l.strip()
                if 12 <= len(s) <= 250: bl.add(hh(s))
            for k in bl: f[k] += 1
    return f
if __name__ == '__main__':
    size = os.path.getsize(SRC); n = 24
    bounds = [size * i // n for i in range(n + 1)]
    FREQ = Counter()
    with Pool(12) as pool:
        for fq in pool.imap_unordered(scan, [(bounds[i], bounds[i+1]) for i in range(n)]):
            FREQ.update(fq)
    DUP = {k: c for k, c in FREQ.items() if c >= 2}
    pickle.dump(DUP, open('reports/train_dupfreq.pkl', 'wb'), protocol=4)
    print(f"train dupfreq: {len(FREQ):,} distinct lines, {len(DUP):,} in >=2 training docs -> reports/train_dupfreq.pkl")
