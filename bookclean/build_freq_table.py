#!/usr/bin/env python3
"""Persist the line-frequency table (line-hash -> distinct-book-count) once, so
the dedup filter and all audits share it. CPU-only."""
import json, sys, re, hashlib, os, pickle
from multiprocessing import Pool
from collections import Counter
def norm(l):
    l=re.sub(r'\s+',' ',l.strip().lower()); return re.sub(r'\d','#',l)
def hh(l): return hashlib.blake2b(norm(l).encode(),digest_size=8).digest()
def scan(task):
    start,end=task; f=Counter()
    with open('/home/josef/data/book.v9.jsonl','rb') as fh:
        fh.seek(start)
        if start: fh.readline()
        pos=fh.tell()
        while pos<end:
            raw=fh.readline()
            if not raw: break
            pos+=len(raw); t=json.loads(raw)['text']; bl=set()
            for l in t.split('\n'):
                s=l.strip()
                if 12<=len(s)<=250: bl.add(hh(s))
            for k in bl: f[k]+=1
    return f
size=os.path.getsize('/home/josef/data/book.v9.jsonl'); n=24
bounds=[size*i//n for i in range(n+1)]
FREQ=Counter()
with Pool(10) as pool:
    for fq in pool.imap_unordered(scan,[(bounds[i],bounds[i+1]) for i in range(n)]): FREQ.update(fq)
# keep only lines in >=2 books (the rest can never be cross-book dup); shrinks the table hugely
DUP={k:c for k,c in FREQ.items() if c>=2}
with open('reports/line_freq.pkl','wb') as f: pickle.dump(DUP, f, protocol=4)
print(f"freq table persisted: {len(FREQ):,} distinct lines, {len(DUP):,} appearing in >=2 books -> reports/line_freq.pkl")
