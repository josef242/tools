#!/usr/bin/env python3
"""Pass 1: rebuild line->distinct-book-count. Pass 2: collect sample TEXT for a
few hashes in each multiplicity bucket, so we can eyeball boilerplate-vs-content
before setting the dedup threshold."""
import json, sys, re, hashlib, random
from multiprocessing import Pool
from collections import Counter
def norm(l):
    l=re.sub(r'\s+',' ',l.strip().lower()); return re.sub(r'\d','#',l)
def hh(l): return hashlib.blake2b(norm(l).encode(),digest_size=8).digest()

def count_scan(task):
    start,end=task; f=Counter()
    with open('/home/josef/data/book.v9.jsonl','rb') as fh:
        fh.seek(start);
        if start: fh.readline()
        pos=fh.tell()
        while pos<end:
            raw=fh.readline()
            if not raw: break
            pos+=len(raw); t=json.loads(raw)['text']; bl=set()
            for l in t.split('\n'):
                s=l.strip()
                if 15<=len(s)<=200: bl.add(hh(s))
            for k in bl: f[k]+=1
    return f

import os
size=os.path.getsize('/home/josef/data/book.v9.jsonl'); n=24
bounds=[size*i//n for i in range(n+1)]
FREQ=Counter()
with Pool(10) as pool:
    for fq in pool.imap_unordered(count_scan,[(bounds[i],bounds[i+1]) for i in range(n)]): FREQ.update(fq)
# pick sample hashes per bucket
buckets={'5-19':(5,20),'20-99':(20,100),'100-999':(100,1000),'1000+':(1000,10**9)}
want={}
for name,(lo,hi) in buckets.items():
    hs=[k for k,c in FREQ.items() if lo<=c<hi]
    random.Random(1).shuffle(hs)
    for k in hs[:40]: want[k]=name
want_set=set(want)
def text_scan(task):
    start,end=task; got={}
    with open('/home/josef/data/book.v9.jsonl','rb') as fh:
        fh.seek(start)
        if start: fh.readline()
        pos=fh.tell()
        while pos<end and len(got)<len(want_set):
            raw=fh.readline()
            if not raw: break
            pos+=len(raw); t=json.loads(raw)['text']
            for l in t.split('\n'):
                s=l.strip()
                if 15<=len(s)<=200:
                    k=hh(s)
                    if k in want_set and k not in got: got[k]=s[:90]
    return got
TEXT={}
with Pool(10) as pool:
    for g in pool.imap_unordered(text_scan,[(bounds[i],bounds[i+1]) for i in range(n)]):
        TEXT.update(g)
from collections import defaultdict
byb=defaultdict(list)
for k,name in want.items():
    if k in TEXT: byb[name].append((FREQ[k],TEXT[k]))
for name in ('1000+','100-999','20-99','5-19'):
    print(f"\n=== bucket {name} (sample) ===")
    for c,txt in sorted(byb[name],reverse=True)[:12]:
        print(f"  x{c:<6} {txt!r}")
