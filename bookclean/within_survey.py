#!/usr/bin/env python3
"""Within-book line-repetition survey. For each book, count how many times each
normalized line repeats WITHIN that book. Bucket by repeat count + capture
samples, to design the within-book dedup filter with Rook's protected classes
(drama speaker tags, poetry refrains, choruses) and the count-threshold insight
(chapter headings ~tens, page furniture ~hundreds)."""
import json, sys, re, random
from multiprocessing import Pool
from collections import Counter
def norm(l):
    l=re.sub(r'\s+',' ',l.strip().lower()); return re.sub(r'\d','#',l)
def scan(task):
    start,end=task; buckets=Counter(); samples={b:[] for b in ('3-9','10-49','50-199','200+')}
    rng=random.Random(start)
    with open('/home/josef/data/book.v10.jsonl','rb') as f:
        f.seek(start)
        if start: f.readline()
        pos=f.tell()
        while pos<end:
            raw=f.readline()
            if not raw: break
            pos+=len(raw); t=json.loads(raw)['text']
            c=Counter(norm(l) for l in t.split('\n') if 3<=len(l.strip())<=120)
            for ln,cnt in c.items():
                if cnt<3: continue
                b='3-9' if cnt<10 else '10-49' if cnt<50 else '50-199' if cnt<200 else '200+'
                buckets[b]+=1
                if len(samples[b])<12 and rng.random()<0.02:
                    samples[b].append((cnt, ln[:60]))
    return buckets, samples
import os
size=os.path.getsize('/home/josef/data/book.v10.jsonl'); n=24
bounds=[size*i//n for i in range(n+1)]
B=Counter(); S={b:[] for b in ('3-9','10-49','50-199','200+')}
with Pool(10) as pool:
    for bk,sm in pool.imap_unordered(scan,[(bounds[i],bounds[i+1]) for i in range(0,n,2)]):  # 12 shards
        B.update(bk)
        for b in S:
            if len(S[b])<20: S[b].extend(sm[b][:20-len(S[b])])
print("=== WITHIN-BOOK repeated lines (sampled ~50% corpus) — distinct (book,line) pairs by repeat count ===")
for b in ('3-9','10-49','50-199','200+'): print(f"  {b:<8} {B[b]:,}")
print("\nsamples by bucket (repeat-count, line):")
for b in ('200+','50-199','10-49','3-9'):
    print(f"\n  {b}:")
    for cnt,ln in sorted(S[b],reverse=True)[:10]: print(f"    x{cnt:<5} {ln!r}")
