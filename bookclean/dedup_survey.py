#!/usr/bin/env python3
"""Survey cross-book duplicated LINES corpus-wide (dedup targeting). CPU-only.
A line appearing in many distinct books is boilerplate (publisher promos, Also-by
lists, license text) — high training harm regardless of position. Measures how
much is dup-catchable and what it looks like, before building the filter."""
import json, sys, re, hashlib
from multiprocessing import Pool
from collections import Counter
sys.path.insert(0,'.')

def norm(l):
    l=re.sub(r'\s+',' ',l.strip().lower())
    l=re.sub(r'\d','#',l)                 # digit-class so ISBN/years/prices collapse
    return l
def hh(l): return hashlib.blake2b(norm(l).encode(),digest_size=8).digest()

def scan(task):
    start,end=task
    seen_local={}                        # hash -> set of book-ids (count distinct books)
    nb=0
    with open('/home/josef/data/book.v9.jsonl','rb') as f:
        f.seek(start)
        if start: f.readline()
        pos=f.tell()
        while pos<end:
            raw=f.readline()
            if not raw: break
            pos+=len(raw); nb+=1
            rec=json.loads(raw); t=rec['text']
            book_lines=set()
            for l in t.split('\n'):
                s=l.strip()
                if 15<=len(s)<=200:      # boilerplate-length lines
                    book_lines.add(hh(s))
            for k in book_lines:
                seen_local[k]=seen_local.get(k,0)+1   # +1 book
    return seen_local, nb

size=__import__('os').path.getsize('/home/josef/data/book.v9.jsonl')
n=24; bounds=[size*i//n for i in range(n+1)]
FREQ=Counter(); NB=0
with Pool(10) as pool:
    for fq,nb in pool.imap_unordered(scan, [(bounds[i],bounds[i+1]) for i in range(n)]):
        FREQ.update(fq); NB+=nb
print(f"scanned {NB:,} books; {len(FREQ):,} distinct normalized lines")
# distribution of cross-book multiplicity
buckets={'2-4':0,'5-19':0,'20-99':0,'100-999':0,'1000+':0}
dup_lines=0
for k,c in FREQ.items():
    if c<2: continue
    dup_lines+=1
    if c<5: buckets['2-4']+=1
    elif c<20: buckets['5-19']+=1
    elif c<100: buckets['20-99']+=1
    elif c<1000: buckets['100-999']+=1
    else: buckets['1000+']+=1
print(f"\nlines appearing in >=2 distinct books: {dup_lines:,}")
print("multiplicity buckets (distinct-book count):", buckets)
# save the high-multiplicity dup set (candidate filter list) + examples
top=FREQ.most_common(4000)
json.dump([[c] for _,c in top], open('reports/dedup_line_freq.json','w'))  # counts only (hashes not reversible)
print(f"\nmost-duplicated line multiplicities (top): {[c for _,c in top[:15]]}")
