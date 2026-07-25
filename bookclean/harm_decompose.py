#!/usr/bin/env python3
"""Decompose the unreachable tail junk by HARM category (Rook's decisive ask):
  (a) cross-book DUPLICATED boilerplate  -- HIGH harm, caught FREE by dedup
  (b) one-off DATA STRUCTURE (index/TOC) -- moderate harm, needs multi-span
  (c) one-off FLUENT PROSE (previews)    -- near-zero harm, maybe not worth cutting
Phase 1 (CPU): paragraph-frequency table over a corpus sample.
Phase 2: attribute each unreachable-junk paragraph in the 140 gold tails."""
import json, sys, re, hashlib
from multiprocessing import Pool
sys.path.insert(0,'.')
from bakeoff import paragraphs
from sentinel3 import tail_boundary

def norm(t):
    return re.sub(r'\s+', ' ', t.strip().lower())[:400]
def h(t):
    return hashlib.blake2b(norm(t).encode(), digest_size=8).digest()

def scan(task):
    start, end = task
    freq = {}
    with open('/home/josef/data/book.v1.jsonl','rb') as f:
        f.seek(start)
        if start: f.readline()
        pos=f.tell()
        while pos < end:
            raw=f.readline()
            if not raw: break
            pos+=len(raw)
            if b'"short_book_title"' in raw[:300] or len(raw)<20000: continue
            rec=json.loads(raw); t=rec['text']
            for zone in (t[:8000], t[-8000:]):
                for _,_,p in paragraphs(zone):
                    pp=p.strip()
                    if len(pp)>=25:
                        k=h(pp); freq[k]=freq.get(k,0)+1
    return freq

# phase 1: frequency table over ~1/4 of the corpus
size=107_837_638_842; n=24
bounds=[size*i//n for i in range(n+1)]
tasks=[(bounds[i],bounds[i+1]) for i in range(0,n,4)]  # scan 6 of 24 shards (~25%)
FREQ={}
with Pool(6) as pool:
    for fq in pool.imap_unordered(scan, tasks):
        for k,v in fq.items(): FREQ[k]=FREQ.get(k,0)+v
print(f"freq table: {len(FREQ):,} distinct paragraphs from ~25% of corpus")

# phase 2: decompose unreachable junk in gold tails
STRUCT=re.compile(r'^[^.!?\n]{2,60}?[,\s]\s*\d{1,4}(?:[-–,\s]\d{1,4}){0,20}\s*$|'
                  r'^#{0,3}\s*(?:index|contents|also by)', re.I|re.M)
SETS=[('reports/gold_tails_books.json','reports/gold_tail_labels_hybrid.json'),
      ('reports/gold_tails_test_books.json','reports/gold_tail_test_hybrid.json'),
      ('reports/gold_tails_val3_books.json','reports/gold_tail_val3_hybrid.json'),
      ('reports/gold_tails_strat_books.json','reports/gold_tail_strat_labels.json')]
cat={'a_dup':0,'b_struct':0,'c_prose':0}
for bf,lf in SETS:
    strat='strat' in bf
    books={(b['offset'] if not strat else i):b for i,b in enumerate(json.load(open(bf)))}
    for g in json.load(open(lf)):
        key=g['index'] if strat else g['offset']
        tail=books[key]['tail']; L=len(tail); truth=g['content_end']
        rb=tail_boundary(tail)                       # rules keep [0,rb)
        lo=truth; hi=min(rb,L)                        # unreachable junk = [content_end, rules_boundary)
        if hi<=lo: continue
        for s,e,p in paragraphs(tail):
            if e<=lo or s>=hi: continue
            pp=p.strip()
            if len(pp)<25: continue
            c=e-s
            if FREQ.get(h(pp),0)>=2:            cat['a_dup']+=c
            elif STRUCT.search(pp) or len(pp)<120: cat['b_struct']+=c
            else:                               cat['c_prose']+=c
tot=sum(cat.values()) or 1
print("\n=== HARM DECOMPOSITION of unreachable tail junk (char-weighted) ===")
print(f"  (a) cross-book DUPLICATED  {cat['a_dup']:>8,}  {100*cat['a_dup']/tot:4.1f}%  <- FREE via dedup, HIGH harm")
print(f"  (b) one-off DATA-STRUCTURE {cat['b_struct']:>8,}  {100*cat['b_struct']/tot:4.1f}%  <- needs multi-span, moderate harm")
print(f"  (c) one-off FLUENT PROSE   {cat['c_prose']:>8,}  {100*cat['c_prose']/tot:4.1f}%  <- near-zero harm, maybe skip")
