#!/usr/bin/env python3
"""Wide CPU-only mine for high-value gold candidates. No GPU (DistilBERT owns
it). Uses a cheap RULES-ONLY proxy for 'ambiguity': books where sentinel3's
tail paragraphs are a churn of keep/cut/neutral transitions (many boundaries)
rather than a clean content->junk split. These are the anthologies / multi-piece
collections the active-learning miner flagged as maximally informative."""
import json, sys, re
from multiprocessing import Pool
sys.path.insert(0, '.')
from bakeoff import paragraphs
from sentinel3 import classify_tail_para
from span_cutter import body_genre_is_riskly

ANTHOLOGY = re.compile(r'(?i)\b(?:selected|collected|complete|best (?:of|american)|'
                       r'anthology|companion to|monologues?|short stories|omnibus|reader)\b')

def scan(task):
    start, end = task
    out = []
    with open('/home/josef/data/book.v1.jsonl','rb') as f:
        f.seek(start)
        if start: f.readline()
        pos = f.tell()
        while pos < end:
            raw = f.readline()
            if not raw: break
            ls = pos; pos += len(raw)
            if b'"short_book_title"' in raw[:300] or len(raw) < 40000: continue
            rec = json.loads(raw)
            t = rec['text']; tail = t[-10000:]
            ps = [x for _,_,x in paragraphs(tail) if len(x.strip())>=15]
            if len(ps) < 6: continue
            k = [classify_tail_para(x) for x in ps]
            # ambiguity proxy: many keep<->cut transitions = non-monotone back matter
            trans = sum(1 for a,b in zip(k,k[1:]) if a!=b and 'neutral' not in (a,b))
            keeps = k.count('keep'); cuts = k.count('cut')
            churn = trans if (keeps and cuts) else 0
            title = rec['meta'].get('title','?')[:50]
            anth = bool(ANTHOLOGY.search(title)) or bool(ANTHOLOGY.search(t[:2000]))
            score = churn + (3 if anth else 0) + (2 if body_genre_is_riskly(t) else 0)
            if score >= 4:
                out.append([score, ls, title, anth])
    return out

size = 107_837_638_842
n = 24
bounds = [size*i//n for i in range(n+1)]
allc = []
with Pool(11) as pool:   # leave a core for DistilBERT's dataloader
    for i, r in enumerate(pool.imap_unordered(scan, list(zip(bounds[:-1], bounds[1:])))):
        allc.extend(r)
        print(f"[{i+1}/{n}] cumulative candidates: {len(allc):,}", flush=True)
allc.sort(reverse=True)
seen = set(); dedup = []
for s, off, title, anth in allc:
    if off in seen: continue
    seen.add(off); dedup.append([s, off, title, anth])
json.dump(dedup[:400], open('reports/wide_candidates.json','w'))
print(f"\nfound {len(dedup):,} high-ambiguity candidates; saved top 400")
print("score histogram:", [sum(1 for c in dedup if c[0]>=t) for t in (4,6,8,12,16)])
print("\ntop 15:")
for s, off, title, anth in dedup[:15]:
    print(f"  {s:>3} {'[anth]' if anth else '     '} {title}")
