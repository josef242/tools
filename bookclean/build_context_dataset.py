#!/usr/bin/env python3
"""Context-aware paragraph dataset. Each example = target paragraph WITH its
neighbors and relative document position -- the information an isolated-paragraph
model lacks (Rook's registered prediction: anthology errors are neighborhood-
determined, not pretraining-determined)."""
import json, sys, random
sys.path.insert(0, '.')
from bakeoff import paragraphs

def fmt(paras, i, wlen):
    s, e, txt = paras[i]
    prev = paras[i-1][2].strip()[-160:] if i > 0 else ""
    nxt = paras[i+1][2].strip()[:160] if i+1 < len(paras) else ""
    pos = min(9, int(10 * s / max(wlen, 1)))          # decile position in window
    return f"P{pos} {prev} || {txt.strip()[:500]} || {nxt}"

def seq_rows(window, boundary, is_head):
    ps = [(s,e,t) for s,e,t in paragraphs(window) if len(t.strip()) >= 15]
    out = []
    for i,(s,e,t) in enumerate(ps):
        if is_head:  y = 1 if e <= boundary else (0 if s >= boundary else None)
        else:        y = 0 if e <= boundary else (1 if s >= boundary else None)
        if y is None: continue
        out.append((fmt(ps, i, len(window)), y))
    return out

rows = []
TAILS = [('reports/gold_tails_books.json','reports/gold_tail_labels_hybrid.json'),
         ('reports/gold_tails_test_books.json','reports/gold_tail_test_hybrid.json'),
         ('reports/gold_tails_val3_books.json','reports/gold_tail_val3_hybrid.json')]
for bf,lf in TAILS:
    books={b['offset']:b for b in json.load(open(bf))}
    for g in json.load(open(lf)):
        rows += [(t,y,'gold_tail') for t,y in seq_rows(books[g['offset']]['tail'], g['content_end'], False)]
strat=json.load(open('reports/gold_tails_strat_books.json'))
for lab in json.load(open('reports/gold_tail_strat_labels.json')):
    rows += [(t,y,'gold_strat') for t,y in seq_rows(strat[lab['index']]['tail'], lab['content_end'], False)]
w2={b['offset']:b for b in json.load(open('reports/gold_wave2_books.json'))}
w3={b['offset']:b for b in json.load(open('reports/gold_wave3_books.json'))}
pilot={b['offset']:b for b in json.load(open('reports/gold_pilot_books.json'))}
for g in json.load(open('reports/gold_labels.json')):
    src=w2.get(g['offset']) or w3.get(g['offset']) or pilot.get(g['offset'])
    cs=len(src['head']) if g['content_start']==-1 else g['content_start']
    rows += [(t,y,'gold_head') for t,y in seq_rows(src['head'], cs, True)]
ngold=len(rows)
rng=random.Random(5)
silver=[json.loads(l) for l in open('reports/silver_books3_heads.jsonl')]
silver=[r for r in silver if r['agree'] and 0<r['sentinel']<r['head_len']]
rng.shuffle(silver)
with open('/home/josef/data/book.v1.jsonl','rb') as f:
    for r in silver[:12000]:
        f.seek(r['offset']); rec=json.loads(f.readline())
        rows += [(t,y,'silver') for t,y in seq_rows(rec['text'][:12000], r['sentinel'], True)]
json.dump(rows, open('reports/para_context_dataset.json','w'))
from collections import Counter
print(f"context dataset: {len(rows):,} rows (gold {ngold:,})  labels={dict(Counter(y for _,y,_ in rows))}")
