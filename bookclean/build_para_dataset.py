#!/usr/bin/env python3
"""Paragraph-level dataset for the neural junk classifier.

GOLD (230 boundaries): tails -> paragraphs before content_end are CONTENT(0),
after are JUNK(1). heads -> before content_start are JUNK(1), after CONTENT(0).
SILVER (104k books3 heads): sentinel/prosefall-agreement boundaries, same rule.

Only paragraphs fully on one side of a boundary are used (straddlers dropped).
"""
import json, sys, random
sys.path.insert(0, '.')
from bakeoff import paragraphs

def tail_rows(tail, content_end):
    out = []
    for s, e, txt in paragraphs(tail):
        t = txt.strip()
        if len(t) < 15: continue
        if e <= content_end:   out.append((t, 0))     # content
        elif s >= content_end: out.append((t, 1))     # junk
    return out

def head_rows(head, content_start):
    out = []
    for s, e, txt in paragraphs(head):
        t = txt.strip()
        if len(t) < 15: continue
        if e <= content_start: out.append((t, 1))     # junk (front matter)
        elif s >= content_start: out.append((t, 0))   # content
    return out

rows = []
# --- gold tails (140) ---
TAILS = [('reports/gold_tails_books.json','reports/gold_tail_labels_hybrid.json'),
         ('reports/gold_tails_test_books.json','reports/gold_tail_test_hybrid.json'),
         ('reports/gold_tails_val3_books.json','reports/gold_tail_val3_hybrid.json')]
for bf, lf in TAILS:
    books = {b['offset']: b for b in json.load(open(bf))}
    for g in json.load(open(lf)):
        rows += [(t, y, 'gold_tail') for t, y in tail_rows(books[g['offset']]['tail'], g['content_end'])]
strat = json.load(open('reports/gold_tails_strat_books.json'))
for lab in json.load(open('reports/gold_tail_strat_labels.json')):
    rows += [(t, y, 'gold_strat') for t, y in tail_rows(strat[lab['index']]['tail'], lab['content_end'])]
# --- gold heads (90) ---
w2 = {b['offset']: b for b in json.load(open('reports/gold_wave2_books.json'))}
w3 = {b['offset']: b for b in json.load(open('reports/gold_wave3_books.json'))}
pilot = {b['offset']: b for b in json.load(open('reports/gold_pilot_books.json'))}
for g in json.load(open('reports/gold_labels.json')):
    src = w2.get(g['offset']) or w3.get(g['offset']) or pilot.get(g['offset'])
    cs = len(src['head']) if g['content_start'] == -1 else g['content_start']
    rows += [(t, y, 'gold_head') for t, y in head_rows(src['head'], cs)]
ngold = len(rows)

# --- silver heads (subsample) ---
rng = random.Random(5)
silver = [json.loads(l) for l in open('reports/silver_books3_heads.jsonl')]
silver = [r for r in silver if r['agree'] and 0 < r['sentinel'] < r['head_len']]
rng.shuffle(silver)
with open('/home/josef/data/book.v1.jsonl','rb') as f:
    for r in silver[:12000]:
        f.seek(r['offset'])
        rec = json.loads(f.readline())
        rows += [(t, y, 'silver') for t, y in head_rows(rec['text'][:12000], r['sentinel'])]

json.dump(rows, open('reports/para_dataset.json','w'))
from collections import Counter
c = Counter(src for _, _, src in rows)
y = Counter(lab for _, lab, _ in rows)
print(f"paragraph dataset: {len(rows):,} rows  (gold: {ngold:,})")
print(f"  by source: {dict(c)}")
print(f"  labels: content={y[0]:,}  junk={y[1]:,}")
