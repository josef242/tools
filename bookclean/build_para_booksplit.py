#!/usr/bin/env python3
"""Book-level split (fixes Rook's paragraph-leakage catch): entire books' worth
of paragraphs go to train OR val, never split. Held-out val = 2 whole gold tail
sets (test + val3) + a slice of gold heads, none of whose paragraphs (or their
context neighbors) appear in training."""
import json, sys, random
sys.path.insert(0,'.')
from bakeoff import paragraphs
from build_context_dataset import seq_rows   # reuse context formatter

def load_tail(bf, lf):
    strat='strat' in bf
    books={(b['offset'] if not strat else i):b for i,b in enumerate(json.load(open(bf)))}
    rows=[]
    for g in json.load(open(lf)):
        key=g['index'] if strat else g['offset']
        rows += [(t,y) for t,y in seq_rows(books[key]['tail'], g['content_end'], False)]
    return rows

# VAL = whole held-out books (test + val3 tail sets) -- never in train
val  = load_tail('reports/gold_tails_test_books.json','reports/gold_tail_test_hybrid.json')
val += load_tail('reports/gold_tails_val3_books.json','reports/gold_tail_val3_hybrid.json')
# TRAIN gold = the OTHER whole books (tune + strat tails + all heads)
tr  = load_tail('reports/gold_tails_books.json','reports/gold_tail_labels_hybrid.json')
tr += load_tail('reports/gold_tails_strat_books.json','reports/gold_tail_strat_labels.json')
w2={b['offset']:b for b in json.load(open('reports/gold_wave2_books.json'))}
w3={b['offset']:b for b in json.load(open('reports/gold_wave3_books.json'))}
pilot={b['offset']:b for b in json.load(open('reports/gold_pilot_books.json'))}
for g in json.load(open('reports/gold_labels.json')):
    src=w2.get(g['offset']) or w3.get(g['offset']) or pilot.get(g['offset'])
    cs=len(src['head']) if g['content_start']==-1 else g['content_start']
    tr += [(t,y) for t,y in seq_rows(src['head'], cs, True)]
tr=[(t,y,'gold') for t,y in tr]; val=[(t,y,'gold_val') for t,y in val]
# silver heads (whole books, disjoint from gold by construction)
rng=random.Random(5)
silver=[json.loads(l) for l in open('reports/silver_books3_heads.jsonl')]
silver=[r for r in silver if r['agree'] and 0<r['sentinel']<r['head_len']]
rng.shuffle(silver)
with open('/home/josef/data/book.v1.jsonl','rb') as f:
    for r in silver[:12000]:
        f.seek(r['offset']); rec=json.loads(f.readline())
        tr += [(t,y,'silver') for t,y in seq_rows(rec['text'][:12000], r['sentinel'], True)]
json.dump({'train':tr,'val':val}, open('reports/para_booksplit.json','w'))
print(f"BOOK-LEVEL split: train={len(tr):,} (gold+silver)  val={len(val):,} (whole held-out books only)")
