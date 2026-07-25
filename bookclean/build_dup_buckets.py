#!/usr/bin/env python3
"""Tag each junk-lexicon span with its TRAINING-SET dup bucket {1,2-4,5-20,21-100,100+},
by the max distinct-training-doc count among its lines (memorization is driven by the
most-repeated line). Parallel array to junk_lexicon.jsonl. Output: reports/junk_dup_bucket.json"""
import json, pickle, re, hashlib
from collections import Counter
def norm(l): l=re.sub(r'\s+',' ',l.strip().lower()); return re.sub(r'\d','#',l)
def hh(l): return hashlib.blake2b(norm(l).encode(),digest_size=8).digest()
FREQ = pickle.load(open('reports/train_dupfreq.pkl','rb'))
lex = [json.loads(l) for l in open('reports/junk_lexicon.jsonl')]
cls = json.load(open('reports/junk_class.json'))
def bucket(c):
    if c>=100: return '100+'
    if c>=21:  return '21-100'
    if c>=5:   return '5-20'
    if c>=2:   return '2-4'
    return '1'
buckets, dupcounts = [], []
for rec in lex:
    counts=[FREQ.get(hh(s), 1) for s in rec['text'].split('\n') if 12<=len(s.strip())<=250]
    dc = max(counts) if counts else 1
    dupcounts.append(dc); buckets.append(bucket(dc))
json.dump({'bucket':buckets,'dupcount':dupcounts}, open('reports/junk_dup_bucket.json','w'))
order=['1','2-4','5-20','21-100','100+']
cb=Counter(buckets[i] for i in range(len(buckets)) if str(cls[i]).upper().startswith('CROSS'))
wb=Counter(buckets[i] for i in range(len(buckets)) if str(cls[i]).upper().startswith('WITHIN'))
print("CROSS-book (n=%d):"%sum(cb.values()), {k:cb.get(k,0) for k in order})
print("WITHIN-book (n=%d):"%sum(wb.values()), {k:wb.get(k,0) for k in order})
