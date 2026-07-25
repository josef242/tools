#!/usr/bin/env python3
"""Audit the bibliography guard: stream v9 vs v9.1 (same doc order, same count), find docs whose
text changed, quantify rescued chars, and sample the rescued spans for eyeballing."""
import json, os, difflib, random
A=os.path.expanduser('~/data/book.v9.jsonl'); B=os.path.expanduser('~/data/book.v9.1.jsonl')
changed=0; rescued=0; samples=[]; rng=random.Random(3); n=0
with open(A) as fa, open(B) as fb:
    for la, lb in zip(fa, fb):
        n+=1
        if len(la)==len(lb) and la==lb: continue
        try:
            ta=json.loads(la)['text']; tb=json.loads(lb)['text']
        except Exception: continue
        if ta==tb: continue
        changed+=1; rescued+=len(tb)-len(ta)
        if len(samples)<12 and rng.random()<0.02:
            # the rescued text = what's in v9.1 but not v9 (guard put it back)
            sm=difflib.SequenceMatcher(None, ta.split('\n'), tb.split('\n'), autojunk=False)
            add=[]
            for tag,i1,i2,j1,j2 in sm.get_opcodes():
                if tag in ('insert','replace'):
                    add += [l for l in tb.split('\n')[j1:j2] if l.strip()]
            if add: samples.append(add[:6])
print(f"docs scanned      : {n:,}")
print(f"docs CHANGED      : {changed:,}  ({changed/n*100:.2f}%)")
print(f"chars rescued     : {rescued:,}  ({rescued/1048576:.1f} MB)")
print(f"avg per changed   : {rescued/max(changed,1):,.0f} chars")
print("\n=== sampled RESCUED spans (should read as bibliography/citations) ===")
for i,s in enumerate(samples[:8]):
    print(f"\n--- doc sample {i+1} ---")
    for l in s[:4]: print("   ", l.strip()[:150])
