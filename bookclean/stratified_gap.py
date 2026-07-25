#!/usr/bin/env python3
"""Per-class x per-dup-bucket A-vs-B NLL gap, from existing eval JSONs' per-span arrays.
Establishes the 132M baseline and (later) the 500M point -> the capacity slope + action trigger.
Usage: stratified_gap.py <armA_eval.json> <armB_eval.json> [label]"""
import json, sys
from collections import defaultdict
cls = json.load(open('reports/junk_class.json'))
buck = json.load(open('reports/junk_dup_bucket.json'))['bucket']
def load(p): return json.load(open(p))['metric1_junkLL']['junk_nll_per_tok']
A, B = load(sys.argv[1]), load(sys.argv[2])
label = sys.argv[3] if len(sys.argv)>3 else ''
n = min(len(A), len(B), len(cls), len(buck))
order = ['1','2-4','5-20','21-100','100+']
for klass in ('CROSS','WITHIN'):
    print(f"\n=== {klass}-book  {label} ===")
    print(f"{'bucket':>8} {'n':>7} {'A_nll':>8} {'B_nll':>8} {'gap(B-A)':>9}")
    for bk in order:
        a=[A[i] for i in range(n) if str(cls[i]).upper().startswith(klass) and buck[i]==bk]
        b=[B[i] for i in range(n) if str(cls[i]).upper().startswith(klass) and buck[i]==bk]
        if not a: continue
        ma, mb = sum(a)/len(a), sum(b)/len(b)
        print(f"{bk:>8} {len(a):>7} {ma:>8.4f} {mb:>8.4f} {mb-ma:>+9.4f}")
