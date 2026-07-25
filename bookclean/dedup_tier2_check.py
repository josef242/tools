#!/usr/bin/env python3
"""Test Josef's hypothesis: does a LINE-LENGTH floor separate legitimately-
repeated content (dialogue/ingredients/headings) from cuttable boilerplate?
Capture all lines in >=100 books, cross-tabulate length x strong-pattern-match."""
import json, sys, re, hashlib
from multiprocessing import Pool
from collections import Counter, defaultdict
def norm(l):
    l=re.sub(r'\s+',' ',l.strip().lower()); return re.sub(r'\d','#',l)
def hh(l): return hashlib.blake2b(norm(l).encode(),digest_size=8).digest()

# STRONG boilerplate patterns (never appear in real content)
STRONG=re.compile(r'(?i)\bISBN\b|(?:©|copyright ©|all rights reserved|library of congress|'
    r'cataloguing?[- ]in[- ]publication|catalogue record|first (?:edition|printing|published)|'
    r'printed (?:and bound )?in|penguin|harpercollins|macmillan|hachette|simon & schuster|'
    r'random house|##\s*contents|copyright page|this ebook|reproduced|retrieval system)')

def count_scan(task):
    start,end=task; f=Counter()
    with open('/home/josef/data/book.v9.jsonl','rb') as fh:
        fh.seek(start)
        if start: fh.readline()
        pos=fh.tell()
        while pos<end:
            raw=fh.readline()
            if not raw: break
            pos+=len(raw); t=json.loads(raw)['text']; bl=set()
            for l in t.split('\n'):
                s=l.strip()
                if 15<=len(s)<=200: bl.add(hh(s))
            for k in bl: f[k]+=1
    return f
import os
size=os.path.getsize('/home/josef/data/book.v9.jsonl'); n=24
bounds=[size*i//n for i in range(n+1)]
FREQ=Counter()
with Pool(10) as pool:
    for fq in pool.imap_unordered(count_scan,[(bounds[i],bounds[i+1]) for i in range(n)]): FREQ.update(fq)
hot={k for k,c in FREQ.items() if c>=100}
def text_scan(task):
    start,end=task; got={}
    with open('/home/josef/data/book.v9.jsonl','rb') as fh:
        fh.seek(start)
        if start: fh.readline()
        pos=fh.tell()
        while pos<end:
            raw=fh.readline()
            if not raw: break
            pos+=len(raw); t=json.loads(raw)['text']
            for l in t.split('\n'):
                s=l.strip()
                if 15<=len(s)<=200:
                    k=hh(s)
                    if k in hot and k not in got: got[k]=s
    return got
TEXT={}
with Pool(10) as pool:
    for g in pool.imap_unordered(text_scan,[(bounds[i],bounds[i+1]) for i in range(n)]): TEXT.update(g)

# cross-tab: length bucket x strong-pattern. "risk" = pattern-NEGATIVE (likely content)
lb=[(0,25),(25,40),(40,60),(60,999)]
tab=defaultdict(lambda:[0,0])   # lenbucket -> [pattern_pos, pattern_neg]
risk_examples=defaultdict(list)
for k,txt in TEXT.items():
    L=len(txt); pat=bool(STRONG.search(txt))
    for lo,hi in lb:
        if lo<=L<hi:
            tab[(lo,hi)][0 if pat else 1]+=1
            if not pat and L>=40:
                risk_examples["long"].append(f"x{FREQ[k]} {txt[:70]!r}")
            break

alllong=risk_examples["long"]
print(f"long (>=40 char) no-pattern high-mult(>=100) lines: {len(alllong)}")
# content heuristics: dialogue (starts with quote), sentence-final punct + no address/url/legal markers
import re as _r
BOILER=_r.compile(r"(?i)publish|edition|manufactured|printed|www\.|http|\beula\b|copyright|"
    r"reserved|\binc\b|\bltd\b|\bllc\b|press\b|books\b|street|avenue|suite|p\.o\.|"
    r"\bfl\b|\bny\b|newsletter|sign up|videos|catalog|distribut|reproduc|\bisbn\b|"
    r"library|available|companies|permission|trademark|patent|division of|imprint")
suspect=[e for e in alllong if not BOILER.search(e)]
print(f"\nof those, NOT matching extended-boilerplate regex (CONTENT SUSPECTS): {len(suspect)}")
for e in sorted(suspect)[:40]: print("  ", e[:95])
