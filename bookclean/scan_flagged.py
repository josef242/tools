#!/usr/bin/env python3
"""Failure-genre discovery: random-sample v10, run within_dedup_filter, and for
each book that trips the CIRCUIT BREAKER (>30% lines would be removed) capture its
title + first lines + removal fraction. The 496 flagged books are furniture-dominated
by definition; this shows WHICH genres (expected: data tables / indexes / catalogs /
OCR dumps; anything else = a new stratum to handle)."""
import json, os, sys, random
sys.path.insert(0, '.')
from within_dedup_filter import within_dedup_book, norm, is_protected, is_furniture, _degenerate
from collections import Counter
V10 = os.path.expanduser('~/data/book.v10.jsonl')
WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 40
size = os.path.getsize(V10)
rng = random.Random(99)
found = []; scanned = 0
while len(found) < WANT and scanned < 60000:
    with open(V10, 'rb') as f:
        f.seek(rng.randrange(size)); f.readline()
        raw = f.readline()
    scanned += 1
    if not raw:
        continue
    try:
        rec = json.loads(raw)
    except Exception:
        continue
    txt = rec['text']
    _, info = within_dedup_book(txt)
    if not info.get('flagged'):
        continue
    # recompute what WOULD have been removed, to characterize the furniture
    lines = txt.split('\n'); ne = [l for l in lines if l.strip()]
    cnt = Counter(norm(l) for l in ne)
    would = []
    for l in ne:
        if _degenerate(l):
            would.append(l); continue
        nl = norm(l); c = cnt[nl]
        if c >= 5 and not is_protected(l, nl) and is_furniture(l, nl, c):
            would.append(l)
    title = rec.get('title') or rec.get('meta', {}).get('title') or '(no title)'
    top_furn = Counter(norm(l)[:40] for l in would).most_common(5)
    found.append({
        'title': str(title)[:70], 'reason': info['reason'],
        'first_lines': [l.strip()[:60] for l in ne[:6]],
        'top_furniture': top_furn,
    })
    print(f"[{len(found)}] {info['reason']:>16}  {str(title)[:60]!r}", flush=True)

print(f"\n=== scanned {scanned} books, found {len(found)} flagged ===\n")
for r in found:
    print(f"TITLE: {r['title']}   ({r['reason']})")
    print(f"  first lines: {r['first_lines']}")
    print(f"  top would-cut furniture: {[(f, c) for f, c in r['top_furniture']]}")
    print()
json.dump(found, open('reports/flagged_genres.json', 'w'), ensure_ascii=True, indent=1)
