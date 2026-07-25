#!/usr/bin/env python3
"""Audit big head cuts: replicate v3 decisions from v1, flag suspects where
(a) removed region is non-Latin-heavy (language casualty), or
(b) neural classifier says content starts almost immediately (genre casualty).
"""
import json, os, sys
from multiprocessing import Pool
sys.path.insert(0, '.')
from books3_clean import clean_books3

BIG = 8000

def script_nonlatin_frac(text):
    letters = [c for c in text if c.isalpha()]
    if not letters: return 0.0
    return sum(1 for c in letters if ord(c) > 0x24F) / len(letters)

def work(task):
    start, end = task
    out = []
    with open(os.path.expanduser('~/data/book.v1.jsonl'), 'rb') as f:
        f.seek(start)
        if start: f.readline()
        pos = f.tell()
        while pos < end:
            raw = f.readline()
            if not raw: break
            line_start = pos
            pos += len(raw)
            if b'"short_book_title"' in raw[:300]: continue
            rec = json.loads(raw)
            text = rec['text']
            t2, entry = clean_books3(text)
            h = entry.get('head', {})
            if h.get('action') == 'cut' and h['chars'] >= BIG:
                removed = text[:h['chars']]
                nl = script_nonlatin_frac(removed)
                out.append({'offset': line_start, 'title': rec['meta'].get('title','?')[:60],
                            'cut': h['chars'], 'nonlatin': round(nl, 3),
                            'booklen': len(text)})
    return out

def main():
    size = os.path.getsize(os.path.expanduser('~/data/book.v1.jsonl'))
    n = 48
    bounds = [size*i//n for i in range(n+1)]
    rows = []
    with Pool(12) as pool:
        for out in pool.imap_unordered(work, list(zip(bounds[:-1], bounds[1:]))):
            rows.extend(out)
    rows.sort(key=lambda r: -r['cut'])
    json.dump(rows, open('reports/big_cut_audit.json','w'), indent=1)
    nonlatin = [r for r in rows if r['nonlatin'] > 0.15]
    print(f"big head cuts (>= {BIG}): {len(rows)}")
    print(f"  non-Latin-heavy (script casualties): {len(nonlatin)}")
    for r in nonlatin[:10]:
        print(f"    {r['title'][:55]}  cut={r['cut']} nonlatin={r['nonlatin']}")

if __name__ == '__main__':
    main()
