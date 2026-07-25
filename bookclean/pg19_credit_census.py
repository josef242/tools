#!/usr/bin/env python3
"""Census of PG19 head credit-block shapes: for each PG19 book, examine the
first 3 paragraphs and classify credit patterns, collecting outliers."""
import json, re, os
from collections import Counter
from multiprocessing import Pool

PATH = os.path.expanduser("~/data/book.v1.jsonl")
PG19_LIMIT = 11_400_000_000

CREDIT = re.compile(r'(?i)^\s*(produced by|e-?text prepared by|transcribed from|'
                    r'this etext was prepared by|prepared by|scanned by|'
                    r'this file was produced from)')

def scan(task):
    start, end = task
    shapes = Counter()
    outliers = []
    with open(PATH, 'rb') as f:
        f.seek(start)
        if start: f.readline()
        pos = f.tell()
        while pos < end:
            raw = f.readline()
            if not raw: break
            pos += len(raw)
            if b'"short_book_title"' not in raw[:300]: continue
            rec = json.loads(raw)
            head = rec['text'][:2500]
            paras = [p.strip() for p in re.split(r'\n\s*\n', head) if p.strip()]
            if not paras:
                shapes['empty'] += 1; continue
            first = paras[0]
            if CREDIT.match(first):
                # normalize shape: which opener + has URL + line count
                opener = CREDIT.match(first).group(1).lower()
                shape = f"{opener}|url={'y' if re.search(r'(?i)www\.|https?://|\.net|\.org|\.com', first) else 'n'}|lines={min(len(first.splitlines()),5)}"
                shapes[shape] += 1
            elif re.search(r'(?i)proofread|gutenberg|internet archive|etext|e-text|distributed', first):
                shapes['credit-ish-nonstd'] += 1
                if len(outliers) < 40: outliers.append((rec['meta']['short_book_title'][:50], first[:160]))
            else:
                shapes['no-credit-first-para'] += 1
                if len(outliers) < 40 and shapes['no-credit-first-para'] % 97 == 1:
                    outliers.append((rec['meta']['short_book_title'][:50], first[:160]))
    return shapes, outliers

def main():
    size = min(os.path.getsize(PATH), PG19_LIMIT)
    n = 48
    bounds = [size*i//n for i in range(n+1)]
    shapes = Counter(); outliers = []
    with Pool(12) as pool:
        for s, o in pool.imap_unordered(scan, list(zip(bounds[:-1], bounds[1:]))):
            shapes.update(s); outliers.extend(o[:40-len(outliers)] if len(outliers)<40 else [])
    total = sum(shapes.values())
    print(f"PG19 books: {total:,}")
    for shape, c in shapes.most_common(30):
        print(f"  {c:>7,}  {shape}")
    json.dump({"shapes": dict(shapes), "outliers": outliers},
              open('reports/pg19_credit_census.json','w'), indent=1)
    print("\nsample outliers:")
    for t, p in outliers[:12]:
        print(f"  [{t}] {' '.join(p.split())[:130]}")

if __name__ == '__main__':
    main()
