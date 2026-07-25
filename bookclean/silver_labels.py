#!/usr/bin/env python3
"""Generate silver boundary labels for all books3 heads: sentinel + prosefall
predictions with agreement flags. Training data for the GPU classifier."""
import json, os, sys
from multiprocessing import Pool
sys.path.insert(0, '.')
from bakeoff import sentinel_head, prosefall_head

PATH = os.path.expanduser('~/data/book.v1.jsonl')
WINDOW = 12000

def scan(task):
    start, end = task
    rows = []
    with open(PATH, 'rb') as f:
        f.seek(start)
        if start: f.readline()
        pos = f.tell()
        idx_off = None
        while pos < end:
            raw = f.readline()
            if not raw: break
            line_start = pos
            pos += len(raw)
            if b'"short_book_title"' in raw[:300]:  # pg19: skip
                continue
            rec = json.loads(raw)
            head = rec['text'][:WINDOW]
            s = sentinel_head(head)
            p = prosefall_head(head)
            s = s if s is not None else len(head)
            p = p if p is not None else len(head)
            rows.append({'offset': line_start, 'sentinel': s, 'prosefall': p,
                         'agree': abs(s - p) <= 150, 'head_len': len(head)})
    return rows

def main():
    size = os.path.getsize(PATH)
    n = 48
    bounds = [size * i // n for i in range(n + 1)]
    total = agree = 0
    with Pool(12) as pool, open('reports/silver_books3_heads.jsonl', 'w') as out:
        for rows in pool.imap(scan, list(zip(bounds[:-1], bounds[1:]))):
            for r in rows:
                out.write(json.dumps(r) + '\n')
                total += 1
                agree += r['agree']
    print(f"silver labels: {total:,} books3 heads; sentinel/prosefall agree(<=150ch): "
          f"{agree:,} ({100*agree/max(total,1):.1f}%)")

if __name__ == '__main__':
    main()
