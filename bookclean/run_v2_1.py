#!/usr/bin/env python3
"""Produce book.v2.new.jsonl: PG19 head/tail boilerplate stripped (ledgered),
books3 records passed through byte-identical."""
import json, os, sys
sys.path.insert(0, '.')
from pg19_clean import clean_book

SRC = os.path.expanduser('~/data/book.v1.jsonl')
DST = os.path.expanduser('~/data/book.v2.new.jsonl')

heads = tails = books = pg19 = 0
chars_removed = 0
with open(SRC, 'rb') as src, open(DST, 'wb') as dst, \
     open('reports/pg19_clean_ledger.jsonl', 'w') as ledger:
    for idx, raw in enumerate(src):
        books += 1
        if b'"short_book_title"' not in raw[:300]:
            dst.write(raw)
            continue
        pg19 += 1
        rec = json.loads(raw)
        text = rec['text']
        cleaned, rem = clean_book(text)
        if rem['head'] or rem['tail']:
            heads += bool(rem['head'])
            tails += bool(rem['tail'])
            chars_removed += len(text) - len(cleaned)
            ledger.write(json.dumps({'book': idx, **rem}) + '\n')
            rec['text'] = cleaned
            dst.write(json.dumps(rec, ensure_ascii=True).encode() + b'\n')
        else:
            dst.write(raw)
        if pg19 % 5000 == 0:
            print(f"  {pg19:,} pg19 processed, heads={heads:,} tails={tails:,}", flush=True)
print(f"DONE: {books:,} books ({pg19:,} pg19); heads stripped {heads:,}, "
      f"tails stripped {tails:,}, {chars_removed/1e6:.1f}M chars removed -> {DST}")
