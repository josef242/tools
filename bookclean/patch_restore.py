#!/usr/bin/env python3
"""Surgical restore: suspects get their v1 text back. v5 -> v6."""
import json, os

suspects = json.load(open('reports/restore_suspects.json'))
restore_offsets = {s['offset'] for s in suspects if s['verdict'] == 'restore'}
# map v1 offsets -> line numbers
lines = set()
off2line = {}
for line in open('reports/book_index_v1.tsv'):
    i, off, ln, reg, title = line.rstrip('\n').split('\t')
    if int(off) in restore_offsets:
        lines.add(int(i)); off2line[int(off)] = int(i)
assert len(lines) == len(restore_offsets), (len(lines), len(restore_offsets))

v1 = open(os.path.expanduser('~/data/book.v1.jsonl'), 'rb')
v5 = open(os.path.expanduser('~/data/book.v5.jsonl'), 'rb')
out = open(os.path.expanduser('~/data/book.v6.jsonl'), 'wb')
restored = 0
for idx, (r1, r5) in enumerate(zip(v1, v5)):
    if idx in lines:
        out.write(r1)   # full v1 record back
        restored += 1
    else:
        out.write(r5)
out.close()
print(f"restored {restored} books -> book.v6.jsonl")
