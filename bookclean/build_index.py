#!/usr/bin/env python3
"""Build a per-book index of book.v1.jsonl: offset, length, region, title."""
import os, re, json
from multiprocessing import Pool

PATH = os.path.expanduser("~/data/book.v1.jsonl")

def get_title(raw):
    m = re.search(rb'"(?:short_book_title|title)":\s*"((?:[^"\\]|\\.){0,100})', raw[:400])
    return (m.group(1).decode('utf-8','replace') if m else '?')[:70].replace('\t',' ')

def scan_range(task):
    start, end = task
    rows = []
    with open(PATH,'rb') as f:
        f.seek(start)
        if start: f.readline()
        pos = f.tell()
        while pos < end:
            raw = f.readline()
            if not raw: break
            region = 'pg19' if b'"short_book_title"' in raw[:300] else 'books3'
            rows.append((pos, len(raw), region, get_title(raw)))
            pos += len(raw)
    return rows

def main():
    size = os.path.getsize(PATH)
    n = 48
    bounds = [size*i//n for i in range(n+1)]
    all_rows = []
    with Pool(12) as pool:
        for rows in pool.imap(scan_range, list(zip(bounds[:-1],bounds[1:]))):
            all_rows.extend(rows)
    all_rows.sort()
    with open('reports/book_index_v1.tsv','w') as out:
        for i,(off,ln,reg,title) in enumerate(all_rows):
            out.write(f"{i}\t{off}\t{ln}\t{reg}\t{title}\n")
    print(f"indexed {len(all_rows):,} books")

if __name__ == '__main__':
    main()
