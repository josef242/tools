#!/usr/bin/env python3
"""Language census: classify every book by stopword voting on a middle slice."""
import json, os, re
from collections import defaultdict
from multiprocessing import Pool

PATH = os.path.expanduser("~/data/book.v1.jsonl")
STOP = {
 'en': [' the ',' and ',' of ',' to ',' was ',' that '],
 'fr': [' le ',' la ',' les ',' des ',' est ',' dans '],
 'de': [' der ',' die ',' und ',' das ',' nicht ',' ist '],
 'es': [' el ',' los ',' que ',' una ',' por ',' como '],
 'it': [' il ',' che ',' della ',' per ',' sono ',' anche '],
}
def scan_range(task):
    start, end = task
    counts = defaultdict(lambda: defaultdict(int))   # region -> lang -> books
    gbs = defaultdict(lambda: defaultdict(int))
    with open(PATH,'rb') as f:
        f.seek(start)
        if start: f.readline()
        pos = f.tell()
        while pos < end:
            raw = f.readline()
            if not raw: break
            pos += len(raw)
            region = "pg19" if b'"short_book_title"' in raw[:300] else "books3"
            mid = len(raw)//2
            s = raw[mid:mid+6000].decode('utf-8','replace').replace('\\n',' ').lower()
            best, bestc = 'other', 1
            for lang, words in STOP.items():
                c = sum(s.count(w) for w in words)
                if c > bestc: best, bestc = lang, c
            counts[region][best] += 1
            gbs[region][best] += len(raw)
    return {r: dict(d) for r,d in counts.items()}, {r: dict(d) for r,d in gbs.items()}

def main():
    size = os.path.getsize(PATH)
    n = 48
    bounds = [size*i//n for i in range(n+1)]
    counts = defaultdict(lambda: defaultdict(int)); gbs = defaultdict(lambda: defaultdict(int))
    with Pool(12) as pool:
        for c, g in pool.imap_unordered(scan_range, list(zip(bounds[:-1],bounds[1:]))):
            for r in c:
                for l, v in c[r].items(): counts[r][l] += v
            for r in g:
                for l, v in g[r].items(): gbs[r][l] += v
    out = {}
    for r in counts:
        tot = sum(counts[r].values()); totg = sum(gbs[r].values())
        print(f"\n=== {r} ({tot:,} books) ===")
        out[r] = {}
        for l in sorted(counts[r], key=lambda x:-counts[r][x]):
            print(f"  {l:<6} {counts[r][l]:>8,} books ({100*counts[r][l]/tot:.1f}%)  {gbs[r][l]/1e9:.1f} GB ({100*gbs[r][l]/totg:.1f}%)")
            out[r][l] = {"books": counts[r][l], "gb": round(gbs[r][l]/1e9,2)}
    json.dump(out, open('reports/lang_census.json','w'), indent=1)

if __name__ == '__main__':
    main()
