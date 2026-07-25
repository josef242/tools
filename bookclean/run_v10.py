#!/usr/bin/env python3
"""v7: books3 from v1 text via midbook+sentinel2 engines; pg19 from v6."""
import json, os, sys
from multiprocessing import Pool
sys.path.insert(0, '.')
from books3_clean_v10 import clean_books3_v10
from midbook_clean import clean_midbook

V1 = os.path.expanduser('~/data/book.v1.jsonl')
V6 = os.path.expanduser('~/data/book.v6.jsonl')
DST = os.path.expanduser('~/data/book.v10.jsonl')
SHARDS = os.path.expanduser('~/data/.v10_shards')

def load_offsets(path):
    offs = []
    for line in open(path):
        f = line.rstrip('\n').split('\t')
        offs.append((int(f[1]), f[3]))
    return offs

def work(task):
    wid, lo, hi, o1, o6 = task
    from collections import Counter
    st = Counter()
    with open(V1,'rb') as f1, open(V6,'rb') as f6, \
         open(f"{SHARDS}/s{wid:03d}.jsonl",'wb') as out, \
         open(f"{SHARDS}/l{wid:03d}.jsonl",'w') as lg:
        for i in range(lo, hi):
            off1, reg = o1[i]
            st['books'] += 1
            if reg == 'pg19':
                f6.seek(o6[i][0]); out.write(f6.readline()); continue
            f1.seek(off1)
            rec = json.loads(f1.readline())
            t, mid_counts = clean_midbook(rec['text'])
            t2, entry = clean_books3_v10(t)
            st['head_cut'] += entry.get('head',{}).get('action') == 'cut'
            st['tail_cut'] += entry.get('tail',{}).get('action') == 'cut'
            st['flagged'] += any(isinstance(v,dict) and v.get('action')=='flagged' for v in entry.values())
            st['chars'] += len(rec['text']) - len(t2)
            if entry or mid_counts:
                lg.write(json.dumps({'line': i, **entry, 'mid': mid_counts}) + '\n')
            rec['text'] = t2
            out.write(json.dumps(rec, ensure_ascii=True).encode() + b'\n')
    return dict(st)

def main():
    os.makedirs(SHARDS, exist_ok=True)
    o1 = load_offsets('reports/book_index_v1.tsv')
    o6 = load_offsets('reports/book_index_v6.tsv')
    n = len(o1); W = 48
    bounds = [n*i//W for i in range(W+1)]
    tasks = [(i, bounds[i], bounds[i+1], o1, o6) for i in range(W)]
    from collections import Counter
    tot = Counter()
    with Pool(12) as pool:
        for i, s in enumerate(pool.imap_unordered(work, tasks)):
            tot.update(s)
            if (i+1) % 12 == 0: print(f"[{i+1}/{W}] {dict(tot)}", flush=True)
    with open(DST,'wb') as out:
        for i in range(W):
            with open(f"{SHARDS}/s{i:03d}.jsonl",'rb') as sh:
                while True:
                    c = sh.read(64_000_000)
                    if not c: break
                    out.write(c)
            os.remove(f"{SHARDS}/s{i:03d}.jsonl")
    with open('reports/v10_ledger.jsonl','w') as L:
        for i in range(W):
            L.write(open(f"{SHARDS}/l{i:03d}.jsonl").read())
            os.remove(f"{SHARDS}/l{i:03d}.jsonl")
    print(f"DONE {dict(tot)} -> {DST}")

if __name__ == '__main__':
    main()
