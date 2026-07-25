#!/usr/bin/env python3
"""Freeze the judged-eval set: 100 pg19 + 100 books3 books, stratified by
length tercile within region, deterministic seed. Extracts head/middle/tail
segments from a given corpus version for judging.

The manifest pins line indices + segment rules (not byte offsets), so the
same logical segments can be re-extracted from any cleaner version.
"""
import json, os, random, argparse

SEED = 20260712
SEG_LEN = 1500

def load_index(path='reports/book_index_v1.tsv'):
    rows = []
    for line in open(path):
        i, off, ln, reg, title = line.rstrip('\n').split('\t')
        rows.append((int(i), int(off), int(ln), reg, title))
    return rows

def freeze(rows, per_region=100):
    rng = random.Random(SEED)
    picked = []
    for reg in ('pg19', 'books3'):
        sub = [r for r in rows if r[3] == reg and r[2] > 20000]  # skip tiny records
        sub.sort(key=lambda r: r[2])
        terciles = [sub[:len(sub)//3], sub[len(sub)//3:2*len(sub)//3], sub[2*len(sub)//3:]]
        quota = [per_region - 2*(per_region//3), per_region//3, per_region//3]
        for t, q in zip(terciles, quota):
            picked.extend(rng.sample(t, q))
    return sorted(picked)

def extract(manifest, corpus, out_path):
    """Extract head/mid/tail segments for each eval book from corpus version."""
    idx = {b['line']: b for b in manifest['books']}
    want = sorted(idx)
    with open(corpus, 'rb') as f, open(out_path, 'w') as out:
        cur = 0
        for line_no in want:
            b = idx[line_no]
            f.seek(b['offset'])          # offset valid only for the frozen version;
            raw = f.readline()           # later versions: reindex or stream
            rec = json.loads(raw)
            text = rec['text']
            mid = max(0, len(text)//2 - SEG_LEN//2)
            for seg, chunk in (('head', text[:SEG_LEN]),
                               ('mid', text[mid:mid+SEG_LEN]),
                               ('tail', text[-SEG_LEN:])):
                out.write(json.dumps({'line': line_no, 'region': b['region'],
                                      'title': b['title'], 'seg': seg,
                                      'text': chunk}) + '\n')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default=os.path.expanduser('~/data/book.v1.jsonl'))
    args = ap.parse_args()
    rows = load_index()
    picked = freeze(rows)
    manifest = {'seed': SEED, 'seg_len': SEG_LEN, 'frozen_from': 'book.v1.jsonl',
                'books': [{'line': i, 'offset': off, 'bytes': ln, 'region': reg,
                           'title': title} for i, off, ln, reg, title in picked]}
    json.dump(manifest, open('reports/eval_set_v1.json', 'w'), indent=1)
    extract(manifest, args.corpus, 'reports/eval_segments_v1.jsonl')
    print(f"frozen {len(picked)} books -> eval_set_v1.json; segments extracted")
