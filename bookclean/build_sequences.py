#!/usr/bin/env python3
"""Sequence-format training data: per-book paragraph feature sequences.
Same 30k silver books as build_training (seed 123) for comparability.
Serves ablation A1 (context-window MLP) and A4 (sequence models)."""
import json, os, sys, random
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, '.')
from bakeoff import paragraphs
from build_training import featurize

WINDOW = 12000
MAX_PARAS = 60

def work(rows_meta):
    seqs = []
    with open(os.path.expanduser('~/data/book.v1.jsonl'), 'rb') as f:
        for off, boundary in rows_meta:
            f.seek(off)
            rec = json.loads(f.readline())
            head = rec['text'][:WINDOW]
            paras = [(s,e,t) for s,e,t in paragraphs(head) if len(t.strip()) >= 15][:MAX_PARAS]
            if len(paras) < 3: continue
            feats = [featurize(t, s, e, len(head), i) for i,(s,e,t) in enumerate(paras)]
            labels = [1.0 if (s+e)/2 < boundary else 0.0 for s,e,t in paras]
            starts = [s for s,e,t in paras]
            seqs.append((np.array(feats,dtype='float32'), np.array(labels,dtype='float32'),
                         np.array(starts,dtype='int32'), boundary))
    return seqs

def main():
    rng = random.Random(123)
    silver = []
    for l in open('reports/silver_books3_heads.jsonl'):
        r = json.loads(l)
        if r['agree'] and 0 < r['sentinel'] < r['head_len']:
            silver.append((r['offset'], r['sentinel']))
    rng.shuffle(silver)
    silver = silver[:30000]
    chunks = [silver[i::12] for i in range(12)]
    all_seqs = []
    with Pool(12) as pool:
        for i, out in enumerate(pool.imap_unordered(work, chunks)):
            all_seqs.extend(out)
            print(f"chunk {i+1}/12 done ({len(all_seqs):,} books)", flush=True)
    X = np.concatenate([s[0] for s in all_seqs])
    y = np.concatenate([s[1] for s in all_seqs])
    starts = np.concatenate([s[2] for s in all_seqs])
    lengths = np.array([len(s[1]) for s in all_seqs], dtype='int32')
    bounds = np.array([s[3] for s in all_seqs], dtype='int32')
    np.savez('reports/train_sequences.npz', X=X, y=y, starts=starts,
             lengths=lengths, boundaries=bounds)
    print(f"sequences: {len(lengths):,} books, {len(y):,} paragraphs, "
          f"mean len {lengths.mean():.1f}")

if __name__ == '__main__':
    main()
