#!/usr/bin/env python3
"""Build paragraph-level training data from silver-agreement books3 heads.

For each agreement book (sentinel~prosefall within 150 chars): paragraphs
ending before the boundary are junk (1), paragraphs starting after are
content (0). 18 structural/lexical/positional features per paragraph."""
import json, os, re, sys
from multiprocessing import Pool
sys.path.insert(0, '.')
from bakeoff import paragraphs, prose_score, lexis_score, _para_markers, \
                    TOC_LINE, LIST_LINE, PURE_QUOTE, ATTRIB

WINDOW = 12000
MAX_PER_SIDE = 10

def featurize(txt, s, e, wlen, idx):
    stripped = txt.strip()
    n = len(stripped)
    lines = [l for l in stripped.split('\n') if l.strip()]
    letters = sum(c.isalpha() for c in stripped) or 1
    words = stripped.split() or ['x']
    sent = stripped.count('. ') + stripped.count('."') + stripped.count('! ') + stripped.count('? ')
    return [
        min(n, 4000) / 4000,
        sum(c.islower() for c in stripped) / letters,
        sum(c.isdigit() for c in stripped) / max(n, 1),
        sum(stripped.count(c) for c in '#*©®™|>=_[]{}') / max(n, 1),
        sent / max(n / 100, 1),
        sum(1 for w in words if len(w) > 1 and w.isupper()) / len(words),
        (sum(len(l) for l in lines) / len(lines)) / 80 if lines else 0,
        min(len(lines), 40) / 40,
        sum(1 for l in lines if TOC_LINE.match(l)) / max(len(lines), 1),
        sum(1 for l in lines if LIST_LINE.match(l)) / max(len(lines), 1),
        lexis_score(txt),
        prose_score(txt),
        min(_para_markers(txt), 5) / 5,
        1.0 if PURE_QUOTE.match(stripped) else 0.0,
        1.0 if ATTRIB.match(stripped) else 0.0,
        s / wlen,
        min(idx, 30) / 30,
        1.0 if re.search(r'(?i)www\.|https?://|isbn|©', stripped) else 0.0,
    ]

def work(task):
    rows_meta, limit = task
    out = []
    with open(os.path.expanduser('~/data/book.v1.jsonl'), 'rb') as f:
        for off, boundary in rows_meta:
            f.seek(off)
            rec = json.loads(f.readline())
            head = rec['text'][:WINDOW]
            paras = paragraphs(head)
            junk = content = 0
            for i, (s, e, txt) in enumerate(paras):
                if len(txt.strip()) < 15: continue
                if e <= boundary and junk < MAX_PER_SIDE:
                    label = 1; junk += 1
                elif s >= boundary and content < MAX_PER_SIDE:
                    label = 0; content += 1
                else:
                    continue
                out.append((label, featurize(txt, s, e, len(head), i)))
    return out

def main():
    import random
    rng = random.Random(123)
    silver = []
    for l in open('reports/silver_books3_heads.jsonl'):
        r = json.loads(l)
        if r['agree'] and r['sentinel'] > 0 and r['sentinel'] < r['head_len']:
            silver.append((r['offset'], r['sentinel']))
    rng.shuffle(silver)
    silver = silver[:30000]
    print(f"using {len(silver):,} agreement books")
    chunks = [silver[i::12] for i in range(12)]
    X, y = [], []
    with Pool(12) as pool:
        for out in pool.imap_unordered(work, [(c, None) for c in chunks]):
            for label, feats in out:
                y.append(label); X.append(feats)
    import numpy as np
    X = np.array(X, dtype='float32'); y = np.array(y, dtype='float32')
    np.savez('reports/train_paragraphs.npz', X=X, y=y)
    print(f"dataset: {len(y):,} paragraphs ({y.mean()*100:.1f}% junk)")

if __name__ == '__main__':
    main()
