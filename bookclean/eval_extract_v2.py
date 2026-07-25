#!/usr/bin/env python3
"""Extract eval segments from any corpus version by LINE NUMBER (stable across
versions, unlike byte offsets)."""
import json, os, sys

def extract(corpus, out_path, seg_len=1500):
    manifest = json.load(open('reports/eval_set_v1.json'))
    want = {b['line']: b for b in manifest['books']}
    remaining = set(want)
    with open(corpus, 'rb') as f, open(out_path, 'w') as out:
        for idx, raw in enumerate(f):
            if idx not in remaining:
                continue
            remaining.discard(idx)
            b = want[idx]
            rec = json.loads(raw)
            text = rec['text']
            mid = max(0, len(text)//2 - seg_len//2)
            for seg, chunk in (('head', text[:seg_len]), ('mid', text[mid:mid+seg_len]),
                               ('tail', text[-seg_len:])):
                out.write(json.dumps({'line': idx, 'region': b['region'],
                                      'title': b['title'], 'seg': seg, 'text': chunk}) + '\n')
            if not remaining:
                break
    print(f"extracted segments -> {out_path}")

if __name__ == '__main__':
    extract(sys.argv[1], sys.argv[2])
