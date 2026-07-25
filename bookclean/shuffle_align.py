#!/usr/bin/env python3
"""Aligned corpus shuffle for the harm ablation. Applies ONE fixed permutation
identically to v9 and v11 so the arms stay record-aligned (arm B = arm A with junk
lines removed, same record order) while distributing Books3 junk throughout the
stream -- otherwise the RNG-free sequential loader would train the ~1.57B-token
arms on the clean PG19 head and never see the junk. Retains the FULL shuffled
files (reusable for scaled training); tokenize a prefix for the ablation.

NVMe (~7GB/s) so random-access reorder is cheap. Seed fixed for reproducibility."""
import json, os, sys, random, array

FILES = {'v9': os.path.expanduser('~/data/book.v9.jsonl'),
         'v11': os.path.expanduser('~/data/book.v11.jsonl')}
OUT = {k: os.path.expanduser(f'~/data/book.{k}.shuf.jsonl') for k in FILES}
SEED = 20260716

def build_offsets(path):
    offs = array.array('q'); f = open(path, 'rb')
    pos = 0
    for line in f:
        offs.append(pos); pos += len(line)
    f.close()
    return offs

def main():
    # offsets (both files are record-aligned: same count, same order)
    offs = {k: build_offsets(p) for k, p in FILES.items()}
    n = len(offs['v9'])
    assert n == len(offs['v11']), f"record count mismatch {n} vs {len(offs['v11'])}"
    print(f"records: {n:,}")

    perm = list(range(n))
    random.Random(SEED).shuffle(perm)

    for k, path in FILES.items():
        o = offs[k]; sz = os.path.getsize(path)
        with open(path, 'rb') as fin, open(OUT[k], 'wb') as fout:
            buf = bytearray(); wrote = 0
            for count, idx in enumerate(perm):
                start = o[idx]
                end = o[idx + 1] if idx + 1 < n else sz
                fin.seek(start)
                buf += fin.read(end - start)
                if len(buf) >= 64_000_000:
                    fout.write(buf); wrote += len(buf); buf = bytearray()
                if (count + 1) % 40000 == 0:
                    print(f"  {k}: {count+1:,}/{n:,}", flush=True)
            if buf:
                fout.write(buf); wrote += len(buf)
        print(f"  {k} DONE -> {OUT[k]} ({wrote/1e9:.1f} GB)")
    # save the permutation so Arm C / future tokenization can reuse the exact order
    with open('reports/shuffle_perm.json', 'w') as f:
        json.dump({'seed': SEED, 'n': n}, f)
    print("perm seed saved (reports/shuffle_perm.json); regenerate with SEED to reuse order")

if __name__ == '__main__':
    main()
