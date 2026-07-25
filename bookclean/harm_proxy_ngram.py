#!/usr/bin/env python3
"""Within-book memorization-magnet harm-proxy: v10 vs v11 (record-aligned).
For a sample of books, find the strongest WITHIN-BOOK repeated 10-gram in each book
(the memorization magnet) and show how within-book dedup reduced it. Also aggregates
the top magnets corpus-wide. A magnet = an n-gram repeated many times inside ONE
book (page furniture, nav strips, OCR loops) — the thing within-book dedup targets."""
import json, os, sys, re
from collections import Counter
V10 = os.path.expanduser('~/data/book.v10.jsonl')
V11 = os.path.expanduser('~/data/book.v11.jsonl')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
SKIP = int(sys.argv[2]) if len(sys.argv) > 2 else 40000  # skip PG19 (clean Gutenberg) to sample Books3
K = 10  # n-gram size

def toks(t):
    return re.sub(r'\d', '#', t.lower()).split()

def top_within_magnet(t):
    """Return (count, gram) of the most-repeated 10-gram within this text."""
    w = toks(t)
    if len(w) < K:
        return (0, '')
    c = Counter()
    for i in range(len(w) - K + 1):
        c[' '.join(w[i:i + K])] += 1
    g, n = c.most_common(1)[0]
    return (n, g)

def main():
    magnet_v10 = Counter()   # gram -> summed within-book repeat across sample
    magnet_v11 = Counter()
    peak_v10 = []; peak_v11 = []
    killed = []              # books where the top magnet shrank a lot
    nb = 0; skipped = 0
    with open(V10) as f10, open(V11) as f11:
        for l10, l11 in zip(f10, f11):
            if skipped < SKIP:
                skipped += 1; continue
            if nb >= N:
                break
            nb += 1
            t10 = json.loads(l10)['text']; t11 = json.loads(l11)['text']
            n10, g10 = top_within_magnet(t10)
            n11, g11 = top_within_magnet(t11)
            peak_v10.append(n10); peak_v11.append(n11)
            if n10 >= 20:
                magnet_v10[g10] += n10
            if n11 >= 20:
                magnet_v11[g11] += n11
            if n10 >= 50 and n10 - n11 >= 30:
                killed.append((n10, n11, g10[:70]))
    peak_v10.sort(); peak_v11.sort()
    def pct(a, p): return a[int(len(a) * p)] if a else 0
    print(f"=== within-book magnet harm-proxy: {nb} record-aligned books (v10 vs v11) ===")
    print(f"peak within-book 10-gram repeat per book (higher = stronger magnet):")
    print(f"  v10:  median={pct(peak_v10,.5)}  p90={pct(peak_v10,.9)}  p99={pct(peak_v10,.99)}  max={peak_v10[-1]}")
    print(f"  v11:  median={pct(peak_v11,.5)}  p90={pct(peak_v11,.9)}  p99={pct(peak_v11,.99)}  max={peak_v11[-1]}")
    n_big10 = sum(1 for x in peak_v10 if x >= 50)
    n_big11 = sum(1 for x in peak_v11 if x >= 50)
    print(f"  books with a >=50x within-book magnet:  v10={n_big10}  v11={n_big11}  "
          f"(-{n_big10-n_big11}, {100*(n_big10-n_big11)/max(n_big10,1):.0f}% reduced)")
    print(f"\nTOP magnets remaining in v11 (want: only legit refrains/dialogue, no furniture):")
    for g, c in magnet_v11.most_common(20):
        print(f"  {c:<8} {g!r}")
    print(f"\nSTRONGEST magnets KILLED by within-book dedup (v10 count -> v11 count):")
    killed.sort(reverse=True)
    for n10, n11, g in killed[:25]:
        print(f"  {n10:>5} -> {n11:<5} {g!r}")

if __name__ == '__main__':
    main()
