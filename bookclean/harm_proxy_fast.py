#!/usr/bin/env python3
"""FAST within-book memorization-magnet harm-proxy: v10 vs v11 (record-aligned).
Two independent metrics per book (neither is what the filter directly optimized):
  (1) DUP-LINE-FRACTION (Gopher/MassiveText standard): fraction of a book's non-empty
      lines that are within-book duplicates (line appears >=2x). Field-standard junk signal.
  (2) PEAK 10-GRAM: the highest within-book repeat count of any 10-gram (the strongest
      memorization magnet in the book). Uses tuple-hash (no string joins) for speed.
Args: N (books to sample) SKIP (books to skip past PG19). Reads v10 & v11 in lockstep
(record-aligned since within-dedup is line-preserving), so the SAME book is compared."""
import json, os, sys, re
from collections import Counter
V10 = os.path.expanduser('~/data/book.v10.jsonl')
V11 = os.path.expanduser('~/data/book.v11.jsonl')
N    = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
SKIP = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
K = 10

def metrics(t):
    lines = [l for l in t.split('\n') if l.strip()]
    nl = len(lines)
    if nl == 0:
        return (0.0, 0)
    lc = Counter(lines)
    dup = sum(c for l, c in lc.items() if c >= 2)          # lines that are duplicates
    dup_frac = dup / nl
    w = re.sub(r'\d', '#', t.lower()).split()
    peak = 0
    if len(w) >= K:
        gc = Counter()
        for i in range(len(w) - K + 1):
            h = hash(tuple(w[i:i + K]))
            c = gc[h] + 1; gc[h] = c
            if c > peak:
                peak = c
    return (dup_frac, peak)

def main():
    df10 = []; df11 = []; pk10 = []; pk11 = []
    nb = 0; skipped = 0
    with open(V10) as f10, open(V11) as f11:
        for l10, l11 in zip(f10, f11):
            if skipped < SKIP:
                skipped += 1; continue
            if nb >= N:
                break
            nb += 1
            d10, p10 = metrics(json.loads(l10)['text'])
            d11, p11 = metrics(json.loads(l11)['text'])
            df10.append(d10); df11.append(d11); pk10.append(p10); pk11.append(p11)
            if nb % 2000 == 0:
                print(f"  ...{nb}", flush=True)
    def stats(a):
        a = sorted(a); n = len(a)
        return (sum(a)/n, a[n//2], a[int(n*.9)], a[int(n*.99)], a[-1])
    m10, md10, p9010, p9910, mx10 = stats(df10)
    m11, md11, p9011, p9911, mx11 = stats(df11)
    print(f"\n=== harm-proxy: {nb} record-aligned Books3 books (skipped {SKIP}) ===")
    print(f"(1) DUP-LINE-FRACTION per book (within-book duplicate lines / non-empty lines):")
    print(f"    v10: mean={m10:.4f} median={md10:.4f} p90={p9010:.4f} p99={p9910:.4f} max={mx10:.4f}")
    print(f"    v11: mean={m11:.4f} median={md11:.4f} p90={p9011:.4f} p99={p9911:.4f} max={mx11:.4f}")
    print(f"    -> mean dup-line-fraction reduced {100*(m10-m11)/max(m10,1e-9):.1f}%")
    a10, amd10, ap9010, ap9910, amx10 = stats(pk10)
    a11, amd11, ap9011, ap9911, amx11 = stats(pk11)
    print(f"(2) PEAK within-book 10-gram repeat per book (memorization magnet strength):")
    print(f"    v10: mean={a10:.1f} median={amd10:.0f} p90={ap9010:.0f} p99={ap9910:.0f} max={amx10:.0f}")
    print(f"    v11: mean={a11:.1f} median={amd11:.0f} p90={ap9011:.0f} p99={ap9911:.0f} max={amx11:.0f}")
    big10 = sum(1 for x in pk10 if x >= 50); big11 = sum(1 for x in pk11 if x >= 50)
    print(f"    books with a >=50x 10-gram magnet: v10={big10} v11={big11} "
          f"(-{big10-big11}, {100*(big10-big11)/max(big10,1):.0f}% fewer)")

if __name__ == '__main__':
    main()
