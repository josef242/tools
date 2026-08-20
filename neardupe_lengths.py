#!/usr/bin/env python3
"""Document length distribution for a tokenized corpus, from the neardupe cardinality file.

A single short record proves nothing; the SHAPE of the length distribution tells you
whether a corpus is healthy, and it decides whether near-duplicate results can be trusted:

  * Below MIN_SHINGLES_FOR_1BIT the 1-bit estimator's large-set assumption fails and
    densification can manufacture similarity between documents that share very little.
    If a large share of the corpus sits down there, the duplicate count is not meaningful.

  * Masses of very short documents also drive the PAIR count up superlinearly -- short
    documents match each other spuriously and each match is a pair -- which is a common
    reason matching appears to hang.

Reads <dataset>/.dataset_explorer_cache/*.neardupe-card.npy, written during sketching, so
this costs nothing: no re-scan of the corpus.

Usage:
    python neardupe_lengths.py <dataset path or .neardupe-card.npy>
"""

import sys
from pathlib import Path

import numpy as np

SHORT = 200          # neardupe.MIN_SHINGLES_FOR_1BIT
BUCKETS = [10, 50, 100, 200, 500, 1000, 5000, 20000, 100000, 10 ** 12]


def find_cards(arg: Path) -> Path:
    if arg.is_file() and arg.name.endswith('.npy'):
        return arg
    cache = (arg if arg.is_dir() else arg.parent) / '.dataset_explorer_cache'
    hits = sorted(cache.glob('*.neardupe-card.npy')) if cache.is_dir() else []
    if not hits:
        sys.exit(f"No *.neardupe-card.npy under {cache}. Run neardupe first "
                 f"(sketching writes it).")
    if len(hits) > 1:
        print(f"note: {len(hits)} cardinality files present, using {hits[0].name}")
    return hits[0]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = find_cards(Path(sys.argv[1]).expanduser())
    cards = np.load(path, mmap_mode='r')
    cards = np.asarray(cards[cards > 0])          # trailing rows of an over-allocated file
    n = cards.size
    if n == 0:
        sys.exit("Cardinality file is empty.")

    print(f"file      : {path.name}")
    print(f"documents : {n:,}")
    print()
    pct = [0, 1, 5, 10, 25, 50, 75, 90, 99, 100]
    vals = np.percentile(cards, pct)
    print("shingle-count percentiles  (a 13-gram doc has ~len-12 shingles)")
    print("  " + "  ".join(f"p{p}={int(v):,}" for p, v in zip(pct, vals)))
    print()

    print("distribution")
    lo = 0
    for hi in BUCKETS:
        c = int(np.count_nonzero((cards > lo) & (cards <= hi)))
        label = f"{lo+1:,}-{hi:,}" if hi < 10 ** 12 else f"{lo+1:,}+"
        bar = '#' * min(58, int(58 * c / n))
        print(f"  {label:>16} {c:>12,} {c/n*100:>6.2f}%  {bar}")
        lo = hi

    short = int(np.count_nonzero(cards < SHORT))
    print()
    print(f"UNDER {SHORT} SHINGLES: {short:,} docs ({short/n*100:.2f}%)")
    if short / n > 0.5:
        print("  More than half the corpus is too short for the 1-bit estimator to be")
        print("  reliable. Near-duplicate results over this corpus are not trustworthy as-is;")
        print("  filter by length before deduping, or raise --nd-perms substantially.")
    elif short / n > 0.1:
        print("  A meaningful minority is below the reliable range. Expect inflated pair")
        print("  counts and check the 'short' flags in the cluster table.")
    else:
        print("  Small enough not to distort results.")

    # Rough sense of how much of the pair space the short tail is responsible for.
    if short > 1000:
        print()
        print(f"  Note: those {short:,} short docs alone span {short*(short-1)//2:,} possible")
        print(f"  pairs. Short documents match each other easily, so this tail is the usual")
        print(f"  cause of a pair count that explodes and a match phase that never ends.")


if __name__ == '__main__':
    main()
