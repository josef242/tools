#!/usr/bin/env python3
"""Audit a saved .neardupe.gz artifact before acting on it.

A duplicate count is only as trustworthy as the clusters behind it. This reads the
artifact produced by `dataset_explorer.py --neardupe` and reports the three ways the
headline number can mislead:

  1. TRANSITIVE CHAINS. Clusters are connected components, so A~B and B~C merge A with C
     even when A and C are unrelated. Measured by EDGE DENSITY -- the share of possible
     within-cluster pairs that actually cleared the threshold. A tight group of n docs has
     all n(n-1)/2 pairs; a chain has roughly n-1. Such a cluster's size overstates real
     duplication, and cutting it as a unit would delete documents that are not duplicates.

  2. SHORT DOCUMENTS. The 1-bit estimator assumes large shingle sets. Below ~200 shingles
     most OPH buckets are empty and get filled by densification, which can manufacture
     agreement between two short docs that share only a few shingles.

  3. GIANT CLUSTERS. One huge component usually means boilerplate (a license, a header)
     rather than genuine document duplication, and it can dominate the total on its own.

Usage:
    python neardupe_audit.py <path to .neardupe.gz>
    python neardupe_audit.py <dataset path>      # finds the artifact beside it
"""

import gzip
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np


# Chains are detected by EDGE DENSITY, not by mean-vs-max similarity. Only pairs at or
# above the threshold are recorded, so mean_sim is bounded below by the threshold and can
# never fall far under max_sim -- a mean-gap test looks reassuring on an actual chain.
# Density is unfooled: a tight cluster of n docs has all n(n-1)/2 pairs, a chain has ~n-1.
CHAIN_DENSITY = 0.5       # below this share of possible pairs => likely a transitive chain
SHORT_SHINGLES = 200      # neardupe.MIN_SHINGLES_FOR_1BIT


def density(c) -> float:
    """Share of possible within-cluster pairs that actually cleared the threshold."""
    n = c['size']
    possible = n * (n - 1) / 2
    return c.get('n_pairs', 0) / possible if possible else 1.0


def find_artifact(arg: Path) -> Path:
    if arg.is_file() and arg.name.endswith('.neardupe.gz'):
        return arg
    cache = (arg if arg.is_dir() else arg.parent) / '.dataset_explorer_cache'
    # Reproduce dataset_explorer._neardupe_artifact_path exactly, so a cache directory
    # holding several datasets' artifacts resolves to the right one instead of the
    # alphabetically-first one.
    tag = hashlib.md5(str(arg.absolute()).encode()).hexdigest()[:8]
    stem = arg.name if arg.is_dir() else arg.stem
    exact = cache / f"{stem}_{tag}.neardupe.gz"
    if exact.exists():
        return exact
    hits = sorted(cache.glob('*.neardupe.gz')) if cache.is_dir() else []
    if not hits:
        sys.exit(f"No .neardupe.gz found for {arg}")
    print(f"note: no exact match for {arg.name}; using {hits[0].name}")
    return hits[0]


def histogram(counts, buckets):
    out, lo = [], 2          # clusters always have at least 2 members
    for hi in buckets:
        n = sum(1 for c in counts if lo <= c <= hi)
        out.append((f"{lo}" if lo == hi else f"{lo}-{hi}" if hi != 10**9 else f"{lo}+", n))
        lo = hi + 1
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = find_artifact(Path(sys.argv[1]).expanduser())
    with gzip.open(path, 'rb') as f:
        art = pickle.load(f)

    clusters = art.get('clusters')
    n_docs = int(art.get('n_docs') or 0)

    # v2 keeps signatures and cardinalities in sibling .npy files; v1 inlined them.
    cards = art.get('cardinalities')
    card_path = path.parent / (path.name[:-len('.neardupe.gz')] + '.neardupe-card.npy')
    if cards is None and card_path.exists():
        cards = np.load(card_path, mmap_mode='r')[:n_docs] if n_docs else np.load(card_path)
    if not n_docs and art.get('signatures') is not None:
        n_docs = int(art['signatures'].shape[0])

    if not clusters:
        print(f"artifact : {path.name}")
        print("No clusters recorded. If this is a checkpoint written straight after "
              "sketching, run neardupe again -- it will reuse the signatures and only "
              "re-run the match.")
        return

    sizes = [c['size'] for c in clusters]
    clustered = sum(sizes)
    removable = sum(s - 1 for s in sizes)

    print(f"artifact : {path.name}")
    print(f"params   : threshold={art['threshold']}  ngram={art['ngram']}  "
          f"perms={art['perms']}")
    print()
    print(f"documents        {n_docs:,}")
    print(f"clusters         {len(clusters):,}")
    print(f"docs in clusters {clustered:,}  ({clustered / max(n_docs, 1) * 100:.2f}%)")
    print(f"removable        {removable:,}  ({removable / max(n_docs, 1) * 100:.2f}%)")
    print("   'removable' = keep one per cluster. This is the number that matters, and it "
          "is always lower than 'docs in clusters'.")

    print()
    print("cluster size distribution")
    for label, n in histogram(sizes, [2, 3, 5, 10, 100, 1000, 10 ** 9]):
        bar = '#' * min(60, int(60 * n / max(len(clusters), 1)))
        print(f"  {label:>8} {n:>8,}  {bar}")

    # --- 1. transitive chains ------------------------------------------------
    chains = [c for c in clusters if c['size'] > 2 and density(c) < CHAIN_DENSITY]
    chain_docs = sum(c['size'] for c in chains)
    print()
    print(f"[1] TRANSITIVE CHAINS  (edge density < {CHAIN_DENSITY}, size > 2)")
    print(f"    {len(chains):,} clusters, {chain_docs:,} docs "
          f"({chain_docs / max(clustered, 1) * 100:.1f}% of clustered)")
    for c in sorted(chains, key=lambda x: -x['size'])[:5]:
        print(f"      size={c['size']:<6} density={density(c):.2f} "
              f"({c.get('n_pairs', 0)}/{c['size']*(c['size']-1)//2} pairs) "
              f"max={c['max_sim']:.3f} first={c['members'][:4]}")
    if not chains:
        print("      none - clusters are internally tight")

    # --- 2. short documents --------------------------------------------------
    print()
    print(f"[2] SHORT DOCUMENTS  (< {SHORT_SHINGLES} shingles: 1-bit estimator unreliable)")
    if cards is None:
        print("    cardinalities not in artifact - cannot check")
    else:
        cards = np.asarray(cards)
        short = set(np.flatnonzero(cards < SHORT_SHINGLES).tolist())
        corpus_pct = len(short) / max(len(cards), 1) * 100
        tainted = [c for c in clusters if any(m in short for m in c['members'])]
        tainted_docs = sum(c['size'] for c in tainted)
        print(f"    corpus-wide      {len(short):,} docs ({corpus_pct:.2f}%)")
        print(f"    clusters touched {len(tainted):,}  covering {tainted_docs:,} docs "
              f"({tainted_docs / max(clustered, 1) * 100:.1f}% of clustered)")
        if tainted_docs / max(clustered, 1) > 0.10:
            print("    WARNING: a large share of your duplicates involve short documents.")
            print("             Re-run with a larger --nd-perms, or exclude short docs, "
                  "before trusting the count.")
        elif not short:
            print("    none - every document is long enough for the estimator")
        med = int(np.median(cards)) if len(cards) else 0
        print(f"    shingle count: min={cards.min():,} median={med:,} max={cards.max():,}")

    # --- 3. giant clusters ---------------------------------------------------
    print()
    print("[3] LARGEST CLUSTERS  (a giant one usually means boilerplate, not duplication)")
    for i, c in enumerate(sorted(clusters, key=lambda x: -x['size'])[:10]):
        share = c['size'] / max(clustered, 1) * 100
        flag = "  <-- CHAIN" if c['size'] > 2 and density(c) < CHAIN_DENSITY else ""
        print(f"    {i+1:>3}. size={c['size']:<7,} density={density(c):.2f} "
              f"max={c['max_sim']:.3f} mean={c['mean_sim']:.3f} "
              f"cont={c.get('max_containment', 0):.2f} {share:>5.1f}%{flag}")

    print()
    solid = removable - sum(c['size'] - 1 for c in chains)
    print(f"BOTTOM LINE: {removable:,} removable; ~{max(solid, 0):,} of those are in "
          f"internally-tight clusters.")
    print("Inspect the largest clusters with 'dupe <n>' before cutting anything.")


if __name__ == '__main__':
    main()
