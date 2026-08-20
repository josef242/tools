#!/usr/bin/env python3
"""Survey a JSONL / JSONL.zst corpus: field shapes, text lengths, and id structure.

Written to answer one question about a corpus full of very short records: are they
FRAGMENTS of larger works, or are they whole records whose text failed to extract?

The distinction decides everything downstream:

  * Fragments (consecutive short records SHARE an id)  -> reassemble by grouping on id.
  * Failed extraction (each short record has its OWN id, often with a real title)
                                                       -> drop them; the rest is intact.
  * Neither (unique ids, no titles, no structure)      -> the dump itself is fragment-level.

Usage:
    python jsonl_survey.py <file.jsonl|file.jsonl.zst> [--max N] [--field text]
"""

import argparse
import collections
import io
import json
import sys
from pathlib import Path

import numpy as np


def open_stream(path: Path):
    if path.name.lower().endswith('.zst'):
        import zstandard as zstd
        fh = open(path, 'rb')
        return io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(fh), encoding='utf-8',
                                errors='replace')
    return open(path, 'r', encoding='utf-8', errors='replace')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--max', type=int, default=200000, help='records to scan (default 200k)')
    ap.add_argument('--field', default='text', help='text field name (default: text)')
    ap.add_argument('--id-field', default='id')
    ap.add_argument('--show', type=int, default=6, help='short examples to print')
    args = ap.parse_args()

    path = Path(args.path).expanduser()
    lengths, ids, titles = [], [], []
    declared = []          # (metadata['words'], actual word count) where both are known
    shortest = []
    field_freq = collections.Counter()
    meta_keys = collections.Counter()
    meta_sample = {}
    n = 0

    with open_stream(path) as fh:
        for line in fh:
            if n >= args.max:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n += 1
            for k in rec:
                field_freq[k] += 1
            md = rec.get('metadata')
            if isinstance(md, dict):
                for k, v in md.items():
                    meta_keys[k] += 1
                    if k not in meta_sample and v not in (None, '', [], {}):
                        meta_sample[k] = repr(v)[:70]
            txt = rec.get(args.field) or ''
            if isinstance(md, dict) and md.get('words') not in (None, ''):
                try:
                    declared.append((int(str(md['words']).replace(',', '').strip()),
                                     len(txt.split())))
                except ValueError:
                    pass
            lengths.append(len(txt))
            ids.append(rec.get(args.id_field))
            titles.append(rec.get('title'))
            if len(txt) < 120:
                shortest.append((n - 1, rec.get(args.id_field), rec.get('title'), txt[:70]))

    if not n:
        sys.exit("No records parsed.")
    L = np.asarray(lengths)

    print(f"file    : {path.name}")
    print(f"records : {n:,}")
    print()
    print("top-level fields")
    for k, c in field_freq.most_common():
        print(f"  {k:<20} {c:>9,}  ({c/n*100:.1f}% of records)")
    if meta_keys:
        print("\nmetadata dict keys")
        for k, c in meta_keys.most_common(25):
            print(f"  {k:<26} {c:>9,}  e.g. {meta_sample.get(k,'')}")

    print()
    print(f"'{args.field}' length in CHARACTERS")
    pct = [0, 1, 5, 10, 25, 50, 75, 90, 99, 100]
    print("  " + "  ".join(f"p{p}={int(v):,}" for p, v in zip(pct, np.percentile(L, pct))))
    for thr in (20, 100, 500, 2000):
        c = int((L < thr).sum())
        print(f"  under {thr:>5,} chars: {c:>9,}  ({c/n*100:5.2f}%)")

    # --- the decisive part -------------------------------------------------
    print()
    print("ID STRUCTURE  (does a short record share its id with its neighbours?)")
    uniq = len({i for i in ids if i is not None})
    print(f"  distinct '{args.id_field}' values : {uniq:,} across {n:,} records")
    if uniq == n:
        print("  -> ids are UNIQUE per record: these are not fragments sharing a work id.")
    else:
        print(f"  -> ids REPEAT ({n - uniq:,} duplicates): records group, reassembly is possible.")

    runs = sum(1 for a, b in zip(ids, ids[1:]) if a is not None and a == b)
    print(f"  adjacent records with the SAME id: {runs:,}")

    have_title = sum(1 for t in titles if t)
    short_idx = {s[0] for s in shortest}
    short_titled = sum(1 for i, t in enumerate(titles) if t and i in short_idx)
    print(f"  records with a non-empty title    : {have_title:,} ({have_title/n*100:.1f}%)")
    if shortest:
        print(f"  short records that have a title   : {short_titled:,} / {len(shortest):,}")

    # ONE verdict, decided by the id evidence first. Titles alone cannot distinguish the
    # cases: a fragment carries its parent work's title just as a failed extraction carries
    # its own, so a title-based test fires in both and would mislead.
    print()
    repeats = uniq < n * 0.95
    print("VERDICT")
    if repeats and runs > n * 0.05:
        print("  FRAGMENTS, CONTIGUOUS. Short records share an id with their neighbours, so")
        print("  they are pieces of a larger work sitting adjacent in the file. Reassemble")
        print("  with a streaming group-by on id -- one pass, no sort needed.")
    elif repeats:
        print("  FRAGMENTS, SCATTERED. Ids repeat but the pieces are not adjacent, so")
        print("  reassembly needs an external sort by id before grouping. Doable but a real")
        print("  job at corpus scale.")
    elif shortest and short_titled > 0.8 * len(shortest):
        print("  FAILED TEXT EXTRACTION. Every id is unique and the short records still")
        print("  carry real titles, so these are whole works whose body did not scrape --")
        print("  not fragments. Drop them by length; the remaining works are intact.")
    elif not shortest:
        print("  HEALTHY. No pathologically short records in the scanned range.")
    else:
        print("  FRAGMENT-LEVEL SOURCE. Ids are unique, yet short records carry no title and")
        print("  no grouping key. Nothing in the data says which pieces belong together, so")
        print("  this dump is likely not reassemblable -- look for a better source.")

    # --- extraction fidelity ------------------------------------------------
    # AO3 records carry metadata['words'], the archive's own declared word count. Comparing
    # it against the text we actually have separates the two cases a length threshold
    # cannot: a legitimately tiny work (an art post with a caption -- declared 8, got 8)
    # from a truncated scrape (declared 5,000, got 200). A blind minimum-length filter gets
    # BOTH of these wrong; this gets both right, per record, from ground truth.
    if declared:
        d = np.asarray([x for x, _ in declared], dtype=np.float64)
        a = np.asarray([y for _, y in declared], dtype=np.float64)
        keep = d > 0
        d, a = d[keep], a[keep]
        ratio = a / d
        print()
        print(f"EXTRACTION FIDELITY  (actual words / metadata['words'], {d.size:,} records)")
        print("  " + "  ".join(f"p{p}={v:.2f}" for p, v in
                               zip([1, 5, 25, 50, 75, 95, 99],
                                   np.percentile(ratio, [1, 5, 25, 50, 75, 95, 99]))))
        bands = [(0.0, 0.10, 'severely truncated'), (0.10, 0.50, 'truncated'),
                 (0.50, 0.90, 'partial'), (0.90, 1.10, 'FAITHFUL'),
                 (1.10, 1e9, 'longer than declared')]
        for lo, hi, label in bands:
            c = int(((ratio >= lo) & (ratio < hi)).sum())
            bar = '#' * min(46, int(46 * c / max(d.size, 1)))
            print(f"  {label:<22} {c:>9,} {c/d.size*100:>6.2f}%  {bar}")

        faithful = int(((ratio >= 0.9) & (ratio < 1.1)).sum())
        broken = int((ratio < 0.5).sum())
        tiny_ok = int(((ratio >= 0.9) & (d < 100)).sum())
        print()
        print(f"  faithful               : {faithful:,} ({faithful/d.size*100:.1f}%)")
        print(f"  truncated (<50%)       : {broken:,} ({broken/d.size*100:.1f}%)")
        print(f"  SHORT BUT FAITHFUL     : {tiny_ok:,} works under 100 declared words that")
        print(f"                           extracted correctly -- real works, not damage.")
        if broken < d.size * 0.05:
            print("  -> extraction is healthy. Filter on DECLARED words, not on extracted")
            print("     length, and drop only records whose ratio says they are truncated.")

    print()
    print(f"shortest records (first {args.show})")
    for idx, rid, title, txt in shortest[:args.show]:
        print(f"  #{idx:<8} id={rid!r:<18} title={str(title)[:34]!r:<36} text={txt!r}")


if __name__ == '__main__':
    main()
