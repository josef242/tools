#!/usr/bin/env python3
"""Noise-marker census v0 for the Gutenberg book corpus (book.jsonl).

Streams complete JSONL records (one book per line) and counts occurrences of
known Project Gutenberg noise markers across the corpus. Safe to run while the
file is still copying: it only reads up to (current_size - margin) and stops at
the last complete line before that limit.

For each marker we track:
  - total occurrences and number of books containing it
  - mean fractional position within the book (0.0 = start, 1.0 = end),
    which confirms whether the noise concentrates in front/back matter
  - a few example snippets with surrounding context, for human judging

A sampled per-line structural scan (every Nth book) additionally measures
line-level noise that literal markers can't catch: standalone page-number
lines, ALL-CAPS lines, and leading blank-line runs.

Usage:
  python census_v0.py [--path ~/data/book.jsonl] [--max-gb 8] [--report out.json]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

# Literal markers, searched case-insensitively (book text is lowercased once
# per record; str.find/str.count are memchr-fast, unlike re alternations).
# Format: key -> list of lowercase literal variants that all count as one marker.
MARKERS = {
    "pg_start_banner":    ["*** start of", "***start of"],
    "pg_end_banner":      ["*** end of", "***end of"],
    "pg_end_text":        ["end of the project gutenberg", "end of project gutenberg",
                           "end of this project gutenberg"],
    "pg_mention":         ["project gutenberg"],
    "pg_url":             ["gutenberg.org", "gutenberg.net"],
    "small_print":        ["small print"],
    "license_block":      ["full license"],
    "produced_by":        ["produced by"],
    "etext_prepared_by":  ["e-text prepared by", "etext prepared by", "text prepared by"],
    "transcriber_note":   ["transcriber's note", "transcribers note", "transcriber’s note",
                           "transcriber note"],
    "distributed_proof":  ["distributed proofread", "pgdp"],
    "internet_archive":   ["internet archive"],
    "scanned_by":         ["scanned by", "scanned images"],
    "proofread_mention":  ["proofread"],
    "illustration_tag":   ["[illustration"],
    "footnote_tag":       ["[footnote"],
    "sidenote_tag":       ["[sidenote"],
    "copyright_line":     ["copyright"],
    "rights_reserved":    ["all rights reserved"],
    "public_domain":      ["public domain"],
    "ebook_mention":      ["ebook", "e-book"],
    "etext_mention":      ["etext", "e-text"],
    "typo_note":          ["typographical error"],
    "toc_heading":        ["\ncontents\n", "table of contents"],
    "index_heading":      ["\nindex\n"],
}

MAX_EXAMPLES_PER_MARKER = 3
EXAMPLE_CONTEXT = 120  # chars either side of the hit


def scan_line_structure(text: str) -> dict:
    """Per-line structural stats for one book (used on a sample of books)."""
    lines = text.split("\n")
    n = len(lines)
    digit_only = 0       # standalone page numbers: "123", "-123-", "[123]"
    all_caps = 0         # shouty lines: running headers, headings
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        core = s.strip("-[]{}() .|*_")
        if core.isdigit():
            digit_only += 1
            continue
        alpha = [c for c in s if c.isalpha()]
        if len(alpha) >= 4 and all(c.isupper() for c in alpha):
            all_caps += 1
    leading_blanks = 0
    for ln in lines:
        if ln.strip():
            break
        leading_blanks += 1
    return {
        "lines": n,
        "digit_only_lines": digit_only,
        "all_caps_lines": all_caps,
        "leading_blank_lines": leading_blanks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.path.expanduser("~/data/book.jsonl"))
    ap.add_argument("--max-gb", type=float, default=8.0,
                    help="scan at most this many GB from the start of the file")
    ap.add_argument("--margin-gb", type=float, default=2.0,
                    help="stay this far behind the current end of file (copy in progress)")
    ap.add_argument("--structure-every", type=int, default=25,
                    help="run the per-line structural scan on every Nth book")
    ap.add_argument("--report", default=None, help="write full JSON report here")
    args = ap.parse_args()

    file_size = os.path.getsize(args.path)
    limit = min(int(args.max_gb * 1e9), max(0, file_size - int(args.margin_gb * 1e9)))
    if limit <= 0:
        sys.exit(f"not enough stable data yet (file={file_size/1e9:.1f}GB, margin={args.margin_gb}GB)")

    occurrences = Counter()          # marker -> total hits
    books_with = Counter()           # marker -> books containing >=1 hit
    pos_sum = defaultdict(float)     # marker -> sum of fractional first-hit positions
    examples = defaultdict(list)     # marker -> [{title, pos_frac, snippet}]
    head_has_pg = 0                  # "project gutenberg" within first 1500 chars
    tail_has_pg = 0                  # "project gutenberg" within last 5000 chars
    structure = Counter()            # aggregated per-line stats over sampled books
    structure_books = 0
    books = 0
    bytes_read = 0
    t0 = time.time()

    with open(args.path, "rb") as f:
        for raw in f:
            bytes_read += len(raw)
            if bytes_read > limit:
                break
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                break  # truncated tail record; treat as end of stable region
            text = rec.get("text", "")
            title = (rec.get("meta") or {}).get("short_book_title", "?")
            low = text.lower()
            tlen = max(len(low), 1)
            books += 1

            for key, variants in MARKERS.items():
                total = 0
                first = -1
                for v in variants:
                    c = low.count(v)
                    if c:
                        total += c
                        p = low.find(v)
                        if first < 0 or p < first:
                            first = p
                if total:
                    occurrences[key] += total
                    books_with[key] += 1
                    pos_sum[key] += first / tlen
                    if len(examples[key]) < MAX_EXAMPLES_PER_MARKER:
                        s = max(0, first - EXAMPLE_CONTEXT)
                        examples[key].append({
                            "title": title,
                            "book_index": books - 1,
                            "pos_frac": round(first / tlen, 4),
                            "snippet": text[s:first + EXAMPLE_CONTEXT],
                        })

            if "project gutenberg" in low[:1500]:
                head_has_pg += 1
            if "project gutenberg" in low[-5000:]:
                tail_has_pg += 1

            if books % args.structure_every == 1:
                st = scan_line_structure(text)
                structure.update(st)
                structure_books += 1

    elapsed = time.time() - t0
    report = {
        "file": args.path,
        "bytes_scanned": bytes_read,
        "books_scanned": books,
        "elapsed_sec": round(elapsed, 1),
        "head_has_pg_frac": round(head_has_pg / max(books, 1), 4),
        "tail_has_pg_frac": round(tail_has_pg / max(books, 1), 4),
        "markers": {
            key: {
                "occurrences": occurrences[key],
                "books_with": books_with[key],
                "books_with_frac": round(books_with[key] / max(books, 1), 4),
                "mean_first_pos": round(pos_sum[key] / books_with[key], 4) if books_with[key] else None,
                "examples": examples[key],
            }
            for key in MARKERS
        },
        "structure_sample": {
            "books_sampled": structure_books,
            **{k: structure[k] for k in structure},
        },
    }

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as out:
            json.dump(report, out, indent=2)

    # Console summary table
    print(f"\nScanned {books:,} books / {bytes_read/1e9:.2f} GB in {elapsed:.0f}s "
          f"({bytes_read/1e6/elapsed:.0f} MB/s)\n")
    print(f"'project gutenberg' in first 1500 chars: {100*head_has_pg/max(books,1):.1f}% of books")
    print(f"'project gutenberg' in last  5000 chars: {100*tail_has_pg/max(books,1):.1f}% of books\n")
    rows = sorted(MARKERS, key=lambda k: -books_with[k])
    print(f"{'marker':<20} {'books%':>7} {'books':>7} {'occurrences':>12} {'mean_pos':>9}")
    for key in rows:
        bw = books_with[key]
        mp = f"{pos_sum[key]/bw:.3f}" if bw else "-"
        print(f"{key:<20} {100*bw/max(books,1):>6.1f}% {bw:>7,} {occurrences[key]:>12,} {mp:>9}")
    if structure_books:
        tot_lines = max(structure["lines"], 1)
        print(f"\nStructural sample ({structure_books} books, {tot_lines:,} lines):")
        print(f"  standalone page-number lines: {structure['digit_only_lines']:,} "
              f"({100*structure['digit_only_lines']/tot_lines:.2f}% of lines)")
        print(f"  ALL-CAPS lines:               {structure['all_caps_lines']:,} "
              f"({100*structure['all_caps_lines']/tot_lines:.2f}% of lines)")
        print(f"  mean leading blank lines:     {structure['leading_blank_lines']/structure_books:.1f}")


if __name__ == "__main__":
    main()
