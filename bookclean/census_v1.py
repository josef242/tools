#!/usr/bin/env python3
"""Census v1: full-corpus, region-aware noise survey of book.v1.jsonl.

Classifies each record as pg19 (meta has short_book_title) or books3 (bare
title), counts an expanded battery of noise markers per region, and harvests
concrete examples (title + snippet) per marker plus random head/tail samples
for human failure-case review.

Parallel over byte ranges; a line is owned by the range containing its first
byte. Pattern matching is byte-level on the lowercased raw JSON line (the
text field's newlines appear as the two-byte sequence '\\n').
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from multiprocessing import Pool

PATH = os.path.expanduser("~/data/book.v1.jsonl")
GB = 1_000_000_000

# marker -> list of lowercase byte literals (any counts as the marker)
PATTERNS = {
    "pg_mention":        [b"project gutenberg"],
    "produced_by":       [b"produced by", b"this file was produced"],
    "pg_end_sentinel":   [b"end of the project gutenberg", b"end of project gutenberg",
                          b"end of this project gutenberg"],
    "transcriber_note":  [b"transcriber"],
    "illustration_tag":  [b"[illustration"],
    "footnote_tag":      [b"[footnote"],
    "sidenote_tag":      [b"[sidenote"],
    "toc_heading":       [b"table of contents", b"\\ncontents\\n"],
    "index_heading":     [b"\\nindex\\n"],
    "copyright":         [b"copyright"],
    "rights_reserved":   [b"all rights reserved"],
    "isbn":              [b"isbn"],
    "url_http":          [b"http://", b"https://"],
    "url_www":           [b"www."],
    "email_at":          [b"@gmail", b"@yahoo", b"@hotmail"],
    "library_congress":  [b"library of congress"],
    "printed_in":        [b"printed in the united states", b"printed in great britain"],
    "first_edition":     [b"first edition", b"first published"],
    "about_author":      [b"about the author"],
    "also_by":           [b"also by ", b"books by "],
    "visit_website":     [b"visit our website", b"visit us at", b"sign up for",
                          b"newsletter"],
    "publisher_big5":    [b"harpercollins", b"penguin", b"random house",
                          b"simon & schuster", b"hachette", b"macmillan",
                          b"harvest house"],
    "kindle_ebook":      [b"kindle", b"epub"],
    "social_media":      [b"facebook", b"twitter.com", b"instagram"],
    "star_separator":    [b"* * *"],
    "html_entity":       [b"&amp;", b"&nbsp;", b"&quot;", b"&#"],
    "html_tag":          [b"</", b"<i>", b"<b>", b"<p>"],
    "bullet_char":       ["•".encode()],
    "copyright_sym":     ["©".encode()],
    "trademark_sym":     ["™".encode()],
    "hyphen_linebreak":  [b"-\\n"],
    "dedication":        [b"\\ndedication\\n", b"for my ", b"to my wife", b"to my husband"],
    "acknowledgments":   [b"acknowledgment", b"acknowledgement"],
    "appendix":          [b"\\nappendix"],
    "glossary":          [b"\\nglossary"],
    "bibliography":      [b"bibliograph"],
    "errata":            [b"errata"],
}

MAX_EX = 4          # examples per (pattern, region)
HEAD_TAIL_N = 14    # random head/tail samples per region


def get_title(raw):
    m = re.search(rb'"(?:short_book_title|title)":\s*"((?:[^"\\]|\\.){0,120})', raw[:400])
    return (m.group(1).decode("utf-8", "replace") if m else "?")[:80]


def snippet_around(raw, pos, radius=110):
    s = raw[max(0, pos - radius):pos + radius].decode("utf-8", "replace")
    return s.replace("\\n", " ").strip()


def scan_range(task):
    start, end, seed = task
    rng = random.Random(seed)
    counts = defaultdict(lambda: [0, 0])        # (region,pat) keyed by tuple -> [occ, books]
    examples = defaultdict(list)
    heads, tails = defaultdict(list), defaultdict(list)
    nbooks = defaultdict(int)
    nbytes = defaultdict(int)

    with open(PATH, "rb") as f:
        f.seek(start)
        if start:
            f.readline()  # skip partial line owned by previous range
        pos = f.tell()
        while pos < end:
            raw = f.readline()
            if not raw:
                break
            pos += len(raw)
            region = "pg19" if b'"short_book_title"' in raw[:300] else "books3"
            low = raw.lower()
            nbooks[region] += 1
            nbytes[region] += len(raw)
            for key, lits in PATTERNS.items():
                total = 0
                first = -1
                for lit in lits:
                    c = low.count(lit)
                    if c:
                        total += c
                        p = low.find(lit)
                        if first < 0 or p < first:
                            first = p
                if total:
                    k = (region, key)
                    counts[k][0] += total
                    counts[k][1] += 1
                    if len(examples[k]) < MAX_EX and rng.random() < 0.25:
                        examples[k].append({"title": get_title(raw),
                                            "snippet": snippet_around(raw, first)})
            if rng.random() < 0.002 and len(heads[region]) < HEAD_TAIL_N:
                t = raw.find(b'"text":')
                body = raw[t + 8:]
                heads[region].append({"title": get_title(raw),
                                      "head": body[:500].decode("utf-8", "replace").replace("\\n", "\n")})
                tails[region].append({"title": get_title(raw),
                                      "tail": body[-600:].decode("utf-8", "replace").replace("\\n", "\n")})
    return {"counts": {f"{r}|{k}": v for (r, k), v in counts.items()},
            "examples": {f"{r}|{k}": v for (r, k), v in examples.items()},
            "heads": dict(heads), "tails": dict(tails),
            "nbooks": dict(nbooks), "nbytes": dict(nbytes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="reports/census_v1.json")
    args = ap.parse_args()

    size = os.path.getsize(PATH)
    n = args.workers * 4
    bounds = [size * i // n for i in range(n + 1)]
    tasks = [(bounds[i], bounds[i + 1], i) for i in range(n)]

    counts = defaultdict(lambda: [0, 0])
    examples = defaultdict(list)
    heads, tails = defaultdict(list), defaultdict(list)
    nbooks, nbytes = defaultdict(int), defaultdict(int)

    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(scan_range, tasks)):
            for k, (o, b) in r["counts"].items():
                counts[k][0] += o
                counts[k][1] += b
            for k, xs in r["examples"].items():
                if len(examples[k]) < MAX_EX * 2:
                    examples[k].extend(xs[:MAX_EX * 2 - len(examples[k])])
            for reg in r["heads"]:
                if len(heads[reg]) < HEAD_TAIL_N * 2:
                    heads[reg].extend(r["heads"][reg])
                    tails[reg].extend(r["tails"][reg])
            for reg, c in r["nbooks"].items():
                nbooks[reg] += c
            for reg, c in r["nbytes"].items():
                nbytes[reg] += c
            print(f"[{i+1}/{n}] done", flush=True)

    report = {
        "file": PATH,
        "regions": {reg: {"books": nbooks[reg], "gb": round(nbytes[reg] / GB, 2)}
                    for reg in nbooks},
        "markers": {k: {"occurrences": counts[k][0], "books_with": counts[k][1],
                        "examples": examples.get(k, [])}
                    for k in sorted(counts)},
        "random_heads": dict(heads),
        "random_tails": dict(tails),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)

    for reg in sorted(nbooks):
        print(f"\n=== {reg}: {nbooks[reg]:,} books, {nbytes[reg]/GB:.1f} GB ===")
        rows = [(k.split("|")[1], counts[k]) for k in counts if k.startswith(reg + "|")]
        rows.sort(key=lambda kv: -kv[1][1])
        print(f"{'marker':<20} {'books%':>7} {'books':>8} {'occurrences':>13}")
        for key, (occ, bw) in rows:
            print(f"{key:<20} {100*bw/max(nbooks[reg],1):>6.1f}% {bw:>8,} {occ:>13,}")


if __name__ == "__main__":
    main()
