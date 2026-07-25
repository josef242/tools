#!/usr/bin/env python3
"""Full-corpus <DWnn> redaction-token census over book.jsonl.

Parallel scan of the raw bytes (no JSON parsing needed): each worker takes a
contiguous byte range (with small overlap so tokens straddling boundaries are
not lost; a match is owned by the range containing its first byte).

Outputs per token id:
  - total count
  - count histogram over 1GB buckets (localizes PG19 vs modern sections)
  - first/last byte offset seen
  - after-character distribution
  - up to K sampled contexts (more for ids on the WANT_MORE list)

Usage: python dw_census_full.py [--workers 12] [--out reports/dw_census_full.json]
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from multiprocessing import Pool

PATH = os.path.expanduser("~/data/book.jsonl")
TOKEN = re.compile(rb"<DW(\d+)>")
OVERLAP = 32          # bytes of overlap between ranges to catch straddlers
CTX = 90              # context bytes either side
K_DEFAULT = 12        # sampled contexts per id per worker
K_MORE = 60           # for ids we still need to decode
WANT_MORE = {8, 9, 40, 56, 67, 75, 76}
GB = 1_000_000_000


def scan_range(task):
    start, end = task
    counts = defaultdict(int)
    gb_hist = defaultdict(lambda: defaultdict(int))
    first_off = {}
    last_off = {}
    after = defaultdict(lambda: defaultdict(int))
    ctx = defaultdict(list)
    rng = random.Random(start)

    with open(PATH, "rb") as f:
        f.seek(start)
        pos = start
        read_end = end + OVERLAP
        while pos < read_end:
            chunk = f.read(min(64_000_000, read_end - pos))
            if not chunk:
                break
            for m in TOKEN.finditer(chunk):
                abs_off = pos + m.start()
                if abs_off >= end:
                    continue  # owned by the next range
                if abs_off < start:
                    continue
                tid = int(m.group(1))
                counts[tid] += 1
                gb_hist[tid][abs_off // GB] += 1
                if tid not in first_off:
                    first_off[tid] = abs_off
                last_off[tid] = abs_off
                nxt = chunk[m.end():m.end() + 1].decode("latin1") or "?"
                after[tid][nxt] += 1
                k = K_MORE if tid in WANT_MORE else K_DEFAULT
                if len(ctx[tid]) < k and rng.random() < 0.5:
                    s = max(0, m.start() - CTX)
                    ctx[tid].append({
                        "offset": abs_off,
                        "text": chunk[s:m.end() + CTX].decode("utf-8", "replace"),
                    })
            # rewind so a token spanning the chunk edge is re-seen next read
            step = len(chunk) - OVERLAP if len(chunk) == 64_000_000 else len(chunk)
            pos += step
            if step < len(chunk):
                f.seek(pos)
    return {
        "counts": dict(counts),
        "gb_hist": {t: dict(h) for t, h in gb_hist.items()},
        "first_off": first_off,
        "last_off": last_off,
        "after": {t: dict(a) for t, a in after.items()},
        "ctx": dict(ctx),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="reports/dw_census_full.json")
    args = ap.parse_args()

    size = os.path.getsize(PATH)
    n = args.workers * 4  # smaller tasks -> better load balance
    bounds = [size * i // n for i in range(n + 1)]
    tasks = list(zip(bounds[:-1], bounds[1:]))

    counts = defaultdict(int)
    gb_hist = defaultdict(lambda: defaultdict(int))
    first_off = {}
    last_off = {}
    after = defaultdict(lambda: defaultdict(int))
    ctx = defaultdict(list)

    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(scan_range, tasks)):
            for t, c in r["counts"].items():
                counts[t] += c
            for t, h in r["gb_hist"].items():
                for g, c in h.items():
                    gb_hist[t][g] += c
            for t, o in r["first_off"].items():
                first_off[t] = min(first_off.get(t, o), o)
            for t, o in r["last_off"].items():
                last_off[t] = max(last_off.get(t, o), o)
            for t, a in r["after"].items():
                for ch, c in a.items():
                    after[t][ch] += c
            for t, xs in r["ctx"].items():
                cap = K_MORE * 3 if t in WANT_MORE else K_DEFAULT * 3
                if len(ctx[t]) < cap:
                    ctx[t].extend(xs[: cap - len(ctx[t])])
            print(f"[{i+1}/{n}] ranges done", flush=True)

    ids = sorted(counts)
    report = {
        "file": PATH,
        "file_size": size,
        "distinct_ids": len(ids),
        "id_range": [ids[0], ids[-1]] if ids else None,
        "total_occurrences": sum(counts.values()),
        "last_token_offset_gb": round(max(last_off.values()) / GB, 2) if last_off else None,
        "ids": {
            str(t): {
                "count": counts[t],
                "first_offset_gb": round(first_off[t] / GB, 3),
                "last_offset_gb": round(last_off[t] / GB, 3),
                "after_chars": dict(sorted(after[t].items(), key=lambda kv: -kv[1])[:8]),
                "gb_hist": {str(g): c for g, c in sorted(gb_hist[t].items())},
                "contexts": ctx[t],
            }
            for t in ids
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)

    print(f"\ndistinct ids: {len(ids)}  range: {ids[0]}..{ids[-1]}")
    print(f"total occurrences: {sum(counts.values()):,}")
    print(f"last token at: {max(last_off.values())/GB:.2f} GB of {size/GB:.2f} GB")
    missing = [i for i in range(ids[0], ids[-1] + 1) if i not in counts]
    print(f"missing ids in range: {missing}")
    for t in ids:
        print(f"  DW{t:<3} count={counts[t]:>9,}  span={first_off[t]/GB:.2f}-{last_off[t]/GB:.2f} GB")


if __name__ == "__main__":
    main()
