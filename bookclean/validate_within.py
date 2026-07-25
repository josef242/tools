#!/usr/bin/env python3
"""Validate within_dedup_filter: (A) gold zero-content-damage audit on the 140
hand-vetted tail books, (B) corpus-wide removed-line audit + per-book removal-rate
+ flagged books on a random v10 sample, (C) timing probe (MANDATORY pre-flight
before any full run)."""
import json, os, sys, time, re, random
sys.path.insert(0, '.')
from within_dedup_filter import within_dedup_book, norm, alpha_words

V1  = os.path.expanduser('~/data/book.v1.jsonl')
V10 = os.path.expanduser('~/data/book.v10.jsonl')

def load_by_offset(path, off):
    with open(path, 'rb') as f:
        f.seek(off)
        return json.loads(f.readline())

# ---- (A) GOLD content-damage audit --------------------------------------------
gold_files = ['gold_tails_books.json', 'gold_tails_test_books.json',
              'gold_tails_val3_books.json', 'gold_tails_strat_books.json']
offsets = []
for gf in gold_files:
    for e in json.load(open('reports/' + gf)):
        offsets.append(e['offset'])
offsets = sorted(set(offsets))
print(f"=== (A) GOLD content-damage audit: {len(offsets)} unique gold books (from v1) ===")

PROSE = re.compile(r"[a-z]{2,}")
def looks_like_prose(line):
    # a removed line is a content-damage ALARM if it reads like a sentence
    s = line.strip()
    return alpha_words(norm(s)) >= 6 and s[-1] in '.!?"”\'' if s else False

gold_removed = 0; gold_alarms = []; touched = 0
per_book = []
for off in offsets:
    try:
        rec = load_by_offset(V1, off)
    except Exception as e:
        print("  skip offset", off, type(e).__name__); continue
    txt = rec['text']
    out, info = within_dedup_book(txt)
    if info.get('flagged'):
        per_book.append((off, 'FLAGGED', info['reason'])); continue
    r = info.get('removed', 0)
    if r:
        touched += 1; gold_removed += r
        kept = set(out.split('\n'))
        removed_lines = [l for l in txt.split('\n') if l and l not in kept]
        # collect distinct removed lines for this book
        for l in set(removed_lines):
            if looks_like_prose(l):
                gold_alarms.append((off, l.strip()[:90]))
        per_book.append((off, r, len(txt.split('\n'))))

print(f"  books touched: {touched}/{len(offsets)}   total lines removed: {gold_removed}")
print(f"  CONTENT-DAMAGE ALARMS (removed line reads like prose): {len(gold_alarms)}")
for off, l in gold_alarms[:40]:
    print(f"    !! {off}  {l!r}")
if not gold_alarms:
    print("  -> ZERO prose-shaped removals on gold. clean.")

# ---- (B) corpus-wide removed-line audit + removal-rate + flagged ---------------
print("\n=== (B) v10 random-sample audit ===")
size = os.path.getsize(V10)
rng = random.Random(1234)
N = 3000
from collections import Counter
audit = Counter(); rates = []; flagged = 0; sampled = 0; degen = 0; furn = 0
removed_samples = []
with open(V10, 'rb') as f:
    for _ in range(N):
        f.seek(rng.randrange(size))
        f.readline()                      # discard partial
        raw = f.readline()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except Exception:
            continue
        sampled += 1
        txt = rec['text']
        out, info = within_dedup_book(txt)
        if info.get('flagged'):
            flagged += 1; continue
        r = info.get('removed', 0)
        if r:
            ne = max(len([l for l in txt.split('\n') if l.strip()]), 1)
            rates.append(r / ne)
            reasons = info.get('reasons', {})
            degen += reasons.get('degen', 0); furn += reasons.get('furniture', 0)
            audit['touched'] += 1
            if len(removed_samples) < 60:
                kept = set(out.split('\n'))
                rl = [l for l in txt.split('\n') if l and l not in kept]
                removed_samples.append(rl[0].strip()[:80] if rl else '')
print(f"  sampled {sampled} books | touched {audit['touched']} ({100*audit['touched']/max(sampled,1):.1f}%)"
      f" | flagged {flagged} ({100*flagged/max(sampled,1):.2f}%)")
print(f"  removed-line reasons: furniture={furn}  degenerate={degen}")
if rates:
    rates.sort()
    print(f"  per-book removal-rate: median={rates[len(rates)//2]:.4f}  "
          f"p90={rates[int(len(rates)*.9)]:.4f}  max={rates[-1]:.4f}")
print("  sample removed lines (want 100% furniture):")
for s in removed_samples[:40]:
    print(f"    - {s!r}")

# ---- (C) timing probe ----------------------------------------------------------
print("\n=== (C) TIMING PROBE ===")
t0 = time.time(); nb = 0; nchar = 0
with open(V10, 'rb') as f:
    for _ in range(2000):
        raw = f.readline()
        if not raw:
            break
        try:
            rec = json.loads(raw)
        except Exception:
            continue
        nb += 1; nchar += len(rec['text'])
        within_dedup_book(rec['text'])
dt = time.time() - t0
rate = nb / dt
print(f"  {nb} books in {dt:.1f}s = {rate:.0f} books/s single-thread")
print(f"  full corpus 205,744 books: ~{205744/rate/60:.1f} min single-thread, "
      f"~{205744/rate/60/10:.1f} min at 10 workers")
