#!/usr/bin/env python3
"""Harm-per-pass: how much LEARNABLE CONTENT does each pass wrongly cut, measured
against the gold test set (140 tail books with a labeled content_end boundary).
This is the asymmetric-cost 'content eaten' term — the thing we care most about.

For each pass we apply its REAL predicates to each gold book and count characters
removed from the CONTENT region [0, content_end) of the tail:
  - STRICT harm  = any content-region char removed (over-counts furniture that sits
    inside the content region, e.g. a page number in an afterword — not real harm).
  - GENUINE harm = only prose-shaped removed lines (the honest 'ate learnable text' number).
Junk cut = content-region's complement removed; reported for context (value, not harm).
Rejected multi-span is included as the anchor: it SHOULD show real harm."""
import json, os, sys, re, pickle
from collections import Counter
sys.path.insert(0, '.')
import sentinel3, dedup_filter, within_dedup_filter as wd
try:
    import span_cutter; HAVE_SPAN = True
except Exception as e:
    HAVE_SPAN = False; print("span_cutter import failed:", e)

V1 = os.path.expanduser('~/data/book.v1.jsonl')
FREQ = pickle.load(open('reports/line_freq.pkl', 'rb'))

# ---- load gold (offset -> content_end) and (offset -> tail) ----
labels = {}
for f in ['gold_tail_labels_hybrid', 'gold_tail_test_hybrid', 'gold_tail_val3_hybrid', 'gold_tail_strat_labels']:
    for e in json.load(open(f'reports/{f}.json')):
        labels[e['offset']] = e['content_end']
books = {}
for f in ['gold_tails_books', 'gold_tails_test_books', 'gold_tails_val3_books', 'gold_tails_strat_books']:
    for e in json.load(open(f'reports/{f}.json')):
        books[e['offset']] = e['tail']
offs = [o for o in labels if o in books]
print(f"gold books with both label + tail: {len(offs)}")

def load_full(off):
    with open(V1, 'rb') as f:
        f.seek(off); return json.loads(f.readline())['text']

def is_prose(s):
    s = s.strip()
    return bool(s) and len(re.findall(r'[a-z]{2,}', s.lower())) >= 6 and s[-1] in '.!?"\'”)'

def real_content_chars(text):
    """Chars of lines that are genuinely LEARNABLE content: prose-shaped AND not
    publisher boilerplate (dedup's STRONG_LINE) AND not furniture-shaped. This
    excludes the false-harm cases — copyright lines and page numbers that happen
    to sit inside the gold content region — and keeps real prose / bibliographies."""
    tot = 0
    for l in text.split('\n'):
        s = l.strip()
        if not s or not is_prose(l):
            continue
        if dedup_filter.STRONG_LINE.search(s):          # publisher/copyright boilerplate
            continue
        nl = wd.norm(l)
        if wd.is_furniture(l, nl, 999):                 # furniture SHAPE (count-agnostic probe)
            continue
        tot += len(l)
    return tot

def tail_start_of(full, tail):
    """Robustly locate the tail inside full (offsets index v1; tail is the v1 suffix)."""
    ts = len(full) - len(tail)
    if ts >= 0 and full[ts:] == tail:
        return ts
    p = full.rfind(tail[:400])            # fallback: search for the tail head
    return p if p >= 0 else None

# ---- per-pass removed-line index sets (using each filter's REAL predicates) ----
def cross_removed(full):
    lines = full.split('\n'); ne = [i for i, l in enumerate(lines) if l.strip()]
    rm = [i for i in ne if 12 <= len(lines[i].strip()) <= 250
          and dedup_filter.STRONG_LINE.search(lines[i].strip())
          and FREQ.get(dedup_filter.hh(lines[i].strip()), 0) >= dedup_filter.MIN_DUP]
    if len(rm) > dedup_filter.BREAKER_FRAC * max(len(ne), 1):
        return []
    return rm

def within_removed(full):
    lines = full.split('\n'); ne = [i for i, l in enumerate(lines) if l.strip()]
    counts = Counter(wd.norm(lines[i]) for i in ne)
    rm = []
    for i in ne:
        s = lines[i]
        if wd._degenerate(s):
            rm.append(i); continue
        nl = wd.norm(s); c = counts[nl]
        if c < wd.T_CREDIT or wd.is_protected(s, nl):
            continue
        if wd.is_furniture(s, nl, c):
            rm.append(i)
    if len(rm) > wd.BREAKER_FRAC * max(len(ne), 1):
        return []
    return rm

def line_removal_harm(full, tail, ce, rm_indices):
    """Map removed line indices to tail offsets; classify content vs junk."""
    ts = tail_start_of(full, tail)
    if ts is None:
        return None
    lines = full.split('\n'); rmset = set(rm_indices)
    harm = junk = 0; content_lines = []; pos = 0
    for i, l in enumerate(lines):
        if i in rmset and pos >= ts:
            t = pos - ts
            if t < ce:
                harm += len(l); content_lines.append(l)
            else:
                junk += len(l)
        pos += len(l) + 1
    return harm, real_content_chars('\n'.join(content_lines)), junk

def boundary_harm(full, tail, ce):
    cut = sentinel3.tail_boundary(tail)
    if cut >= ce:
        return 0, 0, len(tail) - cut          # cut at/after content_end -> 0 content eaten
    seg = tail[cut:ce]                          # content region wrongly removed
    return len(seg), real_content_chars(seg), len(tail) - ce

def span_harm(full, tail, ce):
    spans, _ = span_cutter.find_spans(tail, full)
    harm = harm_content = junk = 0
    for s, e in spans:
        cs, cse = max(s, 0), min(e, ce)
        if cse > cs:
            harm += cse - cs
            harm_content += real_content_chars(tail[cs:cse])
        js, je = max(s, ce), e
        if je > js:
            junk += je - js
    return harm, harm_content, junk

PASSES = [
    ('de-redaction (v1)',        'restore', None),
    ('structural boundary (v9)', 'boundary', None),
    ('cross-book dedup (v10)',   'lines', cross_removed),
    ('within-book dedup (v11)',  'lines', within_removed),
]
if HAVE_SPAN:
    PASSES.append(('multi-span cutter (REJECTED)', 'span', None))

results = {}
worst = {name: (0, '') for name, _, _ in PASSES}
books_hit = {name: 0 for name, _, _ in PASSES}      # gold books with ANY content char removed
skipped = 0
for off in offs:
    full = load_full(off); tail = books[off]; ce = labels[off]
    for name, kind, fn in PASSES:
        if kind == 'restore':
            h = (0, 0, 0)
        elif kind == 'boundary':
            h = boundary_harm(full, tail, ce)
        elif kind == 'span':
            h = span_harm(full, tail, ce)
        else:
            h = line_removal_harm(full, tail, ce, fn(full))
        if h is None:
            skipped += 1; continue
        a = results.setdefault(name, [0, 0, 0])
        a[0] += h[0]; a[1] += h[1]; a[2] += h[2]
        if h[0] > 0:
            books_hit[name] += 1
        if h[1] > worst[name][0]:
            worst[name] = (h[1], f"off={off}")

print(f"\n{'='*78}\nHARM PER PASS — content chars eaten on {len(offs)} gold tail books "
      f"(v1-based)\n{'='*78}")
print(f"{'pass':<32}{'STRICT':>9}{'REAL content':>14}{'books':>7}{'junk cut':>11}")
print(f"{'':<32}{'chars':>9}{'destroyed':>14}{'hit':>7}{'(context)':>11}")
print('-' * 78)
N = len(offs)
for name, _, _ in PASSES:
    a = results.get(name, [0, 0, 0])
    print(f"{name:<32}{a[0]:>9,}{a[1]:>14,}{books_hit[name]:>4}/{N:<2}{a[2]:>11,}")
print('-' * 78)
print("STRICT       = any content-region char removed (incl. boilerplate/furniture inside content).")
print("REAL content = prose AND not publisher-boilerplate AND not furniture = learnable text destroyed.")
print(f"worst single-book REAL-content harm: " +
      " | ".join(f"{n.split('(')[0].strip()}={worst[n][0]}" for n, _, _ in PASSES))
if skipped:
    print(f"(skipped {skipped} pass-book evals: tail not locatable in v1)")
json.dump({n: results.get(n, [0, 0, 0]) for n, _, _ in PASSES},
          open('reports/harm_per_pass.json', 'w'), indent=1)
