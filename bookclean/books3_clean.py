#!/usr/bin/env python3
"""Production books3 front/back-matter cleaner wrapping bake-off sentinel.

Safety rails on top of sentinel:
  - head cut capped at min(15000 chars, 15% of book); tail same. Beyond cap:
    leave text unchanged, flag in ledger for review.
  - head cut requires a strong head anchor within the cut region
  - windows: head 15000, tail 10000 chars
Every decision (cut or skip) is ledgered with reason + removed-text preview.
"""
import json, re, sys
sys.path.insert(0, '.')
from bakeoff import sentinel_head, sentinel_tail, STRONG_TAIL

HEAD_W, TAIL_W = 15000, 10000
STRONG_HEAD = re.compile(
    r'(?i)(?:©|\bisbn\b|\ball rights reserved\b|\bcopyright\b|\blibrary of congress\b|'
    r'\bfirst published\b|\bpublish\w* by\b|\balso by\b|\btable of contents\b|'
    r'^#+ ?contents\b|\bcataloging.in.publication\b)', re.M)

def _nonlatin_frac(t):
    letters = [c for c in t[:6000] if c.isalpha()]
    return (sum(1 for c in letters if ord(c) > 0x24F) / len(letters)) if letters else 0.0


def clean_books3(text):
    """Returns (new_text, ledger_entry)."""
    if _nonlatin_frac(text) > 0.3:
        return text, {'skipped': 'non-latin script - needs language-aware pass'}
    n = len(text)
    entry = {}
    head_cap = min(HEAD_W, max(int(n * 0.15), 3000))
    head = text[:HEAD_W]
    hs = sentinel_head(head)
    cut_start = 0
    if hs and hs > 0:
        if hs >= len(head) - 100:
            entry['head'] = {'action': 'flagged', 'reason': 'window-max verdict (no content found in window)'}
            hs = 0
        elif hs > head_cap:
            entry['head'] = {'action': 'flagged', 'reason': f'cut {hs} > cap {head_cap}'}
        elif not STRONG_HEAD.search(head[:hs]):
            entry['head'] = {'action': 'skipped', 'reason': 'no strong head anchor in cut region'}
        else:
            cut_start = hs
            entry['head'] = {'action': 'cut', 'chars': hs,
                             'removed_preview': ' '.join(head[:hs].split())[:150]}
    tail = text[-TAIL_W:]
    te = sentinel_tail(tail)
    cut_end = n
    if te is not None and te < len(tail):
        removed = len(tail) - te
        tail_cap = min(TAIL_W, max(int(n * 0.15), 3000))
        if removed > tail_cap:
            entry['tail'] = {'action': 'flagged', 'reason': f'cut {removed} > cap {tail_cap}'}
        else:
            cut_end = n - removed
            entry['tail'] = {'action': 'cut', 'chars': removed,
                             'removed_preview': ' '.join(tail[te:].split())[:150]}
    total_cut = cut_start + (n - cut_end)
    if total_cut > 0.40 * n:
        entry['total'] = {'action': 'flagged', 'reason': f'total cut {total_cut} > 40% of {n}'}
        return text, entry
    if cut_start or cut_end < n:
        return text[cut_start:cut_end].strip('\n'), entry
    return text, entry

if __name__ == '__main__':
    # validation over a 300-book random sample
    import os, random
    rng = random.Random(99)
    rows = []
    for line in open('reports/book_index_v1.tsv'):
        i, off, ln, reg, title = line.rstrip('\n').split('\t')
        if reg == 'books3' and int(ln) > 30000:
            rows.append((int(off), title))
    sample = rng.sample(rows, 300)
    from collections import Counter
    actions = Counter(); cut_sizes = []
    examples = []
    with open(os.path.expanduser('~/data/book.v1.jsonl'), 'rb') as f:
        for off, title in sample:
            f.seek(off); rec = json.loads(f.readline())
            t2, entry = clean_books3(rec['text'])
            for side in ('head','tail'):
                if side in entry:
                    actions[f"{side}:{entry[side]['action']}"] += 1
                    if entry[side]['action'] == 'cut':
                        cut_sizes.append(entry[side]['chars'])
                else:
                    actions[f"{side}:none"] += 1
            if len(examples) < 6 and entry.get('head',{}).get('action') == 'cut':
                examples.append((title[:45], entry['head']['chars'], t2[:110]))
    import statistics as st
    print(dict(actions))
    if cut_sizes:
        print(f"cut sizes: median={st.median(cut_sizes):.0f} p90={sorted(cut_sizes)[int(len(cut_sizes)*0.9)]} max={max(cut_sizes)}")
    print("\nhead-cut examples (title | cut chars | new start):")
    for t, c, start in examples:
        print(f"  [{t}] -{c}: {' '.join(start.split())[:95]!r}")
