#!/usr/bin/env python3
"""PG19 v2 cleaning passes: head credits, old-style Etext banners, tail
sentinel, leading-blank normalization. Ledgered and reversible."""
import re

CREDIT_OPEN = re.compile(
    r'(?i)^\s*(?:produced by|e-?text prepared by|transcribed from|prepared by|'
    r'this etext was (?:prepared|produced|created) by|this file was produced (?:by|from)|'
    r'scanned by|etext (?:by|prepared)|typed by|digitized by)')
CREDIT_CONT = re.compile(
    r'(?i)^\s*(?:and the (?:pg )?online distributed|images? (?:of this book|generously)|'
    r'(?:this (?:e-?text|file|book)|which) (?:is|was|includes)|from (?:page )?(?:images|scans)|'
    r'html version by|updated? (?:editions?|by)|special thanks|with thanks to|'
    r'this etext was produced|proofread(?:ing)? by|note: (?:images?|the))')
OLD_BANNER = re.compile(
    r'(?i)^\s*\*{0,4}\s*(?:the project gutenberg\'?s? etext of|'
    r'this is the \d+\w* etext (?:file )?presented by|'
    r'the project gutenberg etext)')
OLD_BANNER_CONT = re.compile(
    r'(?i)^\s*(?:#\d+ in our series|copyright laws are changing|'
    r'please (?:take a look|do not remove)|this (?:should be|header should be) the first|'
    r'we encourage you|title:|author:|release date:|edition:|language:|character set)')
TAIL_SENTINEL = re.compile(
    r'(?i)\n\s*end of (?:the |this )?project gutenberg.{0,300}$', re.S)

def clean_head(text):
    """Strip leading credit/banner paragraphs; normalize leading blanks.
    Returns (cleaned_text, removed_list)."""
    removed = []
    # split off up to the first 10 paragraphs
    paras = re.split(r'(\n\s*\n)', text[:4000], maxsplit=20)
    # paras alternates [para, sep, para, sep, ...]
    consumed = 0
    i = 0
    stripping = True
    mode = None
    strip_budget = 0
    while stripping and i < len(paras):
        p = paras[i]
        if not p.strip():
            consumed += len(p); i += 1; continue
        if mode is None:
            if CREDIT_OPEN.match(p):
                mode, strip_budget = 'credit', 3
            elif OLD_BANNER.match(p):
                mode, strip_budget = 'banner', 8
            else:
                break
            removed.append(p.strip()[:200])
            consumed += len(p); i += 1
            if i < len(paras): consumed += len(paras[i]); i += 1  # separator
            continue
        cont = CREDIT_CONT if mode == 'credit' else OLD_BANNER_CONT
        if strip_budget > 0 and (cont.match(p) or
                                 (mode == 'banner' and OLD_BANNER.match(p))):
            removed.append(p.strip()[:200])
            strip_budget -= 1
            consumed += len(p); i += 1
            if i < len(paras): consumed += len(paras[i]); i += 1
            continue
        break
    rest = text[consumed:] if removed else text
    # normalize leading whitespace to zero blank lines
    rest = rest.lstrip('\n ').rstrip() if removed else rest
    if removed:
        rest = rest  # content starts immediately
    return rest, removed

def clean_tail(text):
    """Truncate from the tail sentinel; also strip dangling *** and blanks."""
    tail_zone = text[-2500:]
    m = TAIL_SENTINEL.search(tail_zone)
    removed = None
    if m:
        cut = len(text) - len(tail_zone) + m.start()
        removed = text[cut:].strip()[:300]
        text = text[:cut]
    text = re.sub(r'[\s*]+$', '', text)
    return text, removed

def clean_book(text):
    removed_head = []
    for _ in range(4):  # iterate: stripping reveals second-layer banners
        text, rem = clean_head(text)
        if not rem:
            break
        removed_head.extend(rem)
    text, removed_tail = clean_tail(text)
    return text, {'head': removed_head, 'tail': removed_tail}

if __name__ == '__main__':
    # sample test over PG19 books
    import json, os, random
    rng = random.Random(42)
    lines = []
    for line in open('reports/book_index_v1.tsv'):
        i, off, ln, reg, title = line.rstrip('\n').split('\t')
        if reg == 'pg19': lines.append((int(off), title))
    sample = rng.sample(lines, 400)
    stats = {'head_stripped': 0, 'tail_stripped': 0, 'neither': 0}
    examples = []
    with open(os.path.expanduser('~/data/book.v1.jsonl'), 'rb') as f:
        for off, title in sample:
            f.seek(off); rec = json.loads(f.readline())
            t2, rem = clean_book(rec['text'])
            if rem['head']: stats['head_stripped'] += 1
            if rem['tail']: stats['tail_stripped'] += 1
            if not rem['head'] and not rem['tail']: stats['neither'] += 1
            if len(examples) < 8 and rem['head']:
                examples.append((title[:45], rem['head'], t2[:90]))
    print(stats)
    print("\nexamples (title | removed | new first chars):")
    for t, rh, start in examples:
        print(f"\n  [{t}]")
        for r in rh: print(f"    REMOVED: {' '.join(r.split())[:110]}")
        print(f"    NOW STARTS: {' '.join(start.split())[:90]!r}")
