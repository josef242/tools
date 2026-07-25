#!/usr/bin/env python3
"""Sentinel v2: consolidated improvements over bakeoff.sentinel_head.
1. content-anchor headings (earliest anchor w/ sustained prose, min w/ prose-run)
2. long praise quotes (up to 700 chars, quote+attribution pairing)
3. front-matter bio blocks (Contributors / About-the-Author prose)
4. chapter-summary TOC lines (numbered entries with prose snippets)
6. adaptive window (caller passes larger window; window-max still flags)
Constants exposed for gold-tuned sweeping.
"""
import re, sys
sys.path.insert(0, '.')
from bakeoff import paragraphs, prose_score, _classify, _para_markers, \
                    _praise_spans, _first_prose_run, _last_prose_run_end

P = dict(GAP_PAD=600, RUN_CHARS=350, PROSE_T=0.55, ANCHOR_PROSE=350,
         FIRST_JUNK_MAX=400, QUOTE_MAX=700)

CONTENT_HEAD = re.compile(
    r'^\s*(?:#+\s*)?(?:\*\*|_)*\s*(?:preface|foreword|introduction|prologue|'
    r"author'?s'? note|acknowledg(?:e?ments)?|chapter\s+(?:one|1)\b|part\s+(?:one|1)\b)"
    r'(?:\*\*|_|\s|:|\.)*', re.I)
BIO = re.compile(
    r'(?:\bis (?:a|an|the) (?:professor|author|writer|lecturer|journalist|editor|'
    r'director|founder|managing|award-winning|bestselling|#1)\b|'
    r'\b(?:s?he|they) (?:lives?|resides?|teaches|divides? (?:his|her|their) time)\b|'
    r'\bis the author of\b|\bhas written\b.{0,40}\bbooks\b)', re.I)
SUMMARY_TOC = re.compile(r'^\s*(?:[IVXLC]+|\d+)\.?\s+[A-Z][a-z].{20,90}\.{3}|'
                         r'^\s*(?:[IVXLC]+|\d+)\s+[A-Z][a-z][^.!?\n]{15,80}(?:\.\.\.|…)?\s*$')
ATTRIB = re.compile(r'^\s*(?:—|--|―)\s*\S')

def _long_quote(t, mx):
    return re.match(r'^\s*["“][^"“”]{8,%d}["”][.!?]?\s*$' % mx, t)

def junk_spans(paras, p=P):
    spans = []
    for i, (s, e, txt) in enumerate(paras):
        t = txt.strip()
        nmark = _para_markers(txt)
        cls = _classify(txt, prose_score, p['PROSE_T'])
        lines = [l for l in t.split('\n') if l.strip()]
        sumtoc = len(lines) >= 3 and sum(1 for l in lines if SUMMARY_TOC.match(l)) / len(lines) > 0.5
        bio = cls == 'prose' and len(BIO.findall(txt)) >= 2 and len(txt) < 1500
        praise = _long_quote(t, p['QUOTE_MAX']) and i + 1 < len(paras) and \
                 ATTRIB.match(paras[i+1][2].strip())
        if nmark >= 2 or cls == 'junk' or sumtoc or bio or praise or \
           (ATTRIB.match(t) and len(t) < 90) or \
           (nmark == 1 and (cls != 'prose' or len(txt) < 400)):
            spans.append((s, e))
    return sorted(set(spans) | set(_praise_spans(paras)))

def head_boundary(head, p=P):
    paras = paragraphs(head)
    if not paras:
        return 0
    spans = junk_spans(paras, p)
    if not spans or spans[0][0] > p['FIRST_JUNK_MAX']:
        return 0
    first_junk = spans[0][0]
    anchor = None
    for i, (s, e, txt) in enumerate(paras):
        if s < first_junk:
            continue
        line = txt.strip().split('\n')[0]
        if len(line) < 90 and CONTENT_HEAD.match(line):
            acc = 0
            for j in range(i, min(i + 5, len(paras))):
                cj = _classify(paras[j][2], prose_score, p['PROSE_T'])
                if cj == 'junk' and j > i:
                    break
                if cj != 'junk':
                    acc += paras[j][1] - paras[j][0]
            if acc >= p['ANCHOR_PROSE']:
                anchor = s
                break
    block_end = spans[0][1]
    for s, e in spans[1:]:
        if s - block_end <= p['GAP_PAD']:
            block_end = max(block_end, e)
        else:
            break
    import bakeoff
    old_run = bakeoff.RUN_CHARS
    bakeoff.RUN_CHARS = p['RUN_CHARS']
    run = _first_prose_run(paras, prose_score, p['PROSE_T'], from_pos=block_end)
    bakeoff.RUN_CHARS = old_run
    run = run if run is not None else len(head)
    if anchor is not None:
        return min(anchor, run) if run > first_junk else anchor
    return run


# ---------------- tail v2.2 ----------------
INDEX_LINE = re.compile(r'^[^.!?\n]{2,60}?\d{1,4}(?:\s*[,–-]\s*\d{1,4}){0,30}\s*$|'
                        r'^[A-Z][^.!?\n]{1,50},\s*$')
CITE_LINE = re.compile(r'(?:_[^_]+_\s*,|\bpp?\.\s*\d|\(\d{4}\)|\d{4}[.,]\s*$|'
                       r'^\s*\d{1,3}\.\s+["“A-Z])')
BACK_HEAD = re.compile(r'^\s*(?:#+\s*)?(?:\*\*|_)*\s*(?:about the author|also by|books by|'
                       r'acknowledg|index|notes|bibliography|thanx|other books|available now|'
                       r'discussion questions|reading group|copyright|ende der leseprobe)', re.I)
JUNK_LINE = re.compile(r'(?i)^\s*(?:isbn\b|©|copyright ©|first (?:north american )?publi)')


def tail_boundary(tail, p_t=0.5):
    """Tail v2.2: index/citation/back-heading detectors + backward junk-cluster
    chaining + line-level refinement for prose glued to junk by single newlines.
    Validated on 20 gold tails: 0 content-cuts >200, medAbsErr 1039 (junk-kept)."""
    import bakeoff as _bk
    paras = paragraphs(tail)
    if not paras:
        return len(tail)
    spans = []
    for i, (s, e, txt) in enumerate(paras):
        t = txt.strip()
        lines = [l for l in t.split('\n') if l.strip()]
        short = [l for l in lines if len(l) <= 120]
        idx_ratio = sum(1 for l in short if INDEX_LINE.match(l)) / max(len(lines), 1)
        cite_ratio = sum(1 for l in short if CITE_LINE.search(l)) / max(len(lines), 1)
        nmark = _para_markers(txt) + sum(1 for rx in _bk.TAIL_RE if rx.search(txt))
        cls = _classify(txt, prose_score, p_t)
        line1 = lines[0] if lines else ''
        if BACK_HEAD.match(line1) or idx_ratio > 0.6 or \
           (cite_ratio > 0.5 and len(lines) >= 2) or nmark >= 2 or cls == 'junk' or \
           (nmark == 1 and (cls != 'prose' or len(txt) < 400)):
            spans.append((s, e))
    if not spans or spans[-1][1] < len(tail) - 400:
        return len(tail)
    block_start = spans[-1][0]
    for s, e in reversed(spans[:-1]):
        if block_start - e <= 800:
            block_start = min(block_start, s)
        else:
            break
    for s, e, txt in paras:
        if s == block_start:
            pos = 0
            for line in txt.split('\n'):
                if JUNK_LINE.match(line.strip()) or BACK_HEAD.match(line.strip()):
                    if pos > 60:
                        return s + pos
                    break
                pos += len(line) + 1
            break
    end = _last_prose_run_end(paras, prose_score, p_t, before_pos=block_start)
    return end if end is not None else 0
