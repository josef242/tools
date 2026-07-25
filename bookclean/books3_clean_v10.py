#!/usr/bin/env python3
"""Production books3 cleaner v9 — structural-hybrid policy.
Heads: sentinel2 (Arena-A winner, gold-tuned content anchors).
Tails: sentinel3 (validated on 3 independent sets: 4.08/4.24/4.08 asym cost,
       zero content-cut on 59/60 books).
All v8 safety rails retained + the zero-cut contract.
"""
import json, re, sys
sys.path.insert(0, '.')
from sentinel2 import head_boundary
from sentinel3 import tail_boundary
from span_cutter import find_spans, apply_spans

TUNED = json.load(open('reports/sentinel2_params.json'))
HEAD_W, HEAD_W_EXT, TAIL_W = 15000, 30000, 10000
STRONG_HEAD = re.compile(
    r'(?i)(?:©|\bisbn\b|\ball rights reserved\b|\bcopyright\b|\blibrary of congress\b|'
    r'\bfirst published\b|\bpublish\w* by\b|\balso by\b|\btable of contents\b|'
    r'^#+ ?contents\b|\bcataloging.in.publication\b)', re.M)

import re as _re
from bakeoff import paragraphs as _paras, prose_score as _ps, _classify as _cl, _para_markers as _pm
_SEP = _re.compile(r'^[\s*#~\-–—•·]+$')

def content_frac(region, p_t=0.5):
    ps = _paras(region)
    if not ps: return 0.0
    keep = 0
    for s, e, txt in ps:
        t = txt.strip()
        if _SEP.match(t) or _pm(txt) >= 1: continue
        cls = _cl(txt, _ps, p_t)
        if cls == 'prose':
            keep += e - s
        elif cls == 'neutral' and (('"' in t or '“' in t) or _re.search(r'[.!?]["”]?$', t)):
            keep += e - s
    return keep / max(len(region), 1)


def leading_content(region, k=5, p_t=0.5):
    """True if any of the first k non-separator paragraphs reads as book content.
    v8 rail -- prevents cutting into short/dialogue-y openings."""
    seen = 0
    for s, e, txt in _paras(region):
        t = txt.strip()
        if _SEP.match(t) or not t: continue
        seen += 1
        if _pm(txt) <= 1:
            cls = _cl(txt, _ps, p_t)
            if cls == 'prose' and len(t) > 150: return True
            if cls == 'neutral' and (('"' in t or '\u201c' in t) or _re.search(r'[.!?]["\u201d]?$', t)):
                if not _re.search(r'(?i)www\.|https?://|isbn|\u00a9|available now|coming soon', t):
                    return True
        if seen >= k: break
    return False


_DEF_LINE = _re.compile(r'^\s*[^|\n]{2,40}\s*[|\u2014\u2013:-]\s+\S.{10,}$')  # "Maquis | Rural resistance fighters"
_BIO = _re.compile(r'(?i)\bis (?:a|an|the) (?:author|writer|professor|lecturer|journalist)|'
                   r'\bis the author of\b|\b(?:she|he|they) lives? in\b|\bteaches\b.{0,30}\buniversity\b')

def learnable_chars(region):
    """Chars of LEARNABLE prose inside a region (hybrid policy): real prose
    paragraphs, author bios, glossary/abbreviation definitions. Epigraphs and
    dedications do NOT count (policy sacrifices them)."""
    total = 0
    for s, e, txt in _paras(region):
        t = txt.strip()
        if not t or _SEP.match(t):
            continue
        lines = [l for l in t.split('\n') if l.strip()]
        # glossary / abbreviations block: majority of lines are "term | definition"
        if len(lines) >= 3 and sum(1 for l in lines if _DEF_LINE.match(l)) / len(lines) > 0.5:
            total += e - s
            continue
        if _pm(txt) >= 1:
            continue
        if _BIO.search(t) and len(t) > 120:
            total += e - s
            continue
        if _cl(txt, _ps, 0.5) == 'prose' and len(t) > 200:
            total += e - s
    return total

def back_off_overshoot(text, hs):
    """Language-agnostic overshoot guard.

    Rather than trying to recognise PROSE (which fails on short openings and on
    languages our anchor regexes don't know -- a Swedish Montaigne's "KAPITEL 1"
    sailed straight through), walk BACK to the last unambiguously-JUNK paragraph
    (marker-bearing, or a multi-line list/TOC) and keep everything after it.
    Headings + the book's opening lines then survive in any language.
    """
    ps = _paras(text[:hs])
    if not ps:
        return hs
    last_junk_end = None
    for s, e, txt in ps:
        t = txt.strip()
        if not t:
            continue
        lines = [l for l in t.split('\n') if l.strip()]
        is_list = len(lines) >= 3          # TOC / illustration list / catalog
        if _pm(txt) >= 1 or is_list or _re.search(r'(?i)^#*\s*(?:inneh|contents|table of contents|sommaire|indice|inhalt)', t):
            last_junk_end = e
    if last_junk_end is None:
        return hs
    return min(hs, last_junk_end)


def _nonlatin_frac(t):
    letters = [c for c in t[:6000] if c.isalpha()]
    return (sum(1 for c in letters if ord(c) > 0x24F) / len(letters)) if letters else 0.0

def clean_books3_v10(text):
    if _nonlatin_frac(text) > 0.3:
        return text, {'skipped': 'non-latin script'}
    n = len(text)
    entry = {}
    head_cap = min(HEAD_W_EXT, max(int(n * 0.15), 3000))
    hs = head_boundary(text[:HEAD_W], TUNED)
    if hs is not None and hs >= HEAD_W - 100 and n > HEAD_W:
        hs = head_boundary(text[:HEAD_W_EXT], TUNED)
        if hs is not None and hs >= HEAD_W_EXT - 100:
            entry['head'] = {'action': 'flagged', 'reason': 'window-max at 30KB'}
            hs = 0
    if hs:
        hs = back_off_overshoot(text, hs)   # fix 1: overshoot into ch.1
    cut_start = 0
    if hs and 0 < hs < HEAD_W_EXT:
        if hs > head_cap:
            entry['head'] = {'action': 'flagged', 'reason': f'cut {hs} > cap {head_cap}'}
        elif not STRONG_HEAD.search(text[:hs]):
            entry['head'] = {'action': 'skipped', 'reason': 'no strong head anchor'}
        elif 2.0 * learnable_chars(text[:hs]) > 0.1 * hs:
            # fix 2: ASYMMETRIC-COST rail -- cutting learnable prose (bios, glossary
            # definitions) costs 20x what leaving the junk costs. Let the metric decide.
            entry['head'] = {'action': 'flagged', 'reason': 'removed region holds learnable prose'}
        elif (content_frac(text[max(0, hs - 1500):hs]) > 0.30
              or leading_content(text[max(0, hs - 1200):hs])):
            entry['head'] = {'action': 'flagged', 'reason': 'removed region content-rich'}
        elif text[hs:hs + 1].islower():
            entry['head'] = {'action': 'flagged', 'reason': 'kept starts mid-sentence'}
        else:
            cut_start = hs
            entry['head'] = {'action': 'cut', 'chars': hs,
                             'removed_preview': ' '.join(text[:hs].split())[:120]}
    tail = text[-TAIL_W:]
    # GENRE GUARD FALLBACK (found via the stratified gold set): when the book's
    # body is caption/pattern/poem-shaped, multi-span cutting is unsafe -- but
    # these books still carry real junk (indexes, copyright, catalogs). Fall back
    # to the conservative SINGLE cut, which cannot punch holes through content.
    from span_cutter import body_genre_is_riskly
    if body_genre_is_riskly(text):
        te = tail_boundary(tail)
        spans = [(te, len(tail))] if te is not None and te < len(tail) else []
    else:
        spans, _ = find_spans(tail, full_text=text)
    tail_removed = 0
    new_tail = tail
    if spans:
        tail_cap = min(TAIL_W, max(int(n * 0.15), 3000))
        span_chars = sum(e - s for s, e in spans)
        if span_chars > tail_cap:
            entry['tail'] = {'action': 'flagged', 'reason': f'spans {span_chars} > cap {tail_cap}'}
        else:
            new_tail = apply_spans(tail, spans)
            tail_removed = len(tail) - len(new_tail)
            entry['tail'] = {'action': 'cut', 'chars': tail_removed, 'spans': len(spans),
                             'removed_preview': ' '.join(tail[spans[0][0]:spans[0][1]].split())[:120]}
    cut_end = n   # tails are now excised in-place, not truncated

    total_cut = cut_start + tail_removed
    if total_cut > 0.40 * n:
        entry['total'] = {'action': 'flagged', 'reason': f'total {total_cut} > 40%'}
        return text, entry
    if cut_start or tail_removed:
        body = text[cut_start:n - TAIL_W] if n > TAIL_W else ''
        head_part = text[cut_start:] if n <= TAIL_W else body
        out = (head_part + new_tail) if n > TAIL_W else new_tail[max(0, cut_start - (n - TAIL_W)):]
        if n > TAIL_W:
            out = text[cut_start:n - TAIL_W] + new_tail
        else:
            out = apply_spans(text, [(s, e) for s, e in spans]) if spans else text
            out = out[cut_start:] if cut_start < len(out) else out
        return out.strip('\n'), entry
    return text, entry
