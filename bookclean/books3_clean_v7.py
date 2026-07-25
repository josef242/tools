#!/usr/bin/env python3
"""Production books3 cleaner v7: sentinel2 head+tail engines, gold-tuned
params, adaptive window extension, all v5 safety rails intact."""
import json, re, sys
sys.path.insert(0, '.')
from sentinel2 import head_boundary, tail_boundary, P as DEFAULT_P


import re as _re
from bakeoff import paragraphs as _paras, prose_score as _ps, _classify as _cl, _para_markers as _pm

_SEP = _re.compile(r'^[\s*#~\-–—•·]+$')

def content_frac(region, p_t=0.5):
    """Fraction of region chars that read as book content: marker-free prose
    paras, or short dialogue/sentence paras (endings are dialogue-heavy)."""
    ps = _paras(region)
    if not ps: return 0.0
    keep = 0
    for s, e, txt in ps:
        t = txt.strip()
        if _SEP.match(t): continue
        if _pm(txt) >= 1: continue
        cls = _cl(txt, _ps, p_t)
        if cls == 'prose':
            keep += e - s
        elif cls == 'neutral' and (('"' in t or '“' in t) or
                                   _re.search(r'[.!?]["”]?$', t)):
            keep += e - s
    return keep / max(len(region), 1)

def leading_content(region, k=5, p_t=0.5):
    """True if any of the first k non-separator paragraphs of region reads as
    book content (marker-free prose, or dialogue/sentence-final short para)."""
    seen = 0
    for s, e, txt in _paras(region):
        t = txt.strip()
        if _SEP.match(t) or not t: continue
        seen += 1
        if _pm(txt) <= 1:
            cls = _cl(txt, _ps, p_t)
            if cls == 'prose' and len(t) > 150: return True
            if cls == 'neutral' and (('"' in t or '\u201c' in t) or _re.search(r'[.!?]["\u201d]?$', t)):
                # exclude obvious junk shapes even when sentence-like
                if not _re.search(r'(?i)www\.|https?://|isbn|\u00a9|available now|coming soon', t):
                    return True
        if seen >= k: break
    return False

TUNED = json.load(open('reports/sentinel2_params.json'))
HEAD_W, HEAD_W_EXT, TAIL_W = 15000, 30000, 10000
STRONG_HEAD = re.compile(
    r'(?i)(?:©|\bisbn\b|\ball rights reserved\b|\bcopyright\b|\blibrary of congress\b|'
    r'\bfirst published\b|\bpublish\w* by\b|\balso by\b|\btable of contents\b|'
    r'^#+ ?contents\b|\bcataloging.in.publication\b)', re.M)

def _nonlatin_frac(t):
    letters = [c for c in t[:6000] if c.isalpha()]
    return (sum(1 for c in letters if ord(c) > 0x24F) / len(letters)) if letters else 0.0

def clean_books3_v7(text):
    if _nonlatin_frac(text) > 0.3:
        return text, {'skipped': 'non-latin script'}
    n = len(text)
    entry = {}
    head_cap = min(HEAD_W_EXT, max(int(n * 0.15), 3000))
    hs = head_boundary(text[:HEAD_W], TUNED)
    if hs is not None and hs >= HEAD_W - 100 and n > HEAD_W:
        hs = head_boundary(text[:HEAD_W_EXT], TUNED)   # adaptive extension
        if hs is not None and hs >= HEAD_W_EXT - 100:
            entry['head'] = {'action': 'flagged', 'reason': 'window-max even at 30KB'}
            hs = 0
    cut_start = 0
    if hs and 0 < hs < HEAD_W_EXT:
        if hs > head_cap:
            entry['head'] = {'action': 'flagged', 'reason': f'cut {hs} > cap {head_cap}'}
        elif not STRONG_HEAD.search(text[:hs]):
            entry['head'] = {'action': 'skipped', 'reason': 'no strong head anchor'}
        elif content_frac(text[max(0,hs-1500):hs]) > 0.30 or leading_content(text[max(0,hs-1200):hs][::-1][::-1][-1200:]):
            entry['head'] = {'action': 'flagged', 'reason': 'removed region content-rich'}
        elif text[hs:hs+1] .islower():
            entry['head'] = {'action': 'flagged', 'reason': 'kept starts mid-sentence'}
        else:
            cut_start = hs
            entry['head'] = {'action': 'cut', 'chars': hs,
                             'removed_preview': ' '.join(text[:hs].split())[:120]}
    tail = text[-TAIL_W:]
    te = tail_boundary(tail)
    cut_end = n
    if te is not None and te < len(tail):
        removed = len(tail) - te
        tail_cap = min(TAIL_W, max(int(n * 0.15), 3000))
        if removed > tail_cap:
            entry['tail'] = {'action': 'flagged', 'reason': f'cut {removed} > cap {tail_cap}'}
        elif content_frac(tail[te:te+1500]) > 0.30 or leading_content(tail[te:te+1500]):
            entry['tail'] = {'action': 'flagged', 'reason': 'removed region content-rich'}
        else:
            cut_end = n - removed
            entry['tail'] = {'action': 'cut', 'chars': removed,
                             'removed_preview': ' '.join(tail[te:].split())[:120]}
    total_cut = cut_start + (n - cut_end)
    if total_cut > 0.40 * n:
        entry['total'] = {'action': 'flagged', 'reason': f'total {total_cut} > 40%'}
        return text, entry
    if cut_start or cut_end < n:
        return text[cut_start:cut_end].strip('\n'), entry
    return text, entry
