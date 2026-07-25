#!/usr/bin/env python3
"""Bake-off: competing boundary-detection methods for books3 front/back matter.

Task: given a book's head window and tail window, predict
  content_start : char offset in head window where real content begins
  content_end   : char offset in tail window where back matter begins
                  (i.e. content runs to tail_window[:content_end])

Methods:
  sentinel   - junk-marker rules: last head marker -> next sustained prose
  prosefall  - structural changepoint on per-paragraph prose scores (no lexicon)
  lexis      - multi-language function-word density changepoint
  hybrid     - sentinel markers gate prosefall's changepoint

All methods share paragraph segmentation. A method may return None
(= no boundary found in window; treat as 0 for heads / len for tails).
"""

import json
import re

# ---------------- shared plumbing ----------------

def paragraphs(window):
    """Split into (start, end, text) paragraphs on blank lines."""
    out = []
    pos = 0
    for block in re.split(r'\n\s*\n', window):
        if not block.strip():
            pos += len(block) + 2
            continue
        start = window.find(block, pos)
        out.append((start, start + len(block), block))
        pos = start + len(block)
    return out


def prose_score(text):
    """Structural prose-likeness in [0,1]-ish. No lexicon."""
    n = len(text)
    if n < 40:
        return 0.0
    letters = sum(c.isalpha() for c in text)
    lower = sum(c.islower() for c in text)
    digits = sum(c.isdigit() for c in text)
    symbols = sum(text.count(c) for c in '#*©®™|>=_[]{}')
    sent_punct = text.count('. ') + text.count('."') + text.count('.\n') + \
        text.count('! ') + text.count('? ') + text.count('," ')
    words = text.split()
    if not words:
        return 0.0
    caps_words = sum(1 for w in words if len(w) > 1 and w.isupper())
    linelens = [len(l) for l in text.split('\n') if l.strip()]
    avg_line = sum(linelens) / len(linelens)

    score = 0.0
    score += min(n / 400.0, 1.0) * 0.25                 # long enough
    score += (lower / max(letters, 1)) * 0.25            # lowercase-dominant
    score += min(sent_punct / max(n / 200.0, 1), 1.0) * 0.30   # sentence flow
    score -= (digits / max(n, 1)) * 2.0                  # digit-heavy = junk
    score -= (symbols / max(n, 1)) * 6.0                 # md/legal symbols
    score -= (caps_words / max(len(words), 1)) * 0.5     # SHOUTY headings
    score += min(avg_line / 60.0, 1.0) * 0.20            # flowing lines
    return max(score, 0.0)


STOP = ("the and of to in was that it he she for with his her they you not "
        "as at be on had have but him from this all we are or so said "
        "le la les des une est dans il elle et que pour der die und das ist "
        "nicht el los una por con como che per sono di anche").split()
STOPSET = set(STOP)


def lexis_score(text):
    """Function-word density across en/fr/de/es/it."""
    words = re.findall(r"[a-zA-Zà-ÿÀ-ß']+", text.lower())
    if len(words) < 25:
        return 0.0
    hits = sum(1 for w in words if w in STOPSET)
    return hits / len(words)


HEAD_MARKERS = [
    r'\bisbn\b', r'©', r'\ball rights reserved\b', r'\blibrary of congress\b',
    r'\bcataloging.in.publication\b', r'\blccn\b', r'\bpages cm\b',
    r'\bfirst published\b', r'\bfirst edition\b', r'\bprinted in\b',
    r'\bmoral rights?\b', r'\breproduc\w+ in any (?:form|manner)\b',
    r'\bwithout (?:the )?(?:prior )?written permission\b',
    r'\bpublish\w* by\b', r'\bwww\.\S+', r'https?://',
    r'\balso by\b', r'\bbooks by\b', r'\bcover (?:design|photograph|art)\b',
    r'\btable of contents\b', r'^#+ ?contents\b', r'^contents$',
    r'\bcopyright\b', r'^\s*10\s+9\s+8\s+7\b', r'\bdc2[0-9]\b',
    r'\ball rights? reserved\b', r'\btrademarks?\b', r'\bimprint\b',
    r'\bebook isbn\b', r'\bv\d+\.\d+\.\d+\b',
    r'^\s*(?:chapter|part)\s+(?:[0-9ivxl]+|one|two|three)\s*$',  # bare TOC entries
    r'\bdistributed by\b', r'\bcatalogue record\b', r'\bbritish library\b',
    r'\bdisclaims?\b', r'\bprinting code\b', r'\bspecial sales\b',
    r'\bregistered offices?\b', r'\bpatent liability\b', r'\bwarrant\w+\b',
    r'\bpenguin (?:group|random house|books)\b', r'\bfirst printing\b',
]
HEAD_RE = [re.compile(p, re.I | re.M) for p in HEAD_MARKERS]

TAIL_MARKERS = [
    r'\babout the author\b', r'\balso by\b', r'\bbooks by\b',
    r'\bisbn\b', r'©', r'\bcopyright\b', r'\ball rights reserved\b',
    r'\blibrary of congress\b', r'\bphoto(?:graph)?s? (?:courtesy|by|©)\b',
    r'\bplease leave a review\b', r'\bthanks for reading\b',
    r'\bvisit (?:our|us|my) (?:web ?site|at)\b', r'\bnewsletter\b',
    r'\bsign up\b', r'\btable of contents\b', r'^#+ ?contents\b',
    r'\backnowledg?e?ments?\b', r'\bbibliography\b', r'^\s*index\s*$',
    r'\bfirst\w* (?:published|edition)\b', r'\bwww\.\S+', r'https?://',
    r'\bjacket design\b', r'\binterior design\b', r'\bspeakers bureau\b',
    r'\bfor more information\b', r'\bcontinue reading\b', r'\bpreview of\b',
    r'\bcoming soon\b', r'\bexcerpt from\b',
]
TAIL_RE = [re.compile(p, re.I | re.M) for p in TAIL_MARKERS]

PROSE_T = 0.55       # prose threshold for structural score
LEXIS_T = 0.22      # function-word density threshold (swept 2026-07-13)
RUN_CHARS = 350      # sustained-prose run size to accept a boundary


TOC_LINE = re.compile(
    r'^\s*(?:#+\s*)?(?:(?:chapter|part)\s+(?:[0-9ivxlc]+|\w+(?:-\w+)?)|'
    r'(?:appendix|epilogue|prologue|foreword|preface|introduction|conclusion|'
    r'acknowledg\w+|index|notes|glossary|bibliography|credits|dedication|'
    r'cover|title page|contents|copyright|about the (?:author|publisher))'
    r')\s*(?:\|.*|\.{2,}.*|\d*)?\s*$', re.I)


LIST_LINE = re.compile(r'^\s*(?:\d+[.)]\s|\d+\s*$|[#>*•▪]+\s|[A-Z0-9 ,.:&\'-]{4,60}$)')


def _classify(txt, scorer, threshold):
    """'prose' | 'junk' | 'neutral' for one paragraph."""
    stripped = txt.strip()
    lines = [l for l in stripped.split('\n') if l.strip()]
    # a paragraph whose lines are mostly TOC/list entries is junk, no matter
    # how long (indented single-paragraph TOCs can run 5KB+)
    tocish = sum(1 for l in lines if TOC_LINE.match(l) or LIST_LINE.match(l))
    if len(lines) >= 2 and tocish / len(lines) > 0.5:
        return 'junk'
    if len(stripped) < 120:
        # short: neutral unless clearly junky (symbols/digits/url/caps/md-list)
        if re.search(r'(?:www\.|https?://|isbn|©|\d{4}|^\s*[#*>|=~-]+\s*$)',
                     stripped, re.I) or (stripped.isupper() and len(stripped) >= 6) \
                or re.match(r'^\s*(?:\d+[.)]\s|\d+\s*$|[#>*•▪]+\s)', stripped):
            return 'junk'
        return 'neutral'
    return 'prose' if scorer(txt) >= threshold else 'junk'


PURE_QUOTE = re.compile(r'^\s*["“][^"“”]{8,300}["”][.!?]?\s*$')
ATTRIB = re.compile(r'^\s*(?:—|--|―)\s*\S')


def _praise_spans(paras, limit_paras=120):
    """Contextual pass: quote+attribution pairs (praise pages). A pure-quote
    paragraph is junk evidence only when an attribution or another pure quote
    sits within 2 paragraphs — lone dialogue openings stay untouched, and
    French em-dash dialogue is safe because it lacks pure-quote neighbors."""
    flags = []
    for i, (s, e, txt) in enumerate(paras[:limit_paras]):
        t = txt.strip()
        flags.append(('q' if PURE_QUOTE.match(t) else
                      'a' if ATTRIB.match(t) and len(t) < 90 else '-'))
    spans = []
    for i, f in enumerate(flags):
        if f == '-':
            continue
        window = flags[max(0, i - 2):i + 3]
        if f == 'q' and ('a' in window or window.count('q') >= 2):
            spans.append((paras[i][0], paras[i][1]))
        elif f == 'a' and 'q' in window:
            spans.append((paras[i][0], paras[i][1]))
    return spans


def _first_prose_run(paras, scorer, threshold, from_pos=0):
    """Start offset of first sustained prose run at/after from_pos, else None.

    Junk paragraphs reset the run; neutral (short) ones extend it weakly, so
    dialogue-heavy fiction still accumulates. Walks back over up to 2 short
    heading paragraphs so chapter titles stay attached to their content."""
    acc = 0
    run_start = None
    run_idx = None
    for i, (s, e, txt) in enumerate(paras):
        if e <= from_pos:
            continue
        cls = _classify(txt, scorer, threshold)
        if cls == 'junk':
            acc = 0
            run_start = run_idx = None
            continue
        if run_start is None:
            if cls == 'neutral' and len(txt.strip()) < 40:
                continue  # don't open a run on a tiny fragment
            run_start, run_idx = max(s, from_pos), i
        acc += (e - max(s, from_pos)) if cls == 'prose' else (e - max(s, from_pos)) // 2
        if acc >= RUN_CHARS:
            # include up to 2 preceding short heading paragraphs
            j = run_idx
            for _ in range(2):
                if j > 0:
                    ps, pe, ptxt = paras[j - 1]
                    line = ptxt.strip()
                    if ps >= from_pos and len(line) < 80 and '\n' not in line and \
                       not TOC_LINE.match(line) and \
                       not re.search(r'(?:isbn|www\.|https?://|©)', line, re.I):
                        j -= 1
                        continue
                break
            return paras[j][0] if j != run_idx else run_start
    return None


def _last_prose_run_end(paras, scorer, threshold, before_pos):
    """End offset of last sustained prose run strictly before before_pos."""
    best = None
    acc = 0
    for s, e, txt in paras:
        if s >= before_pos:
            break
        if scorer(txt) >= threshold and (e - s) > 120:
            acc += e - s
            if acc >= RUN_CHARS:
                best = min(e, before_pos)
        else:
            acc = 0
    return best


# ---------------- methods ----------------

PRAISE_RE = re.compile(
    r'(?:—|--)\s*[A-Z][\w. ]+(?:,?\s*(?:_?New York Times_?|USA Today|'
    r'bestselling|award[- ]winning|author of))|\bpraise for\b', re.I)


def _para_markers(txt):
    return sum(1 for rx in HEAD_RE if rx.search(txt)) + \
           (2 if PRAISE_RE.search(txt) else 0)


GAP_PAD = 600  # junk elements within this many chars chain into one block


def sentinel_head(head):
    """Junk-cluster walk. Junk evidence = marker-bearing paragraphs (weighted)
    + structurally junk paragraphs. Evidence within GAP_PAD chars chains into
    one contiguous front-matter block anchored near offset 0; an isolated
    marker deep in content (URL in the acknowledgments) does not chain.
    Boundary = first sustained prose run after the block."""
    paras = paragraphs(head)
    if not paras:
        return 0
    # junk evidence spans
    spans = []
    for s, e, txt in paras:
        nmark = _para_markers(txt)
        cls = _classify(txt, prose_score, PROSE_T)
        if nmark >= 2 or cls == 'junk' or \
           (nmark == 1 and (cls != 'prose' or len(txt) < 400)):
            spans.append((s, e))
    spans = sorted(set(spans) | set(_praise_spans(paras)))
    if not spans or spans[0][0] > 400:
        return 0  # book starts clean
    # chain spans from the first while gaps are small
    block_end = spans[0][1]
    for s, e in spans[1:]:
        if s - block_end <= GAP_PAD:
            block_end = max(block_end, e)
        else:
            break
    return _first_prose_run(paras, prose_score, PROSE_T, from_pos=block_end)


BIBLIO_LINE = re.compile(
    r'(?:_[^_]+_\.?\s+(?:New York|London|Boston|Chicago|Toronto|[A-Z][a-z]+):'
    r'|(?:\b(?:19|20)\d\d\.?\s*$))')


def _tail_para_markers(txt):
    n = sum(1 for rx in TAIL_RE if rx.search(txt)) + \
        sum(1 for rx in HEAD_RE if rx.search(txt))
    lines = [l for l in txt.strip().split('\n') if l.strip()]
    biblio = sum(1 for l in lines if BIBLIO_LINE.search(l))
    if lines and biblio / len(lines) > 0.5:
        n += 2
    return n


def sentinel_tail(tail):
    """Mirror of sentinel_head: junk-cluster chained backward from window end.
    A long fluent prose paragraph with a single incidental marker (hotel
    listing with a URL) stays content; marker-dense or structurally junky
    paragraphs chain into the back-matter block."""
    paras = paragraphs(tail)
    if not paras:
        return len(tail)
    spans = []
    for s, e, txt in paras:
        nmark = _tail_para_markers(txt)
        cls = _classify(txt, prose_score, PROSE_T)
        if nmark >= 2 or cls == 'junk' or \
           (nmark == 1 and (cls != 'prose' or len(txt) < 400)):
            spans.append((s, e))
    spans = sorted(set(spans) | set(_praise_spans(paras, limit_paras=10**9)))
    if not spans or spans[-1][1] < len(tail) - 400:
        return len(tail)  # book ends clean
    # chain backward from the last span while gaps are small
    block_start = spans[-1][0]
    for s, e in reversed(spans[:-1]):
        if block_start - e <= GAP_PAD:
            block_start = min(block_start, s)
        else:
            break
    # a real back-matter block must contain a strong anchor; content that
    # merely LOOKS junky (hotel listings with URLs/prices) has none
    if not STRONG_TAIL.search(tail[block_start:]):
        return len(tail)
    end = _last_prose_run_end(paras, prose_score, PROSE_T, before_pos=block_start)
    return end if end is not None else 0


STRONG_TAIL = re.compile(
    r'(?:©|\bisbn\b|\babout the author\b|\ball rights reserved\b|'
    r'\bcataloging.in.publication\b|\blibrary of congress\b|'
    r'\balso by\b|\bbooks by\b|\bplease leave a review\b|\bnewsletter\b|'
    r'\bspeakers bureau\b|\bjacket design\b|\bcatalogue record\b)', re.I)


def prosefall_head(head):
    return _first_prose_run(paragraphs(head), prose_score, PROSE_T)


def prosefall_tail(tail):
    paras = paragraphs(tail)
    end = _last_prose_run_end(paras, prose_score, PROSE_T, before_pos=len(tail))
    return end


def lexis_head(head):
    return _first_prose_run(paragraphs(head), lexis_score, LEXIS_T)


def lexis_tail(tail):
    paras = paragraphs(tail)
    return _last_prose_run_end(paras, lexis_score, LEXIS_T, before_pos=len(tail))


def hybrid_head(head):
    s = sentinel_head(head)
    p = prosefall_head(head)
    if s is None:
        return p
    if p is None:
        return s
    return max(s, p) if abs(s - p) < 2000 else s


def hybrid_tail(tail):
    s = sentinel_tail(tail)
    p = prosefall_tail(tail)
    if s is None or s == len(tail):
        return p if p is not None else len(tail)
    if p is None:
        return s
    return min(s, p) if abs(s - p) < 2000 else s


METHODS = {
    'sentinel': (sentinel_head, sentinel_tail),
    'prosefall': (prosefall_head, prosefall_tail),
    'lexis': (lexis_head, lexis_tail),
    'hybrid': (hybrid_head, hybrid_tail),
}


def main():
    rows = [json.loads(l) for l in open('reports/dev_boundaries_books3_ext.jsonl')]
    preds = {}
    for r in rows:
        entry = {}
        for name, (fh, ft) in METHODS.items():
            hs = fh(r['head'])
            te = ft(r['tail'])
            entry[name] = {
                # None from a head = no prose found in window => whole window junk
                'content_start': hs if hs is not None else len(r['head']),
                'content_end': te if te is not None else len(r['tail']),
            }
        preds[str(r['line'])] = entry
    json.dump(preds, open('reports/bakeoff_preds_v1.json', 'w'), indent=1)

    # agreement analysis on heads
    import statistics as st
    spread_h, spread_t = [], []
    for line, e in preds.items():
        hs = [e[m]['content_start'] for m in METHODS]
        ts = [e[m]['content_end'] for m in METHODS]
        spread_h.append((max(hs) - min(hs), int(line)))
        spread_t.append((max(ts) - min(ts), int(line)))
    spread_h.sort(reverse=True)
    spread_t.sort(reverse=True)
    print(f"books: {len(preds)}")
    print(f"head spread median: {st.median(s for s,_ in spread_h):.0f} chars")
    print(f"tail spread median: {st.median(s for s,_ in spread_t):.0f} chars")
    print("\nworst head disagreements (spread, line):", spread_h[:10])
    print("worst tail disagreements (spread, line):", spread_t[:10])
    agree_h = sum(1 for s, _ in spread_h if s <= 120)
    agree_t = sum(1 for s, _ in spread_t if s <= 120)
    print(f"\nheads with <=120 char spread: {agree_h}/{len(preds)}")
    print(f"tails with <=120 char spread: {agree_t}/{len(preds)}")


if __name__ == '__main__':
    main()
