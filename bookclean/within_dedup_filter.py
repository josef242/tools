#!/usr/bin/env python3
"""Within-book dedup filter (SAFE tier): remove page-furniture / nav / OCR-artifact
lines that repeat MANY times inside a SINGLE book. Complements cross-book dedup
(dedup_filter.py): cross-book counts DISTINCT BOOKS, so within-book repeats
(running headers, page numbers, repeated table rows, nav strips) are invisible to
it. Harm-proxy (v9->v10) showed the DOMINANT memorization magnets are within-book:
"here here here..." x2811, "---|---" x156, "###." x397, "corrie wingate/apa
publications" x152.

Two detectors:
  (1) REPEATED-LINE FURNITURE (count x SHAPE). A normalized line repeating >= T
      times within one book is CUT iff it is furniture-shaped (numeric/table with
      <=1 real word, or a photo/credit pattern) AND not in a protected class.
      COUNT ALONE DOES NOT SEPARATE junk from content (survey finding): "* * *"
      scene break repeats x896, recipe instructions repeat, dialogue repeats. So we
      gate on SHAPE, and PROTECT scene breaks, dialogue, headings, and — critically
      (Rook) — DRAMA SPEAKER TAGS (repeat 100s/play; cutting destroys every script).
  (2) DEGENERATE LINE (within-line token repetition). A single line that is one
      short token cycle repeated >= R times ("here here here...", "ref # ref #
      ...") is a nav/OCR artifact -> remove the whole line. Line-dedup can't catch
      this (it occurs as ONE line with thousands of tokens).

Per-book CIRCUIT BREAKER: if removals exceed BREAKER_FRAC of a book's non-empty
lines -> remove NOTHING and flag (protects heavily-paginated / script / poetry /
directory books; flagged list = failure-genre mining, like cross-book dedup)."""
import re
from collections import Counter

T_SHAPE   = 50      # repeated numeric/table furniture: count is the junk signal (page-furniture ~hundreds, chapter headings ~tens)
T_CREDIT  = 5       # photo/credit patterns are positively junk; a few repeats is enough
R_DEGEN   = 30      # a line with >= this many tokens ...
DEGEN_DISTINCT = 3  # ... and <= this many DISTINCT normalized tokens = degenerate nav/OCR junk
BREAKER_FRAC = 0.30 # >30% of a book's non-empty lines removed -> flag, remove nothing

def norm(l):
    l = re.sub(r'\s+', ' ', l.strip().lower())
    return re.sub(r'\d', '#', l)

# photo / design / stock-image credits (positively junk when repeated)
CREDIT = re.compile(
    r"apa publications|/\s*apa\b|shutterstock|istockphoto|\bistock\b|getty images|"
    r"\bgetty\b|\balamy\b|dreamstime|\bcorbis\b|photo(?:graph)? by|photo credit|"
    r"illustration by|design(?:ed)? by|©.*(?:photo|image)|\bthinkstock\b",
    re.I)

# markdown heading:  #, ##, ... ###### followed by text
MD_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
# sequential STRUCTURAL headings (Chapter 3, Day 10, Part II, Act I, Lesson 4 ...).
# Digit-normalization collapses a whole numbered SEQUENCE (Day 1..Day 90) into one
# form with high count, so without this they'd be cut as "repeated furniture" though
# no single line repeats. Rook flagged chapter headings as the count-threshold's job
# to protect. NOT 'page' (page N = furniture) / 'figure'/'table' (captions).
SEQ_HEADING = re.compile(
    r"^\s*(?:chapter|chap|part|book|volume|vol|act|scene|canto|stanza|day|week|lesson|"
    r"session|unit|module|stage|level|phase|appendix|section)\b[\s.:#]*"
    r"(?:[0-9]+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|fifty)\b",
    re.I)
# citation apparatus — policy KEEPS citations/endnotes; bare stubs repeat densely.
CITATION = re.compile(r"\bibid\b|\bop\.?\s*cit\b|\bloc\.?\s*cit\b|\bet\s+seq\b", re.I)
# a token counts as a "real word" if it has >= 2 ascii letters
WORD = re.compile(r"[a-z]{2,}")
# scene-break glyph lines (only break glyphs / spaces, no digits)
SCENE = re.compile(r"^[\s*\-~=_.·•⁂—–#]+$")  # note: caught earlier only if no digit; see is_protected

def alpha_words(nl):
    return len(WORD.findall(nl))

def is_protected(line, nl):
    """Classes that must NEVER be cut even at high repeat count."""
    s = line.strip()
    if not s:
        return True
    # dialogue / quotation
    if s[0] in '"“”‘’\'«':
        return True
    # markdown heading (## Summary, ### Chapter 3 ...)
    if MD_HEADING.match(s):
        return True
    # sequential structural heading (Chapter 3, Day 10, Part II) — short line only
    if len(s.split()) <= 6 and SEQ_HEADING.match(s):
        return True
    # citation apparatus (Ibid., p.113 / op. cit.) — honor KEEP-citations policy
    if CITATION.search(s):
        return True
    # speaker tag / label: ends with ':' and short  (HAMLET:, First Witch:, Cellar sales:)
    if s.endswith(':') and len(s.split()) <= 6:
        return True
    # ALL-CAPS-ish short line: speaker tags / act-scene cues (FIRST WITCH, HAMLET,
    # 1ST MURDERER, ACT I) -> no lowercase letters but has an uppercase letter
    if len(s.split()) <= 6 and not re.search(r'[a-z]', s) and re.search(r'[A-Z]', s):
        return True
    # scene break / horizontal-rule glyph line with NO digit (* * *, ***, ---, ~ ~ ~)
    if '#' not in nl and SCENE.match(nl) and '|' not in nl:
        return True
    # sentence-shaped prose (recipe instructions, real sentences): >=4 real words and
    # ends with terminal punctuation
    if alpha_words(nl) >= 4 and s[-1] in '.!?':
        return True
    return False

def is_furniture(line, nl, count):
    """Return True iff line should be CUT (assumes not protected)."""
    aw = alpha_words(nl)
    # photo / credit line, repeated a few+ times
    if count >= T_CREDIT and CREDIT.search(line):
        return True
    if count >= T_SHAPE:
        # numeric page furniture / numeric table rows: has a digit, <=1 real word
        if '#' in nl and aw <= 1:
            return True
        # table separators / pipe-structured rows with no real words (---|---, a|b|c of digits)
        if '|' in nl and aw <= 1:
            return True
    return False

def _degenerate(line):
    """True iff a single line is a short-cycle token repetition (nav/OCR artifact)."""
    toks = line.split()
    if len(toks) < R_DEGEN:
        return False
    ntoks = [re.sub(r'\d', '#', t.lower()) for t in toks]
    return len(set(ntoks)) <= DEGEN_DISTINCT

def within_dedup_book(text):
    lines = text.split('\n')
    nonempty = [i for i, l in enumerate(lines) if l.strip()]
    counts = Counter(norm(lines[i]) for i in nonempty)

    remove = []
    reasons = Counter()
    for i in nonempty:
        s = lines[i]
        # detector 2: degenerate within-line repetition (independent of count)
        if _degenerate(s):
            remove.append(i); reasons['degen'] += 1; continue
        nl = norm(s)
        c = counts[nl]
        if c < T_CREDIT:                     # cheap reject: nothing fires below the lowest threshold
            continue
        if is_protected(s, nl):
            continue
        if is_furniture(s, nl, c):
            remove.append(i); reasons['furniture'] += 1

    if not remove:
        return text, {'removed': 0}
    if len(remove) > BREAKER_FRAC * max(len(nonempty), 1):
        return text, {'flagged': True, 'reason': f'{len(remove)}/{len(nonempty)} lines',
                      'removed': 0}
    rm = set(remove)
    kept = [l for i, l in enumerate(lines) if i not in rm]
    out = re.sub(r'\n{4,}', '\n\n\n', '\n'.join(kept))
    return out, {'removed': len(remove), 'reasons': dict(reasons),
                 'sample': lines[remove[0]].strip()[:80]}
