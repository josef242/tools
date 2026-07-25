#!/usr/bin/env python3
"""Multi-span tail cutter — v2 design.

WHY v1 FAILED (asym cost 31.45, ate 8/20 books):
    v1 excised ANY run of cut/neutral paragraphs anywhere in the tail window.
    But interior "junk-looking" blocks are often INSIDE content -- photo captions
    in a memoir, scene-break glyphs, a short epigraph between chapters. Cutting
    them mutilated the book's ending.

THE FIX — three structural constraints, each derived from a v1 failure:

  1. JUNK-ZONE CONFINEMENT. Spans may only be excised at or after the first
     *confirmed* back-matter anchor (a strong marker or a detected index run).
     Everything before that anchor is the book's body and is untouchable, no
     matter how junk-shaped an individual paragraph looks. This alone kills the
     photo-caption-in-a-memoir class.

  2. ANCHORED SPANS ONLY. A span must CONTAIN a strong junk marker (ISBN,
     copyright, also-by, praise, promo, TOC-dump, credits) or be a detected
     index run. Neutral paragraphs are swept only as filler BETWEEN anchored
     material -- never excised on their own. (Zero-cut contract: junk must be
     positively proven.)

  3. PROSE ISLANDS SURVIVE. A span stops at any paragraph with body protection
     (prose / dialogue / KEEP-anchor heading). The island is kept, and cutting
     resumes after it -- which is the entire point of going multi-span: it can
     remove [ads] and [copyright] while preserving the [bio] sandwiched between
     them, which a single cut can never do.

  4. COST GATE. Each candidate span is only excised if it pays for itself under
     the asymmetric metric:  0.1 * span_chars > 2.0 * (estimated content chars
     inside it). With well-anchored spans the estimated content is ~0, so this
     is mostly a backstop against a span that swallowed something prose-like.

EXPECTED GAIN: the ~60% of tail junk that single-cut cannot reach because a
prose island (author bio, acknowledgments) sits behind it. On the gold sets the
labelers repeatedly had to choose between a bio and a 5KB index; a span cutter
takes BOTH.

VALIDATION PLAN (non-negotiable, given this saga's history):
    - score on all 3 existing gold tail sets (60 books) for recall + content-eaten
    - the metric to beat: sentinel3 single-cut = 39.2% recall, 2,997 eaten, 3.44 cost
    - ship only if recall rises AND content-eaten does not
    - then a fresh Opus jury on production span cuts before any corpus rebuild
"""

import re
import sys

sys.path.insert(0, '.')
from bakeoff import paragraphs
from sentinel3 import (classify_tail_para, mark_index_runs, body_protection,
                       CUT_MARKERS, KEEP_ANYWAY, SEP)

STRONG = {"isbn", "copyright", "loc_cip", "colophon", "also_by", "praise",
          "publisher_promo", "credits", "nav", "toc_dump", "preview"}

CUT_PENALTY = 2.0
LEAVE_PENALTY = 0.1


def _strong_marks(text):
    return {k for k, rx in CUT_MARKERS.items() if k in STRONG and rx.search(text)}


def find_spans(tail, full_text=None):
    """Return (spans, first_anchor) where spans = [(start,end)] char ranges to
    excise. Everything outside the spans is kept verbatim.

    full_text: the whole book, used by the GENRE GUARD. If the book's body is
    caption/definition/poem-shaped, span-cutting is disabled entirely.
    """
    if full_text is not None and body_genre_is_riskly(full_text):
        return [], None
    paras = paragraphs(tail)
    if not paras:
        return [], None

    kinds = [classify_tail_para(t) for _, _, t in paras]
    kinds = mark_index_runs(paras, kinds)

    # --- constraint 1: locate the junk zone -------------------------------
    # The zone opens at the FIRST paragraph that is a confirmed back-matter
    # anchor. Nothing before it may ever be touched.
    first_anchor = None
    for i, (s, e, txt) in enumerate(paras):
        if kinds[i] == "cut" and (_strong_marks(txt) or True):
            # index runs are already marked 'cut'; strong markers qualify too
            first_anchor = i
            break
    if first_anchor is None:
        return [], None

    # --- constraints 2+3: build anchored spans, stopping at prose islands ---
    spans = []
    i = first_anchor
    n = len(paras)
    while i < n:
        if kinds[i] != "cut":
            i += 1
            continue
        j = i
        anchored = False
        last_cut = i
        while j < n:
            s, e, txt = paras[j]
            t = txt.strip()
            k = kinds[j]
            if k == "keep":
                break                        # constraint 3: prose island -> stop
            if body_protection(txt) > 0:
                break
            if len(t) < 120 and KEEP_ANYWAY.match(t):
                break                        # KEEP-anchor heading -> stop
            if k == "cut":
                anchored = True
                last_cut = j
            j += 1
        if anchored:
            s0 = paras[i][0]
            # LINE REFINEMENT: the span's first paragraph may begin with the
            # book's closing sentence glued to a title list ("In time.\nAlso by
            # Donna VanLiere"). Start the span at the first junk LINE instead.
            txt0 = paras[i][2]
            pos = 0
            for line in txt0.split("\n"):
                st = line.strip()
                if st and (_strong_marks(st) or re.match(r"^[A-Z][A-Za-z' ]{2,60}$", st) is None):
                    if _strong_marks(st) and pos > 8:
                        s0 = paras[i][0] + pos
                        break
                pos += len(line) + 1
            spans.append((s0, paras[last_cut][1]))
        i = max(j, i + 1)

    # --- light heuristics from the v10 jury --------------------------------
    spans = merge_adjacent(paras, kinds, spans)   # block-atomicity
    spans = drop_holes(paras, kinds, spans)       # no holes through prose

    # --- constraint 4: cost gate ------------------------------------------
    kept = []
    for (s, e) in spans:
        region = tail[s:e]
        content_est = sum(pe - ps for ps, pe, ptxt in paragraphs(region)
                          if body_protection(ptxt) > 0)
        if LEAVE_PENALTY * (e - s) > CUT_PENALTY * content_est:
            kept.append((s, e))
    return kept, paras[first_anchor][0]


def apply_spans(tail, spans):
    """Remove spans from the tail, return the surviving text."""
    if not spans:
        return tail
    out, prev = [], 0
    for s, e in spans:
        out.append(tail[prev:s])
        prev = e
    out.append(tail[prev:])
    return "".join(out)


# ---------------------------------------------------------------------------
# LIGHT HEURISTICS (added after the v10 jury: 5 content-cuts in 30 books)
# ---------------------------------------------------------------------------

_DEF_LINE = re.compile(r"^\s*(?:\*\*)?[^|\n]{2,40}(?:\*\*)?\s*[|\u2014\u2013:-]\s+\S.{8,}$")
_CAPTIONISH = re.compile(r"(?i)\b(?:shown|pictured|photo|photograph|image|above|below|left|right)\b")


def body_genre_is_riskly(full_text):
    """GENRE GUARD (jury failures: Images of America -- a caption-bodied history;
    a knitting manual; a poetry collection).

    If the BOOK'S OWN BODY is structurally shaped like the junk we hunt
    (captions, term|definition lines, short poem-like stanzas), then our junk
    detectors cannot be trusted inside it. Sample the middle of the book and
    bail out of span-cutting entirely for these genres.
    """
    n = len(full_text)
    if n < 4000:
        return True                      # too short to judge -> be conservative
    mid = full_text[n // 2: n // 2 + 6000]
    ps = [p for p in paragraphs(mid) if p[2].strip()]
    if len(ps) < 5:
        return False
    lines = [l for _, _, t in ps for l in t.split("\n") if l.strip()]
    if not lines:
        return False
    defish = sum(1 for l in lines if _DEF_LINE.match(l)) / len(lines)
    capish = sum(1 for _, _, t in ps if _CAPTIONISH.search(t)) / len(ps)
    short = sum(1 for l in lines if len(l.strip()) < 60) / len(lines)
    prose = sum(1 for _, _, t in ps if body_protection(t) > 0) / len(ps)
    # definition-heavy (glossary/pattern book), caption-heavy, or poem-like
    return defish > 0.25 or capish > 0.35 or (short > 0.70 and prose < 0.20)


def drop_holes(paras, kinds, spans):
    """NO-HOLES RULE (jury failures: a Spanish glossary sliced in half, poems in a
    collection, a paragraph carved out of the MIDDLE of an acknowledgments).

    A span with genuine content on BOTH sides is a hole punched through prose --
    exactly the failure mode multi-span introduces over single-cut. Keep only
    spans that reach the end of the tail or are backed by junk on one side.
    """
    if not spans:
        return spans
    starts = {s: i for i, (s, e, _) in enumerate(paras)}
    ends = {e: i for i, (s, e, _) in enumerate(paras)}
    last_para = len(paras) - 1
    kept = []
    for (s, e) in spans:
        i0 = starts.get(s, None)
        i1 = ends.get(e, None)
        if i0 is None or i1 is None:
            kept.append((s, e))
            continue
        before_content = any(kinds[k] == "keep" for k in range(max(0, i0 - 2), i0))
        after_content = any(kinds[k] == "keep" for k in range(i1 + 1, min(len(paras), i1 + 3)))
        reaches_end = i1 >= last_para - 1
        if before_content and after_content and not reaches_end:
            continue                     # HOLE -- refuse it
        kept.append((s, e))
    return kept


def merge_adjacent(paras, kinds, spans, gap_paras=2):
    """BLOCK-ATOMICITY (jury: 'the copyright block gets fragmented -- ISBN line
    removed, license paragraph and address dump kept'; indexes split into two
    cuts leaving stranded scraps as fake islands).

    Merge spans separated only by non-content paragraphs, so a junk block is
    excised whole or not at all.
    """
    if len(spans) < 2:
        return spans
    ends = {e: i for i, (s, e, _) in enumerate(paras)}
    starts = {s: i for i, (s, e, _) in enumerate(paras)}
    out = [spans[0]]
    for (s, e) in spans[1:]:
        ps, pe = out[-1]
        i_prev = ends.get(pe)
        i_next = starts.get(s)
        if i_prev is not None and i_next is not None and i_next - i_prev <= gap_paras + 1:
            gap_has_content = any(kinds[k] == "keep"
                                  for k in range(i_prev + 1, i_next))
            if not gap_has_content:
                out[-1] = (ps, e)        # swallow the junk-only gap
                continue
        out.append((s, e))
    return out
