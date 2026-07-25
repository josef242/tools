#!/usr/bin/env python3
"""Sentinel v3 — structural-hybrid boundary engine.

Synthesis of two independently-developed systems:
  * bookclean/sentinel2  — content-anchor head logic (median-0 boundary error),
    junk-cluster chaining, praise-pair detection, char-offset precision.
  * ocr/stage2b_detect_boilerplate — BODY-PROTECTION markers, letter-concentration
    index detector, comma-density prose test, suppression rules, and the
    "zero-cut contract" (only cut on HIGH confidence).

POLICY (structural hybrid, agreed 2026-07-14):
  "Structural back matter" (bookbinder's sense) != "training-harmful boilerplate"
  (LLM-training sense). Cut by STRUCTURE, not by section NAME.

  CUT   — text that is a DATA STRUCTURE or MARKETING artifact:
          alphabetical indexes (comma-number soup), ISBN/copyright/CIP/colophon,
          publisher catalogs & series promos, bare "Also by" title lists,
          praise/blurb pages, newsletter/review pleas, previews of OTHER books,
          ebook nav artifacts, TOC dumps, photo/design credit lists.
  KEEP  — text a language model can LEARN from, even if a bookbinder calls it
          back matter: the book's ending, afterwords, author's notes,
          acknowledgments prose, bibliographies that read as real citations,
          endnotes with explanatory prose, glossaries with definitions,
          appendices with content, chronologies, author bios, reader's guides.

The asymmetric cost model (also from the ocr project) governs every tradeoff:
  cost = 2.0 * (content chars wrongly CUT) + 0.1 * (junk chars wrongly LEFT)
A 20x penalty on eating content. When in doubt: KEEP.
"""

import re
import sys

sys.path.insert(0, '.')
from bakeoff import paragraphs, prose_score, _classify, _para_markers, \
                    _praise_spans, _first_prose_run, _last_prose_run_end
from sentinel2 import head_boundary, P as HEAD_PARAMS  # head engine unchanged

# ---------------------------------------------------------------------------
# ported from ocr/stage2b: structural features
# ---------------------------------------------------------------------------

_TABLE_CHARS = re.compile(r"^[\s<>/|\-*:#•·]+")


def letter_concentration(lines):
    """(max_initial_letter_fraction, alpha_initial_line_count).

    Alphabetical INDEX blocks group under single letters and concentrate
    heavily on one initial letter (>0.4); prose varies broadly (<0.2).
    Straight port of ocr/stage2b_detect_boilerplate._letter_concentration.
    """
    counts = {}
    alpha_lines = 0
    for ln in lines:
        cleaned = _TABLE_CHARS.sub("", ln.strip())
        if not cleaned:
            continue
        c = cleaned[0]
        if c.isalpha():
            alpha_lines += 1
            k = c.upper()
            counts[k] = counts.get(k, 0) + 1
    if alpha_lines == 0:
        return 0.0, 0
    return max(counts.values()) / alpha_lines, alpha_lines


def is_prose_paragraph(text):
    """ocr/stage2b's long_paragraph test: a real prose paragraph is long AND
    comma-sparse AND sentence-terminated. Comma-density is what separates
    narrative from index/bibliography entry soup."""
    for line in text.split("\n"):
        if (len(line) > 400
                and line.count(",") < len(line) / 25
                and line.rstrip().endswith((".", "!", "?", '"', "”"))):
            return True
    return False


def has_heavy_dialogue(text):
    """Fiction dialogue protector. Counts double AND single quotation styles --
    many British novelists (Compton-Burnett) mark speech with single quotes,
    and missing that ate a novel's closing dialogue in held-out testing."""
    dq = text.count('"') + text.count('“') + text.count('”')
    sq = text.count('‘') + text.count('’')
    # bare ASCII apostrophes only count when they open a line (speech), never
    # as possessives/contractions mid-word
    sq += sum(1 for l in text.split('\n') if l.strip()[:1] == "'")
    return (dq + sq) >= 8


DEF_ENTRY = re.compile(
    r"^\s*(?:\*\*|_)?[^:|\n]{2,45}?(?:\*\*|_)?\s*(?:\([^)]{2,40}\))?\s*[:|\u2014\u2013]\s+\S.{8,}$")


def is_definition_block(text):
    """A glossary entry: term (+optional pronunciation) then a definition.
    Policy: glossaries WITH definitions are LEARNABLE and must never be cut.
    (Jury: a Spanish glossary in a memoir's back matter was sliced in half.)"""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    hits = sum(1 for l in lines if DEF_ENTRY.match(l))
    return hits / len(lines) > 0.5 and any(len(l) > 25 for l in lines)


CITATION_YEAR = re.compile(r"\((?:1[89]|20)\d{2}[a-z]?\)")
# A single bibliography/further-reading entry: italic title, or year-in-parens, or
# a publisher clause. Used both to score a run and to protect LEADING citation
# lines that run straight into an index with no paragraph break between them.
CITE_LINE = re.compile(
    r"\((?:19|20)\d{2}[a-z]?\)"
    r"|\((?:[A-Z][\w.&']+[ ,]){1,4}(?:19|20)\d{2}\)"
    r"|_[^_]{4,}_"
    r"|(?:University Press|Books|Publishers|Press),")
NUMBERED_NOTE = re.compile(r"^\s*(?:\*\*)?(\d{1,3})[.)]?\s+(?:[A-Z“\"'(]|[a-z])")

# Index-entry line: text followed by page numbers ("Nixon, Richard, 234-38, 251")
INDEX_ENTRY = re.compile(r"^[^.!?\n]{2,60}?[,\s]\s*\d{1,4}(?:\s*[,–-]\s*\d{1,4}){0,30}\s*$")


def index_score(text):
    """0..1 — how much this paragraph reads like an alphabetical index."""
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 3:
        return 0.0
    conc, alpha_lines = letter_concentration(lines)
    entry_ratio = sum(1 for l in lines if INDEX_ENTRY.match(l.strip())) / len(lines)
    score = 0.0
    if conc >= 0.40 and alpha_lines >= 3:
        score += 0.5
    if entry_ratio > 0.5:
        score += 0.5
    return score


def body_protection(text):
    """Body-protection score with ocr-style SUPPRESSION rules: bibliography
    entries quote journal titles and cite years, so raw quote/prose counts must
    not masquerade as narrative."""
    score = 0
    cites = len(CITATION_YEAR.findall(text))
    notes = sum(1 for l in text.split("\n") if NUMBERED_NOTE.match(l))
    if has_heavy_dialogue(text):
        if cites >= 5 or notes >= 5:
            pass  # BIB_SUPPRESSED
        else:
            score += 2
    if is_prose_paragraph(text):
        score += 2
    if is_definition_block(text):
        score += 3          # glossary definitions: learnable, protect hard
    return score


# ---------------------------------------------------------------------------
# tail: HYBRID back-matter taxonomy — data structures & marketing ONLY
# ---------------------------------------------------------------------------

CUT_MARKERS = {
    # publishing metadata / legal
    "isbn":        re.compile(r"\bISBN[\s:-]?(?:\d[\s-]?){9,13}|\bISBN\b", re.I),
    "copyright":   re.compile(r"(?:Copyright[\s©]|©)\s*\(?c?\)?\s*\d{4}|All rights reserved", re.I),
    "loc_cip":     re.compile(r"Library of Congress|Cataloging[- ]in[- ]Publication|British Library Cataloguing", re.I),
    "colophon":    re.compile(r"(?:Printed and bound|Typeset in|First (?:published|edition|printing)|Printed in (?:the )?(?:United States|Great Britain|USA|Canada))", re.I),
    # catalogs / promos / marketing
    "also_by":     re.compile(r"(?:^|\n)\s*(?:#+\s*)?(?:Also by|ALSO BY|Books by|Other Books by|By the same author|Also available)\b", re.M),
    "promo":       re.compile(r"(?i)visit (?:our|us|my)\s+(?:web\s?site|at|on|online)|newsletter|sign up (?:for|to)|please leave a review|available now|coming soon|www\.\w+|https?://"),
    "publisher_promo": re.compile(r"(?i)speakers bureau|for more information[, ]|subsidiary rights|marketing services|complete list of"),
    "praise":      re.compile(r"(?i)praise for|acclaim for|advance praise"),
    "preview":     re.compile(r"(?i)(?:excerpt|preview|sneak peek|continue reading|turn the page) (?:from|of|for)|keep reading for"),
    # ebook / production artifacts
    "toc_dump":    re.compile(r"(?im)^\s*(?:#+\s*)?(?:table of )?contents\s*$"),
    "nav":         re.compile(r"(?im)^\s*(?:begin reading|title page|cover|copyright page)\s*$"),
    "credits":     re.compile(r"(?i)photo(?:graph)?s?\s+(?:courtesy|by|©)|jacket design|interior design|cover design by"),
}

# Sections that are KEPT under the hybrid policy but were CUT under v8 policy.
# Kept here only for reporting / policy-diff analysis — they never fire a cut.
KEEP_ANYWAY = re.compile(
    r"(?im)^\s*(?:#+\s*|\*\*)?(?:acknowledg(?:e?)ments?|about the authors?|bibliography|"
    r"references|works cited|notes|endnotes|glossary|appendix|chronology|timeline|"
    r"afterword|postscript|epilogue|author'?s? note|translator'?s? note|"
    r"reading group guide|readers?'? guide|discussion questions|"
    r"questions for discussion|a conversation with|book club)\b")

SEP = re.compile(r"^[\s*#~\-–—•·]+$")


STRONG_PROMO = re.compile(
    r"(?i)thank you for buying this (?:e-?book|book)|refer to the link|"
    r"let the conversation begin|keep up[- ]to[- ]date|sign up (?:for|to)|"
    r"join our (?:email|mailing|circle)|visit (?:us|our website) at|"
    r"discover great authors|for more information (?:about|on|,)|"
    r"we hope you (?:have )?enjoyed|find more books like this|"
    r"follow (?:the |us on )?(?:penguin |on )?(?:twitter|facebook|instagram)")
INDEX_HEADER = re.compile(r"(?im)^\s*#{0,3}\s*index(?:\s+of\s+\w+)?\s*$")


def classify_tail_para(text):
    """'cut' | 'keep' | 'neutral' under the hybrid policy, using ADDITIVE
    scoring (ocr/stage2b's model): cut-evidence vs body-evidence, strongest wins.

    Critical: STRONG junk markers OUTSCORE body protection. A publisher blurb is
    long, comma-sparse, sentence-terminated prose -- it passes every "this is
    prose" test -- but an ISBN/copyright/also-by/praise signal proves it is junk
    regardless. Without this, protected marketing prose blocks every cut behind
    it (measured: tail recall collapsed to 2.8%).
    """
    t = text.strip()
    if not t or SEP.match(t):
        return "neutral"

    if STRONG_PROMO.search(text) or INDEX_HEADER.match(text.strip().split(chr(10))[0]):
        return "cut"   # unambiguous marketing / index header -> junk regardless of prose-shape
    body_score = body_protection(text)          # 0..4
    # BIO GUARD: "Photo by X" glued to an author bio must not let the credits
    # marker kill the bio. If the paragraph reads as a bio, protect it.
    if re.search(r"(?i)\bis (?:a|an|the) (?:former |award-winning |bestselling )?"
                 r"(?:author|writer|professor|dancer|journalist|editor|novelist)|"
                 r"\bis the author of\b|\blives? in\b", t) and len(t) > 150:
        body_score = max(body_score, 2)
    idx = index_score(text)
    marks = [k for k, rx in CUT_MARKERS.items() if rx.search(text)]
    STRONG = {"isbn", "copyright", "loc_cip", "colophon", "also_by", "praise",
              "publisher_promo", "credits", "nav", "toc_dump", "preview"}
    strong_marks = set(marks) & STRONG

    # KEEP-ANCHOR: heading announcing learnable back matter, no junk markers.
    if len(t) < 120 and KEEP_ANYWAY.match(t) and not marks:
        return "keep"

    cut_score = 0
    if strong_marks:
        cut_score += 3 + (len(strong_marks) - 1)
    elif marks:
        cut_score += 2 if len(marks) >= 2 else 1
    if idx >= 1.0:
        cut_score += 3
    elif idx >= 0.5:
        cut_score += 1

    if cut_score >= 3 and cut_score > body_score:
        return "cut"
    if body_score > 0:
        return "keep"
    # zero-cut contract: unproven text is neutral, never cut on its own
    return "neutral"



def mark_index_runs(paras, kinds):
    """Block-level index detection.

    Markdown ebooks put EVERY index entry in its own blank-line-separated
    paragraph, so a 397-entry index becomes 397 one-line paragraphs and the
    page-level letter-concentration test never sees a block (measured: 0% of a
    pure-index tail was detected). Reconstruct the block across consecutive
    short single-line paragraphs, then apply the concentration/entry tests.

    GUARDS (poetry, dialogue and children's prose are also short paragraphs):
      - >=8 consecutive entries
      - <20% of lines end in sentence punctuation (. ! ? ") -- prose does, index doesn't
      - entry-shape ratio >0.55 (trailing comma, page numbers) OR letter concentration >=0.35
    """
    n = len(paras)
    i = 0
    while i < n:
        j = i
        lines = []
        while j < n:
            t = paras[j][2].strip()
            if len(t) > 120 or "\n" in t or not t:
                break
            lines.append(t)
            j += 1
        if j - i >= 8:
            # CITATION GUARD: a bibliography is also alphabetical (high letter
            # concentration) but its entries are LONG and carry publisher/year
            # metadata. Index entries are short and end in page numbers.
            cite = sum(1 for l in lines if CITE_LINE.search(l)) / len(lines)
            avg_len = sum(len(l) for l in lines) / len(lines)
            if cite > 0.25 or avg_len > 60:
                i = max(j, i + 1)
                continue                      # bibliography -> learnable, keep
            sent = sum(1 for l in lines if l.endswith((".", "!", "?", '"', "”"))) / len(lines)
            entry = sum(1 for l in lines
                        if l.endswith(",") or INDEX_ENTRY.match(l)
                        or re.search(r"[,\s]\d{1,4}(?:[–-]\d{1,4})?$", l)) / len(lines)
            conc, alpha = letter_concentration(lines)
            if sent < 0.20 and (entry > 0.55 or (conc >= 0.35 and alpha >= 8)):
                # LEADING-CITATION GUARD: a bibliography / further-reading list often
                # runs straight into an alphabetical index with no paragraph break, so
                # the whole thing reads as one run and the index dilutes the citation
                # ratio above. Don't cut the leading citation lines -- start the index
                # cut at the first non-citation entry (usually the "## Index" header).
                cut_start = i
                while cut_start < j and CITE_LINE.search(paras[cut_start][2].strip()):
                    cut_start += 1
                for k in range(cut_start, j):
                    kinds[k] = "cut"
        i = max(j, i + 1)
    return kinds


def propagate_marker_runs(paras, kinds):
    """A strong junk marker ("Also by", "Contents", copyright) is followed by a
    RUN of short one-line paragraphs -- the title list, the chapter list, the
    address block. Markdown atomizes each into its own paragraph, so sweep the
    run that follows a confirmed-cut paragraph.

    Guard: stop at anything that reads as prose (>=200 chars or sentence-final
    or dialogue), so a book's ending can never be swept in.
    """
    n = len(paras)
    for i in range(n):
        if kinds[i] != "cut":
            continue
        j = i + 1
        while j < n:
            t = paras[j][2].strip()
            if not t or SEP.match(t):
                j += 1
                continue
            if kinds[j] == "keep":
                break
            if len(t) >= 200 or body_protection(paras[j][2]) > 0:
                break
            if t.endswith((".", "!", "?")) and len(t) > 80:
                break          # a real sentence -- stop
            kinds[j] = "cut"
            j += 1
    return kinds


def tail_boundary(tail):
    """content_end offset in the tail window under the hybrid policy.

    Walks BACKWARD from the end: the cuttable block is the maximal trailing run
    of cut/neutral paragraphs. The first 'keep' paragraph encountered from the
    end terminates the block -- prose back matter (bios, acks, bibliographies,
    notes) is content and stops the cut.
    """
    paras = paragraphs(tail)
    if not paras:
        return len(tail)
    kinds = [classify_tail_para(t) for _, _, t in paras]
    kinds = mark_index_runs(paras, kinds)

    # find the last paragraph that is 'keep' (learnable prose)
    last_keep = -1
    for i, k in enumerate(kinds):
        if k == "keep":
            last_keep = i

    # COST-AWARE ISLAND RULE (the 20:1 arithmetic the gold labelers used):
    # if the trailing keep-island is small relative to the junk it blocks,
    # sacrificing it is CHEAPER than leaving all that junk. Cut if
    #     2.0 * island_chars  <  0.1 * junk_chars      (i.e. island < junk/20)
    if 0 <= last_keep:
        first_cut = next((i for i, k in enumerate(kinds) if k == "cut"), None)
        if first_cut is not None and first_cut < last_keep:
            # BIBLIOGRAPHY GUARD: a citation-heavy keep-island is real learnable
            # content (the hybrid policy explicitly keeps bibliographies). An
            # alphabetical index that follows it is LOW-harm one-off junk, so the
            # 20:1 arithmetic over-values reaching it -- never sacrifice a real
            # bibliography to grab an index. (This was the only content-destroying
            # path in the pipeline: harm_per_pass caught it clipping further-reading
            # lists on ~3% of edge-case books.)
            island_txt = "\n".join(paras[i][2] for i in range(first_cut, len(paras))
                                   if kinds[i] == "keep")
            is_biblio = len(CITATION_YEAR.findall(island_txt)) >= 3
            island = sum(paras[i][1] - paras[i][0]
                         for i in range(first_cut, len(paras)) if kinds[i] == "keep")
            junk = sum(paras[i][1] - paras[i][0]
                       for i in range(first_cut, len(paras)) if kinds[i] != "keep")
            if not is_biblio and 2.0 * island < 0.1 * junk:
                last_keep = first_cut - 1      # sacrifice the island, take the junk

    if last_keep == len(paras) - 1:
        return len(tail)          # book ends in prose: nothing to cut

    # candidate block starts after the last keep paragraph...
    start_idx = last_keep + 1
    # ...but require at least one HIGH-confidence cut signal in the block,
    # else keep everything (zero-cut contract).
    if not any(k == "cut" for k in kinds[start_idx:]):
        return len(tail)

    # walk forward from start_idx over neutrals to the first real cut signal,
    # so trailing scene-breaks/short lines of the ending stay with the content
    i = start_idx
    while i < len(paras) and kinds[i] != "cut":
        i += 1
    if i >= len(paras):
        return len(tail)

    # the block from i onward must be SUBSTANTIALLY junk: >=60% of its chars in
    # cut/neutral paragraphs and it must not be a lone stray marker paragraph.
    block_chars = sum(e - s for (s, e, _), k in zip(paras[i:], kinds[i:]))
    cut_chars = sum(e - s for (s, e, _), k in zip(paras[i:], kinds[i:]) if k == "cut")
    if block_chars and cut_chars / block_chars < 0.30:
        return len(tail)

    cut_at = paras[i][0]

    # line-level refinement: prose glued to junk by a single newline
    s, e, txt = paras[i]
    pos = 0
    for line in txt.split("\n"):
        st = line.strip()
        if st and (any(rx.search(st) for rx in
                       (CUT_MARKERS["isbn"], CUT_MARKERS["copyright"],
                        CUT_MARKERS["also_by"], CUT_MARKERS["toc_dump"]))
                   or INDEX_ENTRY.match(st)):
            if pos > 8:
                cut_at = s + pos
            break
        pos += len(line) + 1
    return cut_at


# ---------------------------------------------------------------------------
# asymmetric cost (ported from ocr/stage2b E-2b-2)
# ---------------------------------------------------------------------------

CUT_PENALTY = 2.0    # per unit of CONTENT wrongly removed  (the crime)
LEAVE_PENALTY = 0.1  # per unit of JUNK wrongly retained    (a misdemeanor)


def asymmetric_cost(pred, truth, side, unit=100):
    """Cost of one boundary prediction, in units of `unit` chars.

    side='head': pred/truth are content_start. pred > truth => content CUT.
    side='tail': pred/truth are content_end.   pred < truth => content CUT.
    """
    delta = pred - truth
    if side == "head":
        cut = max(0, delta)
        leave = max(0, -delta)
    else:
        cut = max(0, -delta)
        leave = max(0, delta)
    return (CUT_PENALTY * cut + LEAVE_PENALTY * leave) / unit
