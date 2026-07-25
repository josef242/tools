# Bookclean standing policies (agreed with Josef)

- **Non-English books**: KEEP, tag by language. Lower priority than English fixes. (2026-07-13)
- **Markdown in Books3**: KEEP — markdown literacy has training value as a shared doc format.
  Broken conversion artifacts (dead xhtml anchor links, device-setting reader notes) are still
  fair game for a fine-polish pass. (2026-07-13)
- **Single-cut boundary model**: one head cut + one tail cut per book; gray-zone prose
  (acknowledgments, author intros) kept when after the junk; dedications/epigraphs interleaved
  before the boundary are acceptable sacrifices.
- **Catastrophic threshold**: >200 chars of true content removed = tracked failure class; target 0.
- **Books3 cut caps (v5)**: per-end cap = max(15% of book, 3000 chars); total ≤40% of book,
  else flag-don't-cut.
- **Bibliography/endnotes/about-author**: back matter → cut. Indexes in reference books: borderline, cut.
- **Hyphenation repair: DO NOT** — measured 0.9% true broken words vs 57% legitimate compounds.
- **Illustration/sidenote tags**: remove whole tag including caption (captions duplicate adjacent text).
- **Transcriber notes**: remove only structured forms (bracketed or paragraph-initial with punctuation);
  prose *about* transcribers is content.
- **Version retention**: undecided — currently keeping original + all versions (~540GB of 1.9TB).

## 2026-07-14 — STRUCTURAL HYBRID POLICY (supersedes v8 back-matter policy)

Prompted by discovering the OCR project's independent detector (../ocr/stage2b_detect_boilerplate.py)
and its hard-won lesson: **"structural back matter" (bookbinder's sense) is NOT
"training-harmful boilerplate" (LLM-training sense).** They nearly shipped a corpus
that stripped bibliographies/notes/chronologies and documented it as their #1 lesson.

**Cut by STRUCTURE, not by section NAME:**

CUT (data structure / marketing — no learnable language):
  - alphabetical indexes (comma-number soup; detect via letter-concentration + entry ratio)
  - ISBN / copyright / LoC-CIP / colophon / printing history
  - publisher catalogs, series promos, bare "Also by" title lists, praise/blurb pages
  - marketing: newsletter, review pleas, "visit our website"
  - PREVIEWS/EXCERPTS OF OTHER BOOKS (fluent prose, but marketing)
  - ebook nav artifacts, end-dumped TOCs, photo/design credit lists

KEEP (learnable prose, even though "back matter"):
  - the book's ending, epilogues, afterwords, postscripts
  - acknowledgments prose, author's notes, translator's notes
  - bibliographies that read as real citations (citation formatting is learnable)
  - endnotes with explanatory prose; glossaries with definitions
  - appendices with content, chronologies, author bios, reader's guides, Q&As

**METRIC — asymmetric cost (adopted from ocr/stage2b E-2b-2):**
    cost = 2.0 * (content chars wrongly CUT) + 0.1 * (junk chars wrongly LEFT)
  A 20x penalty on eating content. "Better to leave a little extra boilerplate
  than eat a valuable book." When in doubt: KEEP.

**ENGINE:** sentinel3.py — sentinel2 head engine (unchanged, won Arena A) +
new hybrid tail engine (body-protection, letter-concentration index detector,
comma-density prose test, suppression rules, KEEP-anchor headings).

**REJECTED BY EXPERIMENT:** multi-span tail cutting (excises interior boilerplate
but eats content inside memoir/photo-caption regions: asym cost 31.45 vs 3.27).
