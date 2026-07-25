# Overnight Report — Sentinel Improvement Program (2026-07-13/14)

## Headline
**Sentinel-v2 heads, gold-tuned: held-out test medAbsErr 909 → 0 (median book = EXACT
boundary), 0 overcuts.** Tails got their first gold and a v2 that cut median junk-kept
3,726 → 1,039 chars.

## Gold corpus (the fuel for everything)
- **110 gold labels total**: 90 heads (10 pilot + 40 wave2 + 40 wave3) + 20 tails
- All labeled by Opus 4.8 workers, ~2-3 min per 5-10 books, **110/110 mechanically
  validated** (mandatory quote-at-offset protocol; zero failures across all waves)
- Notable hard cases labeled: French (Ricoeur), Swedish, German reading-sample,
  window-limited front matter, acks-before-junk ordering, stage play, anthology,
  empty-md-heading chapters

## Heads: what was done (improvements 1,2,3,4,5 of the program)
1. **Content-anchor logic** (sentinel2.py): boundary = earliest content-heading
   (Preface/Foreword/Introduction/Prologue/Chapter One/Acknowledgments/Author's Note)
   after first junk evidence, validated by sustained prose; reconciled with the junk-chain
   prose-run via min(). This is the single biggest win — ~80% of true boundaries sit at
   such headings.
2. **Long praise quotes** to 700 chars w/ attribution pairing.
3. **Bio-block detector** (BIO regex: "is the author of", "lives in", ...).
4. **Chapter-summary TOC detector** (SUMMARY_TOC).
5. **Constant sweep** on 80 tune golds, composite objective (med + 500*overcuts - 10*within150).
   Best: ANCHOR_PROSE=150, GAP_PAD=800, RUN_CHARS=250, PROSE_T=0.50, FIRST_JUNK_MAX=600.
   Lesson: overcut-first lexicographic objective overfits to useless conservatism.

## Heads: results (10 held-out pilot golds, never touched by tuning)
| version | medAbsErr | within150 | overcut>200 |
|---|---|---|---|
| sentinel-v1 (production) | 909 | 4/10 | 1 |
| sentinel-v2 gold-tuned   | **0** | 6/10 | **0** |
Tune set (80): overcuts 5, med 22, within150 49/80.

## Tails: first-ever gold + v2 (partial; improvement program continues)
- sentinel_tail-v1 vs 20 golds: med 3,726 junk-kept, 5/20 within150, 14 books >500 junk, 1 content-cut
- tail-v2 (+INDEX_LINE, +CITE_LINE, +BACK_HEAD anchors): med 1,039, 8/20 within150,
  3 of 4 total-miss windows fixed, content-cut 1 (-440, needs investigation)

## New discoveries
- **Reading-sample "books"**: entire German Leseprobe with order-now links = whole-book junk class
- Reordered epubs, acks-first orderings, reading-group guides confirmed as tail junk classes

## NOT done yet (queue for next session)
- Improvement 6: adaptive window extension (window-max verdicts rescan at 30KB)
- Tail-v2 iteration 2: fix the 1 content-cut; 11 books still keep >500 junk; tail constant sweep
- 4 remaining head test misses >150 (list-content genres: poetry, headingless starts)
- Wire sentinel2 into books3_clean.py production wrapper + rerun v7 books3 pass
- More gold cheaply available if learning curves justify (Opus fleet validated)

## Files
sentinel2.py (new head engine), reports/sentinel2_params.json (tuned constants),
reports/gold_labels.json (90 heads), reports/gold_tail_labels.json (20 tails),
tail-v2 prototype inline in transcript (needs extraction to sentinel2.py)
