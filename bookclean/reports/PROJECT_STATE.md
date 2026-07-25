# BOOKCLEAN PROJECT STATE — full handoff (read this FIRST after compaction)
Last updated: 2026-07-15. Author: Code. rig-27, ~/valhalla/code/tools/bookclean/

## ⭐ HARM ABLATION VERDICT (2026-07-17, LOCKED 3 seeds/arm) — the standing-debt experiment, done
Trained 132M KEEL toy, arm A=v9 (junk in) vs arm B=v11 (junk removed), identical recipe/order, 3 seeds.
RESULT: junk memorization is DECISIVELY REAL (26 sigma) but MODEST — **gap 0.474±0.018 nats = 1.61x**,
far below the pre-registered 10x bar. Content control NULL (A=B on real text). Harm is HETEROGENEOUS:
within-book furniture (page numbers, degenerate strings) = ZERO gap (low-entropy, B predicts it); the gap
lives in cross-book DUPLICATED boilerplate (worst-2% = "Library and Archives Canada Cataloguing..." = 11x).
Grad-anomaly metric NULL. Gap trajectory still CLIMBING at 5999 (1.20->1.49->1.60x) but decelerating (single-
pass, 0.63 epoch = a LOWER BOUND). IMPLICATIONS: cross-book dedup (v10) removes the real magnets (confirmed);
within-book dedup (v11) shows ~no training-harm benefit at this scale (reframe as corpus hygiene); the 20x
cut-asymmetry / "magnets harmful everywhere" priors were OVERSTATED. NEXT: skip Arm C/span judge (residual
milder than 1.6x); highest-value follow-up = MULTI-EPOCH run (memorization spikes with repeated exposure).
Full detail: reports/ABLATION_RESULTS.md + ABLATION_PREREG.md. Configs: mara_fsdp2/configs/books-abl-*.yaml.
Eval harness: tools/eval_ablation.py (SDPA path for Pascal), aggregate/grad_anomaly/harm_per_pass.

## WHAT THIS IS
Cleaning the 107.8GB RedPajama book corpus (~205,744 books, ~/data/book.jsonl) into
training-viable data for KEEL. First 11.3GB = DeepMind PG19 (Gutenberg); rest = Books3
(modern ebooks). Goal: strip training-harmful junk WITHOUT destroying learnable text.

## ENV / INFRA
- conda env: ~/miniconda3/envs/bookclean (python 3.12, torch 2.5.1+cu121). ALWAYS run
  python via: $HOME/miniconda3/envs/bookclean/bin/python  (NOT system python3, which lacks pip/torch)
- 2x GTX 1070 (Pascal, sm_61; cu121 wheels pinned — newer wheels dropped Pascal). driver 580.
- transformers/tokenizers/sentencepiece/protobuf/tiktoken installed in the env.
- HF model cached: sentence-transformers/all-MiniLM-L6-v2 (22.7M), distilbert-base-uncased (66M).

## CORPUS VERSION LADDER (all in ~/data/, ledgered + reversible)
- book.jsonl        original (untouched)
- book.v1.jsonl     de-redaction: reversed an Ofcom-list word filter (<DWnn> tokens); 270,903
                    restorations, 99.95% word-accuracy vs Gutenberg originals. [mapping: reports/dw_mapping.json]
- book.v2.jsonl     PG19 boilerplate (credits, sentinels)
- ... v3-v8 = iterations, several superseded
- book.v9.jsonl     Books3 front/back-matter via sentinel3 single-cut, hybrid policy.
                    Zero content-cut on 60+ gold books. (superseded as corpus of record by v10)
- book.v10.jsonl    v9 + CROSS-book dedup. 147,770 books touched, 45.3M chars removed, 86 flagged.
                    Gold-safe, audit 100% boilerplate. (superseded as corpus of record by v11.)
                    (NOTE: the old REJECTED span-cutter experiment also briefly used the name
                    v10; that is dead. The dedup v10 is the real one.)
- book.v11.jsonl    **CORPUS OF RECORD** (Josef shipped 2026-07-15) = v10 + WITHIN-book dedup.
                    29,449 books touched (14.3%), 8,554,587 lines / 74,596,845 chars removed
                    (furniture 8,540,867 + degenerate 13,720), 496 flagged (0.24%). 205,744
                    records (aligned). Gold: 0 prose removals. Harm-proxy: peak within-book
                    10-gram magnet HALVED (-53%), severe-magnet books -57%. Filter:
                    within_dedup_filter.py. Copied to ~/brainbox/!BOOKS/book.v11.jsonl.

## THE POLICY (structural hybrid — agreed with Josef, from his OCR project's lesson)
"Structural back matter (bookbinder sense) != training-harmful boilerplate (LLM sense)."
CUT (data-structures/marketing): alphabetical indexes, ISBN/copyright/CIP/colophon, publisher
  catalogs, bare "Also by" lists, praise/blurb pages, marketing/newsletter/review-pleas,
  previews-of-OTHER-books, ebook nav artifacts, end-dumped TOCs, photo/design credits.
KEEP (learnable prose): the book's ending, epilogues, afterwords, acknowledgments prose,
  author's notes, bibliographies-as-citations, endnotes-with-prose, glossaries-WITH-definitions,
  appendices, chronologies, author bios, reader's guides, discussion questions.
Full policy: reports/POLICIES.md

## METRIC (asymmetric cost, adopted from OCR project's E-2b-2)
cost = 2.0 * (content chars wrongly CUT) + 0.1 * (junk chars wrongly LEFT). 20x penalty on
eating content. ALWAYS report MARGINAL-over-rules recall AND content-eaten, never absolute
recall or AUC alone (you live at the far right of the ROC curve).

## THE ENGINES
- bakeoff.py       shared paragraph/scoring primitives
- sentinel2.py     HEAD engine (content-anchor logic; won Arena A vs OCR's stage2b 1.98 vs 10.90).
                   Params: reports/sentinel2_params.json. Head boundary median error ~20-40 chars.
- sentinel3.py     TAIL engine (structural-hybrid, zero-cut contract). tail_boundary(tail).
                   Just patched with STRONG_PROMO + INDEX_HEADER (override body-protection).
- span_cutter.py   multi-span tail cutter — REJECTED 3x (eats content). find_spans(tail, full_text).
                   Has genre-guard/no-holes/block-atomicity/glossary guards but still unsafe.
- books3_clean_v9.py  production wrapper for v9 (sentinel2 heads + sentinel3 tails + rails).
- OCR project's independent detector: ~/valhalla/code/ocr/stage2b_detect_boilerplate.py
  (page-granular, rules, has BODY-PROTECTION markers + letter-concentration index detector we ported).

## GOLD SETS (all book-level, hand-labeled by Opus fleet, mechanically validated)
- 90 HEAD labels: reports/gold_labels.json (pilot+wave2+wave3), books in gold_{pilot,wave2,wave3}_books.json
- 140 TAIL labels (hybrid policy):
    reports/gold_tail_labels_hybrid.json (20)  <- gold_tails_books.json
    reports/gold_tail_test_hybrid.json (20)    <- gold_tails_test_books.json
    reports/gold_tail_val3_hybrid.json (20)    <- gold_tails_val3_books.json
    reports/gold_tail_strat_labels.json (80)   <- gold_tails_strat_books.json  [STRATIFIED: poetry/
       captions/craft/cookbook/travel/plays/technical/reference/guard_fires/random — the edge-case set
       that finally caught the span cutter. body_sample field = mid-book text for genre calibration.]
- NOTE: tail labels are single-cut (content_end offset). A span-aware relabel would credit
  multi-span better. Old pre-hybrid tail labels (reports/gold_tail_labels.json) are OBSOLETE.

## KEY FINDINGS (the hard-won lessons — do not re-derive)
1. RULES BEAT LLMs for boundary detection (both us AND the independent OCR project concluded this;
   per-item neural classification loses cross-item coherence).
2. WE OPTIMIZED THE WRONG TARGET FOR 2 DAYS: v8 cut bibliographies/bios/notes (~35M tokens of
   LEARNABLE prose); all 3 juries blessed it because we wrote the rubric. OCR project's independent
   view exposed it. LESSON: the hardest problem is KNOWING whether you're right.
3. THE METRIC CAN REWARD DOING NOTHING (20x penalty): OCR engine cut 0% of our tails and scored
   "great". Always report marginal recall + content-eaten.
4. MULTI-SPAN FAILED 3x, each after looking good on the preceding metric. A single cut can be wrong
   in one place; multi-span punches holes through prose. Killed it.
5. NEURAL (MiniLM-22M paragraph classifier, raw text): 99.6% junk-precision @0.99 BUT wired as a
   promoter it EATS CONTENT (paragraph-precision != boundary-safety; one FP moves the boundary).
   Under our own asymmetric cost it's NET-NEGATIVE (+9,028 cost). 
6. ARCHITECTURAL CEILING (triangulated 3 ways, all ~25%): single-cut recall is capped by NON-MONOTONE
   back matter (junk interleaved with content — bio after ads). confined-wiring ceiling = 24.9%
   (oracle), rules = 24.1-25.2%, multi-span = higher but eats content. NOT a classifier/gold/rules
   problem — architectural. 75/140 gold books have junk but NO rule anchor.
7. **HARM DECOMPOSITION (THE DECISIVE RESULT, 2026-07-15)**: of the unreachable 75% tail junk,
   char-weighted: (a) cross-book DUPLICATED 23.9%+ [HIGH harm, FREE via dedup] | (b) one-off
   DATA-STRUCTURE 24.3% [moderate, needs multi-span] | (c) one-off FLUENT PROSE 51.8% [near-zero
   harm — previews of other books, "nearly just more book"]. (a) is a LOWER BOUND (25% corpus sample).
   => DEDUP catches the dangerous junk free, half the residual isn't worth cutting, only ~18% of
   total tail junk genuinely needs multi-span (and it's moderate harm). This REDUCES multi-span to
   a footnote and INVERTS the sequencing: DEDUP FIRST.

## CURRENT DECISION (converged with Rook, pending Josef's final go)
1. v9 is corpus of record. KEEL preprocessing can start against it.
2. BUILD DEDUP PASS NEXT (highest value): global exact-dup line/paragraph filter + corpus-level
   near-dup book detection (Books3 multi-editions, PG19/Books3 classic overlap). No GPU, rules-grade.
   Catches high-harm cross-book boilerplate regardless of position, incl. non-monotone tails.
3. Re-measure residual AFTER dedup. Multi-span/neural go/no-go against THAT number (likely footnote).
4. Neural/context = research track, off critical path.

## WHAT'S RUNNING RIGHT NOW (background jobs)
- cuda:0: context-MiniLM training (train_context.py). NOTE: trained on LEAKY paragraph-level split
  — its classifier metrics are INFLATED; the ARCHITECTURAL findings are unaffected (whole-book eval).
- cuda:1: DistilBERT-66M ep2 (train_neural_distilbert.py). Size-axis datapoint.
- reports/para_booksplit.json: LEAKAGE-FREE book-level split BUILT (val = whole held-out test+val3
  books). Use build_para_booksplit.py output to RETRAIN cleanly before trusting any context-cell number.

## ROOK'S STANDING CORRECTIONS (adopted)
- Referee must anchor to HUMAN (Josef) labels, not Opus labels (else external-to-model not external-to-loop).
- Recalibrate threshold per retrain (or use split-conformal for a distribution-free per-cut guarantee).
- Edition/near-dup screening between gold and training pools before trusting holdout numbers.
- Keep epsilon of pure-random in every mining batch (active-learning is blind to SHARED blind spots).
- If multi-span ever built: defense-in-depth = spans-as-atomic-units + transition-priors(HMM/CRF) +
  independent decorrelated veto (perplexity scorer / OCR engine). NOT "classifier accurate enough to not need cages".
- Anthology stratum (12,846 high-churn books, reports/wide_candidates.json): write explicit inter-piece
  policy BEFORE labeling (per-piece titles+editorial intros = KEEP; inter-piece catalogs+previews = CUT).
- STANDING DEBT: everything (20x asymmetry, "training-harmful") encodes UNTESTED beliefs about training.
  The toy harm ablation (v9 vs v9+X: boilerplate-emission rate + val loss on pristine held-out books)
  is the ground truth this all proxies. Do it before conclusions calcify.

## MINING / FLYWHEEL ASSETS
- reports/active_candidates.json (180 books, GPU model-vs-rules disagreement; ~1.4% of books meaningfully disagree)
- reports/wide_candidates.json (400 books, CPU keep<->cut churn proxy; anthologies/collections concentrate here)
- harm_decompose.py, mine_active.py, mine_wide.py

## NEXUS THREAD
Rook briefing thread: messages #206 (my brief) -> #207,#209,#211 (Rook) / #208,#210 (my replies).
Rook is Fable — keep messages to clean cleaning-methodology facts (avoid the redaction topic).
Memories #32-39 have the technical trail.

## DEDUP PASS (2026-07-15) — THE SURPRISING WIN, likely book.v10.jsonl
Built + validated + running the full corpus pass. This is the highest value/effort move of the project.

FILTER (dedup_filter.py): remove cross-book boilerplate LINES corpus-wide. A line is removed iff
(STRONG boilerplate pattern) AND (digit-normalized line appears in >=3 distinct books). Per-book
CIRCUIT BREAKER: if removals >25% of a book's non-empty lines -> remove NOTHING, flag (protects
cookbooks/anthologies/quotation-dicts). Freq table persisted: reports/line_freq.pkl (16.5M lines in
>=2 books, 215MB, built by build_freq_table.py from v9).

DESIGN JOURNEY (all measure-first saves):
- Naive "cut any high-frequency line" is UNSAFE: high-mult buckets contain CONTENT — common dialogue
  ("What are you doing?" x4390), recipe ingredients ("1 cup buttermilk" x799), chapter headings.
  Frequency identifies anything COMMON, most short common lines are content.
- Josef's LENGTH-FLOOR insight (confirmed): content-risk is 75% <25 chars, ~99% <60. Long+frequent =
  boilerplate. But length alone insufficient (recipe INSTRUCTIONS are long).
- Rook's conceptual core: frequency conflates LANGUAGE (regenerates everywhere, high LM prob) vs
  PUBLISHING PROCESS (stamped, low prob + high freq). Signal = multiplicity IN EXCESS of linguistic
  expectation. Length is a cheap proxy (regen prob decays exp. with length).
- Rook's gates (adopted): (1) RUN-DENSITY — junk arrives in runs; isolated dup = quote/epigraph (protect),
  dense dup block = copyright page (cut). (2) CIRCUIT BREAKER for dup-heavy BOOKS. (3) decaying mult
  threshold w/ length. Missing protected class Rook named: QUOTATION (scripture/poems/recipes/legal).
- Chose the SAFE increment: Tier-1 patterns + breaker only. Deferred Tier-2 (length+mult) — suspect
  audit showed it risks recipes/dialogue.

VALIDATION (all passed): gold 140-tail zero GENUINE content damage (616 chars removed were boilerplate-
in-kept-region = single-cut label artifacts, a FEATURE); removed-line audit 100% boilerplate; on 3k
sample 76.3% books touched, 2/3000 flagged. Full pass tracking identically (76.6% touched, 0.05% flagged).

**THE SURPRISING RESULT (the project's biggest lesson):** dedup (~hours, CPU-only) attacks the HIGH-harm
junk (cross-book duplicated boilerplate = memorization magnets) MORE DIRECTLY than the elaborate boundary
machinery (sentinel/bake-off/neural = DAYS + GPUs, hit ~25% architectural ceiling). Comparison:
- volume: v9 boundary removes MORE raw chars (bulky front/back matter). BUT volume != value.
- harm-weighted: dedup wins — targets the HIGH-harm category; boundary removes mix incl. near-zero-harm.
- reach: dedup 76% of books (scattered boilerplate near-universal) vs boundary (contiguous-block-only, capped).
- COMPLEMENTARY not competing: dedup catches what boundary structurally CAN'T (buried/scattered/no-anchor).
- effort: dedup hours/CPU vs boundary days/GPU. If we'd measured the HARM denominator first, dedup comes
  BEFORE the 2 days of boundary engineering. (Rook's sequencing thesis, borne out.)

## PENDING POST-DEDUP VALIDATION (Rook's asks, do after pass completes)
- pristine-set FP test: run filter over v1 Gutenberg-verified books (free FP check)
- top-100 books by removal FRACTION (any cookbooks/anthologies = failure genre)
- before/after top-100 10-grams (harm-proxy: publisher templates should vanish -> purely linguistic)
- exact side-by-side char deltas: v1->v9 (boundary) vs v9->v10 (dedup)

## STANDING DEBT (Rook, unchanged, important): everything encodes UNTESTED beliefs about training.
The toy harm ablation (v9 vs v9+X: boilerplate-emission rate + val loss on pristine held-out books) is
the real ground truth. Do before conclusions calcify.

## DEDUP DONE + HARM-PROXY REDIRECT (2026-07-15, latest)
book.v10.jsonl BUILT (205,744 lines, = v9 + cross-book dedup). Full pass: 147,770 books touched (71.8%),
854,391 lines / 45.3M chars removed, 86 flagged. Gold-safe, audit 100% boilerplate.

HARM-PROXY (top repeated 10-grams, v9 vs v10, Rook's memorization-magnet metric) — THE REDIRECT:
- cross-book dedup helped MODESTLY: "all rights reserved no part of this book may be" (x686) fell out of
  top-12; copyright-clause fragments down ~10-15%.
- BUT the DOMINANT magnets are WITHIN-BOOK repetition, UNTOUCHED: "here here here here..." x2811 (ebook
  nav/OCR artifact, #1 offender), Dutch legal citations repeated, "ref 1 ref 2 ref 3 ref 4 ref 5" x581.
  Cross-book dedup counts DISTINCT BOOKS so within-book repeats (running headers, page furniture, nav) are
  invisible.
=> NEXT HIGH-VALUE MOVE: WITHIN-BOOK line dedup (line repeated >N times in ONE book = furniture/nav = junk).
  Nearly free, CPU. Needs content guard (legit refrains, repeated dialogue, song choruses) — freq + shape
  (short/no-sentence-structure/page-num/header). The "here here here" and "ref 1 ref 2" look like OCR/
  conversion artifacts, maybe own cleanup. AWAITING Josef's go to build.
Recipe now: v9 boundary + cross-book dedup + WITHIN-book dedup (all complementary, CPU).

## DISTILBERT SIZE-AXIS RESULT (research record, off critical path; LEAKY split so absolute #s inflated,
## but the delta between models is meaningful since they share the leakage)
Held-out gold-paragraph junk precision/recall @0.99:
  MiniLM-22M ep2:     0.9959 / 0.2159
  DistilBERT-66M ep2: 0.9983 / 0.2651
=> 3x params buys ~+5 pts recall at fixed precision AT MATCHED EPOCHS. Rook's confound (MiniLM ep2 0.216
  beat DistilBERT ep1 0.180) resolved: at matched ep2 size wins, so the effect is real not just duration.
  Modest, and MOOT for production (confined ceiling +0.8, unconfined net-negative). Models saved:
  reports/para_clf.pt (MiniLM), reports/para_clf_distilbert.pt. Both GPUs now idle.

## NN-AS-JUDGE RESULT (Josef's reframe — I had conflated promoter vs judge; he was right)
I tested NN as PROMOTER (adds cuts -> ate content). Josef's JUDGE config: aggressive operator proposes,
NN VETOES spans it isn't confident are all-junk. Tested (span-cutter + MiniLM min-prob veto, 4 gold sets):
  judge@0.5: recall 1.6% / eaten 0 | @0.9: 0.1%/0 | @0.99: 0.0%/0
  baselines: rules single-cut 25.2%/2,755 | span-alone 3.0%/3,550
=> VETO WORKS FOR SAFETY (0 eaten always). But recall collapses: min-prob-over-span veto needs EVERY para
  to clear thr, NN recall ~22%, so P(all k clear)~0.22^k~0. Binding constraint is NN RECALL not precision.
  Judge is production-relevant IF recall lifts (= clean book-split retrain + more gold + context). Likely
  real fix (Rook's defense-in-depth #1): a SPAN-LEVEL judge (classify whole span as unit, conformal
  guarantee), not paragraph-min. Neural track reframed: aggressive-operator + high-precision veto, NOT dead.

## ROOK'S CORRECTIONS ON THE JUDGE (#216 — important, changes the conclusion)
1. I DIAGNOSED THE WRONG BOTTLENECK. final_recall = proposer_coverage x judge_pass_rate. Span-alone recall
   is 3.0% -> the PROPOSER only finds 3% of junk; a perfect judge caps the config at 3%. Binding constraint
   is PROPOSER COVERAGE, not NN recall. In propose-verify, proposer PRECISION barely matters — crank the
   proposer toward max recall, let the judge kill garbage. My timid-proposer+strict-judge = double-filter = 1.6%.
2. VETO SEMANTICS BUG (the fix): min-junk-prob-over-span conflates "no evidence of content" with "not certain
   junk." Uncertainty != content evidence. INVERT: veto iff ANY paragraph shows POSITIVE content evidence
   (max content-prob > tau), not "pass only if unanimously confident junk." This DECOUPLES the requirements:
   the judge needs CONTENT-DETECTION (easy, majority class), NOT junk-recall (my 22% = hardest ROC corner).
   Answers Josef's "why not just a better NN" loop-back: judge needs only content-detection, likely strong today.
   MEASURE (free w/ clean retrain): fraction of gold CONTENT paragraphs with content-prob > tau. That's the
   judge's real capability, not junk-recall.
3. Leaky-result epistemics: leak biases OPTIMISTIC, so a failing optimistic measurement is CONCLUSIVE (truth
   is worse). Only optimistic PASSES are untrustworthy. Judge "not viable yet" stands qualitatively.
4. Judge status upgraded: not "research track" but "PRODUCTION-VIABLE CONFIG, GATED on the toy harm ablation
   showing the post-dedup residual is worth cutting." Feasibility solved (veto=safe); VALUE now unmeasured &
   shrinking (dedup keeps eating the target).

## ROOK'S RANKING (adopted)
1. WITHIN-BOOK DEDUP (do next). Field-standard (Gopher/MassiveText dup-line-fraction + top-n-gram rules, C4,
   RefinedWeb). Massively-repeated within-doc tokens = loss-spike/gradient-anomaly source, not just memorization.
   PROTECTED CLASSES: **DRAMA speaker tags** (repeat 100s/play — cutting destroys every script!), poetry
   refrains, song choruses, liturgical responses. KEY DESIGN: separate running-headers from chapter-headings
   by COUNT — chapter headings repeat ~TENS (chapter count), page furniture ~HUNDREDS (page count); threshold
   lives between. Bonus signal: page furniture recurs QUASI-PERIODICALLY by char offset; nothing legit does.
2. CLEAN NN RETRAIN (book-split, reports/para_booksplit.json built). Config change + GPU-hrs; unblocks judge +
   context prediction + all registered bets at once. Measure the content-detection operating point (free).
3. SPAN-LEVEL JUDGE. Only on clean base; only if post-within-book-dedup residual justifies per harm gate.
   Train on spans-as-units: gold junk spans = positives; synth negatives by SHIFTING gold spans to STRADDLE
   the content boundary (the exact error the judge exists to catch). Fuse dup-multiplicity + pattern + position,
   not NN-alone.

## v10 FOLLOW-UPS (Rook): (a) feed the 616-char gold corrections back into gold (versioned). (b) READ the 86
## circuit-breaker books = failure-genre discovery (quotation dicts/cookbooks expected; anything else = new stratum).

## WITHIN-BOOK SURVEY DONE (within_survey.py) — filter design ready to BUILD
Distinct (book,line) pairs by within-book repeat count (~50% corpus): 3-9: 1.92M | 10-49: 314k |
50-199: 54k | 200+: 9.4k.
KEY FINDING: COUNT ALONE DOES NOT SEPARATE junk from content. Examples:
  high-repeat FURNITURE (CUT): "---|---" x156 (md table sep), "####| **##.##**" x456, "###."/"##." x397
    (page numbers), "corrie wingate/apa publications" x152 (photo credit), running headers.
  high-repeat CONTENT (PROTECT): "* * *" scene break x896 (!), recipe instr "preheat the oven to ###°f" x34,
    dialogue '"no."' x6, section headings "## summary", refrains, DRAMA SPEAKER TAGS (Rook).
FILTER DESIGN (count x SHAPE, not count alone):
  CUT a line iff within-book-repeat >= T AND furniture-shaped:
    - markdown table rows (^[\s|:-]+$ with |, or "---|---"), pure digit/punct lines ("###.", "##."),
    - photo/credit patterns (".../apa publications", "istock", "getty", "photo by"),
    - short running-header-ish repeated at ~page-count frequency (high count + short + no sentence punct).
  PROTECT (never cut): "* * *"/"***"/scene-break glyphs, dialogue (starts with quote), sentence-shaped prose,
    markdown headings "#..#" (low repeat = chapter count), recipe/ingredient lines, drama SPEAKER: tags,
    poetry refrains/choruses.
  Validate like cross-book dedup: gold zero-content-damage + removed-line audit (100% furniture?) +
    per-book removal-rate (spike = bug) + circuit breaker for repetition-heavy books (scripts/poetry).
Note: "here here here" x2811 from harm-proxy is within-LINE token repetition (a single nav/OCR line), may
need its own tiny detector (line = same token repeated >=N times).

## WITHIN-BOOK DEDUP DONE -> book.v11.jsonl (2026-07-15) — validated, awaiting Josef's ship call
FILTER within_dedup_filter.py (SAFE tier), two detectors:
  (1) REPEATED-LINE FURNITURE (count x SHAPE): normalized line repeating >= T_SHAPE(50) within one book,
      cut iff furniture-shaped [numeric/table with <=1 real word, e.g. "###."/"---|---"; OR photo/credit
      pattern at >= T_CREDIT(5)] AND not protected. Digit-normalization (\d->#) means a numbered SEQUENCE
      (Day1..Day90, page 1..400) collapses to one form w/ high count -> all cut. Right for page numbers;
      would over-cut semantic numbered headings, so those are PROTECTED (see below).
  (2) DEGENERATE LINE: single line with >=30 tokens and <=3 distinct normalized tokens ("here here...",
      "ref # ref # ...") -> whole line removed. Catches the v9->v10 harm-proxy #1 magnet (within-LINE, not
      within-line-repeat, so line-dedup alone missed it). Only 13,720 such lines corpus-wide (rare).
  PROTECTED (never cut): dialogue(quote-start), markdown headings, SEQUENTIAL STRUCTURAL headings
    (Chapter/Day/Part/Act/Scene/Lesson/Week/Book/Vol/Canto + num — Rook's chapter-heading flag), citation
    apparatus (Ibid/op.cit/loc.cit — KEEP-citations policy), speaker tags (ends ':' short, or ALLCAPS short
    -> drama), scene-break glyphs (* * *), sentence-shaped prose (>=4 words + terminal punct -> recipes).
  CIRCUIT BREAKER: >30% of a book's non-empty lines removed -> remove NOTHING, flag. 496 books flagged
    (furniture-dominated: data tables/indexes/catalogs/OCR dumps = failure-genre mining target).
RESULTS: 29,449 touched (14.3%), 74.6M chars removed (65% MORE than cross-book dedup's 45.3M!), all
  within-book furniture. TOP REMOVED corpus-wide (reports/within_dedup_topremoved.json): "###." x2,970,657
  + "##." x1,522,535 (page numbers = the biggest memorization magnets in the corpus), "---|---" x382,644
  + "|"/"##"/"| |" (table/md syntax), "(#.##)"/"fig. #.##"/phone-numbers (numeric refs). ZERO prose in top-30.
VALIDATION (validate_within.py): (A) gold 140-book audit -> 0 prose-shaped removals (30 touched, 6813
  furniture lines). (B) 3k v10 sample -> 14.2% touched, 5 flagged, removal-rate median 7.9%/p90 16.8%/max
  29.4% (breaker well-calibrated), removed lines furniture-only. (C) timing probe passed (17 books/s, no hang).
BORDERLINE CLASSES cut (flag for Josef): "serves #" x44k (recipe yield label), "##. chapter ##" x21k (TOC
  entries). Both non-prose structural labels; defensible as furniture but a stricter policy might keep them.

## v11 POST-BUILD VALIDATION (2026-07-15) — both strong; Josef asked to HOLD until these ran
HARM-PROXY (harm_proxy_fast.py, 10k record-aligned Books3 books, skip 50k PG19; two INDEPENDENT metrics):
  (2) PEAK within-book 10-gram repeat/book (memorization-magnet strength; tuple-hash, not the line metric
      the filter used -> independent): v10 mean=31.6/p90=81/p99=373/max=10346 -> v11 mean=14.8/p90=32/p99=182/
      max=4013. MEAN MAGNET HALVED (-53%), p90 -60%. Books with >=50x magnet: 1329 -> 577 (-57%).
  (1) DUP-LINE-FRACTION (Gopher std, all lines repeating >=2x): v10 mean .0815 -> v11 .0793 (only -2.7%).
  INTERPRETATION (design working as intended): dup-fraction counts ALL repeats equally incl. protected
  dialogue/scene-breaks/refrains; magnet metric is dominated by HIGH-multiplicity lines (page numbers x100s).
  We surgically removed high-mult furniture (magnet halves) while preserving benign repetition (fraction flat).
FLAGGED-GENRE SCAN (scan_flagged.py, 40 of 496 breaker books, reports/flagged_genres.json): 3 clean genres,
  NO bug. (A) classics/verse/drama in MARKDOWN-TABLE layout (Horace Odes, Sophocles, Measure for Measure,
  Rumi Masnavi, Pulci Morgante |x30072, Iliad, Marquez) - ebook->md wrapped verse in 1-col tables, bare "|"/
  "---|---" scaffolding x1000s. Breaker protected all. CRUCIAL: plays flagged on PIPES not speaker tags ->
  NO drama-tag leak (protects held). (B) ebook nav/pagelist-heavy small books ("##."x90 = ~90-entry PageList
  nav; Berenstain Bears). (C) genuine DATA TABLES (2013 Car Price Guide, cookbook nutrition) - breaker
  correctly refuses to gut a catalog. => breaker is doing its job; conservative (leaves furniture in 496 books).
  v12 IDEA (not a v11 defect): PURE-SYNTAX lines (bare "|","---|---","##." = ZERO alpha words) can NEVER damage
  prose, so EXEMPT them from the breaker (or use content-bearing-line fraction as the breaker denominator) ->
  would clean Morgante's 30k pipe-lines safely. Registered, not built.

## BIBLIOGRAPHY GUARD FIX (sentinel3, 2026-07-16) — Rook #219 highest harm-per-effort, DONE (code)
ROOT CAUSE: a "Further Reading" bibliography (citation lines, alphabetical by author) that runs STRAIGHT
into an alphabetical index with no paragraph break -> mark_index_runs reconstructs them as ONE run; the big
index dilutes the citation ratio (17 cites / 285 lines = 6% < 25% guard) so the whole block (bibliography
included) is marked cut. NOT the island rule (never entered for these).
FIX: (1) added CITE_LINE module regex (italic title | year-in-parens | publisher clause). (2) mark_index_runs
LEADING-CITATION GUARD: even when a run is a genuine index, walk past leading citation lines and start the
cut at the first non-citation entry (the "## Index" header). (3) belt-and-suspenders: island rule won't
sacrifice a citation-heavy keep-island (CITATION_YEAR>=3).
RESULT (harm_per_pass, 100 gold): v9 REAL content destroyed 1,638 -> 681 (-58%), books hit 3 -> 2, worst
957 -> 631, junk-cut UNCHANGED (114,356 -> 114,335 = no regression). Main mechanism eliminated (the original
book now cuts exactly at content_end).
RESIDUAL (2 books, logged not chased — different/deeper mechanisms, rabbit-hole risk pre-ablation):
  - off=13937345547: 631 chars, AUTHOR BIO ("In 2010, Gail and David developed IMAGINE...") classified 'cut'
    by classify_tail_para itself (marketing-phrase precision issue, not structural). FOLLOW-UP.
  - off=105930491637: 50 chars, a sentence fragment at the boundary (line-split artifact). Minor.
NOT YET PROPAGATED TO CORPUS: this is a sentinel3 (boundary) change; the live corpus of record (v11) was built
with the OLD boundary. Baking it in requires regenerating boundary -> cross-book -> within-book (multi-hour,
boundary is the expensive pass). DECISION PENDING (Josef): regen now, or batch with the v12 pure-syntax pass.

## HARM-PER-PASS (harm_per_pass.py, 2026-07-15) — content eaten vs gold, the metric we care most about
Applied each pass's REAL filter to 100 gold tail books (content_end labels), counted chars removed from
the CONTENT region. STRICT = any content-region char; REAL content = prose AND not boilerplate AND not
furniture (the honest "ate learnable text" number). Results:
  de-redaction v1:        0 / 0        (0/100 books)
  boundary v9:        1,955 / 1,638    (3/100)  <- REAL harm: clips BIBLIOGRAPHIES (further-reading lists,
                                                  the hardest reference-vs-backmatter call). 1 book = 957 chars.
  cross-book dedup v10:    608 / 0     (2/100)  <- 608 strict = Random-House copyright lines (boilerplate,
                                                  correctly removed); REAL content destroyed = ZERO.
  within-book dedup v11:   159 / 0     (4/100)  <- 159 strict = page numbers in content region; REAL = ZERO.
  multi-span REJECTED:   2,776 / 429   (6/100)  <- widest blast radius (2x the books of v9) = why it was killed.
KEY: BOTH DEDUP PASSES DESTROY 0 LEARNABLE CONTENT (the receipt). Residual content-risk lives in the BOUNDARY
pass = bibliography clipping (small, isolated, edge-cases). Harm-SPREAD (books hit) not harm-VOLUME is what makes
a pass dangerous (multi-span: fewer chars than v9 but 2x books). NOTE: corrects the "v9 0-cut/60+ gold" claim —
that held on the ORIGINAL gold; the STRATIFIED 80-book edge set surfaces v9's bibliography clip. ACTION (someday):
strengthen sentinel3 bibliography guard for further-reading lists. Table lives in reports/COMPARISON.md (harm axis).

## CORPUS LADDER BYTE DELTAS (exact, for the big board)
orig 107,837,638,842 -> v1 107,837,477,721 (-161KB de-redact, length-preserving) -> v9 107,441,886,827
(-395MB boundary) -> v10 107,394,733,116 (-47MB cross-book dedup) -> v11 107,309,860,661 (-85MB within-book).
Total cleaned ~528MB = ~0.49% of corpus. Volume: boundary biggest (395MB, low harm-density); dedups smaller
(132MB total, HIGH harm-density = the magnets). On-disk v2-v6,v8,v9,v10,v11 present; v7 gone.

## IMMEDIATE NEXT ACTION
v11 SHIPPED as corpus of record (2026-07-15). Copied to ~/brainbox/!BOOKS/book.v11.jsonl (stale v8 still
there = flag for Josef to delete). Recipe = de-redaction -> v9 boundary -> cross-book dedup -> within-book
dedup. THE BIG BOARD is built: reports/bookclean_board.html (published artifact) — corpus ladder + methods
bake-off + harm-proxy + lessons. Update it (v11->corpus of record, v10->superseded) as versions advance.
OPEN THREADS: (1) report v11 to Rook thread (clean methodology facts only, Rook=Fable). (2) v12 idea:
exempt PURE-SYNTAX lines from the within-book breaker (cleans pipe-scaffolded classics safely). (3) clean
NN retrain on para_booksplit.json -> unblocks span-level judge. (4) housekeeping: 616-char gold corrections
into gold; read v10(86)+v11(496) breaker books (genres known: pipe-scaffolded classics / ebook-nav / data tables).
STANDING DEBT (still the real ground truth): toy harm ablation (boilerplate-emission + val loss on pristine held-out). (v10 remains it until then; v11 fully reversible,
ledgered reports/within_dedup_ledger.jsonl.) If yes: recipe = v9 boundary + cross-book dedup + within-book
dedup (all complementary, CPU). Optional confirm: n-gram harm-proxy v10-vs-v11 (harm_proxy_ngram.py; the
naive 10-gram-join version is SLOW ~20min/15k books — rewrite to count only furniture-pattern n-grams if run;
the corpus-wide top-removed list already IS the before/after magnet evidence). Then: copy final to
~/brainbox/!BOOKS/ (Josef usually asks); report to Rook thread.
STANDING DEBT: toy harm ablation = real ground truth AND the value gate for multi-span/judge. Also pending:
feed 616-char gold corrections back to gold; read the v10(86)+v11(496) circuit-breaker books (failure-genre).
