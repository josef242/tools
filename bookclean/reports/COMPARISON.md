# Bookclean — version comparison (plain numbers)
rig-27 · updated 2026-07-15 · corpus of record = **v11**

Each version = the previous one **plus one more pass** (cumulative, not alternatives).
So v11 already contains v9's boundary cuts and v10's cross-book dedup.

## THE UNIFIED CHART — every version × every axis
```
                                            VOLUME                  VALUE            HARM (vs gold)
version  pass added            corpus    cumulative   % of     magnet strength   REAL content   books
                               size      cut          orig     (peak 10-gram)    DESTROYED*      harmed*
--------------------------------------------------------------------------------------------------------
book     raw RedPajama         107.84G   —            —        (baseline)        —              —
v1       + de-redaction        107.84G   ~0           0.00%    —                 0              0/100
v9       + boundary            107.44G   395 MB       0.37%    not targeted **   1,638          3/100
v10      + cross-book dedup    107.39G   442 MB       0.41%    31.6              0              0/100  ✓
v11      + within-book dedup   107.31G   527 MB       0.49%    14.8  (−53%)      0              0/100  ✓
--------------------------------------------------------------------------------------------------------
ref: multi-span cutter (REJECTED — never shipped)                                429            6/100  ✗ killed
```
\*  HARM measured per-pass in isolation on 100 gold tail books (the filter's own content-cut).
   REAL content DESTROYED = prose, not boilerplate, not furniture = learnable text lost.
\** boundary removes bulk VOLUME but not memorization magnets — those are within-book furniture
    (page numbers etc.) that only the dedup passes reach. So the magnet number is flat through v9.

**How to read one row of progress:** cumulative-cut goes UP (more junk gone), magnet strength goes
DOWN (fewer memorization magnets), REAL content destroyed stays at ZERO for the shipped dedup passes.
That's forward motion on all three axes at once. v11 is the current corpus of record.

---

## Volume axis — same metrics across runs
```
                                        cut THIS    cumulative   % of orig   books touched
version  pass added            size     pass        cut          removed     (this pass)
--------------------------------------------------------------------------------------------
book     raw RedPajama         107.84G   —           —            —           —
v1       + de-redaction        107.84G   ~0 *        ~0           0.00 %      ~all (restores text)
v9       + structural boundary 107.44G   395 MB      395 MB       0.37 %      boundary blocks
v10      + cross-book dedup     107.39G   47 MB      442 MB       0.41 %      147,770  (71.8 %)
v11      + within-book dedup    107.31G   85 MB      527 MB       0.49 %       29,449  (14.3 %)
```
\* de-redaction swaps tokens back in (length-preserving), so net size ≈ 0 — it adds value, not deletions.

Cumulative-cut only ever goes **up** → that column is the real "forward progress" line.

## Value axis — why the small passes matter more than the big one
Volume and harm are different things. The boundary pass removes the **most bytes** but the
**least dangerous** junk; the dedup passes remove **fewer bytes** of the **memorization magnets**.

```
                          removed junk       harm-density   memorization-magnet effect
pass                      (bytes)            of that junk   (peak within-book 10-gram)
-----------------------------------------------------------------------------------------
structural boundary (v9)  395 MB  (biggest)  LOW–MOD        ~none — magnets are within-book,
                                                            boundaries can't reach them
cross-book dedup (v10)     47 MB             HIGH           cross-book templates ↓
within-book dedup (v11)    85 MB             HIGH           peak magnet 31.6 → 14.8  (−53 %),
                                                            books w/ ≥50× magnet 1,329 → 577 (−57 %)
```

Takeaway: **v9 wasn't abandoned — it's the foundation.** But bulk-bytes-removed plateaued at v9;
the dedup passes are what actually crushed the magnets, at a fraction of the volume.

## Harm axis — how much LEARNABLE CONTENT each pass wrongly cuts (vs the gold test set)
The thing we care most about. Measured on 100 gold tail books (each hand-labeled with a
content_end boundary), applying each pass's real filter and counting characters removed from
the CONTENT side. `harm_per_pass.py`.
```
pass                          STRICT   REAL content   books   junk cut
                              chars    DESTROYED      hit     (for context)
--------------------------------------------------------------------------------
de-redaction (v1)                 0            0      0/100           0
structural boundary (v9)      1,955        1,638      3/100     114,356
cross-book dedup (v10)          608            0      2/100      15,531
within-book dedup (v11)         159            0      4/100      10,029
multi-span cutter (REJECTED)  2,776          429      6/100     123,413
```
- STRICT = any content-region char removed (includes boilerplate/furniture that merely SITS
  inside the content region — a page number in an afterword, a copyright line before the end).
- REAL content DESTROYED = prose, and NOT publisher boilerplate, and NOT furniture = the honest
  "ate learnable text" number.

Read it like this:
- **Both dedup passes destroy ZERO learnable content.** v10's 608 strict chars are Random-House
  copyright lines; v11's 159 are page numbers — both correctly removed, they just live inside the
  labeled content region. This is the receipt: the recent work is provably content-safe.
- **Boundary (v9) carries the only real residual risk** — 1,638 chars of *bibliographies* clipped
  across 3/100 edge-case books (the hardest tail call: reference-list vs cuttable back-matter).
  Small, isolated, and exactly what the stratified gold set exists to surface.
- **Multi-span (REJECTED) damages the MOST books (6/100)** — widest blast radius, which is why it
  was killed. Note it destroys FEWER total chars than v9 but across TWICE the books: harm-spread,
  not harm-volume, is what makes a pass dangerous.

(Measured per-pass in isolation on the v1 base, so each number is "if this filter alone saw this
gold book." Production order is v1→v9→v10→v11; the numbers are for comparison, not a live audit.)

> Magnet numbers are measured v10→v11 on a 10k-book aligned sample. A full magnet-per-row column
> (v1/v9 too) can be computed if we want the value axis filled for every version — one ~8-min pass.
