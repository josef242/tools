# Multi-Epoch + Capacity Ablation — Pre-Registration (v2)
**Locked before resuming any checkpoint.** rig-31/rig-27 · 2026-07-17 · design: Code + Rook (#221), Josef.
Supersedes the single-pass pre-reg's threshold, which had a spec bug (Rook): it never defined the
POPULATION. This one does — that's the whole fix.

## AMENDMENT (2026-07-18, Code + Josef) — epoch axis {1,2,3,5} → {1,2,3}
The 5-epoch mark is CUT. Justification: production (wizard101 4.6B) does ~1 epoch over the books data on
the current schedule; if a run is worth extending it might reach 2–3 epochs, essentially never more (Josef).
So the decision-relevant reps range is [1,3] — which we measure DIRECTLY at marks {1,2,3}. On the reps axis
there is therefore NO extrapolation; it is a fully-measured interpolation over the entire production range.
The 5ep point lives in a regime production never visits → confirmatory at best, and costs double (both arms,
matched steps). Freed budget (the 28.8k→48k stretch, ~40% of the planned run, entirely on arm B which hadn't
started) is REALLOCATED to the capacity axis, where the real extrapolation uncertainty lives (4.6B ≈ 35× the
132M toy). Arm A marks 9600/19200/28800 were all already banked before this amendment; arm A training stopped
here. Arm B capped at max_steps=28_800.

## Why (production-regime matching, not curve-watching)
The single-pass ablation ran 0.63 epochs and found a modest 1.61x memorization gap, still climbing at the
end. Production (wizard101 ~4.6B params on this corpus) runs MULTIPLE passes. Memorization scales
~log-linearly in repetitions and increases with capacity — so the decision-relevant number lives at
(4.6B params) x (multi-epoch), where no ablation can run. **The only path is a scaling law: measure
gap(reps, params) at a handful of cheap points, fit, extrapolate.** Output = predicted memorization gap
at wizard101's real scale + diet. That single number sets corpus policy for the real run.

## Populations (DEFINED this time — the bug fix)
Junk is NOT one thing. Three pre-declared, ARM-INDEPENDENT populations, tied to what removed the span:
- **HARM class = CROSS-BOOK duplicated junk** — the v9->v10 diff (what cross-book dedup removed:
  CIP/publisher boilerplate, "…Cataloguing in Publication Data…"). Frequent + slot-uncontested +
  INCOMPRESSIBLE. This is the class the single-pass run convicted at 11x on its worst tail.
- **NULL-CONTROL class = WITHIN-BOOK furniture** — the v10->v11 diff (page numbers, degenerate strings).
  Frequent + uncontested but COMPRESSIBLE. Single-pass gap 1.00x.
- **CONTENT control** — the v11 prose spans (present in both arms). Single-pass gap null.
(Build: split junk_lexicon into cross-book vs within-book subsets via the v9/v10 and v10/v11 record diffs;
secondary cross-check = the token-diversity partition already built.)

## The 3-axis harm model this tests (Rook, for the board)
    JUNK HARM  ∝  FREQUENCY  x  SLOT-UNCONTESTEDNESS  x  INCOMPRESSIBILITY
Pinned by three experiments: <DWnn> natural experiment (fails freq+contestedness -> 0 emission across
271k tokens/years) · this ablation (furniture compressible -> 0 gap; boilerplate incompressible -> 11x) ·
slot-competition arithmetic. The multi-epoch grid tests the FREQUENCY axis's coefficient directly.

## Design
- **Gate (DONE, free): absolute-NLL census.** Arm A aggregate junk PPL ~16 (not verbatim). Spans <1 nat/tok
  = 5.0% (mostly benign compressible furniture — numbered lists, pipes); the harmful incompressible
  boilerplate sits at MODERATE NLL. Verbatim x arbitrary risk is narrow. Recorded; watch these spans across epochs.
- **Epoch axis:** resume s42 (both arms) checkpoints; continue to epoch marks {1, 2, 3}
  (9600 / 19200 / 28800 steps at 262k tok/step over the 2.51B sample; all on the 4800 save boundary).
  PER-CLASS gap at each mark. (5ep mark cut — see AMENDMENT: production ceiling is 3 epochs → measured
  interpolation, no reps-axis extrapolation.)
- **Capacity axis:** 2-3 model sizes (12L/768 have; + a smaller e.g. 6L/512 and a larger e.g. 24L/1024,
  same data/recipe) x the epoch marks. Fit gap(reps, params); both ~log-linear -> usable law from a handful of points.
- **Emission (targeted, deployment-relevant):** skip unconditional sampling (slot competition -> ~0, and
  Pascal can't afford it). PROMPT with junk-context prefixes (feed a copyright-page opening; measure the
  continuation for verbatim boilerplate). A few hundred prompts, cheap on the eval rig. Per-class.

## PRE-REGISTERED PREDICTIONS (population-defined)
- **P1:** HARM class (cross-book) gap grows ~LOG-LINEARLY with repetitions (slope significantly > 0).
- **P2:** NULL class (within-book furniture) gap stays ~1.00x, FLAT across all epoch marks.
- **P3:** CONTENT control stays null (A=B) at every mark and every capacity.
- **P4:** junk-prefix EMISSION of verbatim boilerplate rises with reps for the HARM class, ~0 for furniture.
- **P5:** gap is ~log-linear in BOTH reps and params -> the fit extrapolates.

## DECISION THRESHOLDS (locked)
- **Repetition-scaling CONFIRMED** iff P1 holds (monotone, log-linear slope > 0 across {1,2,3}) AND P2 holds
  (furniture flat). A flat HARM-class curve = the single-pass 1.6x is the ceiling, not a floor -> junk harm
  is genuinely minor even in production regime.
- **The decision number:** gap(4.6B, wizard101-reps) from the fit. If it exceeds [threshold TBD w/ Josef —
  a memorization/emission level that would change corpus policy], the cross-book cleaning is load-bearing at
  scale; if not, v10's value is confirmed-but-small and dup>=2 tightening is unnecessary.
- **CONTINGENT follow-up (not the span judge — that's dead):** if repetition-scaling confirmed on the
  duplicated class, lower the cross-book dup threshold from >=3 toward >=2 — the 2-books-only bucket
  (13.8M lines = 92% of all dup lines) is currently UNTOUCHED and is the proven harm axis.

## Standing corrections carried in (Rook #221)
- Killed claim: "within-book magnets cause gradient anomalies/loss-spikes" (Rook #216) — did NOT replicate.
- The 20x cut-asymmetry is VINDICATED, not indicted: smaller junk-harm -> leaving junk is cheaper ->
  true asymmetry > 20x -> "when in doubt, keep" empirically backed.
