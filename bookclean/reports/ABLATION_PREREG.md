# Harm Ablation — Pre-Registration
**Locked before any training run.** rig-27 · drafted 2026-07-16 · design: Code + Rook (#219), reviewed by Josef.

The whole cleaning project encodes one untested belief: that the junk we remove actually harms KEEL
training. Every shipped number (the 20:1 asymmetry, "memorization magnets are harmful," v11's −53% magnet
win) *assumes* this. This experiment tests the assumption directly, and — via Arm 3 — decides whether the
next pass (span judge / v12) is worth building. Pre-registering the predictions and decision thresholds is
what converts the result from "suggestive" to "decisive": we cannot move the goalposts after seeing data.

---

## Arms (3)
All arms train the **same architecture, same recipe, same sampling order** — the *only* difference is which
text the tokens come from.

| arm | corpus | what it isolates |
|-----|--------|------------------|
| **A — v9**  | boundary-only | baseline: junk still present (cross-book + within-book furniture) |
| **B — v11** | boundary + cross-book + within-book dedup | our shipped cleaning; the removed junk is absent |
| **C — v11 + spike-in** | v11 with gold-labeled **residual-category** junk injected at ~10× natural concentration | gates the span judge: does the junk we *didn't* remove matter? |

**Why Arm C is the one that matters for what's next:** A-vs-B measures the harm of *removed* junk (the
high-duplication magnets — the worst category). But the span-judge / v12 decision is about *residual* junk
(one-off data structures, fluent previews), which sits in **both** A and B and cancels out of that
comparison. A positive A-vs-B result says nothing about the residual. Arm C overdoses the residual category
so a null becomes decisive: **if even 10× residual junk produces no measurable signal, the span judge is
dead with confidence** and the residual stays in the corpus, unmourned.

---

## Recipe (config risk held out of the comparison)
- **Model:** reuse the **dirty-paws Gate-1 toy** (~25M KEEL, known-good config). Reusing a validated recipe
  removes config risk — any arm difference is the cleaning delta, not a training artifact.  *(TO CONFIRM w/ Josef: recipe path / launch command.)*
- **Tokens:** ~1–2B per run.
- **Sampling order:** IDENTICAL across arms — same shuffle seed over **record-aligned** data, so arm B is
  arm A with exactly the junk spans removed and nothing else reordered. Arm C = arm B's stream with the
  spike-in records interleaved at the same positions each seed.
- **Seeds:** 3 per arm (2 if compute-tight). Report mean ± spread; "inside seed-noise" is a first-class outcome.
- **Compute:** 2× GTX 1070 available. Per Rook, the **clean NN retrain runs CONCURRENTLY** on the second GPU
  if free (different resource class; it gates the judge's content-detection, not this ablation). Serialize only if forced.

---

## Metrics (descending statistical power — val loss is NOT the harm detector)
**Structural fact:** v9→v11 removed ~120M chars from a ~27B-token corpus = **0.1% of the data**. A 0.1%
delta will not move val loss at toy scale. Val loss is the *sanity check* (cleaning didn't hurt), not the
harm signal. The harms actually claimed — emission contamination, memorization skew, gradient anomalies —
get measured directly:

1. **Junk log-likelihood ratio** *(most powerful, zero sampling noise).*
   Teacher-force the removed-junk sequences through arm A vs arm B models; compare mean assigned log-prob.
   **Control:** matched-*frequency* content sequences (so the ratio reflects junk-ness, not rarity).
   **The eval set is free:** the v9→v11 diff *is* a perfectly labeled junk lexicon by construction —
   ~120M chars of positives, zero annotation. (Build from the boundary-removed text + the dedup ledgers
   `reports/dedup_ledger.jsonl` + `reports/within_dedup_ledger.jsonl`.) The reversible ledger pays off again.

2. **Emission rate.** Fraction of generations containing removed-junk 10-grams — both **unconditional** and
   **prompted from pristine-book prefixes**. Directly measures regurgitation contamination.

3. **Loss-spike / grad-norm anomaly count** during training. Free telemetry; directly tests the "magnets
   cause gradient anomalies" claim we've all been carrying. Log per-step grad-norm; count outliers (>Nσ).

4. **Val loss** on pristine held-out books — held out from **both** corpora, **edition-screened** for
   near-dups (a held-out book whose sibling edition is in training invalidates the number — same lesson as
   the gold sets). The sanity check, not the verdict.  *(TO CONFIRM: held-out set + edition-screen source.)*

---

## PRE-REGISTERED PREDICTIONS (write them down before the data)
- **P1 (val loss):** indistinguishable between arms A/B/C — inside seed-noise. *(If val loss DID move on a
  0.1% delta, something is wrong, not right.)*
- **P2 (junk-LL, A vs B):** arm A assigns **higher** log-prob to removed-junk sequences than arm B
  (A memorized the junk it saw). Directional; magnitude is the test.
- **P3 (emission, A vs B):** arm A emits removed-junk 10-grams at higher rate than arm B.
- **P4 (grad anomalies, A vs B):** arm A shows **more** loss-spikes / grad-norm outliers than arm B.
- **P5 (Arm C):** *no registered direction* — this is the open question. C-vs-B tests whether overdosed
  residual junk produces any of the above signals. This is what we genuinely do not know.

## PRE-REGISTERED DECISION THRESHOLDS (locked)
- **Harm of removed junk CONFIRMED** if: junk-LL ratio **>10×** on the top magnets (A over B) **OR** emission
  delta **>5×** (A over B), outside seed-noise.
- **Null** if the A-vs-B differences sit **inside seed-noise** on all of metrics 1–3. (Optimistic-bias note:
  the toy scale biases toward null, so a *confirmed* result is strong; a null is "not detectable at this scale,"
  not "provably zero.")
- **SPAN JUDGE / v12 GATE (Arm C):** if C shows **no** signal above B on metrics 1–3 (overdosed residual junk
  is indistinguishable from clean), the span judge is **not worth building** — residual stays in the corpus.
  If C *does* show signal, the residual is real and the judge is back on the table, sized by the effect.

## Outputs
- Results table (arms × metrics, mean ± seed-spread) added to the big board and COMPARISON.md.
- The board/COMPARISON get these predictions + thresholds **now**, before runs, timestamped — the pre-registration.
- Verdict lines: "harm of removed junk: confirmed/null," "span judge: build/dead."

## Open dependencies (need before launch)
1. dirty-paws Gate-1 recipe path + launch command (Josef).
2. Pristine held-out set + edition-screening source (Josef / reuse gold near-dup screen).
3. Confirm 2nd GPU free for concurrent clean NN retrain, else serialize.
4. Build the junk lexicon + matched-frequency content control (Code — from ledgers + boundary diff; can start now).
