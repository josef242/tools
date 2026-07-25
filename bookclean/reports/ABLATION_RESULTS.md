# Harm Ablation — Results (LOCKED, 3 seeds/arm)
rig-31 training + rig-27 eval · 2026-07-17 · pre-registration: ABLATION_PREREG.md

## Design (as pre-registered)
Does the junk we strip from the books corpus actually harm KEEL training? Two arms, identical
132M recipe (12L×768, minimal recipe, tied→untied head, KEEL+doc-mask+SWA, mtp off), identical
data ORDER (RNG-free loader), only difference = the junk:
- **Arm A = v9** (boundary only) — junk present. **Arm B = v11** (+cross- & within-book dedup) — junk removed.
- 3 seeds each (42/43/44), 6000 steps = ~1.57B tokens = **0.63 epochs** over the 2.51B-token corpus sample.
- Junk eval set = the v9→v11 diff (200k removed spans = free labeled lexicon). Content control = 3.2k
  v11 prose spans (present in both arms). Data made junk-representative via one aligned shuffle (same perm
  both arms) so the sequential loader doesn't train only the clean PG19 head.

## Headline (matched final step 5999, 200k junk)
```
                    junk NLL/tok        content NLL/tok
arm A (v9 dirty)    2.754 ± 0.018       4.022
arm B (v11 clean)   3.228 ± 0.004       4.019
GAP = 0.474 ± 0.018 nats  ->  1.61x  (26.3 sigma)
content control:  A - B = -0.004  ->  NULL  (arms equal on real text)
```
**Junk memorization is decisively real (26 sigma) but MODEST (1.6x), far below the pre-registered 10x bar.
The content control is a perfect null -> the gap is junk-specific, not an A>B artifact.**

## Where the harm lives (arm-independent token-diversity partition, seed-averaged)
```
degenerate <0.2   n=11,641    A 0.96  B 0.96   gap +0.00   1.00x   <- ZERO (low-entropy, B predicts it)
low 0.2-0.4       n=10,581    A 2.57  B 2.51   gap -0.07   0.94x   <- ZERO
mid 0.4-0.7       n=22,393    A 2.79  B 2.90   gap +0.11   1.12x
diverse >0.7      n=155,385   A 2.90  B 3.49   gap +0.60   1.82x   <- the gap
worst 2% by gap:                                            11x    <- cross-book cataloguing boilerplate
   e.g. "Library and Archives Canada Cataloguing...", "A CIP catalogue record for this book..."
```
The genuine memorization harm is **cross-book DUPLICATED publisher boilerplate** — the v10 (cross-book
dedup) target, and exactly what the harm-decomposition flagged as HIGH-harm. **Within-book furniture
(the v11 target) shows ZERO memorization gap** — low-entropy, the clean model reconstructs it from pattern.

## Other metrics
- **Metric #3 grad anomalies: NULL.** A (3 seeds) grad_outliers 229 / loss_spikes 33 vs B 211 / 40;
  nrm_median identical. "Within-book magnets cause loss-spikes" NOT supported at this scale.
- **Metric #4 val loss:** (as pre-registered) not the harm detector; 0.25%-token delta won't move it.
- **Gap TRAJECTORY (matched steps, 10k sample):** 1.20x (1k) -> 1.49x (3k) -> 1.60x (5999). Monotone,
  **still climbing at the end** (slope +0.023 nats/1000) but **decelerating** -> single-pass-limited.

## Verdict vs pre-registered thresholds
- junk-LL >10x on top magnets = harm confirmed -> **NOT met in aggregate (1.6x)**; only worst-2%
  cross-book boilerplate reaches 11x. Emission (#2) not run (naive sampler too slow on Pascal; capacity-axis).
- Real effect, decisively above noise, but modest magnitude. **The 20x cut-asymmetry / "magnets harmful
  everywhere" priors were OVERSTATED at this scale.** Caveat: 130M, 0.63 epoch = a LOWER BOUND (memorization
  grows with exposure + capacity).

## Implications
1. Cross-book dedup (v10) removes the real memorization magnets — value empirically confirmed.
2. Within-book dedup (v11) shows ~no training-HARM benefit at this scale — reframed as corpus hygiene.
3. SKIP Arm C / span judge: if the WORST junk gives 1.6x, the residual is milder -> judge not worth building
   (Arm C was the gate; it reads "don't").
4. Highest-value follow-up = MULTI-EPOCH run (Condition 1): gap still climbing at 0.63 epoch; memorization
   spikes with repeated exposure -> 3-5 epochs is where harm could re-accelerate. Cheap (resume checkpoints).
   Second: larger-capacity KEEL (Condition 2) + metric #2 emission (capacity is where regurgitation bites).

## Artifacts
eval_v9-s{42,43,44}.json, eval_v11-s{42,43,44}.json (full 200k per-junk NLL); traj_{A,B}3000.json +
smoke.json + s1000 (trajectory); junk_lexicon.jsonl (200k); content_control.jsonl; magnet subsets.
Checkpoints on ~/brainbox/checkpoints/current/books-abl-*; local copies ~/data/abl_ckpts/.
