# Modern Eval Suite — Design & Implementation Plan

> Replaces the aging HellaSwag/MMLU-letter/WikiText-PPL battery with a 2026-era checkpoint
> evaluation suite: OLMES-style cloze scoring, bits-per-byte loss metrics, tiered fast/full
> runs, and a comparability contract that survives tokenizer changes across runs.

Status: **PLAN** (not yet implemented). Written 2026-07-03 from a research pass over
OLMES / DCLM / OLMo 2+3 / SmolLM2+3 / Signal-and-Noise / DataDecide practice, verified
against this repo's actual capabilities and against a clone of lm-eval-harness v0.4.12.

---

## 0. Pre-existing bug found during review (fix independently of this plan)

Both existing MMLU call sites pad the loss **masks** with `pad_id` instead of 0:
[generate_neo.py:176](../generate_neo.py#L176) and eval.py:192
(`pad_and_stack(all_masks, pad_id)`). With the llama adapter `pad_id == eos_id == 2`
(tokenizer_abstraction.py:133-135), padded positions get mask weight 2 — pad-token losses
leak into the answer NLL with double weight, and the contamination varies with row length
within a batch. (`score_hellaswag_batch` and eval.py's ARC path correctly pad masks with 0.)
All historical letter-MMLU numbers carry this bug — one more reason old and new numbers
never share a plot (§7). Two-token fix if the legacy path runs again before Phase 3.

## 1. Why change anything

The current suite ([eval.py](../eval.py) + [generate_neo.py](../generate_neo.py)) has three
structural problems, independent of which benchmarks are in it:

1. **MMLU is scored in the wrong form for our scale.** We score the answer *letter*
   (`" A"`/`" B"`…) by NLL — the "MCF" (multiple-choice form). MCF sits at chance (25%)
   until a model is quite strong; every open lab (OLMo, SmolLM, DCLM) monitors base
   checkpoints with the *cloze form* (CF: score each full answer string, length-normalized)
   precisely because CF gives smooth signal from early training. Our MMLU curve is likely
   flat-at-chance for most of a run not because the model learns nothing but because the
   formulation can't see it.
2. **Per-token perplexity is incomparable across our own runs.** We compare SuperBPE runs
   against Llama-tokenizer runs; SuperBPE encodes the same text in far fewer tokens, so
   per-token PPL between those runs is meaningless. The field's answer is **bits-per-byte
   (BPB)**: `total_nll_nats / (ln(2) · utf8_bytes)` — same text, same denominator, any
   tokenizer. (The Pile, Paloma, OLMo, and the SuperBPE paper itself all report BPB.)
3. **Our two noisiest metrics are the expensive ones.** GSM8K 8-shot generative accuracy has
   SNR ≈ 1.2 at the 1B scale (AI2 Signal-and-Noise, arXiv:2508.13144); HumanEval pass@1 on
   164 problems has stderr ≈ ±3–4pp anywhere in the 15–85% range. Both have continuous
   replacements — BPB of the *gold* answer/solution — that lift SNR by 5–20× (GSM8K 1.2→7.0,
   MBPP 2.0→41.8) and cost one forward pass instead of a 256-token generation.

Also: WikiText-103 is contaminated in every modern corpus (it's curated Wikipedia) and
single-domain; and we're missing the highest-SNR task family entirely (closed-book knowledge:
TriviaQA/Jeopardy-style gold-answer likelihood).

## 2. What stays

- **The sweep pattern** in `generate_neo.run_hella_sweep` (token milestones from
  `val_log.txt`, resume-by-step, append-only logs) — this generalizes to the whole suite.
- **The batched NLL path** — `pad_and_stack` + `get_batch_loss`/`get_batch_nll`
  ([generate_neo.py:220-330](../generate_neo.py#L220-L330)) using the full non-cached
  forward. Right-padding is safe under causal attention; this is the workhorse for every
  loglikelihood task and it already exists. (Batched *generation* does not exist —
  `stream_generate_kv` is strictly bsz=1 and the cached-attention path has no padding mask
  — and we do not need it; see tiering.)
- **The Windows-safe HumanEval sandbox** in eval.py (subprocess + nulled builtins) — kept as
  the code-exec scorer on Windows; native `code_eval` works on the Linux rigs.
- **The coherence sweep** — orthogonal to this plan, unchanged.
- **eval.py's shape** (one checkpoint → JSON results) survives, but its internals are rebuilt
  around a task registry and a new engine.

## 3. Engine decision: lm-eval-harness + thin adapter (+ two native modules)

**Adopt EleutherAI lm-evaluation-harness, pinned at v0.4.12, driven programmatically via a
custom `TemplateLM` adapter.** Rationale:

- From v0.4.10 onward the base package has **no torch/transformers dependency** (verified on
  PyPI); it installs cleanly into `trainenv` (torch 2.5.1 untouched, datasets 4.5.0 ≥ its
  `datasets>=2.16` floor).
- Hand-rolled scoring keeps costing us correctness: no length normalization on MMLU/ARC
  (acc_norm is a multi-point mover), letter-MCF-only MMLU, the §0 mask bug, bespoke
  answer-extraction regexes for GSM8K. The harness's formulations are the reference
  implementations papers cite.
- The adapter is small because the harness asks for three primitives, and we already have
  the hard one (batched NLL). Estimated ~250–350 lines.
- `use_cache` (SQLite, per checkpoint — the cache key ignores model identity, so
  per-checkpoint DBs are mandatory, not just tidy) gives request-level resume;
  `samples=` (fixed index lists, also a `simple_evaluate` kwarg) gives frozen subsets.

**Two native modules stay outside the harness:**

1. **BPB scorer** (new, §5) — per-domain bits-per-byte over frozen raw-text panels, **plus
   the gold-continuation BPB tasks** (GSM8K-BPB, MBPP/HumanEval-BPB, TriviaQA gold-answer
   NLL). Verified: none of these exist as built-in lm-eval tasks in v0.4.12, and a plain
   task YAML *cannot* express "loglikelihood of gold continuation normalized by bytes"
   (for `output_type: loglikelihood` the harness supports only `perplexity` and `acc`;
   `bits_per_byte` is wired only to `loglikelihood_rolling`, which can't condition on a
   few-shot context). Custom task packages with `process_results: !function` hooks are the
   lm-eval-native alternative; we go native instead so one frozen boundary-attribution rule
   (§5) covers every continuation-NLL metric.
2. **Code-execution scoring** — harness's `code_eval` metric is Unix-only
   (`signal.SIGALRM`/`resource`, plus an explicit `os.name == "nt"` NotImplementedError).
   Worse: the stock humaneval/mbpp tasks fail **at task-load time** on Windows (their
   utils.py runs `evaluate.load('code_eval')` at import). Windows path: custom copies of the
   task YAMLs (same prompts/stops, pass-through metric) under `eval_artifacts/tasks/`,
   harvest completions via `log_samples`, score with eval.py's sandbox. Linux rigs: native
   path with `HF_ALLOW_CODE_EVAL=1` + `--confirm_run_unsafe_code`.

Rejected alternatives: **ai2-olmes** (HF-models-only; pins torch≥2.8 / datasets<4 /
lm_eval==0.4.3 — all conflict with trainenv; verified in its pyproject), **lighteval**
(custom models supported but mid-architectural-pivot two releases running). If we want exact
OLMES formulations later, we port their YAMLs as custom tasks via
`TaskManager(include_path=…)` rather than adopting the package.

### 3.1 The adapter (`keel_lm.py`)

Subclass `lm_eval.api.model.TemplateLM`:

| method | implementation |
|---|---|
| `tok_encode` / `tok_decode` | delegate to `BaseTokenizer.encode`/`.decode`; `tok_encode` must accept the harness's `add_special_tokens=None/bool` kwarg and map it onto the frozen BOS policy (never silently ignore it) |
| `eot_token_id` | `tokenizer.eos_id` |
| `prefix_token_id` | `tokenizer.bos_id` (≠ eos_id for llama; training packs docs with BOS, and the model's doc-position logic keys on bos_id — an EOS prefix is off-distribution). Subject to the Phase-0 A/B, then frozen |
| `_loglikelihood_tokens` | sort-by-length, chunk, `pad_and_stack` → full non-cached `model(tokens)` → gather log-softmax over continuation span; also `is_greedy` (argmax match — LAMBADA needs it) |
| `loglikelihood_rolling` | reuse `lm_eval.utils.get_rolling_token_windows` (it prepends `prefix_token_id` itself — do **not** also add BOS in tok_encode) |
| `generate_until` | wrap `nc.stream_generate_kv` (bsz=1, greedy) with stop-string truncation |
| `max_length` | **pinned suite constant 2048** (fleet minimum), *not* per-checkpoint `max_seq_len` — see §7 |

BOS policy is implemented at exactly **one layer**: inside `_loglikelihood_tokens` (after
`_encode_pair`) and in `generate_until` — never inside `tok_encode`, or rolling windows get
double-prefixed. Note `stream_generate_kv` hardcodes `bos=True`
([neo_common.py:739](../../common_fsdp2/neo_common.py#L739)); either add a `bos=` kwarg or
document in policy.md that generation is always BOS-prefixed and the A/B governs
loglikelihood only.

Adapter-specific hazards, handled in Phase 0/1:

- **SuperBPE boundary splitting.** `TemplateLM._encode_pair` encodes `context+continuation`
  as one string and splits at `len(tok_encode(context))`. SuperBPE superword tokens merge
  *across* the context/continuation whitespace boundary far more often than plain BPE,
  mis-attributing boundary tokens. This biases every continuation-NLL metric — worst for
  LAMBADA (single-word continuation) and the gold-BPB tasks. Fix: one frozen attribution
  rule in policy.md (a token whose byte span begins inside the continuation belongs wholly
  to it; denominator = exact continuation bytes), unit-tested on both tokenizers with
  boundary-merge cases including LAMBADA and GSM8K-answer shapes. (Byte-offset splits change
  conditioning, not just attribution — the rule must say which tokenization is scored.)
- **torch.compile.** `load_model_and_tokenizer` unconditionally calls
  `torch.compile(model, mode="reduce-overhead", dynamic=True)` on single-GPU CUDA loads —
  currently a no-op only because the handle isn't reassigned (neo_common.py:530). There is
  no off switch today. Phase 1 adds a `compile=` parameter and eval callers pass
  `compile=False`, rather than relying on the accident staying broken.
- **Import path.** keel_lm.py uses a file-relative sys.path insertion
  (`os.path.dirname(__file__)`), not the CWD-relative `'../common_fsdp2'` literal the other
  tools use — lm-eval machinery may import it from any working directory.

## 4. The suite (tiered)

Design rules synthesized from the research:

- **CF (cloze) char-normalized scoring for all multiple-choice tasks.** lm-eval's `acc_norm`
  is the same logprob-per-character family OLMES uses for its CF scores (lm-eval excludes
  the leading target-delimiter space from the char count; OLMES includes it — close but not
  identical, so our numbers are OLMES-*style*, not OLMES-comparable until their YAMLs are
  ported). OLMES also uses pmi normalization for ARC-C/CSQA/OBQA — deferred (§9).
- **Letter-MCF MMLU is kept but demoted** to a separate "emergence" curve in Tier 2 — never
  averaged into the headline number, and a flat-at-chance line doesn't need dense sampling.
- **Fixed everything**: fixed sample indices (`samples=` JSON committed to repo, with HF
  dataset *revision* pins and per-item content hashes — indices alone are meaningless if the
  Hub dataset is re-uploaded), fixed curated few-shot exemplars (per-subject for MMLU),
  fixed seed, pinned lm_eval version, pinned n per task. A subsample compared across runs on
  identical items is a paired comparison — sampling error mostly cancels.
- **Sample size is part of a series' identity.** `eval_log.jsonl` records carry `n`, and the
  dashboard groups by `(task, metric, n)` — an n=200 GSM8K point and a full-1,319 point are
  different series, never one curve. Each task's n is frozen in Phase 0; changing it bumps
  `suite_version`.
- **Headline aggregate**: DCLM-style centered accuracy `(acc − chance)/(1 − chance)` averaged
  over Tier-1 MC tasks, with **per-task chance** = mean(1/k_i) over the frozen sample (ARC
  has 3–5 options; a flat 25% mis-centers it), plus a separate BPB composite. Never mix.

### Tier 0 — loss fit (every checkpoint; ~minutes; fully offline; native BPB module)

| eval | metric | notes |
|---|---|---|
| Own-corpus domain panels ×7 | **BPB** per domain + macro-avg | the 7 dn4 training-mix groups (of 17 shard dirs present): ao3, books, stories, preselect, code_python, code_c, **edufineweb_1.5TT** — exact dir names pinned in the panel MANIFEST; panel set is fixed globally across runs regardless of each run's mix (comparability first) |
| External slices ×3 (FineWeb-Edu val, C4-en val, a code slice) | BPB | catches data-mix confounds when comparing runs (in-distribution val loss flatters whichever run matches its own mix) |
| WikiText-103 | BPB (legacy column) | via the same native module, same windowing; labeled contaminated; token-PPL retired from cross-run use |

### Tier 1 — capability tracking (every sweep checkpoint; budget ≤ ~45 min @ 4.6B, sized by Phase 0 benchmark)

All loglikelihood — no generation. Engine noted per row.

| task | engine | form / metric | n | shots | why |
|---|---|---|---|---|---|
| HellaSwag | lm-eval | CF, acc_norm | **full 10,042** | 0 | lowest step-noise curve; current sweep already runs full val — a 2k cut would be a noise regression (±0.50→±1.12pp) |
| ARC-Easy | lm-eval | CF, acc_norm | full 2,376 | 5 fixed | among the best small-scale discriminators (DataDecide) |
| ARC-Challenge | lm-eval | CF, acc_norm | full 1,172 | 5 fixed | high signal; small n → don't over-read 1pt moves |
| MMLU-CF (`mmlu_continuation`) | lm-eval | CF, acc_norm, **macro over subjects computed by our runner** | fixed ~2k sample (or the 16 high-SNR subjects) | 5 fixed, per-subject | the informative MMLU at our scale. Built-in group aggregate is size-weighted micro-`acc` — wrong on both axes; we macro-average per-subject acc_norm ourselves. `samples=` file needs all 57 leaf-task entries |
| LAMBADA (openai) | lm-eval | last-word acc + logprob | full 5,153 | 0 | free smooth LM-quality curve; exercises is_greedy |
| GSM8K-BPB | **native** | BPB of gold 8-shot CoT answer | full 1,319 | 8 fixed (shared prefix) | continuous math signal (SNR 1.2 → 7.0 at 1B scale) |
| MBPP-BPB / HumanEval-BPB | **native** | BPB of canonical solution | full 500 / 164 | 3 / **0** (match Tier-2 prompt forms) | code signal visible long before pass@1 leaves zero |
| TriviaQA gold-answer NLL | **native** | BPB of gold answer (canonical alias frozen in artifact) | fixed 1k sample | 0–5 | highest-SNR family (knowledge); OLMo 2's `*_ppl` pattern |

### Tier 2 — milestones (every Nth checkpoint / stage boundaries / run-vs-run decisions)

| task | metric | n | notes |
|---|---|---|---|
| MMLU-MCF (letter) | separate emergence curve | same fixed 2k sample | continuity with historical numbers (which carry the §0 mask bug — re-baselined, not spliced); watch for liftoff above 25% |
| GSM8K 8-shot CoT generative (`gsm8k_cot`) | strict + flexible EM | fixed 200 sample; full 1,319 is a **separate series** at end-of-run | ±3.5pp stderr at n=200 → run-vs-run claims use paired per-item deltas from archived samples; bsz=1 generation → rigs preferred |
| HumanEval pass@1, MBPP pass@1 | exec-based | full | Windows: custom pass-through-metric task copies + our sandbox (§3); Linux: native `code_eval` |
| OLMES core-9 completion: PIQA, CSQA, OpenBookQA, BoolQ, SocialIQA, WinoGrande | CF, acc_norm | full | cheap but noisy/late movers — tracked, not weighted in headline. **SocialIQA is known-broken under datasets 4.x** (script-only Hub repo) — needs a parquet-mirror task override or gets dropped |
| TriviaQA (generative, closed-book) | EM/F1 | fixed 1k | the quotable knowledge number |

### Tier 3 — release / cross-run comparison tables (end of run; never for recipe decisions)

Full OLMES protocol (CF **and** MCF, report max) on core-9 + MMLU for apples-to-apples
comparison with published OLMo/SmolLM tables; plus a held-out set never used during training
decisions: **MMLU-Pro** (10-way MC — expect ~chance until ~7B-class; gate on observed
liftoff), **AGIEval-English**, **GSM-Plus** (report the GSM8K−GSM-Plus gap as a memorization
check), **BBH** (only ≥7B), **NaturalQuestions**.

**Skip entirely at our scale (base models):** GPQA, MuSR, MATH/MATH-500 (unless a math-heavy
mix), IFEval (SFT-only — revisit for chat checkpoints), ARC-AGI, HLE.

## 5. Native BPB module (`bpb_eval.py`)

Two modes sharing one scoring core and one frozen boundary-attribution rule:

**(a) Document BPB** (Tier 0). The tokenized val shards are llama-tokenized uint16 streams —
unusable for cross-tokenizer BPB directly. One-time extraction: decode a fixed sample of
documents per domain (or re-sample from raw source) → **frozen raw-text panels**, ~1–2M
tokens' worth per domain, JSONL `{doc_id, text}` + SHA-256 manifest (panels live next to the
datasets, manifest in git).

**(b) Gold-continuation BPB** (Tier-1 native rows). Frozen `(context, continuation)` panels
built once from HF datasets: GSM8K gold CoT answers behind the fixed 8-shot prefix, MBPP
(3-shot) / HumanEval (0-shot) canonical solutions, TriviaQA canonical-alias answers. Score =
NLL of continuation / UTF-8 bytes of continuation, under the frozen attribution rule (§3.1).

Scoring protocol, both modes:

- Per-document, BOS-prefixed, **windows pinned to the suite constant (2048 tokens)** — *not*
  the checkpoint's max_seq_len. The early-token bias only cancels if window size is
  identical; our fleet mixes 2048/8192/12288 contexts and a floating window would hand
  long-context checkpoints ~0.01–0.1 BPB for free on long-doc domains (books/ao3/code) —
  the same order as the deltas we're trying to detect. Long docs additionally split at fixed
  **byte offsets (~6 KB)** into independently scored chunks so window boundaries land at the
  same text positions regardless of tokenizer (a token-count window spans ~40% more bytes
  under SuperBPE — a residual SuperBPE-favoring bias otherwise). 6 KB keeps llama
  tokenizations safely under 2048 tokens. `window_tokens` and `chunk_bytes` recorded in the
  MANIFEST and in every BPB record; changing either bumps `suite_version`. An optional
  clearly-labeled native-context diagnostic column may be emitted alongside, never in its
  place.
- Batched by length-bucket through the non-cached forward; bf16, `torch.no_grad()`.
- Per-doc records `{doc_id, nll_nats, n_tokens, n_bytes}` appended to JSONL → free resume
  (skip scored doc_ids) and micro/macro re-aggregation without re-running.
- BOS excluded from both scored NLL and byte count; bytes = UTF-8 bytes of exactly the
  scored text.

Runs standalone (no lm-eval dependency) — also the zero-network smoke eval for any box.
Optional later: a Paloma subset (gated HF dataset, ImpACT license).

## 6. Outputs & dashboard integration

The dashboard ([dashboard.py](../../dashboard/dashboard.py)) does **not** read eval.py's
`*_results.json` — it parses `val_log.txt`, `gen_log.txt`, `hellaswag_log.txt` (CSV
`step, tokens, score`), and `coherence_log.jsonl`. Contract for the new suite:

- **`eval_log.jsonl`** written next to checkpoints (same dir as `val_log.txt`), one record
  per (checkpoint, task, metric): `{step, tokens, suite_version, tier, task, metric, value,
  stderr, n}`. Append-only; the sweep driver resumes by `(step, task, suite_version)`.
- **New `EvalLogParser`** in dashboard.py (mirroring `CoherenceParser`), grouping series by
  `(task, metric, n)` so subsample and full runs never merge into one curve.
  `hellaswag_log.txt` stays for continuity.
- Full lm-eval results JSON + per-sample logs (`log_samples=True`) archived per checkpoint
  under an `evals/` subdir — required for paired per-item run-vs-run deltas (paired analysis
  cuts required n by 1/(1−ρ) — ≥2× for related runs, typically much more; Miller, "Adding
  Error Bars to Evals", arXiv:2411.00640).

## 7. Comparability contract (frozen, versioned artifacts)

Everything that can silently move a score becomes a repo artifact, referenced by
`suite_version` (e.g. `suite_v1`):

| artifact | content |
|---|---|
| `eval_artifacts/samples/*.json` | fixed instance-index lists per task (lm-eval `samples=` format; 57 leaf entries for MMLU), generated once with seed 1234, **plus HF repo + revision pins and per-item SHA-256 content hashes**, verified at load, hard-fail on mismatch |
| `eval_artifacts/fewshot/*.yaml` | curated fixed few-shot exemplars (label-balanced; per-subject for MMLU; OLMES-published shots where usable) |
| `eval_artifacts/tasks/` | custom task overrides: Windows humaneval/mbpp pass-through-metric copies, SocialIQA parquet-mirror override, any OLMES ports |
| `eval_artifacts/bpb_panels/MANIFEST.json` | SHA-256 + doc/byte counts + `window_tokens`/`chunk_bytes` of every BPB panel |
| `eval_artifacts/policy.md` | BOS policy (Phase-0 A/B, then frozen; insertion-point rule), boundary-attribution rule, **fleet-wide `max_length=2048` pin**, truncation policy, byte-counting rules, lm_eval version pin |

Rules: pin `lm_eval==0.4.12` for the life of the comparison series; any artifact change
bumps `suite_version`; numbers from different suite_versions never share a plot; old
hand-rolled numbers never share a plot with new numbers (systematic offsets guaranteed —
formulation differences plus the §0 mask bug — re-baseline instead, Phase 6).

**Context length is pinned fleet-wide, not per-checkpoint.** With per-checkpoint
`max_length`, only 2048-ctx runs (dn4/kv3) would get silently left-truncated on long items
(MMLU professional_law 5-shot prompts routinely exceed 2048 llama tokens even with short
shots) — a run-vs-run comparability hole. Pinning `max_length=2048` everywhere makes
truncation identical for all checkpoints. Phase 0 audits the **max prompt length over all
items × both tokenizers** per task (not just shot length) and trims per-subject shots until
long-item truncation is rare and documented.

## 8. Implementation phases

**Phase 0 — install, artifacts, measurements, policies (~1 day + short GPU runs)**
1. `pip install lm_eval==0.4.12` in trainenv; smoke-test each chosen task with `--limit 8`
   (datasets 4.5.0 dropped script-datasets; verified migrated: piqa→baber/piqa,
   hellaswag→Rowan/hellaswag, winogrande→allenai/winogrande, boolq→aps/super_glue,
   csqa→tau/commonsense_qa, wikitext→EleutherAI/wikitext_document_level; **known broken:
   social_iqa** → §4 Tier-2 override). First run online to populate the HF cache.
2. Build `eval_artifacts/`: generate `samples=` index JSONs (seed 1234) with revision pins +
   content hashes; curate per-subject few-shot YAMLs; policy.md skeleton; custom task
   overrides (Windows code tasks, SocialIQA).
3. Throughput benchmark: batched non-cached forward tok/s for a dn4 checkpoint (bf16,
   seq 2048, B ∈ {4, 8, 16}) on the 4080 → derive per-task minutes → confirm Tier-1 fits the
   ≤45 min budget (analytic estimate: ~24M forward tokens ≈ ~40 min at ~10k tok/s; full
   5-shot MMLU-CF alone would be 1.5–2h — hence the 2k subsample).
4. BOS A/B on one checkpoint (HellaSwag ± BOS-prefixed contexts); freeze in policy.md.
5. SuperBPE `_encode_pair` boundary unit tests (both tokenizers; whitespace-merge, LAMBADA
   single-word, GSM8K-answer cases); freeze the attribution rule.
6. Context-fit audit per §7 (max prompt length over all items × both tokenizers).
7. Contamination audit: the corpus contains `mid_gsm8k` and `mid_mmlu` midtraining shards —
   verify test splits were excluded before any GSM8K/MMLU delta is used for run-vs-run
   claims. The held-out Tier-3 set hedges this regardless.

**Phase 1 — `keel_lm.py` adapter (~1–2 days)**
`TemplateLM` subclass per §3.1 (file-relative imports). Includes the `compile=` parameter on
`load_model_and_tokenizer` and (optionally) a `bos=` kwarg on `stream_generate_kv`.
**Parity gate at the primitive level**: identical (context, continuation) token pairs
through the adapter's `_loglikelihood_tokens` and through `get_batch_nll` must agree
near-exactly — same math, same model, no formulation differences. (A task-level HellaSwag
comparison is a sanity check only: compare lm-eval's *unnormalized* `acc` against
`score_hellaswag_batch` with BOS matched, expecting a small offset from lm-eval's
activity_label prefix and detokenizer cleanup — acc_norm has no legacy counterpart.)
Loglikelihood-only support for MoE/GDN checkpoints initially (the non-cached forward is
architecture-agnostic); generation on GDN stays excluded until validated.

**Phase 2 — native BPB module (~1.5–2 days)**
Panel extraction + both scoring modes per §5, including the gold-continuation panels
(GSM8K/MBPP/HumanEval/TriviaQA), external slices, and the WikiText-BPB legacy column.

**Phase 3 — suite runner / eval.py rework (~1–2 days)**
Replace the if/elif ladder ([eval.py:663-683](../eval.py#L663-L683)) with a registry:
`{name: TaskSpec(engine=lm_eval|bpb|code_exec, tier, task_args)}`. Tier presets
(`--tier 0|1|2|3` plus `--test` à-la-carte). Per-checkpoint flow: load model once → run
tiers → write `eval_log.jsonl` + archive full results + per-sample logs. Suite-runner-side
MMLU macro-acc_norm aggregation. `use_cache` SQLite per checkpoint. Fix the §0 mask bug in
passing. Keep `TestResult` JSON as a secondary artifact.

**Phase 4 — sweep driver (~half a day)**
Generalize `run_hella_sweep` → `run_eval_sweep` (same val_log.txt token-milestone logic,
resume by `(step, task, suite_version)` from `eval_log.jsonl`, `--tier` selection). The
hella/coherence sweeps remain but Tier-1 supersedes hella_sweep for new runs.

**Phase 5 — rig bring-up (~half a day + rig time)**
Sharded-load path in keel_lm (`shard_strategy` models: device-map-aware input placement,
as `score_hellaswag_batch` already does); documented HF cache sync to the offline rigs
(`HF_HOME` rsync + `HF_DATASETS_OFFLINE=1`); one Tier-0+1 validation run on a rig. 7B-class
checkpoints (~7.4B for 7B-MUON) and all heavy/generative tiers run here.

**Phase 6 — dashboard parser (~half a day)**
`EvalLogParser` in dashboard.py grouping by `(task, metric, n)`; default plotted curves:
BPB macro, centered-acc aggregate, HellaSwag, ARC-E, MMLU-CF, GSM8K-BPB, MMLU-MCF emergence.

**Phase 7 — re-baseline & rollout (~half a day + GPU time)**
Run Tier 0+1 on 2–3 reference checkpoints of **every active run — including MoE/GDN runs
(loglikelihood tiers only)** — to seed the new curves. Old numbers stay on old plots.
Retire letter-MMLU as a headline metric.

Total: roughly **6–8 working days** plus Phase-0 GPU time; incremental and useful from
Phase 2 onward (the BPB module alone already fixes the cross-tokenizer problem).

## 9. Risks / open questions

- **Runtime estimates are analytic until Phase 0 measures them.** The 101-layer deep-narrow
  dn4 topology (no flash-attn in the eval path) makes per-token cost worse than parameter
  count suggests. Tier-1 membership flexes to the measured budget.
- **7B-class checkpoints don't fit the 16GB 4080** without sharding/offload — they eval on
  the rigs (Phase 5).
- **Generative tasks are bsz=1.** Accepted: confined to Tier 2 milestones. Left-padded
  masked batch-decode is real model-side work and explicitly out of scope.
- **SFT/chat checkpoint evals** (IFEval etc.) deferred; needs chat-template hooks the
  tokenizer abstraction doesn't expose yet.
- **OLMES pmi normalization** (ARC-C/CSQA/OBQA) differs from lm-eval's `acc_mutual_info`
  (unconditional context `"Answer:"` vs empty). v1 uses acc_norm everywhere; exact OLMES
  ports are a later nice-to-have and bump suite_version.
- **Panel choice is global, not per-run-mix.** Runs whose training mix differs from the dn4
  7 (e.g. climb-heavy configs) still eval on the same panels — that's the point
  (comparability), but expect mix-mismatched runs to look worse on out-of-mix domains;
  read the per-domain breakdown, not just the macro.
