# tools/ — Utility & Probe Scripts

This directory holds the research probes, inference/chat tools, data-pipeline scripts, and
coherence metrics built for the `mara_fsdp2` / KEEL-NorMuon training work. The probes drive the
body-norm-ramp / WD-waste / Newton-Schulz / z-loss investigations; the rest are general-purpose
inference, eval, and data-prep utilities. Each tool has its own doc here — this page is just the
landing index. (Anything that touches generated or dataset text follows the black-box rule: no
sample text appears in these docs.)

## Training-research probes (KEEL / NorMuon)

These probes tell the body-norm-ramp / WD-waste investigation roughly in sequence — from cheap
log parses, through forward-only and gradient probes, to the Newton-Schulz radial-mechanism work
and the WD "taming dial". Canonical writeup: `mara_fsdp2/docs/WD_WASTE_ANALYSIS.md`.

| tool | what it does | doc |
|------|--------------|-----|
| `wd_waste_probe.py` | Part A: pure log-parse of `diagnostics.jsonl` estimating each matrix's wasted decoupled-WD gradient share vs. real loss signal, and whether `\|\|W\|\|` is still ramping or at equilibrium. | [`wd_waste_probe.md`](wd_waste_probe.md) |
| `wd_waste_partb.py` | Part B: per-matrix `cos(g_loss, W)` on CE-only gradients to test whether the body-norm ramp is loss-null (WD wasted-but-harmless) or loss-coupled (WD fighting real signal). | [`wd_waste_partb.md`](wd_waste_partb.md) |
| `keel_radial_probe.py` | Read-only checkpoint probe for *why* pure CE yields an anti-radial body gradient — train-vs-eval gradient, branch-gain derivative, ε-sensitivity, and a bf16/fp32-accum discriminator. | [`keel_radial_probe.md`](keel_radial_probe.md) |
| `ns_spectral_audit.py` | Audits the singular-value spectrum of the real CE gradient on body matrices to prove Newton-Schulz turns a radial-null gradient into an anti-radial update. | [`ns_spectral_audit.md`](ns_spectral_audit.md) |
| `ns_radial_isolation.py` | FSDP-free probe that runs a radial-null gradient through the real Muon stages (Newton-Schulz, scaling, NorMuon) to pinpoint which stage injects the anti-radial lean. | [`ns_radial_isolation.md`](ns_radial_isolation.md) |
| `normuon_update_decomp.py` | Decomposes the actual NorMuon step per body matrix and compares `cos(update,W)` vs `cos(grad,W)` to test whether the normalized update injects the radial drift driving the ramp. | [`normuon_update_decomp.md`](normuon_update_decomp.md) |
| `finite_rescale_probe.py` | Forward-only probe testing whether scaling a whole weight class by a finite factor `c` leaves CE unchanged, mapping the "safe range" for body renorm. | [`finite_rescale_probe.md`](finite_rescale_probe.md) |
| `wd_equilibrium.py` | Computes the equilibrium body-weight norm each WD strength would produce from a measured per-step update magnitude, sizing the body-only WD "taming dial". | [`wd_equilibrium.md`](wd_equilibrium.md) |
| `gpm.py` | Measures whether above-trend grad-norm spikes predict above-trend loss improvement on the next step, via detrended Spearman correlation over a window (gradient-productivity metric). | [`gpm.md`](gpm.md) |
| `gpm_retrofit_inject.py` | Edits a historical `gen_log.txt` in place to append the live-trainer `\| gpm: +S/+L` field to each step line, computed identically to the in-trainer `GPMTracker`. | [`gpm_retrofit_inject.md`](gpm_retrofit_inject.md) |
| `zloss_row_center_probe.py` | Forward-only probe that row-centers a checkpoint's output head to measure how much of an inflated logZ is a CE-invisible common-mode gauge vs. a real centered margin, and checks for low-rank head collapse. | [`zloss_row_center_probe.md`](zloss_row_center_probe.md) |
| `rare_token_nll_probe.py` | Audits a checkpoint's rare-token tail by computing per-token NLL on a fixed held-out panel bucketed by on-panel frequency, to detect z-loss over-compression damage. | [`rare_token_nll_probe.md`](rare_token_nll_probe.md) |

## Inference / chat / generation / eval

| tool | what it does | doc |
|------|--------------|-----|
| `chat_neo.py` | Interactive streaming chat/roleplay REPL for testing a checkpoint (custom `.pt` via neo_common or GGUF via llama_cpp), with raw-completion and chat-template (mara/llama/chatml) modes plus slash-commands. | [`chat_neo.md`](chat_neo.md) |
| `generate_neo.py` | Interactive REPL and non-interactive driver for generation, chat, standard benchmarks (HellaSwag/MMLU/GSM8K/WikiText), checkpoint sweeps, and a per-layer KEEL activation-RMS probe. | [`generate_neo.md`](generate_neo.md) |
| `eval.py` | Runs standard LLM benchmarks (HellaSwag, MMLU, ARC-Easy/Challenge, GSM8K, HumanEval) on a checkpoint and writes a JSON results file. | [`eval.md`](eval.md) |
| `test_kv_cache.py` | Benchmark harness that runs `chat_neo.py` twice (KV cache off vs on) and reports the wall-clock speedup. | [`test_kv_cache.md`](test_kv_cache.md) |

## Data pipeline

| tool | what it does | doc |
|------|--------------|-----|
| `dataset_explorer.py` | Read-only inspector for large Parquet/JSONL/`.jsonl.zst` dataset files (and directories) with metadata caching, byte-offset indexing, sampling, search, and stats. | [`dataset_explorer.md`](dataset_explorer.md) |
| `pre_tokenize.py` | Tokenises JSON/JSONL/Parquet/Scriptorium datasets into fixed-size `.npy` token shards with a resumable manifest and token cache. | [`pre_tokenize.md`](pre_tokenize.md) |

## Coherence metrics

The coherence-metrics family scores text degeneration on already-generated output (no model load):
`coherence_metrics.py` is the library, `recompute_coherence_metrics.py` re-runs it over a stored
log, and `redact_coherence_log.py` strips the raw text before a log leaves the box.

| tool | what it does | doc |
|------|--------------|-----|
| `coherence_metrics.py` | Pure-Python library of text-degeneration metrics (repetition, diversity, entity continuity, semantic drift, nonword rate, entropy ratio) computed on already-generated text — no model load. | [`coherence_metrics.md`](coherence_metrics.md) |
| `recompute_coherence_metrics.py` | Re-runs the coherence-metric library over an existing `coherence_log.jsonl` using the frozen stored text (no model load), rewriting per-prompt metrics + the aggregate, then auto-redacts the output. | [`recompute_coherence_metrics.md`](recompute_coherence_metrics.md) |
| `redact_coherence_log.py` | Strips raw generated text from `coherence_log.jsonl` files by replacing `per_prompt[*].text` with `<redacted>`, preserving all metrics and metadata. | [`redact_coherence_log.md`](redact_coherence_log.md) |
