# generate_neo.py
> Interactive REPL + non-interactive batch driver for loading a model checkpoint and doing inference: free-form/chat generation, standard benchmarks (HellaSwag, MMLU, GSM8K, WikiText perplexity), checkpoint sweeps, and a per-layer activation-RMS probe for KEEL diagnostics.

## What it does
Loads a checkpoint via `neo_common.load_model_and_tokenizer` and the tokenizer abstraction, then either drops into an interactive command loop (`CommandFramework`) or runs one of several non-interactive batch modes and exits. Interactive commands cover text/chat generation, single-shot benchmarks, and interactive sweeps. Non-interactive modes (selected by flags) cover the HellaSwag sweep, the coherence-metric sweep / single-step coherence eval, and the `activation-probe` mode. The activation probe is part of the KEEL diagnostics family — it dumps per-block residual + sub-layer forward RMS to JSON (siblings: the WD-waste / Newton-Schulz / body-norm investigation tooling that reasons about KEEL branch gains and activation scale).

This is the main hands-on inference/eval entry point for mara checkpoints. Reach for it to chat with a model, score a checkpoint on a benchmark, sweep a run's checkpoints over training, or inspect activation magnitudes for a KEEL run.

## Usage
```bash
# Interactive REPL (auto-selects most recent model_*.pt if given a dir)
python generate_neo.py --model_path /path/to/run_dir
python generate_neo.py --model_path /path/to/run_dir/model_step_010000.pt --temp 0.7 --top_p 0.9

# Non-interactive HellaSwag sweep over a run's checkpoints (500M-token milestones)
python generate_neo.py --model_path /path/to/run_dir --hella_sweep --token_interval 500

# Coherence-metric sweep, then a one-off single-step coherence eval
python generate_neo.py --model_path /path/to/run_dir --coherence_sweep
python generate_neo.py --model_path /path/to/run_dir --coherence_step 10000 --coherence_force

# KEEL activation-RMS probe on the latest checkpoint -> JSON
python generate_neo.py --model_path /path/to/run_dir --mode activation-probe \
    --probe_seq_len 8192 --probe_batch_size 8
```

Inside the interactive REPL, type `help` for commands. Notable ones: `prompt` / `chat` / `dprompt` (generate), `hella`, `mmlu`, `ppl`, `gsm8k`, `hella_sweep`, `coherence_sweep`, `load` (swap checkpoint), `export` (model `.pt` → raw `.bin`), `temp`, `top_p`, `size`, `batch`, `user`, `ls`/`cd` (browse prompt dir), `cls`, `exit`.

## Key arguments
| flag | default | meaning |
|---|---|---|
| `--model_path` | None (required) | Checkpoint `.pt` file, or a directory (auto-selects highest-step `model_*.pt`). Must be a directory for the sweep/coherence modes. |
| `--gen_size` | `"1024"` | Tokens to generate (interactive default generation length). |
| `--prompt_file` | `ev4.yaml` | Prompt YAML loaded from the prompt dir (`../xn/gen/`). |
| `--temp` | `0.7` | Sampling temperature. |
| `--top_p` | `0.9` | Top-p nucleus sampling. |
| `--full` | off | Use fp32 instead of half precision (half is the default; `--full` disables it). |
| `--tok_kind` | None (auto) | Tokenizer kind: `llama` or `hf`; auto-detected from checkpoint if omitted. |
| `--tok_path` | None (auto) | Tokenizer files path; auto-detected from checkpoint if omitted. |
| `--special_tokens` | None (auto) | Path to special-tokens JSON (needed for chat-format generation). |
| `--batch_size` | `16` | Batch size for evaluations. |
| `--gpu` | None | GPU index (`-1` = last). |
| `--shard_strategy` | `none` | `auto` / `balanced` / `none` (Accelerate multi-GPU sharding). |
| `--max_memory` | None | Per-GPU memory cap when sharding, e.g. `14GiB`. |
| `--qk_norm_mode` | None | `none` / `before_rope` / `after_rope_legacy` / `after_rope_fixed`. |
| `--use_keel` | off | Force-enable KEEL (Highway-style Post-LN) when the checkpoint config omits it. |
| `--hella_sweep` | off | Non-interactive HellaSwag sweep over checkpoints in the dir; uses `--token_interval`. |
| `--token_interval` | `500` | Sweep milestone spacing in millions of tokens. |
| `--dp_groups` | None | Data-parallel sweep (with `--hella_sweep`): `auto`, or explicit GPU worker groups like `0;5,1;6,2` (`;` between groups, `,` within, biggest GPU first in a group). Spawns one worker subprocess per group (`CUDA_VISIBLE_DEVICES`-pinned; multi-GPU groups use balanced sharding), shards the examples across workers, merges partial scores into one log line. `auto` sizes groups from the checkpoint file size and per-GPU VRAM. A step with any failed worker is not recorded and re-evaluates next run. Host RAM must fit (workers × on-disk checkpoint size) during concurrent loads. |
| `--coherence_sweep` | off | Non-interactive coherence-metric sweep over checkpoints. |
| `--coherence_step N` | None | One-off coherence eval on a single step (append to `coherence_log.jsonl`). |
| `--coherence_force` | off | Re-evaluate a step already present in the coherence log. |
| `--coherence_prompts` | `./coherence_prompts.json` | Coherence prompt-bank JSON. |
| `--coherence_gen_size` / `--coherence_temp` / `--coherence_top_p` / `--coherence_seed` | `512` / `0.7` / `0.9` / `42` | Coherence sampling params (seed makes trajectories deterministic across checkpoints). |
| `--mode activation-probe` | None | KEEL per-layer forward-RMS probe; dumps JSON and exits. |
| `--probe_seq_len` / `--probe_batch_size` | `8192` / `8` | Activation-probe calibration batch shape. |
| `--probe_data_root` | `../../notebooks/datasets/tokenized/llama/` | Tokenized data root for the probe (resolved from CWD or ckpt dir). |
| `--probe_output` | None | Probe JSON path; default `<ckpt_dir>/activation_probe_step_<N>.json`. |

## Notes
- Run from the `tools/` directory: it inserts `../common_fsdp2` (and `../saved_code` for FSDP1 ckpts) on `sys.path`, writes logs under `./logs`, and resolves several paths (prompt dir `../xn/gen/`, HellaSwag data `./hellaswag/`, leaderboard `hellaswag/hella_leaderboard.json`) relative to CWD.
- Sweep modes parse `val_log.txt` in the run dir for step→token mapping, auto-detect the checkpoint save interval from `model_step_*.pt` filenames (needs ≥2 checkpoints), append results (HellaSwag → `hellaswag_log.txt`; coherence → `coherence_log.jsonl`), and skip already-evaluated steps. The coherence sweep also writes a redacted copy via `redact_coherence_log` — relevant because coherence generations are produced from unfiltered prompt-bank text; treat `coherence_log.jsonl` and any generated output as sensitive.
- Benchmarks pull datasets from HuggingFace (`datasets`): MMLU = `cais/mmlu`, GSM8K = `gsm8k`, WikiText = `wikitext-{2,103}-raw-v1`; first run downloads/caches them. GPU + bf16 autocast assumed for CUDA paths.
- Generation, chat rendering, and per-token stats live in `neo_common` (`stream_generate_kv`, `generate_with_stats`, `load_prompt`, `load_yaml_chat_prompt`, `render_chat_for_completion`); coherence metrics come from `coherence_metrics.compute_all` / `aggregate`. Chat mode warns if the required special tokens aren't registered in the tokenizer.
- `--model_path` is required for every mode. Sweep/coherence/probe modes additionally require `--model_path` to be a directory.
